#!/usr/bin/env python3
"""Generate ligands for a user-supplied apo pocket.

The released Apo2Mol model was trained with paired apo/holo structures, but the
holo coordinates are not used as a condition in ``sample_diffusion``.  For
custom inference this script fills the compatibility-only holo fields with the
apo coordinates and asks the user for the desired ligand atom count.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_scatter import scatter_mean

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import (
    FOLLOW_BATCH,
    ApoHoloLigandData,
    torchify_dict,
)
from graphbap.bapnet import BAPNet
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.data import PDBProtein, compute_residue_transforms


MAX_LIGAND_ATOMS = 150


def _resolve_repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _dummy_ligand_dict(num_atoms: int, center: np.ndarray) -> dict[str, Any]:
    """Create shape-compatible ligand fields without supplying a reference ligand."""
    return {
        "smiles": "",
        "element": np.full(num_atoms, 6, dtype=np.int64),
        "pos": np.repeat(center[None, :], num_atoms, axis=0).astype(np.float32),
        "bond_index": np.empty((2, 0), dtype=np.int64),
        "bond_type": np.empty((0,), dtype=np.int64),
        "center_of_mass": center.astype(np.float32),
        "atom_feature": np.zeros((num_atoms, 8), dtype=np.int64),
        "hybridization": ["SP3"] * num_atoms,
    }


def build_custom_data(apo_pocket: str | Path, ligand_num_atoms: int):
    """Parse and featurize an apo pocket as an Apo2Mol inference data object."""
    apo_pocket = Path(apo_pocket).expanduser().resolve()
    if not apo_pocket.is_file():
        raise FileNotFoundError(f"Apo pocket PDB does not exist: {apo_pocket}")
    if not 1 <= ligand_num_atoms <= MAX_LIGAND_ATOMS:
        raise ValueError(
            f"--ligand-num-atoms must be between 1 and {MAX_LIGAND_ATOMS}; "
            f"got {ligand_num_atoms}."
        )

    try:
        protein = PDBProtein(str(apo_pocket))
    except KeyError as exc:
        raise ValueError(
            "The apo pocket contains an ATOM record with a non-standard residue "
            f"name ({exc.args[0]}). Keep only the 20 standard amino acids."
        ) from exc

    if not protein.atoms:
        raise ValueError(f"No protein ATOM records were found in {apo_pocket}.")

    incomplete = []
    for index, residue in enumerate(protein.residues):
        atom_names = {protein.atom_name[i] for i in residue["atoms"]}
        missing = sorted({"N", "CA", "C"} - atom_names)
        if missing:
            incomplete.append((index, residue["name"], missing))
    if incomplete:
        preview = ", ".join(
            f"#{idx} {name} missing {'/'.join(missing)}"
            for idx, name, missing in incomplete[:5]
        )
        raise ValueError(
            "Every pocket residue needs N, CA, and C atoms for Apo2Mol's residue "
            f"transform. Invalid residues: {preview}"
        )

    apo_dict = protein.to_dict_atom()
    center = apo_dict["pos"].mean(axis=0)
    ligand_dict = _dummy_ligand_dict(ligand_num_atoms, center)
    data = ApoHoloLigandData.from_apo_holo_ligand_dicts(
        apo_dict=torchify_dict(apo_dict),
        holo_dict=torchify_dict(apo_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )

    (
        data.protein_rotations,
        _,
        data.protein_translations,
        data.protein_chi_apo,
        data.protein_chi_holo,
        data.protein_chi_mask,
    ) = compute_residue_transforms(
        protein_pos_apo=data.protein_pos,
        protein_pos_holo=data.protein_pos_holo,
        protein_atom_name=data.protein_atom_name,
        protein_atom_to_aa_name=data.protein_atom_to_aa_name,
        protein_atom_to_aa_group=data.protein_atom_to_aa_group,
    )

    num_residues = int(data.protein_atom_to_aa_group.max()) + 1
    if data.protein_translations.size(0) != num_residues:
        raise ValueError(
            "Could not build transforms for every pocket residue: "
            f"expected {num_residues}, got {data.protein_translations.size(0)}."
        )

    data.apo_filename = str(apo_pocket)
    data.holo_filename = str(apo_pocket)
    data.ligand_filename = f"custom_{ligand_num_atoms}_atoms"
    data.custom_inference = True
    data.custom_ligand_num_atoms = ligand_num_atoms

    data = trans.FeaturizeProteinAtom()(data)
    data = trans.FeaturizeLigandAtom("add_aromatic")(data)
    return data


def load_models(
    sampling_config,
    training_config,
    device: torch.device,
    checkpoint: str | Path | None = None,
    pminet_checkpoint: str | Path | None = None,
):
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(
        training_config.data.transform.ligand_atom_mode
    )

    main_checkpoint = _resolve_repo_path(
        checkpoint if checkpoint is not None else sampling_config.model.checkpoint
    )
    pminet_path = _resolve_repo_path(
        pminet_checkpoint
        if pminet_checkpoint is not None
        else training_config.net_cond.ckpt_path
    )
    if not main_checkpoint.is_file():
        raise FileNotFoundError(f"Apo2Mol checkpoint not found: {main_checkpoint}")
    if not pminet_path.is_file():
        raise FileNotFoundError(f"PMINet checkpoint not found: {pminet_path}")

    net_cond = BAPNet(
        ckpt_path=str(pminet_path),
        hidden_nf=training_config.net_cond.hidden_dim,
    ).to(device)
    model = ScorePosNet3D(
        training_config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)

    checkpoint_data = torch.load(main_checkpoint, map_location=device)
    model_state = {
        key.removeprefix("model."): value
        for key, value in checkpoint_data["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state, strict=True)
    model.eval()
    net_cond.eval()
    return model, net_cond, main_checkpoint, pminet_path


def _to_numpy(tensor: torch.Tensor, dtype=None) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    return array.astype(dtype) if dtype is not None else array


def sample_custom_data(
    model: ScorePosNet3D,
    net_cond: BAPNet,
    data,
    ligand_num_atoms: int,
    num_samples: int,
    batch_size: int,
    num_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run fixed-size custom sampling synchronously and return eval-compatible data."""
    if num_samples < 1:
        raise ValueError("--num-samples must be at least 1.")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if num_steps < 1 or num_steps > model.num_timesteps:
        raise ValueError(
            f"--num-steps must be in [1, {model.num_timesteps}]; got {num_steps}."
        )

    output: dict[str, Any] = {
        "data": data,
        "pred_protein_pos": [],
        "pred_protein_pos_rmsd": [],
        "pred_protein_pos_tmscore": [],
        "pred_ligand_pos": [],
        "pred_ligand_v": [],
        "pred_protein_pos_traj": [],
        "pred_ligand_ligand_pos_traj": [],
        "pred_ligand_v_traj": [],
        "time": [],
    }

    completed = 0
    while completed < num_samples:
        current_batch_size = min(batch_size, num_samples - completed)
        batch = Batch.from_data_list(
            [data.clone() for _ in range(current_batch_size)],
            follow_batch=FOLLOW_BATCH,
        ).to(device)
        batch_protein = batch.protein_element_batch
        batch_ligand = torch.repeat_interleave(
            torch.arange(current_batch_size, device=device), ligand_num_atoms
        )
        protein_counts = torch.bincount(
            batch_protein, minlength=current_batch_size
        ).tolist()
        protein_offsets = np.cumsum([0] + protein_counts)
        ligand_offsets = np.arange(current_batch_size + 1) * ligand_num_atoms

        center = scatter_mean(batch.protein_pos, batch_protein, dim=0)
        init_ligand_pos = center[batch_ligand] + torch.randn(
            (current_batch_size * ligand_num_atoms, 3), device=device
        )
        uniform_logits = torch.zeros(
            (current_batch_size * ligand_num_atoms, model.num_classes),
            device=device,
        )
        init_ligand_v = log_sample_categorical(uniform_logits)

        started = time.time()
        with torch.no_grad():
            sampled = model.sample_diffusion(
                data=batch,
                protein_pos_apo=batch.protein_pos,
                protein_pos_holo=batch.protein_pos_holo,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=False,
                center_pos_mode="protein",
                net_cond=net_cond,
                cond_dim=model.config.cond_dim,
            )
        elapsed = time.time() - started

        final_protein = _to_numpy(sampled["protein_pos"], np.float32)
        final_ligand = _to_numpy(sampled["ligand_pos"], np.float64)
        final_types = _to_numpy(sampled["v"])
        protein_steps = sampled["protein_pos_traj"]
        ligand_steps = sampled["ligand_pos_traj"]
        type_steps = sampled["v_traj"]

        for local_index in range(current_batch_size):
            p_start, p_end = protein_offsets[local_index : local_index + 2]
            l_start, l_end = ligand_offsets[local_index : local_index + 2]
            output["pred_protein_pos"].append(final_protein[p_start:p_end])
            output["pred_ligand_pos"].append(final_ligand[l_start:l_end])
            output["pred_ligand_v"].append(final_types[l_start:l_end])
            output["pred_protein_pos_rmsd"].append(
                _to_numpy(sampled["protein_pos_rmsd"][local_index])
            )
            output["pred_protein_pos_tmscore"].append(
                _to_numpy(sampled["protein_pos_tmscore"][local_index])
            )
            output["pred_protein_pos_traj"].append(
                np.stack(
                    [_to_numpy(step[p_start:p_end], np.float64) for step in protein_steps]
                )
            )
            output["pred_ligand_ligand_pos_traj"].append(
                np.stack(
                    [_to_numpy(step[l_start:l_end], np.float64) for step in ligand_steps]
                )
            )
            output["pred_ligand_v_traj"].append(
                np.stack([_to_numpy(step[l_start:l_end]) for step in type_steps])
            )
            output["time"].append(elapsed / current_batch_size)
        completed += current_batch_size

    return output


def write_predicted_pocket(
    source_pdb: str | Path, coordinates: np.ndarray, destination: str | Path
) -> None:
    """Write generated pocket coordinates while retaining the input PDB records."""
    source_pdb = Path(source_pdb)
    destination = Path(destination)
    output_lines = []
    atom_index = 0
    saw_end = False
    for line in source_pdb.read_text().splitlines(keepends=True):
        record = line[0:6].strip()
        if record == "ENDMDL":
            output_lines.append(line)
            saw_end = True
            break
        if record == "ATOM":
            if atom_index >= len(coordinates):
                raise ValueError("Predicted pocket has fewer coordinates than the input PDB.")
            x, y, z = coordinates[atom_index]
            line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            atom_index += 1
        output_lines.append(line)
    if atom_index != len(coordinates):
        raise ValueError(
            "Predicted pocket coordinate count does not match the input PDB: "
            f"{len(coordinates)} versus {atom_index}."
        )
    if not saw_end and not any(line.startswith("END") for line in output_lines):
        output_lines.append("END\n")
    destination.write_text("".join(output_lines))


def export_structures(
    result: dict[str, Any], apo_pocket: str | Path, output_dir: str | Path
) -> list[dict[str, Any]]:
    """Reconstruct generated ligands and write predicted protein pockets."""
    from utils import reconstruct

    output_dir = Path(output_dir)
    structure_dir = output_dir / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (ligand_pos, ligand_v, protein_pos) in enumerate(
        zip(
            result["pred_ligand_pos"],
            result["pred_ligand_v"],
            result["pred_protein_pos"],
        )
    ):
        atom_numbers = trans.get_atomic_number_from_index(
            ligand_v, mode="add_aromatic"
        )
        aromatic = trans.is_aromatic_from_index(ligand_v, mode="add_aromatic")
        ligand_path = structure_dir / f"generated_{index:03d}.sdf"
        pocket_path = structure_dir / f"predicted_pocket_{index:03d}.pdb"
        record: dict[str, Any] = {
            "sample_index": index,
            "ligand_sdf": None,
            "predicted_pocket_pdb": str(pocket_path),
        }
        write_predicted_pocket(apo_pocket, protein_pos, pocket_path)
        try:
            molecule = reconstruct.reconstruct_from_generated(
                ligand_pos, atom_numbers, aromatic
            )
            molecule.SetProp("_Name", f"Apo2Mol_custom_{index:03d}")
            writer = Chem.SDWriter(str(ligand_path))
            writer.write(molecule)
            writer.close()
            check = Chem.SDMolSupplier(str(ligand_path), removeHs=False)[0]
            if check is None:
                raise ValueError("RDKit could not read the reconstructed molecule.")
            record.update(
                {
                    "ligand_sdf": str(ligand_path),
                    "smiles": Chem.MolToSmiles(check),
                    "num_atoms_with_hydrogen": check.GetNumAtoms(),
                }
            )
        except Exception as exc:  # keep raw tensors when chemical reconstruction fails
            record["reconstruction_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    return records


def add_vina_scores(
    structures: list[dict[str, Any]], output_dir: str | Path
) -> dict[str, Any]:
    """Score every reconstructed final pocket-ligand pair with AutoDock Vina."""
    summary: dict[str, Any] = {
        "enabled": True,
        "scoring_function": "vina",
        "mode": "score_only",
        "units": "kcal/mol",
        "receptor": "predicted_pocket_pdb",
        "num_successful": 0,
        "num_failed": 0,
    }
    output_dir = Path(output_dir)

    try:
        from utils.evaluation.docking_vina import VinaDockingTask
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for record in structures:
            record["vina"] = {"mode": "score_only", "error": error}
        summary["num_failed"] = len(structures)
        summary["error"] = error
        return summary

    for record in structures:
        ligand_path = record.get("ligand_sdf")
        pocket_path = record.get("predicted_pocket_pdb")
        if ligand_path is None:
            error = "Ligand reconstruction failed; no SDF is available for scoring."
            record["vina"] = {"mode": "score_only", "error": error}
            summary["num_failed"] += 1
            continue

        try:
            molecule = Chem.SDMolSupplier(str(ligand_path), removeHs=False)[0]
            if molecule is None:
                raise ValueError("RDKit could not read the reconstructed ligand SDF.")

            # VinaDockingTask creates receptor preparation files beside its receptor.
            # Work on a temporary copy so custom output folders contain only final files.
            with tempfile.TemporaryDirectory(
                prefix="apo2mol_vina_", dir=output_dir
            ) as temporary_dir:
                temporary_dir = Path(temporary_dir)
                receptor_copy = temporary_dir / "receptor.pdb"
                shutil.copyfile(pocket_path, receptor_copy)
                task = VinaDockingTask(
                    str(receptor_copy), molecule, tmp_dir=str(temporary_dir)
                )
                result = task.run(mode="score_only")[0]

            record["vina"] = {
                "mode": "score_only",
                "affinity_kcal_mol": float(result["affinity"]),
            }
            summary["num_successful"] += 1
        except Exception as exc:
            record["vina"] = {
                "mode": "score_only",
                "error": f"{type(exc).__name__}: {exc}",
            }
            summary["num_failed"] += 1

    return summary


def _metric_summary(values: list[float]) -> dict[str, float | int | None]:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": int(finite_values.size),
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
    }


def evaluate_custom_samples(
    result: dict[str, Any], structures: list[dict[str, Any]]
) -> dict[str, Any]:
    """Calculate selected reference-free metrics from ``eval_split.py``."""
    from utils.evaluation import analyze, scoring_func

    records = {int(record["sample_index"]): record for record in structures}
    num_samples = len(result["pred_ligand_pos"])
    stable_molecules = 0
    stable_atoms = 0
    total_atoms = 0
    reconstructed = 0
    complete = 0
    chemistry_results = []
    ring_sizes = []
    vina_scores = []

    for index, (ligand_pos, ligand_v) in enumerate(
        zip(result["pred_ligand_pos"], result["pred_ligand_v"])
    ):
        atom_numbers = trans.get_atomic_number_from_index(
            ligand_v, mode="add_aromatic"
        )
        molecule_stable, sample_stable_atoms, sample_num_atoms = (
            analyze.check_stability(ligand_pos, atom_numbers)
        )
        stable_molecules += int(molecule_stable)
        stable_atoms += int(sample_stable_atoms)
        total_atoms += int(sample_num_atoms)

        sample_metrics: dict[str, Any] = {
            "molecule_stable": bool(molecule_stable),
            "stable_atoms": int(sample_stable_atoms),
            "num_atoms": int(sample_num_atoms),
            "atom_stability": (
                float(sample_stable_atoms / sample_num_atoms)
                if sample_num_atoms
                else None
            ),
            "reconstruction_success": False,
            "complete": False,
        }
        record = records.get(index)
        if record is None:
            continue
        record["evaluation"] = sample_metrics
        vina = record.get("vina", {})
        if "affinity_kcal_mol" in vina:
            vina_scores.append(float(vina["affinity_kcal_mol"]))

        ligand_path = record.get("ligand_sdf")
        if ligand_path is None:
            continue
        try:
            molecule = Chem.SDMolSupplier(str(ligand_path), removeHs=False)[0]
            if molecule is None:
                raise ValueError("RDKit could not read the reconstructed ligand SDF.")
            reconstructed += 1
            sample_metrics["reconstruction_success"] = True
            smiles = Chem.MolToSmiles(molecule)
            is_complete = "." not in smiles
            sample_metrics["complete"] = is_complete
            if not is_complete:
                continue
            complete += 1

            chemistry = scoring_func.get_chem(molecule)
            sample_ring_sizes = {
                str(size): int(count)
                for size, count in sorted(chemistry["ring_size"].items())
            }
            sample_metrics["chemistry"] = {
                "qed": float(chemistry["qed"]),
                "sa": float(chemistry["sa"]),
                "logp": float(chemistry["logp"]),
                "lipinski": int(chemistry["lipinski"]),
                "ring_sizes": sample_ring_sizes,
            }
            chemistry_results.append(sample_metrics["chemistry"])
            ring_sizes.append({int(size) for size in chemistry["ring_size"]})
        except Exception as exc:
            sample_metrics["chemistry_error"] = f"{type(exc).__name__}: {exc}"

    def fraction(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    chemistry_summary = {
        name: _metric_summary([metrics[name] for metrics in chemistry_results])
        for name in ("qed", "sa", "logp", "lipinski")
    }
    ring_size_ratio = {
        str(size): fraction(
            sum(size in sample_ring_sizes for sample_ring_sizes in ring_sizes),
            len(ring_sizes),
        )
        for size in range(3, 10)
    }
    return {
        "num_samples": num_samples,
        "molecule_stability": fraction(stable_molecules, num_samples),
        "atom_stability": fraction(stable_atoms, total_atoms),
        "reconstruction_rate": fraction(reconstructed, num_samples),
        "completeness_rate": fraction(complete, num_samples),
        "num_chemistry_evaluated": len(chemistry_results),
        "chemistry": chemistry_summary,
        "ring_size_ratio": ring_size_ratio,
        "vina_score_only_kcal_mol": _metric_summary(vina_scores),
    }


def run_custom_inference(
    *,
    apo_pocket: str | Path,
    ligand_num_atoms: int,
    output_dir: str | Path,
    num_samples: int = 5,
    batch_size: int = 5,
    num_steps: int | None = None,
    device_name: str = "cuda:0",
    config_path: str | Path = REPO_ROOT / "configs" / "sampling.yaml",
    train_config_path: str | Path = REPO_ROOT / "configs" / "training.yaml",
    checkpoint: str | Path | None = None,
    pminet_checkpoint: str | Path | None = None,
    seed: int | None = None,
    reconstruct_structures: bool = True,
    calculate_vina: bool = True,
    overwrite: bool = True,
) -> dict[str, Any]:
    apo_pocket = Path(apo_pocket).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    result_path = output_dir / "result_0.pt"
    if result_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {result_path}. Remove --no-overwrite or use "
            "another directory."
        )

    config_path = _resolve_repo_path(config_path)
    train_config_path = _resolve_repo_path(train_config_path)
    sampling_config = misc.load_config(str(config_path))
    training_config = misc.load_config(str(train_config_path))
    actual_seed = sampling_config.sample.seed if seed is None else seed
    actual_steps = sampling_config.sample.num_steps if num_steps is None else num_steps
    misc.seed_all(actual_seed)

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device_name} was requested, but CUDA is not available."
        )
    device = torch.device(device_name)
    data = build_custom_data(apo_pocket, ligand_num_atoms)
    model, net_cond, main_checkpoint, pminet_path = load_models(
        sampling_config,
        training_config,
        device,
        checkpoint=checkpoint,
        pminet_checkpoint=pminet_checkpoint,
    )
    result = sample_custom_data(
        model=model,
        net_cond=net_cond,
        data=data,
        ligand_num_atoms=ligand_num_atoms,
        num_samples=num_samples,
        batch_size=batch_size,
        num_steps=actual_steps,
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        generated_files = [
            result_path,
            output_dir / "metadata.json",
            output_dir / "sampling.yaml",
        ]
        structure_dir = output_dir / "structures"
        if structure_dir.is_dir():
            generated_files.extend(structure_dir.glob("generated_*.sdf"))
            generated_files.extend(structure_dir.glob("predicted_pocket_*.pdb"))
        for generated_file in generated_files:
            if generated_file.is_file():
                generated_file.unlink()
    torch.save(result, result_path)
    shutil.copyfile(config_path, output_dir / "sampling.yaml")
    structures = (
        export_structures(result, apo_pocket, output_dir)
        if reconstruct_structures
        else []
    )
    vina_scoring = (
        add_vina_scores(structures, output_dir)
        if calculate_vina
        else {"enabled": False}
    )
    evaluation = evaluate_custom_samples(result, structures)
    metadata = {
        "mode": "custom_apo_pocket",
        "apo_pocket": str(apo_pocket),
        "ligand_num_atoms": ligand_num_atoms,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "num_steps": actual_steps,
        "seed": actual_seed,
        "device": str(device),
        "checkpoint": str(main_checkpoint),
        "pminet_checkpoint": str(pminet_path),
        "result": str(result_path),
        "structures": structures,
        "vina_scoring": vina_scoring,
        "evaluation": evaluation,
        "note": (
            "protein_pos_holo is set equal to the supplied apo pocket for custom "
            "inference; reported protein RMSD is therefore displacement from apo, "
            "not accuracy against an unseen holo structure."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ligand-num-atoms", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Diffusion steps (default: value from configs/sampling.yaml, normally 1000).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "sampling.yaml"
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=REPO_ROOT / "configs" / "training.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--pminet-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-reconstruct", action="store_true")
    parser.add_argument(
        "--no-vina",
        action="store_true",
        help="Skip Vina score-only evaluation of reconstructed structures.",
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite custom outputs (default).",
    )
    overwrite_group.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail if result_0.pt already exists in the output directory.",
    )
    parser.set_defaults(overwrite=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fixed-size ligands from a user-supplied apo pocket PDB."
    )
    parser.add_argument("--apo-pocket", required=True, type=Path)
    add_common_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = run_custom_inference(
        apo_pocket=args.apo_pocket,
        ligand_num_atoms=args.ligand_num_atoms,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        device_name=args.device,
        config_path=args.config,
        train_config_path=args.train_config,
        checkpoint=args.checkpoint,
        pminet_checkpoint=args.pminet_checkpoint,
        seed=args.seed,
        reconstruct_structures=not args.no_reconstruct,
        calculate_vina=not args.no_vina,
        overwrite=args.overwrite,
    )
    print(f"Custom sampling completed. Results saved to {metadata['result']}")


if __name__ == "__main__":
    main()
