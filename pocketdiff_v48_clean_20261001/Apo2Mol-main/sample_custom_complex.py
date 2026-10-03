#!/usr/bin/env python3
"""Extract an apo pocket from a matched holo-ligand complex and run Apo2Mol.

The holo structure and ligand are used only to identify pocket residue sequence
positions.  The corresponding residues are taken from the apo structure and
are the sole structural condition supplied to ligand generation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from Bio.PDB import PDBIO, PDBParser, Select
from rdkit import Chem

from sample_custom_pocket import (
    add_common_arguments,
    run_custom_inference,
)
from utils.data import PDBProtein, get_align_rotran_kabsch


STANDARD_RESIDUES = set(PDBProtein.AA_NAME_SYM)


def _protein_chains(structure):
    model = next(structure.get_models())
    chains = []
    for chain in model:
        residues = [
            residue
            for residue in chain
            if residue.id[0] == " " and residue.resname in STANDARD_RESIDUES
        ]
        if residues:
            sequence = "".join(PDBProtein.AA_NAME_SYM[r.resname] for r in residues)
            chains.append((chain.id, residues, sequence))
    return chains


def resolve_exact_chain_pair(apo_structure, holo_structure):
    apo_chains = _protein_chains(apo_structure)
    holo_chains = _protein_chains(holo_structure)
    matches = [
        (apo_item, holo_item)
        for apo_item in apo_chains
        for holo_item in holo_chains
        if apo_item[2] == holo_item[2]
    ]
    if not matches:
        apo_summary = ", ".join(f"{c}:{len(s)}" for c, _, s in apo_chains)
        holo_summary = ", ".join(f"{c}:{len(s)}" for c, _, s in holo_chains)
        raise ValueError(
            "No apo/holo chain pair has an exactly identical standard-amino-acid "
            f"sequence. Apo chains [{apo_summary}]; holo chains [{holo_summary}]."
        )
    max_length = max(len(pair[0][2]) for pair in matches)
    longest = [pair for pair in matches if len(pair[0][2]) == max_length]
    if len(longest) != 1:
        candidates = ", ".join(
            f"apo {a[0]!r} / holo {h[0]!r} ({len(a[2])} aa)"
            for a, h in longest
        )
        raise ValueError(
            "Multiple equally long exact chain matches were found; the chain pair "
            f"cannot be selected unambiguously. Candidates: {candidates}"
        )
    return longest[0]


def load_ligand(path: str | Path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Ligand structure does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        molecule = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
    elif suffix == ".mol":
        molecule = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
    elif suffix == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
    elif suffix == ".pdb":
        molecule = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False)
    else:
        raise ValueError("--ligand must be an SDF, MOL, MOL2, or PDB file.")
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"RDKit could not read 3D ligand coordinates from {path}.")
    conformer = molecule.GetConformer()
    heavy_indices = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    if not heavy_indices:
        raise ValueError(f"No ligand heavy atoms were found in {path}.")
    heavy_coordinates = np.asarray(
        [list(conformer.GetAtomPosition(i)) for i in heavy_indices], dtype=np.float64
    )
    return path, molecule, heavy_coordinates


def _residue_center_of_mass(residue) -> np.ndarray:
    atoms = [
        atom
        for atom in residue.get_atoms()
        if (atom.element or "").strip().upper() not in {"H", "D"}
    ]
    if not atoms:
        return np.empty((0,), dtype=np.float64)
    coordinates = np.asarray([atom.coord for atom in atoms], dtype=np.float64)
    masses = np.asarray([atom.mass for atom in atoms], dtype=np.float64)
    return np.sum(coordinates * masses[:, None], axis=0) / masses.sum()


def select_pocket_positions(
    holo_residues: Iterable,
    ligand_coordinates: np.ndarray,
    radius: float,
) -> list[int]:
    if radius <= 0:
        raise ValueError("--pocket-radius must be positive.")
    selected = []
    for sequence_index, residue in enumerate(holo_residues):
        center_of_mass = _residue_center_of_mass(residue)
        if center_of_mass.size == 0:
            continue
        distances = np.linalg.norm(
            ligand_coordinates - center_of_mass[None, :], axis=1
        )
        if float(distances.min()) < radius:
            selected.append(sequence_index)
    if not selected:
        raise ValueError(
            f"No holo protein residues were found within {radius:.2f} A of the ligand."
        )
    return selected


def _validate_backbone(residues: list, label: str) -> None:
    invalid = []
    for residue in residues:
        names = {atom.name for atom in residue.get_atoms()}
        missing = sorted({"N", "CA", "C"} - names)
        if missing:
            invalid.append(
                f"{residue.parent.id}:{residue.id[1]}{residue.id[2].strip()} "
                f"{residue.resname} missing {'/'.join(missing)}"
            )
    if invalid:
        raise ValueError(
            f"{label} pocket residues need complete N/CA/C backbones. "
            + "; ".join(invalid[:5])
        )


class _ResidueSelector(Select):
    def __init__(self, model_id, chain_id, residue_ids):
        self.model_id = model_id
        self.chain_id = chain_id
        self.residue_ids = set(residue_ids)

    def accept_model(self, model):
        return int(model.id == self.model_id)

    def accept_chain(self, chain):
        return int(chain.id == self.chain_id)

    def accept_residue(self, residue):
        return int(residue.id in self.residue_ids)

    def accept_atom(self, atom):
        altloc = atom.get_altloc()
        return int(altloc in {" ", "A", "1"})


def save_selected_pocket(structure, residues: list, path: str | Path) -> None:
    if not residues:
        raise ValueError("Cannot write an empty pocket.")
    model_id = residues[0].parent.parent.id
    chain_id = residues[0].parent.id
    residue_ids = [residue.id for residue in residues]
    io = PDBIO()
    io.set_structure(structure)
    io.save(
        str(path),
        _ResidueSelector(model_id, chain_id, residue_ids),
        preserve_atom_numbering=False,
    )


def align_holo_to_apo(
    holo_residues: list, apo_residues: list
) -> tuple[np.ndarray, np.ndarray, float, float]:
    holo_ca = np.asarray([residue["CA"].coord for residue in holo_residues])
    apo_ca = np.asarray([residue["CA"].coord for residue in apo_residues])
    before = float(np.sqrt(np.mean(np.sum((holo_ca - apo_ca) ** 2, axis=1))))
    translation, rotation = get_align_rotran_kabsch(holo_ca, apo_ca)
    aligned = holo_ca @ rotation.T + translation
    after = float(np.sqrt(np.mean(np.sum((aligned - apo_ca) ** 2, axis=1))))
    return translation, rotation, before, after


def transform_structure(structure, translation, rotation):
    transformed = copy.deepcopy(structure)
    for atom in transformed.get_atoms():
        atom.coord = atom.coord @ rotation.T + translation
    return transformed


def save_aligned_ligand(molecule, translation, rotation, path: str | Path) -> None:
    molecule = Chem.Mol(molecule)
    conformer = molecule.GetConformer()
    for index in range(molecule.GetNumAtoms()):
        coordinate = np.asarray(conformer.GetAtomPosition(index), dtype=np.float64)
        x, y, z = coordinate @ rotation.T + translation
        conformer.SetAtomPosition(index, (float(x), float(y), float(z)))
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()


def prepare_apo_pocket(
    *,
    apo_pdb: str | Path,
    holo_pdb: str | Path,
    ligand: str | Path,
    preparation_dir: str | Path,
    pocket_radius: float = 10.0,
) -> dict:
    apo_pdb = Path(apo_pdb).expanduser().resolve()
    holo_pdb = Path(holo_pdb).expanduser().resolve()
    if not apo_pdb.is_file():
        raise FileNotFoundError(f"Apo protein PDB does not exist: {apo_pdb}")
    if not holo_pdb.is_file():
        raise FileNotFoundError(f"Holo protein PDB does not exist: {holo_pdb}")
    preparation_dir = Path(preparation_dir).expanduser().resolve()
    preparation_dir.mkdir(parents=True, exist_ok=True)

    parser = PDBParser(QUIET=True)
    apo_structure = parser.get_structure("apo", str(apo_pdb))
    holo_structure = parser.get_structure("holo", str(holo_pdb))
    (apo_chain_id, apo_residues, apo_sequence), (
        holo_chain_id,
        holo_residues,
        holo_sequence,
    ) = resolve_exact_chain_pair(apo_structure, holo_structure)
    assert apo_sequence == holo_sequence

    ligand_path, ligand_molecule, ligand_coordinates = load_ligand(ligand)
    selected_positions = select_pocket_positions(
        holo_residues, ligand_coordinates, pocket_radius
    )
    selected_apo = [apo_residues[index] for index in selected_positions]
    selected_holo = [holo_residues[index] for index in selected_positions]
    _validate_backbone(selected_apo, "Apo")
    _validate_backbone(selected_holo, "Holo")
    if len(selected_positions) < 3:
        raise ValueError("At least three pocket residues are required for alignment.")

    translation, rotation, rmsd_before, rmsd_after = align_holo_to_apo(
        selected_holo, selected_apo
    )
    aligned_holo_structure = transform_structure(
        holo_structure, translation, rotation
    )
    aligned_model = next(aligned_holo_structure.get_models())
    aligned_holo_chain = aligned_model[holo_chain_id]
    aligned_holo_residue_map = {residue.id: residue for residue in aligned_holo_chain}
    aligned_holo_residues = [
        aligned_holo_residue_map[residue.id] for residue in selected_holo
    ]

    apo_pocket_path = preparation_dir / "apo_pocket.pdb"
    holo_pocket_path = preparation_dir / "holo_pocket_aligned_to_apo.pdb"
    ligand_aligned_path = preparation_dir / "ligand_aligned_to_apo.sdf"
    save_selected_pocket(apo_structure, selected_apo, apo_pocket_path)
    save_selected_pocket(
        aligned_holo_structure, aligned_holo_residues, holo_pocket_path
    )
    save_aligned_ligand(
        ligand_molecule, translation, rotation, ligand_aligned_path
    )

    metadata = {
        "apo_pdb": str(apo_pdb),
        "holo_pdb": str(holo_pdb),
        "ligand": str(ligand_path),
        "apo_chain": apo_chain_id,
        "holo_chain": holo_chain_id,
        "sequence_identical": True,
        "sequence_length": len(apo_sequence),
        "pocket_radius_angstrom": pocket_radius,
        "pocket_distance_criterion": "residue_center_of_mass_to_ligand_heavy_atom",
        "pocket_sequence_positions_zero_based": selected_positions,
        "pocket_num_residues": len(selected_positions),
        "alignment_ca_rmsd_before": rmsd_before,
        "alignment_ca_rmsd_after": rmsd_after,
        "apo_pocket": str(apo_pocket_path),
        "holo_pocket_aligned_to_apo": str(holo_pocket_path),
        "ligand_aligned_to_apo": str(ligand_aligned_path),
        "generation_condition": "apo_pocket_only",
    }
    (preparation_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract corresponding apo pocket residues from an exact-sequence "
            "apo/holo pair, then generate fixed-size ligands from the apo pocket."
        )
    )
    parser.add_argument("--apo-pdb", required=True, type=Path)
    parser.add_argument("--holo-pdb", required=True, type=Path)
    parser.add_argument(
        "--ligand",
        required=True,
        type=Path,
        help="Bound holo ligand with 3D coordinates (SDF, MOL, MOL2, or PDB).",
    )
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Write aligned pocket inputs without running GPU generation.",
    )
    add_common_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    pocket_metadata = prepare_apo_pocket(
        apo_pdb=args.apo_pdb,
        holo_pdb=args.holo_pdb,
        ligand=args.ligand,
        preparation_dir=output_dir / "input_preparation",
        pocket_radius=args.pocket_radius,
    )
    if args.extract_only:
        print(
            "Pocket extraction completed. Files saved to "
            f"{output_dir / 'input_preparation'}"
        )
        return

    inference_metadata = run_custom_inference(
        apo_pocket=pocket_metadata["apo_pocket"],
        ligand_num_atoms=args.ligand_num_atoms,
        output_dir=output_dir,
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
    inference_metadata["mode"] = "custom_apo_holo_complex"
    inference_metadata["pocket_extraction"] = pocket_metadata
    (output_dir / "metadata.json").write_text(
        json.dumps(inference_metadata, indent=2) + "\n"
    )
    print(
        "Custom sampling completed. Results saved to "
        f"{inference_metadata['result']}"
    )


if __name__ == "__main__":
    main()
