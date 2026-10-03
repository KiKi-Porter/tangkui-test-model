# PocketDiff v4.8 Clean Release

This directory is the self-contained release of the latest full-data
PocketDiff training run. It is deliberately separated from the development
worktree so later experiments do not mix code, checkpoints, and caches.

## Contents

```text
data/
  prepared_diffusion_full_v47.pt       # authoritative prepared cache
  prepared_diffusion_full_smoke.pt     # small smoke cache
runs/
  v48_full_dataset_epoch_equiv_20260930/
    checkpoint_best.pt                 # recommended checkpoint
    checkpoint_last.pt
    evaluation_20step.json
    ddp_report.json
  bridge_phase1_20261001/              # TargetDiff/PocketDiff bridge outputs
pocketdiff/                            # training, inference, geometry, tests
targetdiff-main/targetdiff-main/       # local TargetDiff source and weights
Apo2Mol-main/                          # adapter/source utilities
env/
  targetdiff_environment.yaml
  runtime.txt
```

The raw Apo2Mol structure directory is not copied into this release. The
prepared cache contains the canonical tensors required by the v4.8 training
and inference path. The split pickle is retained in `data/` for scripts that
need sample metadata.

## Model Contract

v4.8 is the released name for the `v47` architecture variant trained in
`v48_full_dataset_epoch_equiv_20260930`.

The model is a deterministic bridge-velocity model, not a classical DDPM.
The input protein is the apo structure at normalized bridge time `t=1`.
The target endpoint is holo at `t=0`. Training constructs a label-independent
current protein state and supervises the local velocity from that state toward
the next point on the apo-to-holo bridge.

At inference, the model performs 20 Euler updates:

```text
apo state
  -> model predicts residue-local translation/rotation velocity
  -> apply 1/20 of the update with local SE(3) geometry
  -> recompute residue frames
  -> repeat 20 times
```

The model is therefore closer to conditional flow matching / bridge
integration than to a noisy reverse DDPM chain. It does not start from random
protein noise. The ligand condition is supplied as a 13-class atom cloud.

### Architecture

1. TargetDiff-compatible protein atom input: 27 features per protein atom.
2. TargetDiff-compatible ligand input: 13 categorical atom classes and
   coordinates.
3. Read-only TargetDiff `UniTransformerO2TwoUpdateGeneral` encoder with
   `fix_x=True`; the encoder mixes atom-level protein-ligand context without
   changing coordinates.
4. Learned atom-to-residue attention pooling.
5. N-CA-C residue frames for local SE(3) actions.
6. Explicit local directional and ligand contact messages.
7. Peptide sequence-neighbor messages and gated sequence motion smoothing.
8. Residue motion head predicting local translation and rotation.
9. χ head exists in the architecture, but the released v48 checkpoint was
   trained backbone-only with χ updates disabled.
10. Updates preserve each residue's internal rigid geometry, but independent
    residue motions do not guarantee perfect global peptide geometry.

The v48 checkpoint has approximately 5.84 million parameters and uses
`hidden_dim=128`, `architecture_variant=v47`, `encoder_backend=targetdiff`,
`prediction_type=bridge_velocity`, and `target_steps=20`.

## Environment

The validated environment is the existing `targetdiff` conda environment.
Its exact package specification is copied to:

```text
env/targetdiff_environment.yaml
```

The validation runtime is recorded in `env/runtime.txt`. From the release
root:

```bash
conda env create -f env/targetdiff_environment.yaml
conda activate targetdiff
export PYTHONPATH="$PWD"
```

If the environment already exists, only activate it and set `PYTHONPATH`.

## Validate the Release

Run the complete clean-directory validation:

```bash
PYTHONPATH="$PWD" python scripts/validate_release.py
```

This validates the cache hash and split sizes, reloads the checkpoint,
executes one TargetDiff-compatible model forward, applies one local SE(3)
update, runs a one-step trajectory, and performs one CPU training update.
The report is written to `validation_report.json`.

## Training

The full prepared cache is already built. A four-GPU full-data run is:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
  PYTHONPATH="$PWD" \
  bash pocketdiff/run_v48_full_ddp.sh
```

The script uses the release-relative cache:

```text
data/prepared_diffusion_full_v47.pt
```

The default output directory is:

```text
runs/v48_full_dataset_epoch_equiv_20260930/
```

For a small local training smoke:

```bash
PYTHONPATH="$PWD" python pocketdiff/scripts/train_diffusion.py \
  --prepared-cache data/prepared_diffusion_full_smoke.pt \
  --output-dir runs/smoke \
  --device cpu \
  --train-count 1 \
  --valid-count 0 \
  --test-count 0 \
  --holdout-count 0 \
  --updates 1 \
  --batch-size 1 \
  --encoder-backend targetdiff \
  --architecture-variant v47 \
  --prediction-type bridge_velocity \
  --target-parameterization bridge_velocity \
  --target-steps 20 \
  --backbone-only \
  --disable-chi
```

## Inference and Evaluation

Evaluate the released checkpoint with the fixed 20-step protocol:

```bash
PYTHONPATH="$PWD" python pocketdiff/scripts/evaluate_diffusion.py \
  --checkpoint runs/v48_full_dataset_epoch_equiv_20260930/checkpoint_best.pt \
  --prepared-cache data/prepared_diffusion_full_v47.pt \
  --output runs/v48_full_dataset_epoch_equiv_20260930/evaluation_release.json \
  --device cuda \
  --valid-count 985 \
  --test-count 450 \
  --steps 20 \
  --batch-size 1
```

Export multi-model PDB trajectories:

```bash
PYTHONPATH="$PWD" python pocketdiff/scripts/export_diffusion_trajectory_pdb.py \
  --checkpoint runs/v48_full_dataset_epoch_equiv_20260930/checkpoint_best.pt \
  --prepared-cache data/prepared_diffusion_full_v47.pt \
  --output-dir runs/v48_full_dataset_epoch_equiv_20260930/trajectories_release \
  --device cuda \
  --steps 20
```

## TargetDiff Bridge

The bridge experiment uses official TargetDiff for ligand generation and
invokes PocketDiff once per ten TargetDiff reverse steps during `t=199..0`.
It compares TargetDiff-only and coupled sampling using the same random trace:

```bash
PYTHONPATH="$PWD" python pocketdiff/scripts/run_bridge_phase1.py --full \
  --prepared-cache data/prepared_diffusion_full_v47.pt \
  --targetdiff-checkpoint \
    targetdiff-main/targetdiff-main/pretrained_models/pretrained_diffusion.pt \
  --pocketdiff-checkpoint \
    runs/v48_full_dataset_epoch_equiv_20260930/checkpoint_best.pt \
  --output runs/bridge_phase1_20261001
```

Each sample contains one multi-model PDB for TargetDiff-only and one for the
coupled trajectory, plus a serialized `.pt` trajectory and
`comparison.json`.

## Important Limitations

The released checkpoint is an engineering baseline, not a successful
apo-to-holo predictor. In the ten-sample bridge experiment it produced finite,
nonzero protein updates, but the mean coupled backbone RMSD was worse than
the apo baseline. It was trained with clean/reference ligand conditions, so
TargetDiff late-stage noisy ligand conditioning is an out-of-distribution
bridge test. Do not interpret the bridge output as validated induced-fit
generation.

The authoritative checkpoint and cache are:

```text
runs/v48_full_dataset_epoch_equiv_20260930/checkpoint_best.pt
data/prepared_diffusion_full_v47.pt
```
