# PocketDiff and TargetDiff Internal Cross Baselines

Code-only release of the two isolated baseline projects used in the PocketDiff bridge work.

This repository intentionally excludes model weights, `.pt/.ckpt` caches, raw datasets, generated molecules, and training runs. The original clean project directories remain available locally for those artifacts.

## Projects

- `pocketdiff_v48_clean_20261001/`: PocketDiff v4.8 clean baseline. This is the residue-level protein relaxation / flow-style PocketDiff codebase and includes its training, sampling, evaluation, geometry, and bridge interfaces.
- `targetdiff_internal_cross_final_20261003/`: TargetDiff baseline with the conservative Internal Cross modification. It keeps the original TargetDiff 27D protein atom representation, 13-class ligand representation, 1000-step Gaussian/categorical diffusion protocol, and UniTransformer backbone, while adding the time-conditioned layerwise protein-ligand residual branch.

## Artifacts excluded from GitHub

Weights and prepared data are intentionally not committed. They include multi-gigabyte files that are unsuitable for ordinary GitHub storage.

Local artifact locations:

- PocketDiff clean release: `/data/Model_coding/PocketDiff/pocketdiff_v48_clean_20261001/`
- TargetDiff clean release: `/data/Model_coding/PocketDiff/targetdiff_internal_cross_final_20261003/`

Use each project README for environment setup, training, sampling, and evaluation commands. Commands that require weights or prepared caches must be pointed at the corresponding local clean release.

## Upload policy

This repository contains source code, configuration, tests, scripts, and documentation only. No trained parameters or private/raw data are included.
