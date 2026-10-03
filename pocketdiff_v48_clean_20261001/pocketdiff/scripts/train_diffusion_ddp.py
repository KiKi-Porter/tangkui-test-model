"""Train the PocketDiff diffusion core with torchrun/DDP.

The cache is loaded once per rank on CPU.  Each rank receives a deterministic
slice of the training complexes, while DDP synchronizes gradients every
update.  Only rank zero evaluates and writes the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from pocketdiff.diffusion import DiffusionMotionAdapter
from pocketdiff.scripts.train_diffusion import (
    _parse_float_tuple,
    _parse_int_tuple,
    build_parser,
)
from pocketdiff.training.diffusion import (
    DiffusionTrainConfig,
    evaluate_diffusion_loss,
    evaluate_diffusion_sampler,
    load_apo2mol_slices,
    save_diffusion_checkpoint,
    train_diffusion_model,
)


def _ddp_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--backend",
        default="nccl",
        choices=("nccl", "gloo"),
        help="Distributed backend; nccl is required for multi-GPU training.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save a resumable rank-0 checkpoint and progress JSON every N updates.",
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=200,
        help="Evaluate validation loss every N updates and update checkpoint_best.pt.",
    )
    return parser


def _config_from_args(args: argparse.Namespace, device: str) -> DiffusionTrainConfig:
    return DiffusionTrainConfig(
        data_root=str(args.data_root),
        split_path=str(args.split_path),
        source_cache=str(args.source_cache) if args.source_cache else None,
        prepared_cache=str(args.prepared_cache) if args.prepared_cache else None,
        device=device,
        train_split=args.train_split,
        valid_split=args.valid_split,
        test_split=args.test_split,
        holdout_split=args.holdout_split,
        train_count=args.train_count,
        valid_count=args.valid_count,
        test_count=args.test_count,
        holdout_count=args.holdout_count,
        train_seed=args.train_seed,
        holdout_seed=args.holdout_seed,
        diffusion_seed=args.diffusion_seed,
        updates=args.updates,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        self_state_probability=args.self_state_probability,
        self_state_steps=args.self_state_steps,
        self_state_rollout_steps=args.self_state_rollout_steps,
        self_state_warmup=args.self_state_warmup,
        apo_boundary_probability=args.apo_boundary_probability,
        cache_manifest=str(args.cache_manifest) if args.cache_manifest else None,
        verify_cache_sources=not args.no_verify_cache_sources,
        times=_parse_float_tuple(args.times),
        time_min=args.time_min,
        time_max=args.time_max,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        encoder_backend=args.encoder_backend,
        architecture_variant=args.architecture_variant,
        endpoint_weight=args.endpoint_weight,
        next_state_weight=args.next_state_weight,
        next_state_steps=args.next_state_steps,
        chi_loss_weight=args.chi_loss_weight,
        backbone_only=args.backbone_only,
        use_chi=not args.disable_chi,
        score_normalize=args.score_normalize,
        score_floor=args.score_floor,
        motion_parameterization=args.motion_parameterization,
        sampler_steps=_parse_int_tuple(args.sampler_steps),
        remaining_step_offset=args.remaining_step_offset,
        translation_noise_scale=args.translation_noise_scale,
        rotation_noise_scale=args.rotation_noise_scale,
        chi_noise_scale=args.chi_noise_scale,
        prediction_type=args.prediction_type,
        target_parameterization=args.target_parameterization,
        target_steps=args.target_steps,
        backbone_endpoint_weight=args.backbone_endpoint_weight,
        continuity_weight=args.continuity_weight,
        direction_weight=args.direction_weight,
        direction_threshold=args.direction_threshold,
        backbone_progress_direction_weight=args.backbone_progress_direction_weight,
        backbone_progress_magnitude_weight=args.backbone_progress_magnitude_weight,
        backbone_progress_threshold=args.backbone_progress_threshold,
        motion_bucket_count=args.motion_bucket_count,
        motion_bucket_balance=not args.no_motion_bucket_balance,
    )


def _reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor / dist.get_world_size()).cpu())


def _jsonable_args(args: argparse.Namespace) -> dict:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def main() -> None:
    args = _ddp_parser().parse_args()
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.validate_every <= 0:
        raise ValueError("--validate-every must be positive")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError("train_diffusion_ddp.py must be launched with torchrun")
    if not torch.cuda.is_available() and args.backend == "nccl":
        raise RuntimeError("NCCL DDP requires CUDA")
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=args.backend, init_method="env://")
    try:
        config = _config_from_args(args, str(device))
        slices = load_apo2mol_slices(config)
        train_values = slices["train"].values[rank::world_size]
        if not train_values:
            raise RuntimeError(f"rank {rank} received no training samples")
        model = DiffusionMotionAdapter(
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            encoder_backend=config.encoder_backend,
            prediction_type=config.prediction_type,
            architecture_variant=config.architecture_variant,
        ).to(device)
        model._pocketdiff_use_chi = bool(config.use_chi)
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        # backbone-only runs intentionally leave the χ head inactive.
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
        ddp_model._pocketdiff_use_chi = bool(config.use_chi)

        output_dir = Path(args.output_dir)
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        start_time = time.time()
        best_validation_loss = float("inf")
        best_validation_report = None

        def on_update(step, row, current_model, current_optimizer):
            nonlocal best_validation_loss, best_validation_report
            if rank != 0:
                return
            completed = int(step) + 1
            should_validate = (
                completed % args.validate_every == 0
                or completed == config.updates
            )
            should_checkpoint = (
                completed % args.checkpoint_every == 0
                or completed == config.updates
            )
            validation = None
            if should_validate:
                validation = evaluate_diffusion_loss(
                    current_model.module
                    if hasattr(current_model, "module")
                    else current_model,
                    slices["valid"].values,
                    config,
                    batch_size=config.batch_size,
                )
                validation_loss = float(validation["total_loss"])
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_validation_report = {
                        "update": completed,
                        "validation": validation,
                    }
                    save_diffusion_checkpoint(
                        output_dir / "checkpoint_best.pt",
                        current_model,
                        current_optimizer,
                        config=config,
                        report={
                            "format": "pocketdiff-diffusion-ddp-best-v1",
                            "best_validation_loss": best_validation_loss,
                            "best_validation": validation,
                            "completed_updates": completed,
                            "config": _jsonable_args(args),
                        },
                    )
            if not should_checkpoint:
                return
            progress = {
                "format": "pocketdiff-diffusion-ddp-progress-v1",
                "world_size": world_size,
                "completed_updates": completed,
                "total_updates": config.updates,
                "elapsed_seconds": time.time() - start_time,
                "last_row": row,
                "validation": validation,
                "best_validation_loss": best_validation_loss,
                "best_validation": best_validation_report,
                "config": _jsonable_args(args),
            }
            save_diffusion_checkpoint(
                output_dir / "checkpoint_last.pt",
                current_model,
                current_optimizer,
                config=config,
                report=progress,
            )
            (output_dir / "progress.json").write_text(
                json.dumps(progress, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        "update": completed,
                        "total_updates": config.updates,
                        "elapsed_minutes": progress["elapsed_seconds"] / 60.0,
                        "loss": row["loss"],
                        "self_state_count": row["self_state_count"],
                        "validation_loss": (
                            None
                            if validation is None
                            else validation["total_loss"]
                        ),
                        "best_validation_loss": best_validation_loss,
                    }
                ),
                flush=True,
            )

        _, optimizer, rows = train_diffusion_model(
            train_values,
            config,
            model=ddp_model,
            optimizer=optimizer,
            update_callback=on_update,
        )
        dist.barrier()
        if rank == 0:
            trained_model = ddp_model.module
            trained_model.eval()
            report = {
                "format": "pocketdiff-diffusion-ddp-run-v1",
                "world_size": world_size,
                "rank_train_counts": [len(slices["train"].values[r::world_size]) for r in range(world_size)],
                "training_rows_rank0": rows,
                "config": {
                    **{
                        key: value
                        for key, value in _jsonable_args(args).items()
                        if key != "output_dir"
                    },
                    "device": str(device),
                },
                "all_rank_losses_finite": True,
                "best_validation_loss": best_validation_loss,
                "best_validation": best_validation_report,
                "best_checkpoint": str(output_dir / "checkpoint_best.pt"),
                "evaluation": "deferred_to_independent_process",
            }
            save_diffusion_checkpoint(
                output_dir / "checkpoint.pt",
                trained_model,
                optimizer,
                config=config,
                report=report,
            )
            (output_dir / "ddp_report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        "world_size": world_size,
                        "completed_updates": config.updates,
                        "checkpoint": str(output_dir / "checkpoint.pt"),
                        "best_checkpoint": str(output_dir / "checkpoint_best.pt"),
                        "best_validation_loss": best_validation_loss,
                        "evaluation": "deferred_to_independent_process",
                    },
                    indent=2,
                ),
                flush=True,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
