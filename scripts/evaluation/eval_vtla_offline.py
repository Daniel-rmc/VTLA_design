#!/usr/bin/env python3
"""Evaluate deterministic VTLA deployment predictions on recorded UniVTAC data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from dataloader import VTLADataset, discover_episode_files
from models.vtla_policy import VTLAPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation"),
        default="all",
        help=(
            "Episode split to evaluate. train/validation reuse the exact episode "
            "names stored in the checkpoint run configuration."
        ),
    )
    return parser.parse_args()


def select_episode_files(dataset_dir: Path, split: str, run_config: dict) -> list[Path]:
    """Resolve an evaluation split without silently mixing train and validation data."""
    discovered = discover_episode_files(dataset_dir)
    if split == "all":
        return discovered

    dataset_config = run_config.get("dataset", {})
    episode_names = dataset_config.get(f"{split}_episodes")
    if not episode_names:
        raise ValueError(
            f"Checkpoint does not record a non-empty {split}_episodes split; "
            "use --split all only if evaluating every trajectory is intentional"
        )

    by_name = {path.name: path for path in discovered}
    if len(by_name) != len(discovered):
        raise ValueError(f"Episode filenames are not unique under {dataset_dir}")
    missing = [name for name in episode_names if name not in by_name]
    if missing:
        raise ValueError(
            f"{len(missing)} checkpoint {split} episodes are missing under "
            f"{dataset_dir}: {missing[:5]}"
        )
    return [by_name[name] for name in episode_names]


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_config = checkpoint["run_config"]
    training = dict(run_config["training"])
    training["pretrained_backbones"] = False
    model = VTLAPolicy(SimpleNamespace(**training), stage=training["stage"])
    state_dict = checkpoint["model_state_dict"]
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    stats = checkpoint["dataset_stats"]
    episode_files = select_episode_files(args.dataset_dir, args.split, run_config)
    dataset = VTLADataset(
        str(args.dataset_dir),
        training["camera_names"],
        training["tactile_names"],
        chunk_size=training["chunk_size"],
        state_dim=training["state_dim"],
        joint_indices=stats.get("joint_indices"),
        episode_files=episode_files,
        normalization_stats=stats,
        normalize_tactile=training.get("normalize_tactile", True),
        verbose=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    action_std = torch.tensor(
        stats.get("action_std", stats["joint_std"]),
        dtype=torch.float32,
        device=device,
    )

    normalized_abs_sum = 0.0
    raw_abs_sum = 0.0
    valid_scalar_count = 0
    first_raw_abs_sum = torch.zeros(training["state_dim"], dtype=torch.float64)
    all_raw_abs_sum = torch.zeros(training["state_dim"], dtype=torch.float64)
    all_joint_count = 0
    first_count = 0
    prediction_norm_abs_max = 0.0

    with torch.inference_mode():
        for batch in loader:
            qpos = batch["qpos"].to(device, non_blocking=True)
            cameras = batch["cam_image"].to(device, non_blocking=True)
            tactile = batch["tac_image"].to(device, non_blocking=True)
            targets = batch["actions"].to(device, non_blocking=True)
            is_pad = batch["is_pad"].to(device, non_blocking=True)

            predictions = model(qpos, cameras, tactile)
            valid = ~is_pad
            normalized_error = (predictions - targets).abs()
            raw_error = normalized_error * action_std.view(1, 1, -1)
            valid_3d = valid.unsqueeze(-1)

            normalized_abs_sum += (normalized_error * valid_3d).sum().item()
            raw_abs_sum += (raw_error * valid_3d).sum().item()
            valid_scalar_count += valid.sum().item() * training["state_dim"]
            all_raw_abs_sum += (raw_error * valid_3d).sum((0, 1)).double().cpu()
            all_joint_count += valid.sum().item()
            first_raw_abs_sum += raw_error[:, 0].sum(0).double().cpu()
            first_count += qpos.shape[0]
            prediction_norm_abs_max = max(
                prediction_norm_abs_max, predictions.abs().max().item()
            )

    per_joint_all = (all_raw_abs_sum / all_joint_count).tolist()
    per_joint_first = (first_raw_abs_sum / first_count).tolist()
    result = {
        "evaluation_type": "offline deterministic inference on recorded UniVTAC trajectories",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch_zero_based": checkpoint.get("epoch"),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "split": args.split,
        "episode_count": len(episode_files),
        "episodes": [path.name for path in episode_files],
        "samples": len(dataset),
        "valid_action_steps": all_joint_count,
        "normalized_mae": normalized_abs_sum / valid_scalar_count,
        "raw_joint_mae": raw_abs_sum / valid_scalar_count,
        "raw_joint_mae_per_dimension": per_joint_all,
        "first_step_raw_joint_mae": sum(per_joint_first) / len(per_joint_first),
        "first_step_raw_joint_mae_per_dimension": per_joint_first,
        "prediction_normalized_abs_max": prediction_norm_abs_max,
        "settings": {
            "device": str(device),
            "batch_size": args.batch_size,
            "camera_names": training["camera_names"],
            "tactile_names": training["tactile_names"],
            "chunk_size": training["chunk_size"],
            "state_dim": training["state_dim"],
            "joint_indices": stats.get("joint_indices", list(range(training["state_dim"]))),
            "normalize_tactile": training.get("normalize_tactile", True),
            "latent": "zero (deployment path)",
        },
        "limitations": [
            "Offline action error is reported separately from held-out simulator task success.",
            "This check validates deployment inference and action errors; simulator task success is measured separately by UniVTAC.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved offline evaluation: {args.output}")


if __name__ == "__main__":
    main()
