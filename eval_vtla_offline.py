#!/usr/bin/env python3
"""Evaluate deterministic VTLA deployment predictions on recorded UniVTAC data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import VTLADataset
from models.vtla_policy import VTLAPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training = dict(checkpoint["run_config"]["training"])
    training["pretrained_backbones"] = False
    model = VTLAPolicy(SimpleNamespace(**training), stage=training["stage"])
    state_dict = checkpoint["model_state_dict"]
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    dataset = VTLADataset(
        str(args.dataset_dir),
        training["camera_names"],
        training["tactile_names"],
        chunk_size=training["chunk_size"],
        state_dim=training["state_dim"],
        verbose=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    stats = checkpoint["dataset_stats"]
    joint_mean = torch.tensor(stats["joint_mean"], dtype=torch.float32, device=device)
    joint_std = torch.tensor(stats["joint_std"], dtype=torch.float32, device=device)
    if not np.allclose(dataset.joint_mean, joint_mean.cpu().numpy(), atol=1e-6) or not np.allclose(
        dataset.joint_std, joint_std.cpu().numpy(), atol=1e-6
    ):
        raise RuntimeError("Dataset normalization statistics differ from the checkpoint")

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
            raw_error = normalized_error * joint_std.view(1, 1, -1)
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
            "latent": "zero (deployment path)",
        },
        "limitations": [
            "The five recorded trajectories were also used for training, so this is not a held-out score.",
            "This check validates deployment inference and action errors; simulator task success is measured separately by UniVTAC.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved offline evaluation: {args.output}")


if __name__ == "__main__":
    main()
