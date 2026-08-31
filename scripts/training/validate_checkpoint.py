#!/usr/bin/env python3
"""Load a saved VTLA checkpoint and run one real LeRobot dataset sample."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

import lerobot_policy_vtla  # noqa: F401  # Register the external policy plugin.
from lerobot_policy_vtla.modeling_vtla import VTLAPolicy


def prepare_dataset_sample_for_policy(
    sample: dict[str, torch.Tensor], image_keys: list[str]
) -> dict[str, torch.Tensor]:
    """Mirror LeRobot's train/eval uint8-to-[0,1] conversion."""
    prepared = dict(sample)
    for image_key in image_keys:
        image = prepared.get(image_key)
        if image is not None and image.dtype == torch.uint8:
            prepared[image_key] = image.to(dtype=torch.float32) / 255.0
    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--repo-id", default="local/manipulationnet_peg_in_hole_15holes"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"Expected a local LeRobot dataset at {dataset_root}; meta/info.json is missing"
        )
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    if config.type != "vtla":
        raise RuntimeError(f"Expected a VTLA checkpoint, got {config.type!r}")

    policy = VTLAPolicy.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
        strict=True,
    )
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
    )
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        video_backend="pyav",
        return_uint8=True,
    )

    raw_sample = dataset[args.sample_index]
    for camera_key in dataset.meta.camera_keys:
        if raw_sample[camera_key].dtype != torch.uint8:
            raise RuntimeError(
                f"Expected uint8 input for {camera_key!r}, got {raw_sample[camera_key].dtype}"
            )
    sample = prepare_dataset_sample_for_policy(
        raw_sample, list(dataset.meta.camera_keys)
    )
    batch = preprocessor(sample)

    for camera_key in dataset.meta.camera_keys:
        processed_image = batch.get(camera_key)
        if processed_image is None:
            continue
        dynamic_range = processed_image.max() - processed_image.min()
        if not torch.isfinite(processed_image).all() or dynamic_range.item() <= 0:
            raise RuntimeError(
                f"Processed camera image {camera_key!r} lost its dynamic range: "
                f"min={processed_image.min().item()}, max={processed_image.max().item()}"
            )
    with torch.inference_mode():
        normalized_action = policy.select_action(batch)
        action = postprocessor(normalized_action)

    if tuple(action.shape) != (1, config.action_feature.shape[0]):
        raise RuntimeError(f"Unexpected action shape: {tuple(action.shape)}")
    if not all(math.isfinite(value) for value in action.flatten().tolist()):
        raise RuntimeError("Checkpoint inference returned a non-finite action")

    print(f"checkpoint={checkpoint}")
    print(f"sample_index={args.sample_index}")
    print(f"state_dim={config.robot_state_feature.shape[0]}")
    print(f"action_dim={config.action_feature.shape[0]}")
    print(f"action_shape={tuple(action.shape)}")
    print(f"action_min={action.min().item():.6f}")
    print(f"action_max={action.max().item():.6f}")
    first_camera = dataset.meta.camera_keys[0]
    print(f"raw_camera_dtype={raw_sample[first_camera].dtype}")
    print(f"raw_camera_min={raw_sample[first_camera].min().item()}")
    print(f"raw_camera_max={raw_sample[first_camera].max().item()}")
    print(f"processed_camera={first_camera}")
    print(f"processed_camera_dtype={batch[first_camera].dtype}")
    print(f"processed_camera_min={batch[first_camera].min().item():.6f}")
    print(f"processed_camera_max={batch[first_camera].max().item():.6f}")


if __name__ == "__main__":
    main()
