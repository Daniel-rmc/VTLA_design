#!/usr/bin/env python3
"""Fail fast when a training image does not match the pinned runtime."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
from dataclasses import fields

import torch
from lerobot.configs.train import TrainPipelineConfig

import lerobot_policy_vtla

EXPECTED = {
    "lerobot": "0.6.1",
    "lerobot_policy_vtla": "0.1.0",
}


def main() -> None:
    for distribution, expected in EXPECTED.items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise RuntimeError(f"{distribution}={actual}, expected {expected}")
        print(f"{distribution}={actual}")
    print(f"torch={torch.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the training container")
    print(f"gpu_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
    print(f"NCCL_CUMEM_ENABLE={os.environ.get('NCCL_CUMEM_ENABLE')}")
    print(f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE')}")
    train_fields = {field.name for field in fields(TrainPipelineConfig)}
    if "eval_num_workers" not in train_fields:
        raise RuntimeError("Pinned LeRobot eval-worker overlay is missing")
    print("lerobot_eval_worker_overlay=True")
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL)
    print(f"plugin={lerobot_policy_vtla.__file__}")


if __name__ == "__main__":
    main()
