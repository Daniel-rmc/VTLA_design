#!/usr/bin/env python3
"""Minimal two-or-more GPU NCCL health check for the training container."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"rank={local_rank} phase=init gpu={torch.cuda.get_device_name(local_rank)}", flush=True)
    dist.init_process_group(backend="nccl")
    value = torch.tensor([local_rank + 1.0], device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"all_reduce returned {value.item()}, expected {expected}")
    print(f"rank={local_rank} phase=ok all_reduce={value.item():.1f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
