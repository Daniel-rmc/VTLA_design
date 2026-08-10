#!/usr/bin/env python3
"""Minimal NCCL health check for the selected CUDA_VISIBLE_DEVICES."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=30))
    value = torch.tensor([rank + 1.0], device=f'cuda:{local_rank}')
    dist.all_reduce(value)
    torch.cuda.synchronize(local_rank)
    print(f'rank={rank} local_rank={local_rank} all_reduce={value.item()}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
