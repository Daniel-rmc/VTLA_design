"""Reproducibility helpers shared by single- and multi-GPU trainers."""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _git_metadata(project_dir: Path) -> Dict[str, Any]:
    def run_git(*args: str) -> str:
        result = subprocess.run(
            ['git', *args], cwd=project_dir, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ''

    return {
        'commit': run_git('rev-parse', 'HEAD') or None,
        'branch': run_git('branch', '--show-current') or None,
        'dirty': bool(run_git('status', '--porcelain')),
    }


def build_run_config(
    args,
    world_size: int,
    dataset_size: int,
    batches_per_rank: int,
    dataset_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    project_dir = Path(__file__).resolve().parent
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_info = []
    if torch.cuda.is_available():
        for logical_id in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(logical_id)
            gpu_info.append({
                'logical_id': logical_id,
                'name': props.name,
                'total_memory_gib': round(props.total_memory / 1024 ** 3, 2),
            })

    return {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': [sys.executable, *sys.argv],
        'project_dir': str(project_dir),
        'git': _git_metadata(project_dir),
        'training': {
            **_jsonable(vars(args)),
            'world_size': world_size,
            'effective_batch_size': args.batch_size * world_size,
            'dataset_size': dataset_size,
            'batches_per_rank': batches_per_rank,
        },
        'dataset_stats': _jsonable(dataset_stats),
        'runtime': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'torchvision': _torchvision_version(),
            'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'cuda_visible_devices': gpu_ids,
            'gpus': gpu_info,
        },
    }


def _torchvision_version() -> Optional[str]:
    try:
        import torchvision
        return torchvision.__version__
    except Exception:
        return None


def write_run_config(run_dir: str, config: Dict[str, Any]) -> Path:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    config_path = path / 'config.json'
    temporary_path = path / '.config.json.tmp'
    with temporary_path.open('w', encoding='utf-8') as stream:
        json.dump(_jsonable(config), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary_path, config_path)
    return config_path


def append_epoch_metrics(run_dir: str, stage: str, epoch: int, losses: Dict[str, Any]) -> None:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    record = {
        'stage': stage,
        'epoch': epoch + 1,
        'losses': {
            key: float(np.mean(values)) if len(values) else None
            for key, values in losses.items()
        },
    }
    with (path / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
