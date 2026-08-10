#!/usr/bin/env python3
"""Validate and fingerprint a published UniVTAC HDF5 task subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py

from dataloader import CAMERA_PATHS, discover_episode_files, infer_grasp_classify_label, resolve_tactile_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-episodes', type=int, default=100)
    parser.add_argument('--episode-limit', type=int, default=0,
                        help='Fingerprint only the first N episodes after validating availability')
    parser.add_argument('--task-name', default=None)
    parser.add_argument('--camera-names', nargs='+', default=['cam_high'])
    parser.add_argument('--source', default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    available_episode_files = discover_episode_files(dataset_dir)
    if len(available_episode_files) != args.expected_episodes:
        raise RuntimeError(
            f'Expected {args.expected_episodes} episodes, found '
            f'{len(available_episode_files)} in {dataset_dir}'
        )
    if args.episode_limit < 0 or args.episode_limit > len(available_episode_files):
        raise RuntimeError(
            f'Invalid episode limit {args.episode_limit} for '
            f'{len(available_episode_files)} available episodes'
        )
    episode_files = (
        available_episode_files[:args.episode_limit]
        if args.episode_limit else available_episode_files
    )
    task_name = args.task_name or dataset_dir.parent.parent.name
    camera_paths = []
    for name in args.camera_names:
        if name not in CAMERA_PATHS:
            raise RuntimeError(f'Unsupported camera name: {name}')
        camera_paths.append(CAMERA_PATHS[name])
    required_paths = ['embodiment/joint', *camera_paths]

    metadata_path = dataset_dir / 'metadata.json'
    if not metadata_path.is_file():
        raise RuntimeError(f'Missing official metadata: {metadata_path}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if len(metadata) != len(available_episode_files):
        raise RuntimeError(
            f'Expected {len(available_episode_files)} metadata records, found {len(metadata)}'
        )

    records = []
    total_samples = 0
    raw_joint_dims = set()
    for episode_path in episode_files:
        with h5py.File(episode_path, 'r') as root:
            tactile_paths = [
                resolve_tactile_path(root, name, episode_path)
                for name in ('tac_left', 'tac_right')
            ]
            episode_required_paths = [*required_paths, *tactile_paths]
            missing = [path for path in episode_required_paths if path not in root]
            if missing:
                raise RuntimeError(f'{episode_path} is missing required datasets: {missing}')
            joint = root['embodiment/joint']
            if joint.ndim != 2 or joint.shape[1] < 8 or joint.shape[0] < 2:
                raise RuntimeError(f'{episode_path} has invalid joint shape {joint.shape}')
            frames = int(joint.shape[0])
            for data_path in episode_required_paths[1:]:
                if len(root[data_path]) != frames:
                    raise RuntimeError(
                        f'{episode_path}: {data_path} has {len(root[data_path])} frames, '
                        f'expected {frames}'
                    )
            raw_joint_dims.add(int(joint.shape[1]))

        episode_id = episode_path.stem
        episode_metadata = metadata.get(episode_id)
        if episode_metadata is None:
            raise RuntimeError(f'Metadata has no record for episode {episode_id}')
        record = {
            'episode': episode_path.name,
            'bytes': episode_path.stat().st_size,
            'sha256': sha256_file(episode_path),
            'frames': frames,
            'training_samples': frames - 1,
            'seed': episode_metadata.get('seed'),
            'result': episode_metadata.get('result'),
        }
        if task_name == 'grasp_classify':
            record['class_label'] = infer_grasp_classify_label(episode_path)
        records.append(record)
        total_samples += frames - 1

    if raw_joint_dims != {9}:
        raise RuntimeError(f'Expected official raw 9D joint observations, found {raw_joint_dims}')
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': args.source or f'huggingface://datasets/byml/UniVTAC/{task_name}/clean',
        'task_name': task_name,
        'dataset_dir': str(dataset_dir),
        'available_episode_count': len(available_episode_files),
        'episode_count': len(records),
        'selection': f'first {len(records)} numerically sorted episodes',
        'camera_names': args.camera_names,
        'training_sample_count': total_samples,
        'raw_joint_dim': 9,
        'model_joint_indices': list(range(8)),
        'control_layout': '7 arm joints + 1 gripper command',
        'metadata_sha256': sha256_file(metadata_path),
        'total_hdf5_bytes': sum(record['bytes'] for record in records),
        'episodes': records,
        'result_counts': {
            result: sum(record['result'] == result for record in records)
            for result in sorted({record['result'] for record in records}, key=str)
        },
    }
    if task_name == 'grasp_classify':
        manifest['class_counts'] = {
            label: sum(record['class_label'] == label for record in records)
            for label in sorted({record['class_label'] for record in records})
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(
        f"Validated {len(records)} episodes, {total_samples} samples, "
        f"{manifest['total_hdf5_bytes'] / 1024 ** 3:.2f} GiB"
    )
    print(f'Wrote manifest: {args.output.resolve()}')


if __name__ == '__main__':
    main()
