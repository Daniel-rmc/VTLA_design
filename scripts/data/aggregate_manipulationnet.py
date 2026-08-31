#!/usr/bin/env python3
"""Aggregate the 15 local LeRobot v3 datasets for official lerobot-train."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.datasets.aggregate import aggregate_datasets


def _file_indices(paths: list[Path]) -> set[tuple[int, int]]:
    return {
        (int(path.parent.name.removeprefix("chunk-")), int(path.stem.removeprefix("file-")))
        for path in paths
    }


def _repair_episode_file_references(source: Path, destination: Path) -> list[str]:
    """Repair stale data and episode-metadata references in a copied metadata tree.

    The affected source datasets already contain every episode in one parquet file
    with correct global dataset offsets, but some episode rows still reference the
    pre-merge file number.  No frame values or video references are changed.
    """
    existing_data = _file_indices(sorted((source / "data").glob("*/*.parquet")))
    existing_meta = _file_indices(sorted((source / "meta" / "episodes").glob("*/*.parquet")))
    if not existing_data:
        raise FileNotFoundError(f"No data parquet files found below {source}")
    if not existing_meta:
        raise FileNotFoundError(f"No episode metadata parquet files found below {source}")
    repairs: list[str] = []
    for episode_path in sorted((destination / "meta" / "episodes").glob("*/*.parquet")):
        table = pq.read_table(episode_path)
        for prefix, existing in (("data", existing_data), ("meta/episodes", existing_meta)):
            referenced = set(
                zip(
                    table[f"{prefix}/chunk_index"].to_pylist(),
                    table[f"{prefix}/file_index"].to_pylist(),
                    strict=True,
                )
            )
            missing = referenced - existing
            if not missing:
                continue
            if len(existing) != 1:
                raise ValueError(
                    f"{source.name} has missing {prefix} references {sorted(missing)} and multiple "
                    f"candidate files {sorted(existing)}; automatic repair is unsafe"
                )
            chunk_index, file_index = next(iter(existing))
            row_count = table.num_rows
            chunk_column = table.schema.get_field_index(f"{prefix}/chunk_index")
            file_column = table.schema.get_field_index(f"{prefix}/file_index")
            table = table.set_column(
                chunk_column,
                f"{prefix}/chunk_index",
                pa.array([chunk_index] * row_count, type=table.schema.field(chunk_column).type),
            )
            table = table.set_column(
                file_column,
                f"{prefix}/file_index",
                pa.array([file_index] * row_count, type=table.schema.field(file_column).type),
            )
            repairs.append(
                f"{source.name}: mapped stale {prefix} references {sorted(missing)} to "
                f"{(chunk_index, file_index)}"
            )
        pq.write_table(table, episode_path)
    return repairs


@contextmanager
def repaired_dataset_views(roots: list[Path]):
    with tempfile.TemporaryDirectory(prefix="vtla_manipulationnet_") as temp_dir:
        staging_root = Path(temp_dir)
        repaired_roots = []
        repairs = []
        for source in roots:
            destination = staging_root / source.name
            destination.mkdir()
            shutil.copytree(source / "meta", destination / "meta")
            (destination / "data").symlink_to(source.resolve() / "data", target_is_directory=True)
            (destination / "videos").symlink_to(source.resolve() / "videos", target_is_directory=True)
            repairs.extend(_repair_episode_file_references(source, destination))
            repaired_roots.append(destination)
        yield repaired_roots, repairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/workspace/datasets/manipulationNet/peg_in_hole/15holes_merged"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/datasets/manipulationNet/peg_in_hole/15holes_v3"),
    )
    parser.add_argument("--repo-id", default="local/manipulationnet_peg_in_hole_15holes")
    parser.add_argument("--expected-datasets", type=int, default=15)
    parser.add_argument("--concatenate-videos", action="store_true")
    parser.add_argument("--concatenate-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite {args.output_root}. Move it aside or choose a new --output-root."
        )
    roots = sorted(path for path in args.source_root.iterdir() if (path / "meta" / "info.json").is_file())
    if len(roots) != args.expected_datasets:
        raise ValueError(
            f"Expected {args.expected_datasets} datasets, found {len(roots)} "
            f"below {args.source_root}"
        )

    repo_ids = [f"local/{root.name}" for root in roots]
    with repaired_dataset_views(roots) as (repaired_roots, repairs):
        for repair in repairs:
            print(f"REPAIR: {repair}")
        aggregate_datasets(
            repo_ids=repo_ids,
            roots=repaired_roots,
            aggr_repo_id=args.repo_id,
            aggr_root=args.output_root,
            concatenate_videos=args.concatenate_videos,
            concatenate_data=args.concatenate_data,
        )
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.output_root)
    expected_episodes = sum(
        LeRobotDatasetMetadata(repo_id, root=root).total_episodes
        for repo_id, root in zip(repo_ids, roots, strict=True)
    )
    if metadata.total_episodes != expected_episodes:
        raise RuntimeError(
            f"Aggregated dataset has {metadata.total_episodes} episodes; expected {expected_episodes}"
        )
    print(
        f"Created {args.output_root}: {metadata.total_episodes} episodes, "
        f"{metadata.total_frames} frames, {len(metadata.tasks)} tasks"
    )


if __name__ == "__main__":
    main()
