#!/usr/bin/env python3
"""Combine per-task UniVTAC aggregates into one benchmark summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


TASKS = (
    "grasp_classify",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--checkpoint-policy",
        default="per-task stage2_last.ckpt at optimizer step 4000",
    )
    parser.add_argument(
        "--evaluation-type",
        default="UniVTAC simulator task suite",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    missing = []
    for task in args.tasks:
        aggregate_path = args.group_dir / task / "eval" / "univtac" / "aggregate_result.json"
        if not aggregate_path.is_file():
            missing.append(task)
            continue
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "task": task,
                "successes": aggregate["successes"],
                "episodes": aggregate["completed_episodes"],
                "success_rate_percent": aggregate["success_rate_percent"],
                "complete": aggregate["complete"],
                "aggregate_path": str(aggregate_path.resolve()),
            }
        )

    complete = not missing and all(row["complete"] for row in rows)
    total_successes = sum(row["successes"] for row in rows)
    total_episodes = sum(row["episodes"] for row in rows)
    summary = {
        "evaluation_type": args.evaluation_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_policy": args.checkpoint_policy,
        "seed_range_per_task": {"start": 1000000, "end": 1000099},
        "complete": complete,
        "missing_tasks": missing,
        "tasks": rows,
        "macro_average_success_rate_percent": (
            sum(row["success_rate_percent"] for row in rows) / len(rows) if rows else None
        ),
        "micro_average_success_rate_percent": (
            100.0 * total_successes / total_episodes if total_episodes else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"UniVTAC suite: tasks={len(rows)}/{len(args.tasks)}, complete={complete}, "
        f"macro={summary['macro_average_success_rate_percent']}"
    )
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
