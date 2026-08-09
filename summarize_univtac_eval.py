#!/usr/bin/env python3
"""Merge sharded UniVTAC logs into one seed-level evaluation result."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


RESULT_RE = re.compile(
    r"\[(?P<index>\d+)\s*\]\s+Seed\s+(?P<seed>\d+)\s+"
    r"(?P<status>success|failed)\s+after\s+(?P<seconds>[\d.]+)\s+s\."
)
DETAIL_RE = re.compile(r"steps:\s*(?P<steps>\d+)\s*,\s*actions:\s*(?P<actions>\d+)")
ERROR_RE = re.compile(r"Seed\s+(?P<seed>\d+)\s+occurred exception:\s*(?P<error>.*)")


def parse_log(log_path: Path) -> tuple[list[dict], list[dict]]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    records = []
    errors = []
    for index, line in enumerate(lines):
        result_match = RESULT_RE.search(line)
        if result_match:
            detail_match = DETAIL_RE.search(lines[index + 1]) if index + 1 < len(lines) else None
            records.append(
                {
                    "seed": int(result_match.group("seed")),
                    "status": result_match.group("status"),
                    "duration_seconds": float(result_match.group("seconds")),
                    "steps": int(detail_match.group("steps")) if detail_match else None,
                    "actions": int(detail_match.group("actions")) if detail_match else None,
                    "source_log": str(log_path.resolve()),
                }
            )
        error_match = ERROR_RE.search(line)
        if error_match:
            errors.append(
                {
                    "seed": int(error_match.group("seed")),
                    "error": error_match.group("error"),
                    "source_log": str(log_path.resolve()),
                }
            )
    return records, errors


def summarize(result_root: Path, start_seed: int, end_seed: int) -> dict:
    if end_seed < start_seed:
        raise ValueError("end_seed must be greater than or equal to start_seed")
    log_paths = sorted(result_root.glob("*/log.log"))
    all_records = []
    all_errors = []
    for log_path in log_paths:
        records, errors = parse_log(log_path)
        all_records.extend(records)
        all_errors.extend(errors)

    requested = set(range(start_seed, end_seed + 1))
    by_seed: dict[int, list[dict]] = {}
    for record in all_records:
        if record["seed"] in requested:
            by_seed.setdefault(record["seed"], []).append(record)

    # If a seed was deliberately resumed, prefer the last completed record and
    # retain every duplicate in diagnostics.
    selected = [records[-1] for _, records in sorted(by_seed.items())]
    duplicates = {
        str(seed): records for seed, records in sorted(by_seed.items()) if len(records) > 1
    }
    missing_seeds = sorted(requested - set(by_seed))
    successes = sum(record["status"] == "success" for record in selected)
    failures = sum(record["status"] == "failed" for record in selected)
    completed = len(selected)

    return {
        "evaluation_type": "UniVTAC simulator task success",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_root": str(result_root.resolve()),
        "seed_range": {"start": start_seed, "end": end_seed},
        "expected_episodes": len(requested),
        "completed_episodes": completed,
        "successes": successes,
        "failures": failures,
        "success_rate_percent": (100.0 * successes / completed) if completed else None,
        "complete": completed == len(requested) and not missing_seeds,
        "missing_seeds": missing_seeds,
        "duplicate_completed_seeds": duplicates,
        "error_events": [error for error in all_errors if error["seed"] in requested],
        "records": selected,
        "source_logs": [str(path.resolve()) for path in log_paths],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--end-seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = summarize(args.result_root, args.start_seed, args.end_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"UniVTAC: {result['successes']}/{result['completed_episodes']} success "
        f"({result['success_rate_percent']}%), missing={len(result['missing_seeds'])}"
    )
    print(f"Saved aggregate evaluation: {args.output}")


if __name__ == "__main__":
    main()
