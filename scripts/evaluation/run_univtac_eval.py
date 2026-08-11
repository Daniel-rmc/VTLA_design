#!/usr/bin/env python3
"""Run UniVTAC evaluation and archive the exact request and resulting paths."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = PROJECT_ROOT / "runs" / "stage2" / "20260809_155942_1073ae9_gpu123"
SEED_RESULT_RE = re.compile(r"\[\d+\s*\]\s+Seed\s+(\d+)\s+(?:success|failed)\s+after")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--univtac-root", type=Path, default=Path("/home/rmc/workspace/UniVTAC"))
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/rmc/miniconda/envs/UniVTAC/bin/python"),
    )
    parser.add_argument(
        "--deploy-config",
        type=Path,
        default=PROJECT_ROOT / "univtac_adapter" / "deploy.yml",
    )
    parser.add_argument("--task", default="grasp_classify")
    parser.add_argument("--task-config", default="demo")
    parser.add_argument("--gpu", default="3", help="Physical GPU ID exposed as cuda:0")
    parser.add_argument("--total-num", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=-1)
    parser.add_argument("--max-seed", type=int, default=-1)
    parser.add_argument("--expert-check", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def eula_file(python_path: Path) -> Path:
    env_root = python_path.parent.parent
    candidates = sorted(
        {path.resolve() for path in (env_root / "lib").glob("python*/site-packages/omni/kit_app.py")}
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Could not uniquely locate omni/kit_app.py under {env_root}")
    return candidates[0].with_name("EULA_ACCEPTED")


def main():
    args = parse_args()
    accepted_path = eula_file(args.python)
    accepted = accepted_path.is_file() and accepted_path.read_text(encoding="utf-8").strip().lower() in {
        "y", "yes", "1"
    }
    if not accepted:
        raise SystemExit(
            "NVIDIA Omniverse EULA has not been accepted by the user. Run this interactively, "
            "review the displayed agreement, and answer the prompt:\n"
            f"  {args.python} -c \"import isaacsim\""
        )

    result_root = (
        args.univtac_root
        / "eval_result"
        / "VTLA"
        / args.task
        / args.deploy_config.stem
    )
    result_root.mkdir(parents=True, exist_ok=True)
    lock_file = (result_root / ".evaluation.lock").open("a+", encoding="utf-8")
    if os.environ.get("VTLA_EVAL_LOCK_HELD") != "1":
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

    # Re-check persisted results only after acquiring the task lock. This makes
    # separately scheduled GPU lanes safe: a later lane waits for the first,
    # then skips seeds that were completed while it was waiting.
    if (
        args.start_seed >= 0
        and args.max_seed >= args.start_seed
        and args.max_seed - args.start_seed + 1 == args.total_num
    ):
        completed_seeds = set()
        for result_log in result_root.glob("*/log.log"):
            completed_seeds.update(
                int(seed)
                for seed in SEED_RESULT_RE.findall(
                    result_log.read_text(encoding="utf-8", errors="replace")
                )
            )
        requested_seeds = set(range(args.start_seed, args.max_seed + 1))
        if requested_seeds <= completed_seeds:
            print(
                f"Skipping {args.task}: all requested seeds "
                f"{args.start_seed}-{args.max_seed} are already complete"
            )
            return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = args.run_dir / "eval" / "univtac" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.deploy_config, archive_dir / "deploy.yml")
    task_config_path = args.univtac_root / "task_config" / f"{args.task_config}.yml"
    if task_config_path.is_file():
        shutil.copy2(task_config_path, archive_dir / "task_config.yml")

    command = [
        str(args.python),
        "scripts/eval_policy.py",
        args.task,
        args.task_config,
        str(args.deploy_config.resolve()),
        "--total_num",
        str(args.total_num),
        "--start_seed",
        str(args.start_seed),
        "--max_seed",
        str(args.max_seed),
        "--device",
        "cuda:0",
    ]
    if args.expert_check:
        command.append("--expert_check")
    if args.headless:
        command.append("--headless")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    environment["PYTHONUNBUFFERED"] = "1"
    # A detached tmux launched from an SSH session can retain a stale
    # DISPLAY=localhost:* value. Isaac Sim is headless here and must not bind
    # its lifecycle to that X11 forwarding socket.
    if args.headless:
        environment.pop("DISPLAY", None)
        environment.pop("XAUTHORITY", None)
    # An absolute Python path does not activate its Conda environment. Add the
    # environment's bin directory explicitly so UniVTAC can launch ffmpeg and
    # other runtime tools installed alongside Python.
    environment["PATH"] = os.pathsep.join(
        (str(args.python.parent.resolve()), environment.get("PATH", ""))
    )
    adapter_root = str((PROJECT_ROOT / "univtac_adapter").resolve())
    python_paths = [adapter_root, str(args.univtac_root.resolve())]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    request = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "cwd": str(args.univtac_root.resolve()),
        "cuda_visible_devices": args.gpu,
        "environment_bin": str(args.python.parent.resolve()),
        "task": args.task,
        "task_config": str(task_config_path.resolve()),
        "deploy_config": str(args.deploy_config.resolve()),
        "total_num": args.total_num,
        "start_seed": args.start_seed,
        "max_seed": args.max_seed,
        "expert_check": args.expert_check,
        "headless": args.headless,
        "display_removed": args.headless,
    }
    (archive_dir / "request.json").write_text(
        json.dumps(request, indent=2) + "\n", encoding="utf-8"
    )

    preexisting = set(result_root.iterdir()) if result_root.is_dir() else set()
    output_tail = ""
    interrupted = False
    with (archive_dir / "stdout.log").open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=args.univtac_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
                log_file.flush()
                output_tail = (output_tail + line)[-1_000_000:]
            exit_code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            process.terminate()
            try:
                exit_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            exit_code = 130

    new_results = sorted(set(result_root.iterdir()) - preexisting) if result_root.is_dir() else []
    # Concurrent shards share result_root and can all appear as "new" to each
    # wrapper. Associate a shard with this request by its explicit start seed.
    matching_results = []
    result_log_text = ""
    for result_dir in new_results:
        result_log = result_dir / "log.log"
        text = result_log.read_text(encoding="utf-8", errors="replace") if result_log.is_file() else ""
        if args.start_seed == -1 or re.search(rf"Seed\s+{args.start_seed}\b", text):
            matching_results.append(result_dir)
            result_log_text += text
    final_match = re.findall(
        r"Final Result:\s*(\d+)/(\d+)\(([\d.]+)%\) success",
        result_log_text or output_tail,
    )
    summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": exit_code,
        "interrupted": interrupted,
        "univtac_result_dirs": [str(path.resolve()) for path in matching_results],
        "final_result": None,
    }
    if final_match:
        successes, episodes, rate = final_match[-1]
        summary["final_result"] = {
            "successes": int(successes),
            "episodes": int(episodes),
            "success_rate_percent": float(rate),
        }
    (archive_dir / "result.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (archive_dir / "exit_code").write_text(f"{exit_code}\n", encoding="utf-8")
    print(f"Evaluation archive: {archive_dir}")
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
