#!/usr/bin/env bash
# Re-evaluate the released ACT + UniVTAC Encoder checkpoints on fixed seeds.

set -euo pipefail

PROJECT_DIR="/home/rmc/workspace/VTLA_design"
UNIVTAC_DIR="/home/rmc/workspace/UniVTAC"
PYTHON_BIN="/home/rmc/miniconda/envs/UniVTAC/bin/python"
DEPLOY_CONFIG="${PROJECT_DIR}/univtac_adapter/act_univtac_encoder_repro.yml"
RESULT_STEM="act_univtac_encoder_repro"
TASKS=(grasp_classify insert_HDMI insert_hole insert_tube lift_bottle lift_can pull_out_key put_bottle_in_shelf)

cd "${PROJECT_DIR}"

train_config_for() {
    case "$1" in
        insert_tube|lift_can) printf 'train_config_all\n' ;;
        *) printf 'train_config\n' ;;
    esac
}

run_eval() {
    local group_dir="$1"
    local gpu_id="$2"
    local task="$3"
    local total_num="$4"
    local start_seed="$5"
    local max_seed="$6"
    local train_config
    train_config=$(train_config_for "${task}")

    TRAIN_CONFIG="${train_config}" CKPT_CONFIG="train_config" EP_NUM=50 \
        "${PYTHON_BIN}" -m scripts.evaluation.run_univtac_eval \
        --run-dir "${group_dir}/${task}" \
        --univtac-root "${UNIVTAC_DIR}" \
        --deploy-config "${DEPLOY_CONFIG}" \
        --result-policy-name ACT \
        --task "${task}" \
        --task-config demo \
        --gpu "${gpu_id}" \
        --total-num "${total_num}" \
        --start-seed "${start_seed}" \
        --max-seed "${max_seed}"
}

summarize_task() {
    local group_dir="$1"
    local task="$2"
    local aggregate="${group_dir}/${task}/eval/univtac/aggregate_result.json"
    "${PYTHON_BIN}" -m scripts.evaluation.summarize_univtac_eval \
        --result-root "${UNIVTAC_DIR}/eval_result/ACT/${task}/${RESULT_STEM}" \
        --start-seed 1000000 \
        --end-seed 1000099 \
        --output "${aggregate}"
}

missing_seeds() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
print(" ".join(str(seed) for seed in result["missing_seeds"]))
PY
}

ensure_task_complete() {
    local group_dir="$1"
    local gpu_id="$2"
    local task="$3"
    local aggregate="${group_dir}/${task}/eval/univtac/aggregate_result.json"
    local result_root="${UNIVTAC_DIR}/eval_result/ACT/${task}/${RESULT_STEM}"
    local attempt missing seed

    mkdir -p "${group_dir}/${task}"
    if [[ ! -d "${result_root}" ]] || ! find "${result_root}" -mindepth 2 -maxdepth 2 -name log.log -print -quit | grep -q .; then
        echo "[$(date --iso-8601=seconds)] ${task}: full evaluation on GPU ${gpu_id}"
        if ! run_eval "${group_dir}" "${gpu_id}" "${task}" 100 1000000 1000099; then
            echo "[$(date --iso-8601=seconds)] ${task}: evaluator returned non-zero; checking persisted outcomes"
        fi
    fi

    for attempt in 1 2 3; do
        summarize_task "${group_dir}" "${task}"
        missing=$(missing_seeds "${aggregate}")
        [[ -z "${missing}" ]] && break
        echo "[$(date --iso-8601=seconds)] ${task}: repair ${attempt}, missing: ${missing}"
        for seed in ${missing}; do
            if ! run_eval "${group_dir}" "${gpu_id}" "${task}" 1 "${seed}" "${seed}"; then
                echo "[$(date --iso-8601=seconds)] ${task} seed ${seed}: non-zero; will verify persisted outcome"
            fi
        done
    done

    summarize_task "${group_dir}" "${task}"
    "${PYTHON_BIN}" - "${aggregate}" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
if not result["complete"] or result["unresolved_error_events"]:
    raise SystemExit(
        f"Incomplete result: missing={result['missing_seeds']}, "
        f"unresolved_errors={result['unresolved_error_events']}"
    )
PY
}

if [[ "${1:-}" == "--worker" ]]; then
    gpu_id="$2"
    group_dir="$3"
    shift 3
    for task in "$@"; do
        ensure_task_complete "${group_dir}" "${gpu_id}" "${task}"
    done
    exit 0
fi

if [[ "${1:-}" == "--finalize" ]]; then
    group_dir="$2"
    while [[ ! -f "${group_dir}/lane_gpu0.exit_code" || ! -f "${group_dir}/lane_gpu3.exit_code" ]]; do
        sleep 30
    done
    code0=$(<"${group_dir}/lane_gpu0.exit_code")
    code3=$(<"${group_dir}/lane_gpu3.exit_code")
    if [[ "${code0}" != 0 || "${code3}" != 0 ]]; then
        echo "Lane failure: GPU0=${code0}, GPU3=${code3}" >&2
        printf '1\n' > "${group_dir}/exit_code"
        exit 1
    fi
    "${PYTHON_BIN}" -m scripts.evaluation.summarize_univtac_suite \
        --group-dir "${group_dir}" \
        --output "${group_dir}/aggregate_result.json" \
        --evaluation-type "UniVTAC released ACT + UniVTAC Encoder reproduction" \
        --checkpoint-policy "released per-task policy_last.ckpt; EP_NUM=50; official encoder-initialized training"
    printf '0\n' > "${group_dir}/exit_code"
    echo "[$(date --iso-8601=seconds)] ACT + UniVTAC Encoder reproduction complete"
    exit 0
fi

for required in "${PYTHON_BIN}" "${DEPLOY_CONFIG}" "${UNIVTAC_DIR}/scripts/eval_policy.py"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done
for task in "${TASKS[@]}"; do
    ckpt_dir="${UNIVTAC_DIR}/policy/ACT/act_ckpt/act-${task}/demo-50/train_config"
    for required in "${ckpt_dir}/policy_last.ckpt" "${ckpt_dir}/dataset_stats.pkl"; do
        [[ -f "${required}" ]] || { echo "Missing released artifact: ${required}" >&2; exit 1; }
    done
done

for gpu_id in 0 3; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')
    (( free_mb >= 16000 )) || { echo "GPU ${gpu_id} has only ${free_mb} MiB free" >&2; exit 1; }
done

timestamp=$(date +"%Y%m%d_%H%M%S")
git_sha=$(git -C "${PROJECT_DIR}" rev-parse --short HEAD)
group_dir="${PROJECT_DIR}/runs/baselines/act_univtac_encoder_${timestamp}_${git_sha}"
mkdir -p "${group_dir}"
cp "${DEPLOY_CONFIG}" "${group_dir}/deploy.yml"

"${PYTHON_BIN}" - "${group_dir}" <<'PY'
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json, sys

group = Path(sys.argv[1])
root = Path("/home/rmc/workspace/UniVTAC/policy/ACT/act_ckpt")
tasks = ["grasp_classify", "insert_HDMI", "insert_hole", "insert_tube",
         "lift_bottle", "lift_can", "pull_out_key", "put_bottle_in_shelf"]
records = []
for task in tasks:
    ckpt = root / f"act-{task}" / "demo-50" / "train_config" / "policy_last.ckpt"
    digest = sha256()
    with ckpt.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    records.append({
        "task": task,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": digest.hexdigest(),
        "train_config": "train_config_all" if task in {"insert_tube", "lift_can"} else "train_config",
        "checkpoint_storage_config": "train_config",
    })
plan = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "method": "ACT + UniVTAC Encoder (released checkpoints)",
    "seeds_per_task": {"start": 1000000, "end": 1000099, "count": 100},
    "ep_num": 50,
    "lanes": {
        "gpu0": ["grasp_classify", "insert_HDMI", "insert_hole", "insert_tube"],
        "gpu3": ["lift_bottle", "lift_can", "pull_out_key", "put_bottle_in_shelf"],
    },
    "artifacts": records,
    "notes": [
        "Published package stores all policy files under train_config even when the embedded evaluation log names train_config_all.",
        "The released put_bottle_in_shelf log names clean.yml, which is absent locally; current official demo.yml is used with the released demo-50 checkpoint path.",
    ],
}
(group / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
PY

# The released package stores the two multi-view checkpoints in train_config/.
# Read-only symlinks let unmodified official deployment code load those files
# while still selecting train_config_all.yml for model inputs.
for task in insert_tube lift_can; do
    parent="${UNIVTAC_DIR}/policy/ACT/act_ckpt/act-${task}/demo-50"
    if [[ ! -e "${parent}/train_config_all" ]]; then
        ln -s train_config "${parent}/train_config_all"
    fi
done

prefix="act_univtac_repro_${timestamp}"
tmux new-session -d -s "${prefix}_gpu0" \
    "bash -lc '${BASH_SOURCE[0]} --worker 0 \"${group_dir}\" grasp_classify insert_HDMI insert_hole insert_tube 2>&1 | tee \"${group_dir}/lane_gpu0.log\"; code=\${PIPESTATUS[0]}; printf \"%s\\n\" \"\${code}\" > \"${group_dir}/lane_gpu0.exit_code\"; exit \"\${code}\"'"
tmux new-session -d -s "${prefix}_gpu3" \
    "bash -lc '${BASH_SOURCE[0]} --worker 3 \"${group_dir}\" lift_bottle lift_can pull_out_key put_bottle_in_shelf 2>&1 | tee \"${group_dir}/lane_gpu3.log\"; code=\${PIPESTATUS[0]}; printf \"%s\\n\" \"\${code}\" > \"${group_dir}/lane_gpu3.exit_code\"; exit \"\${code}\"'"
tmux new-session -d -s "${prefix}_finalize" \
    "${BASH_SOURCE[0]} --finalize ${group_dir}"

printf '%s\n' "${group_dir}" > "${PROJECT_DIR}/runs/baselines/latest_act_univtac_encoder_run.txt"
echo "Started ACT + UniVTAC Encoder reproduction"
echo "  group: ${group_dir}"
echo "  GPU 0: grasp_classify -> insert_HDMI -> insert_hole -> insert_tube"
echo "  GPU 3: lift_bottle -> lift_can -> pull_out_key -> put_bottle_in_shelf"
echo "  tmux prefix: ${prefix}"
