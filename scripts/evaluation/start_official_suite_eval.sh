#!/usr/bin/env bash
# Evaluate every paper-aligned VTLA checkpoint on 100 fixed UniVTAC rollouts.

set -euo pipefail

PROJECT_DIR="/home/rmc/workspace/VTLA_design"
SCRIPT_PATH="${PROJECT_DIR}/scripts/evaluation/start_official_suite_eval.sh"
GROUP_DIR="${GROUP_DIR:-${PROJECT_DIR}/runs/stage2/official_20260810_141830_f27987c_gpu123}"
UNIVTAC_DIR="/home/rmc/workspace/UniVTAC"
PYTHON_BIN="/home/rmc/miniconda/envs/UniVTAC/bin/python"
GPU_ID="${GPU_ID:-0}"
MIN_FREE_MEMORY_MB="${MIN_FREE_MEMORY_MB:-12000}"
CONFIG_DIR="${PROJECT_DIR}/univtac_adapter/official4000"
SUITE_DIR="${GROUP_DIR}/eval/univtac_suite"
TASKS=(insert_HDMI insert_hole insert_tube lift_bottle lift_can pull_out_key put_bottle_in_shelf)

cd "${PROJECT_DIR}"

run_eval() {
    local task="$1"
    local total_num="$2"
    local start_seed="$3"
    local max_seed="$4"

    "${PYTHON_BIN}" -m scripts.evaluation.run_univtac_eval \
        --run-dir "${GROUP_DIR}/${task}" \
        --deploy-config "${CONFIG_DIR}/${task}.yml" \
        --task "${task}" \
        --task-config demo \
        --gpu "${GPU_ID}" \
        --total-num "${total_num}" \
        --start-seed "${start_seed}" \
        --max-seed "${max_seed}"
}

summarize_task() {
    local task="$1"
    local aggregate="${GROUP_DIR}/${task}/eval/univtac/aggregate_result.json"
    local result_root="${UNIVTAC_DIR}/eval_result/VTLA/${task}/${task}"
    "${PYTHON_BIN}" -m scripts.evaluation.summarize_univtac_eval \
        --result-root "${result_root}" \
        --start-seed 1000000 \
        --end-seed 1000099 \
        --output "${aggregate}"
}

missing_seeds() {
    local aggregate="$1"
    "${PYTHON_BIN}" - "${aggregate}" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
print(" ".join(str(seed) for seed in result["missing_seeds"]))
PY
}

ensure_task_complete() {
    local task="$1"
    local aggregate="${GROUP_DIR}/${task}/eval/univtac/aggregate_result.json"
    local result_root="${UNIVTAC_DIR}/eval_result/VTLA/${task}/${task}"
    local attempt seed missing

    if [[ ! -d "${result_root}" ]] || ! find "${result_root}" -mindepth 2 -maxdepth 2 -name log.log -print -quit | grep -q .; then
        echo "[$(date --iso-8601=seconds)] Full evaluation: ${task}, seeds 1000000-1000099"
        if ! run_eval "${task}" 100 1000000 1000099; then
            echo "[$(date --iso-8601=seconds)] ${task} evaluator returned non-zero; checking persisted seeds"
        fi
    fi

    for attempt in 1 2 3; do
        summarize_task "${task}"
        missing=$(missing_seeds "${aggregate}")
        if [[ -z "${missing}" ]]; then
            break
        fi
        echo "[$(date --iso-8601=seconds)] ${task} repair attempt ${attempt}; missing seeds: ${missing}"
        for seed in ${missing}; do
            if ! run_eval "${task}" 1 "${seed}" "${seed}"; then
                echo "[$(date --iso-8601=seconds)] ${task} seed ${seed} returned non-zero; persisted outcome will be checked"
            fi
        done
    done

    summarize_task "${task}"
    "${PYTHON_BIN}" - "${aggregate}" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
if not result["complete"] or result["unresolved_error_events"]:
    raise SystemExit(
        f"Incomplete UniVTAC result: complete={result['complete']}, "
        f"missing={result['missing_seeds']}, errors={result['unresolved_error_events']}"
    )
PY
}

if [[ "${1:-}" == "--worker" ]]; then
    mkdir -p "${SUITE_DIR}"
    if [[ ! -f "${SUITE_DIR}/smoke_exit_code" || "$(<"${SUITE_DIR}/smoke_exit_code")" != 0 ]]; then
        echo "[$(date --iso-8601=seconds)] Smoke test: grasp_classify seed 9000000"
        run_eval grasp_classify 1 9000000 9000000
        printf '0\n' > "${SUITE_DIR}/smoke_exit_code"
    else
        echo "[$(date --iso-8601=seconds)] Reusing successful smoke test"
    fi

    for task in "${TASKS[@]}"; do
        ensure_task_complete "${task}"
    done

    "${PYTHON_BIN}" -m scripts.evaluation.summarize_univtac_suite \
        --group-dir "${GROUP_DIR}" \
        --tasks "${TASKS[@]}" \
        --output "${SUITE_DIR}/aggregate_result.json"
    printf '0\n' > "${SUITE_DIR}/exit_code"
    echo "[$(date --iso-8601=seconds)] All eight UniVTAC evaluations completed"
    exit 0
fi

for required in "${PYTHON_BIN}" "${GROUP_DIR}" "${UNIVTAC_DIR}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Error: required path is missing: ${required}" >&2
        exit 1
    fi
done
for task in "${TASKS[@]}"; do
    checkpoint="${GROUP_DIR}/${task}/checkpoints/stage2_last.ckpt"
    config="${CONFIG_DIR}/${task}.yml"
    exit_file="${GROUP_DIR}/${task}/exit_code"
    if [[ ! -f "${exit_file}" || "$(<"${exit_file}")" != 0 ]]; then
        echo "Error: training did not finish successfully: ${task}" >&2
        exit 1
    fi
    for required in "${checkpoint}" "${config}"; do
        if [[ ! -f "${required}" ]]; then
            echo "Error: required file is missing: ${required}" >&2
            exit 1
        fi
    done
done

free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')
if (( free_mb < MIN_FREE_MEMORY_MB )); then
    echo "Error: GPU ${GPU_ID} has only ${free_mb} MiB free" >&2
    exit 1
fi
if tmux has-session -t vtla_official4000_eval_gpu0 2>/dev/null; then
    echo "Error: tmux session vtla_official4000_eval_gpu0 already exists" >&2
    exit 1
fi

mkdir -p "${SUITE_DIR}"
cat > "${SUITE_DIR}/plan.json" <<EOF
{
  "created_at": "$(date --iso-8601=seconds)",
  "gpu": ${GPU_ID},
  "tasks": ["insert_HDMI", "insert_hole", "insert_tube", "lift_bottle", "lift_can", "pull_out_key", "put_bottle_in_shelf"],
  "excluded_tasks": {"grasp_classify": "already evaluated separately at user request"},
  "checkpoint": "per-task stage2_last.ckpt at step 4000",
  "smoke_seed": 9000000,
  "official_seed_start": 1000000,
  "official_seed_end": 1000099,
  "rollouts_per_task": 100,
  "git_commit": "$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
}
EOF

tmux new-session -d -s vtla_official4000_eval_gpu0 \
    "bash -lc '${SCRIPT_PATH} --worker 2>&1 | tee -a \"${SUITE_DIR}/suite.log\"; code=\${PIPESTATUS[0]}; printf \"%s\\n\" \"\${code}\" > \"${SUITE_DIR}/worker_exit_code\"; exit \"\${code}\"'"

echo "Started UniVTAC remaining-task evaluation"
echo "  tmux: vtla_official4000_eval_gpu0"
echo "  GPU: ${GPU_ID}"
echo "  smoke: grasp_classify seed 9000000"
echo "  full: 7 remaining tasks, 100 fixed seeds per task"
echo "  log: ${SUITE_DIR}/suite.log"
