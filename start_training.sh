#!/usr/bin/env bash
# Launch a reproducible single-GPU VTLA run in tmux.

set -euo pipefail

PROJECT_DIR="/home/rmc/workspace/VTLA_design"
PYTHON_BIN="/home/rmc/miniconda/envs/UniVTAC/bin/python"
DATASET_DIR="${DATASET_DIR:-/home/rmc/workspace/UniVTAC/data/official/grasp_classify/clean}"
STAGE="${1:-stage2}"
GPU_ID="${2:-1}"
MIN_FREE_MEMORY_MB="${MIN_FREE_MEMORY_MB:-20000}"

if [[ ! "${STAGE}" =~ ^stage[123]$ ]]; then
    echo "Error: stage must be stage1, stage2, or stage3" >&2
    exit 1
fi
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')
if (( free_mb < MIN_FREE_MEMORY_MB )); then
    echo "Error: GPU ${GPU_ID} has only ${free_mb} MiB free" >&2
    exit 1
fi

case "${STAGE}" in
    stage1)
        BATCH_SIZE="${BATCH_SIZE:-16}"
        NUM_EPOCHS="${NUM_EPOCHS:-100}"
        EXTRA_ARGS="--tactile_supervise rgb marker"
        ;;
    stage2)
        BATCH_SIZE="${BATCH_SIZE:-64}"
        NUM_EPOCHS="${NUM_EPOCHS:-150}"
        EXTRA_ARGS="--camera_names cam_high cam_wrist --chunk_size 50 --joint_indices 0 1 2 3 4 5 6 7"
        ;;
    stage3)
        BATCH_SIZE="${BATCH_SIZE:-32}"
        NUM_EPOCHS="${NUM_EPOCHS:-200}"
        EXTRA_ARGS="--camera_names cam_high cam_wrist --chunk_size 50 --joint_indices 0 1 2 3 4 5 6 7"
        latest_stage2=$(ls -t "${PROJECT_DIR}"/runs/stage2/*/checkpoints/stage2_epoch_*.ckpt 2>/dev/null | head -n 1 || true)
        if [[ -z "${latest_stage2}" ]]; then
            echo "Error: Stage 3 requires a Stage 2 checkpoint" >&2
            exit 1
        fi
        EXTRA_ARGS+=" --stage2_ckpt ${latest_stage2}"
        ;;
esac

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
GIT_SHA=$(git -C "${PROJECT_DIR}" rev-parse --short HEAD 2>/dev/null || echo nogit)
RUN_NAME="${TIMESTAMP}_${GIT_SHA}_gpu${GPU_ID}"
RUN_DIR="${PROJECT_DIR}/runs/${STAGE}/${RUN_NAME}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LOG_FILE="${RUN_DIR}/train.log"
EXIT_FILE="${RUN_DIR}/exit_code"
SESSION_NAME="vtla_${STAGE}_${RUN_NAME}"
mkdir -p "${CKPT_DIR}"

TRAIN_CMD="set -o pipefail; cd ${PROJECT_DIR}; env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/../UniVTAC:${PROJECT_DIR}/../UniVTAC/policy/ACT ${PYTHON_BIN} train_vtla.py --stage ${STAGE} --dataset_dir ${DATASET_DIR} --tactile_names tac_left tac_right --state_dim 8 --batch_size ${BATCH_SIZE} --num_epochs ${NUM_EPOCHS} --ckpt_dir ${CKPT_DIR} --run_dir ${RUN_DIR} --device cuda:0 --num_workers ${NUM_WORKERS:-8} --save_freq ${SAVE_FREQ:-10} ${EXTRA_ARGS} 2>&1 | tee ${LOG_FILE}; code=\${PIPESTATUS[0]}; echo \${code} > ${EXIT_FILE}; exit \${code}"
printf '%s\n' "${TRAIN_CMD}" > "${RUN_DIR}/launch_command.sh"
chmod +x "${RUN_DIR}/launch_command.sh"
tmux new-session -d -s "${SESSION_NAME}" "bash -lc '${TRAIN_CMD}'"

echo "VTLA training started"
echo "  tmux session: ${SESSION_NAME}"
echo "  physical GPU: ${GPU_ID}"
echo "  run directory: ${RUN_DIR}"
echo "  log: ${LOG_FILE}"
