#!/usr/bin/env bash
# Launch reproducible VTLA DDP training in a detached tmux session.

set -euo pipefail

PROJECT_DIR="/home/rmc/workspace/VTLA_design"
PYTHON_BIN="/home/rmc/miniconda/envs/UniVTAC/bin/python"
DATASET_DIR="${DATASET_DIR:-/home/rmc/workspace/UniVTAC/data/official/grasp_classify/clean}"
DATASET_MANIFEST="${DATASET_MANIFEST:-${PROJECT_DIR}/data_manifests/grasp_classify_clean_modelscope.json}"
STAGE="${1:-stage2}"
NUM_GPUS="${2:-3}"
REQUESTED_GPU_IDS="${3:-${GPU_IDS:-}}"
MIN_FREE_MEMORY_MB="${MIN_FREE_MEMORY_MB:-20000}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

if [[ ! "${STAGE}" =~ ^stage[123]$ ]]; then
    echo "Error: stage must be stage1, stage2, or stage3" >&2
    exit 1
fi
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_GPUS must be a positive integer" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: Python environment not found at ${PYTHON_BIN}" >&2
    exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is not installed" >&2
    exit 1
fi

declare -a GPU_LIST
if [[ -n "${REQUESTED_GPU_IDS}" ]]; then
    IFS=',' read -r -a GPU_LIST <<< "${REQUESTED_GPU_IDS}"
else
    mapfile -t GPU_LIST < <(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
        awk -F',' -v threshold="${MIN_FREE_MEMORY_MB}" '
            {gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 >= threshold) print $1}
        ' | head -n "${NUM_GPUS}"
    )
fi

if [[ "${#GPU_LIST[@]}" -ne "${NUM_GPUS}" ]]; then
    echo "Error: requested ${NUM_GPUS} GPUs, but selected ${#GPU_LIST[@]} with >= ${MIN_FREE_MEMORY_MB} MiB free" >&2
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
    exit 1
fi

GPU_IDS_CSV=$(IFS=','; echo "${GPU_LIST[*]}")
for gpu_id in "${GPU_LIST[@]}"; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')
    if (( free_mb < MIN_FREE_MEMORY_MB )); then
        echo "Error: GPU ${gpu_id} has only ${free_mb} MiB free (need ${MIN_FREE_MEMORY_MB} MiB)" >&2
        exit 1
    fi
done

case "${STAGE}" in
    stage1)
        BATCH_SIZE="${BATCH_SIZE:-16}"
        NUM_EPOCHS="${NUM_EPOCHS:-100}"
        EXTRA_ARGS="--tactile_supervise rgb marker"
        ;;
    stage2)
        BATCH_SIZE="${BATCH_SIZE:-64}"
        NUM_EPOCHS="${NUM_EPOCHS:-150}"
        EXTRA_ARGS="--camera_names cam_high cam_wrist --chunk_size 50 --joint_indices 0 1 2 3 4 5 6 7 --val_fraction 0.1 --val_seed 20260809 --val_freq 5 --amp_dtype bfloat16"
        latest_stage1=$(ls -t "${PROJECT_DIR}"/runs/stage1/*/checkpoints/stage1_epoch_*.ckpt 2>/dev/null | head -n 1 || true)
        if [[ -n "${latest_stage1}" ]]; then
            EXTRA_ARGS+=" --stage1_ckpt ${latest_stage1}"
        fi
        ;;
    stage3)
        BATCH_SIZE="${BATCH_SIZE:-32}"
        NUM_EPOCHS="${NUM_EPOCHS:-200}"
        EXTRA_ARGS="--camera_names cam_high cam_wrist --chunk_size 50 --joint_indices 0 1 2 3 4 5 6 7 --val_fraction 0.1 --val_seed 20260809 --val_freq 5 --amp_dtype bfloat16"
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
GPU_TAG=${GPU_IDS_CSV//,/}
RUN_NAME="${TIMESTAMP}_${GIT_SHA}_gpu${GPU_TAG}"
RUN_DIR="${PROJECT_DIR}/runs/${STAGE}/${RUN_NAME}"
CKPT_DIR="${RUN_DIR}/checkpoints"
LOG_FILE="${RUN_DIR}/train.log"
EXIT_FILE="${RUN_DIR}/exit_code"
SESSION_NAME="vtla_${STAGE}_${RUN_NAME}"
SAVE_FREQ="${SAVE_FREQ:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ "${STAGE}" != "stage1" ]]; then
    if [[ ! -f "${DATASET_MANIFEST}" ]]; then
        echo "Error: validated dataset manifest not found: ${DATASET_MANIFEST}" >&2
        exit 1
    fi
    episode_count=$(find "${DATASET_DIR}" -maxdepth 1 -type f -name '*.hdf5' | wc -l)
    if [[ "${episode_count}" -ne 100 ]]; then
        echo "Error: expected 100 official episodes in ${DATASET_DIR}, found ${episode_count}" >&2
        exit 1
    fi
fi

mkdir -p "${CKPT_DIR}"

TRAIN_CMD="set -o pipefail; cd ${PROJECT_DIR}; env PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} CUDA_VISIBLE_DEVICES=${GPU_IDS_CSV} PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/../UniVTAC:${PROJECT_DIR}/../UniVTAC/policy/ACT ${PYTHON_BIN} train_vtla_multigpu.py --stage ${STAGE} --num_gpus ${NUM_GPUS} --dataset_dir ${DATASET_DIR} --dataset_manifest ${DATASET_MANIFEST} --tactile_names tac_left tac_right --state_dim 8 --batch_size ${BATCH_SIZE} --num_epochs ${NUM_EPOCHS} --ckpt_dir ${CKPT_DIR} --run_dir ${RUN_DIR} --num_workers ${NUM_WORKERS} --save_freq ${SAVE_FREQ} ${EXTRA_ARGS} 2>&1 | tee ${LOG_FILE}; code=\${PIPESTATUS[0]}; echo \${code} > ${EXIT_FILE}; exit \${code}"

printf '%s\n' "${TRAIN_CMD}" > "${RUN_DIR}/launch_command.sh"
chmod +x "${RUN_DIR}/launch_command.sh"
tmux new-session -d -s "${SESSION_NAME}" "bash -lc '${TRAIN_CMD}'"

echo "VTLA training started"
echo "  tmux session: ${SESSION_NAME}"
echo "  physical GPUs: ${GPU_IDS_CSV}"
echo "  batch/GPU: ${BATCH_SIZE}"
echo "  effective batch: $((BATCH_SIZE * NUM_GPUS))"
echo "  NCCL_P2P_DISABLE: ${NCCL_P2P_DISABLE}"
echo "  epochs: ${NUM_EPOCHS}"
echo "  run directory: ${RUN_DIR}"
echo "  log: ${LOG_FILE}"
echo "  config: ${RUN_DIR}/config.json"
echo ""
echo "Monitor with: tail -f ${LOG_FILE}"
