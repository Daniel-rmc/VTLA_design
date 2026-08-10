#!/usr/bin/env bash
# Launch the paper-aligned UniVTAC task suite on physical GPUs 1, 2, and 3.

set -euo pipefail

PROJECT_DIR="/home/rmc/workspace/VTLA_design"
UNIVTAC_DIR="/home/rmc/workspace/UniVTAC"
PYTHON_BIN="/home/rmc/miniconda/envs/UniVTAC/bin/python"
PROFILE_PATH="${PROJECT_DIR}/configs/univtac_paper_aligned_stage2.json"
ENCODER_CKPT="${UNIVTAC_DIR}/policy/ACT/encoder/checkpoints/resnet18/official/encoder.pth"
MIN_FREE_MEMORY_MB="${MIN_FREE_MEMORY_MB:-20000}"

run_task() {
    local gpu_id="$1"
    local group_dir="$2"
    local task_name="$3"
    local dataset_dir="${UNIVTAC_DIR}/data/official/${task_name}/clean"
    local source_manifest="${PROJECT_DIR}/data_manifests/${task_name}_official_first50.json"
    local run_dir="${group_dir}/${task_name}"
    local ckpt_dir="${run_dir}/checkpoints"
    local -a camera_args=(--camera_names cam_high)

    if [[ "${task_name}" == "insert_tube" ]]; then
        camera_args=(--camera_names cam_high cam_wrist)
    fi

    mkdir -p "${ckpt_dir}"
    cp "${PROFILE_PATH}" "${run_dir}/protocol.json"
    cp "${source_manifest}" "${run_dir}/dataset_manifest.json"

    local -a command=(
        env
        PYTHONUNBUFFERED=1
        NCCL_P2P_DISABLE=1
        CUDA_VISIBLE_DEVICES="${gpu_id}"
        PYTHONPATH="${PROJECT_DIR}:${UNIVTAC_DIR}:${UNIVTAC_DIR}/policy/ACT"
        "${PYTHON_BIN}" -m scripts.training.train_vtla_multigpu
        --stage stage2
        --num_gpus 1
        --task_name "${task_name}"
        --experiment_profile univtac_paper_aligned_v1
        --dataset_dir "${dataset_dir}"
        --dataset_manifest "${run_dir}/dataset_manifest.json"
        --episode_limit 50
        --split_strategy official_univtac
        --val_fraction 0.2
        --val_seed 1
        --normalization_scope selected
        "${camera_args[@]}"
        --tactile_names tac_left tac_right
        --no-normalize_tactile
        --state_dim 8
        --joint_indices 0 1 2 3 4 5 6 7
        --chunk_size 50
        --temporal_agg
        --hidden_dim 512
        --dim_feedforward 3200
        --nheads 8
        --enc_layers 4
        --dec_layers 7
        --dropout 0.1
        --backbone resnet18
        --tactile_backbone resnet18
        --tactile_position_embedding learned
        --tactile_backbone_ckpt "${ENCODER_CKPT}"
        --freeze_tactile_batchnorm
        --kl_weight 10
        --pad_weight 0
        --l1_reduction official_mean
        --batch_size 64
        --max_steps 4000
        --num_epochs 1000
        --lr 0.00001
        --lr_backbone 0.00001
        --lr_vision_backbone 0.00001
        --lr_tactile 0.00001
        --weight_decay 0.0001
        --lr_scheduler none
        --grad_clip 0
        --amp_dtype bfloat16
        --seed 0
        --num_workers 8
        --val_freq 1
        --checkpoint_freq_steps 1000
        --ckpt_dir "${ckpt_dir}"
        --run_dir "${run_dir}"
    )

    printf '%q ' "${command[@]}" > "${run_dir}/launch_command.sh"
    printf '\n' >> "${run_dir}/launch_command.sh"
    chmod +x "${run_dir}/launch_command.sh"

    set +e
    "${command[@]}" 2>&1 | tee "${run_dir}/train.log"
    local code=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "${code}" > "${run_dir}/exit_code"
    return "${code}"
}

if [[ "${1:-}" == "--worker" ]]; then
    gpu_id="$2"
    group_dir="$3"
    shift 3
    for task_name in "$@"; do
        run_task "${gpu_id}" "${group_dir}" "${task_name}"
    done
    exit 0
fi

for required in "${PYTHON_BIN}" "${PROFILE_PATH}" "${ENCODER_CKPT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Error: required path is missing: ${required}" >&2
        exit 1
    fi
done
if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is not installed" >&2
    exit 1
fi

tasks=(grasp_classify insert_HDMI insert_hole insert_tube)
for task_name in "${tasks[@]}"; do
    dataset_dir="${UNIVTAC_DIR}/data/official/${task_name}/clean"
    manifest="${PROJECT_DIR}/data_manifests/${task_name}_official_first50.json"
    episode_count=$(find "${dataset_dir}" -maxdepth 1 -type f -name '*.hdf5' 2>/dev/null | wc -l)
    if (( episode_count < 50 )); then
        echo "Error: ${task_name} has ${episode_count} episodes; at least 50 are required" >&2
        exit 1
    fi
    if [[ ! -f "${manifest}" ]]; then
        echo "Error: validated manifest is missing: ${manifest}" >&2
        exit 1
    fi
done

for gpu_id in 1 2 3; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')
    if (( free_mb < MIN_FREE_MEMORY_MB )); then
        echo "Error: GPU ${gpu_id} has only ${free_mb} MiB free" >&2
        exit 1
    fi
done

timestamp=$(date +"%Y%m%d_%H%M%S")
git_sha=$(git -C "${PROJECT_DIR}" rev-parse --short HEAD 2>/dev/null || echo nogit)
group_dir="${PROJECT_DIR}/runs/stage2/official_${timestamp}_${git_sha}_gpu123"
mkdir -p "${group_dir}"
cp "${PROFILE_PATH}" "${group_dir}/protocol.json"

session_prefix="vtla_official_${timestamp}"
tmux new-session -d -s "${session_prefix}_gpu1" \
    "${BASH_SOURCE[0]} --worker 1 ${group_dir} grasp_classify insert_tube"
tmux new-session -d -s "${session_prefix}_gpu2" \
    "${BASH_SOURCE[0]} --worker 2 ${group_dir} insert_HDMI"
tmux new-session -d -s "${session_prefix}_gpu3" \
    "${BASH_SOURCE[0]} --worker 3 ${group_dir} insert_hole"

echo "Started paper-aligned UniVTAC training suite"
echo "  group: ${group_dir}"
echo "  GPU 1 queue: grasp_classify -> insert_tube"
echo "  GPU 2: insert_HDMI"
echo "  GPU 3: insert_hole"
echo "  global batch/task: 64"
echo "  optimizer steps/task: 4000"
echo "  tmux prefix: ${session_prefix}"
