#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config_path="${VTLA_CONFIG:-${repo_root}/configs/lerobot/vtla_manipulationnet.yaml}"
dataset_root="${VTLA_DATASET_ROOT:-/workspace/datasets/manipulationNet/peg_in_hole/15holes_v3}"
visible_gpus="${VTLA_GPUS:-}"
output_dir="${VTLA_OUTPUT_DIR:-${repo_root}/outputs/train/$(date +%Y-%m-%d_%H-%M-%S)_vtla}"
wandb_enable="${VTLA_WANDB_ENABLE:-true}"
wandb_project="${VTLA_WANDB_PROJECT:-vtla-manipulationnet-30k}"
wandb_mode="${VTLA_WANDB_MODE:-online}"
wandb_entity="${VTLA_WANDB_ENTITY:-}"
wandb_run_name="${VTLA_WANDB_RUN_NAME:-$(basename "${output_dir}")}"
wandb_disable_artifact="${VTLA_WANDB_DISABLE_ARTIFACT:-true}"
wandb_env_file="${VTLA_WANDB_ENV_FILE:-${repo_root}/.secrets/wandb.env}"

if [[ -z "${visible_gpus}" ]]; then
  echo "VTLA_GPUS must be set explicitly (for example: VTLA_GPUS=0 or VTLA_GPUS=0,3)." >&2
  exit 1
fi
if [[ ! "${visible_gpus}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "VTLA_GPUS must be a comma-separated list of physical GPU indices: ${visible_gpus}" >&2
  exit 1
fi

IFS=',' read -r -a gpu_indices <<< "${visible_gpus}"
declare -A seen_gpus=()
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to validate VTLA_GPUS." >&2
  exit 1
fi
gpu_count="$(nvidia-smi -L | wc -l)"
if [[ "${gpu_count}" -eq 0 ]]; then
  echo "No NVIDIA GPUs are visible to the training process." >&2
  exit 1
fi
for gpu_index in "${gpu_indices[@]}"; do
  if [[ -n "${seen_gpus[${gpu_index}]:-}" ]]; then
    echo "VTLA_GPUS contains a duplicate GPU index: ${gpu_index}" >&2
    exit 1
  fi
  if (( 10#${gpu_index} >= gpu_count )); then
    echo "VTLA_GPUS requests GPU ${gpu_index}, but only indices 0-$((gpu_count - 1)) are visible." >&2
    exit 1
  fi
  seen_gpus["${gpu_index}"]=1
done

num_processes="${VTLA_NUM_PROCESSES:-${#gpu_indices[@]}}"
if [[ ! "${num_processes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "VTLA_NUM_PROCESSES must be a positive integer: ${num_processes}" >&2
  exit 1
fi
if [[ "${num_processes}" -ne "${#gpu_indices[@]}" ]]; then
  echo "VTLA_NUM_PROCESSES=${num_processes} does not match ${#gpu_indices[@]} GPU(s) in VTLA_GPUS=${visible_gpus}." >&2
  exit 1
fi

if [[ "${wandb_enable}" != "true" && "${wandb_enable}" != "false" ]]; then
  echo "VTLA_WANDB_ENABLE must be true or false: ${wandb_enable}" >&2
  exit 1
fi
if [[ "${wandb_disable_artifact}" != "true" && "${wandb_disable_artifact}" != "false" ]]; then
  echo "VTLA_WANDB_DISABLE_ARTIFACT must be true or false: ${wandb_disable_artifact}" >&2
  exit 1
fi
if [[ "${wandb_enable}" == "true" ]]; then
  if [[ ! -f "${wandb_env_file}" ]]; then
    echo "W&B is enabled but its environment file is missing: ${wandb_env_file}" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${wandb_env_file}"
  set +a
  export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "W&B is enabled but WANDB_API_KEY is not set in ${wandb_env_file}." >&2
    exit 1
  fi
  export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
  python - <<'PY'
import os

import wandb

if not wandb.login(
    key=os.environ["WANDB_API_KEY"],
    host=os.environ["WANDB_BASE_URL"],
    verify=True,
    timeout=30,
):
    raise RuntimeError("W&B authentication failed")
print("W&B authentication verified (API key redacted).")
PY
fi

# Create only the parent. LeRobot intentionally rejects an existing run
# directory unless the invocation is a resume.
mkdir -p "$(dirname "${output_dir}")"
if [[ ! -w "$(dirname "${output_dir}")" ]]; then
  echo "Output parent is not writable: $(dirname "${output_dir}")" >&2
  echo "Fix the bind-mount ownership before starting training." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${visible_gpus}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

common_args=(
  "--config_path=${config_path}"
  "--dataset.root=${dataset_root}"
  "--output_dir=${output_dir}"
  "--job_name=${wandb_run_name}"
  "--wandb.enable=${wandb_enable}"
  "--wandb.project=${wandb_project}"
  "--wandb.mode=${wandb_mode}"
  "--wandb.disable_artifact=${wandb_disable_artifact}"
)

if [[ -n "${wandb_entity}" ]]; then
  common_args+=("--wandb.entity=${wandb_entity}")
fi

if [[ "${num_processes}" -eq 1 ]]; then
  exec lerobot-train "${common_args[@]}" "$@"
fi

exec accelerate launch \
  --multi_gpu \
  --num_processes="${num_processes}" \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  "$(command -v lerobot-train)" \
  "${common_args[@]}" \
  "$@"
