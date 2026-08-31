#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
session_name="${VTLA_TMUX_SESSION:-vtla_30k}"
project_name="${VTLA_WANDB_PROJECT:-vtla-manipulationnet-30k}"
output_dir="${VTLA_OUTPUT_DIR:-${repo_root}/outputs/train/${timestamp}_${project_name}}"
log_file="${VTLA_TMUX_LOG:-${repo_root}/outputs/tmux/${session_name}_${timestamp}.log}"

if [[ -z "${VTLA_GPUS:-}" ]]; then
  echo "VTLA_GPUS must be set explicitly (for example: VTLA_GPUS=0,3)." >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed in this training container." >&2
  exit 1
fi
if tmux has-session -t "${session_name}" 2>/dev/null; then
  echo "tmux session already exists: ${session_name}" >&2
  exit 1
fi
if [[ -e "${output_dir}" ]]; then
  echo "Output directory already exists: ${output_dir}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output_dir}")" "$(dirname "${log_file}")"

export VTLA_OUTPUT_DIR="${output_dir}"
export VTLA_WANDB_ENABLE="${VTLA_WANDB_ENABLE:-true}"
export VTLA_WANDB_PROJECT="${project_name}"
export VTLA_WANDB_MODE="${VTLA_WANDB_MODE:-online}"
export VTLA_WANDB_RUN_NAME="${VTLA_WANDB_RUN_NAME:-$(basename "${output_dir}")}"
export VTLA_WANDB_DISABLE_ARTIFACT="${VTLA_WANDB_DISABLE_ARTIFACT:-true}"

printf -v tmux_command \
  'cd %q && set -o pipefail; ./scripts/training/train_lerobot_vtla.sh 2>&1 | tee -a %q; training_status=${PIPESTATUS[0]}; echo "VTLA training exited with status ${training_status}" | tee -a %q; exec bash' \
  "${repo_root}" "${log_file}" "${log_file}"

tmux new-session -d -s "${session_name}" "bash -lc $(printf '%q' "${tmux_command}")"

echo "Started VTLA training in tmux session ${session_name}."
echo "  GPUs: ${VTLA_GPUS}"
echo "  W&B project: ${project_name}"
echo "  Output: ${output_dir}"
echo "  Log: ${log_file}"
echo "Attach with: tmux attach -t ${session_name}"
