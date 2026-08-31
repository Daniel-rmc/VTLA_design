#!/usr/bin/env bash
set -euo pipefail

workspace_root="${WORKSPACE_ROOT:-/home/rmc/workspace}"
datasets_root="${VTLA_DATASETS_ROOT:-${workspace_root}/datasets}"
image_name="${VTLA_IMAGE:-vtla-train:lerobot-0.6.1}"
container_name="${VTLA_CONTAINER_NAME:-vtla_train}"
shm_size="120g"

if [[ ! -d "${datasets_root}" ]]; then
  echo "Dataset root does not exist: ${datasets_root}" >&2
  echo "Set VTLA_DATASETS_ROOT to the host directory mounted at /workspace/datasets." >&2
  exit 1
fi

if docker container inspect "${container_name}" >/dev/null 2>&1; then
  required_image="$(docker image inspect --format '{{.Id}}' "${image_name}")"
  existing_image="$(docker inspect --format '{{.Image}}' "${container_name}")"
  existing_shm="$(docker inspect --format '{{.HostConfig.ShmSize}}' "${container_name}")"
  existing_restart="$(docker inspect --format '{{.HostConfig.RestartPolicy.Name}}' "${container_name}")"
  existing_gpu_count="$(docker inspect --format '{{range .HostConfig.DeviceRequests}}{{.Count}}{{end}}' "${container_name}")"
  if [[ "${existing_image}" != "${required_image}" || "${existing_shm}" != "128849018880" || \
        "${existing_restart}" != "unless-stopped" || "${existing_gpu_count}" != "-1" ]]; then
    echo "Existing ${container_name} does not match the required image/GPU/120G SHM/restart configuration." >&2
    echo "Remove or rename it explicitly before recreating the training container." >&2
    exit 1
  fi
  if [[ "$(docker inspect --format '{{.State.Running}}' "${container_name}")" != "true" ]]; then
    docker start "${container_name}" >/dev/null
  fi
else
  docker run --detach \
    --name "${container_name}" \
    --restart unless-stopped \
    --gpus all \
    --shm-size="${shm_size}" \
    --init \
    --workdir /workspace/VTLA_design \
    --volume "${workspace_root}:/workspace" \
    --volume "${datasets_root}:/workspace/datasets" \
    --volume vtla-cache:/home/vtla/.cache \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --env "NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}" \
    --env "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}" \
    "${image_name}" \
    sleep infinity >/dev/null
fi

echo "${container_name} is running. Enter it with:"
echo "  docker exec -it ${container_name} bash"
