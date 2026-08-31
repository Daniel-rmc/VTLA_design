#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${VTLA_IMAGE:-vtla-train:lerobot-0.6.1}"

docker build \
  --file "${repo_root}/docker/Dockerfile.train" \
  --build-arg "LOCAL_UID=$(id -u)" \
  --build-arg "LOCAL_GID=$(id -g)" \
  --tag "${image_name}" \
  "${repo_root}"
