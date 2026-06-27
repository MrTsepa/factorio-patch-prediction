#!/usr/bin/env bash
# Build the training image and run it on a LOCAL NVIDIA GPU (sanity check before
# pushing to the cloud). Requires Docker + the NVIDIA Container Toolkit.
# Artifacts land in ./cloud_out (runs/ + outputs/).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-factorio-patch:latest}"

echo "Building $IMAGE for linux/amd64 ..."
docker buildx build --platform linux/amd64 -f cloud/Dockerfile -t "$IMAGE" --load .

mkdir -p cloud_out
echo "Running training on local GPU ..."
docker run --rm --gpus all \
  -e EPOCHS="${EPOCHS:-24}" -e EMPTY_WEIGHT="${EMPTY_WEIGHT:-0.12}" \
  -v "$PWD/cloud_out:/workspace" \
  "$IMAGE"

echo "Done. See ./cloud_out/runs and ./cloud_out/outputs"
