#!/usr/bin/env bash
# Start Triton serving the address_ner model. Run from the REPO ROOT.
#
#   bash triton_serving/run_server.sh
#
# The whole repo is mounted at /workspace so the Python backend can import
# vn_ner/ and serving/, and reuse the already-extracted .model_cache/ model.
# Drop `--gpus all` on a CPU-only host. The HF cache mount lets the base
# vinai/phobert-base weights be reused offline after the first download.
set -euo pipefail

IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.08-py3}"

docker run --rm -it \
  --gpus all \
  --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$(pwd)":/workspace \
  -v "${HOME}/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace \
  "${IMAGE}" \
  bash -lc "pip install -q -r triton_serving/requirements-server.txt && \
            tritonserver --model-repository=/workspace/triton_serving/model_repository"
