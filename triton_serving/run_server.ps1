# Start Triton serving the address_ner model. Run from the REPO ROOT:
#
#   .\triton_serving\run_server.ps1
#
# The whole repo is mounted at /workspace so the Python backend can import
# vn_ner/ and serving/, and reuse the already-extracted .model_cache/ model.
# Remove "--gpus","all" on a CPU-only host. Requires Docker Desktop.

$Image = if ($env:TRITON_IMAGE) { $env:TRITON_IMAGE } else { "nvcr.io/nvidia/tritonserver:24.08-py3" }

docker run --rm -it `
  --gpus all `
  --shm-size=1g `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v "${PWD}:/workspace" `
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" `
  -w /workspace `
  $Image `
  bash -lc "pip install -q -r triton_serving/requirements-server.txt && tritonserver --model-repository=/workspace/triton_serving/model_repository"
