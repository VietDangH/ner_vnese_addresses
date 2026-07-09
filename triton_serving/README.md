# Address NER — Triton Inference Server

The **Triton** counterpart to [`serving/`](../serving/) (FastAPI). Same model,
same predictions — different serving stack.

**Reuses the exact same `serving/model_service.py`** the FastAPI app uses
(segmentation → PhoBERT → softmax decode → token→char alignment). So a given
address returns a byte-for-byte identical JSON from either server. The only thing
that changes is the transport.

## Layout

```
triton_serving/
├── model_repository/
│   └── address_ner/
│       ├── config.pbtxt        # backend, batching, I/O tensors, instances
│       └── 1/                  # model version 1
│           └── model.py        # Python backend: reuses serving.model_service
├── client.py                   # HTTP client + tiny benchmark
├── requirements-client.txt     # tritonclient (your machine)
├── requirements-server.txt     # transformers/sentencepiece (inside container)
├── run_server.ps1 / .sh        # docker run helpers
└── README.md
```

## 1. Start the server

Triton runs in NVIDIA's container. **Run from the repo root** (`d:\ner_vnese_addresses`)
so the whole repo mounts at `/workspace` and the backend can import `PhoBERT`,
`serving`, and reuse the already-extracted `.model_cache/` model:

```powershell
# Windows PowerShell (Docker Desktop)
.\triton_serving\run_server.ps1
```
```bash
# Linux / macOS
bash triton_serving/run_server.sh
```

The script `pip install`s `requirements-server.txt` into the container, then
launches `tritonserver`. Wait for the `READY` table:

```
+-------------+---------+--------+
| Model       | Version | Status |
+-------------+---------+--------+
| address_ner | 1       | READY  |
+-------------+---------+--------+
```

Ports: **8000** HTTP · **8001** gRPC · **8002** Prometheus metrics.

> **First load** downloads the base `vinai/phobert-base` from the HF Hub (needs
> internet once; the mounted `~/.cache/huggingface` makes it offline after). This
> is the same requirement as the FastAPI service.
> **CPU-only?** Remove `--gpus all` from the run script.

## 2. Call it

On your machine (not the container):

```bash
pip install -r triton_serving/requirements-client.txt
```

```bash
# readiness — like GET /health
python triton_serving/client.py health

# tag one address — like POST /predict
python triton_serving/client.py predict "123 đường lê lợi, phường bến nghé, quận 1, tp hcm"

# tag a list — like POST /batch_predict
python triton_serving/client.py batch "204/9 hải phòng, cát hải" "8 tràng tiền, hoàn kiếm"

# tag a .txt/.csv upload — like POST /batch_predict/file
python triton_serving/client.py file addresses.txt
```

`predict` returns the same JSON as FastAPI's `/predict` (`start`/`end` are
character offsets):

```json
{
  "text": "123 đường lê lợi, phường bến nghé, quận 1, tp hcm",
  "entities": [
    { "text": "123", "label": "HOUSE_NO", "start": 0, "end": 3, "confidence": 0.9999 },
    { "text": "đường lê lợi", "label": "STREET", "start": 4, "end": 16, "confidence": 0.9999 },
    { "text": "phường bến nghé", "label": "WARD", "start": 18, "end": 33, "confidence": 0.9999 }
  ],
  "latency_ms": 12.3
}
```
