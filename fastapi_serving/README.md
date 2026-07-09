# Address NER — FastAPI Serving

## Project structure

```
serving/
├── app.py            # FastAPI app: lifespan (loads model once) + routes
├── model_service.py  # NERService: in-memory model + token→char offset mapping
├── schemas.py        # Pydantic request/response models (drive validation + docs)
├── files.py          # parse uploaded .txt / .csv into a list of texts
├── settings.py       # config via env vars; resolves a model dir or .zip
└── requirements.txt
```

Separation of concerns: routes/validation (`app.py`, `schemas.py`) are independent
of *how* predictions are made (`model_service.py`). To serve a different model
later, change only `model_service.py` — the API contract stays the same.

## 1. Install

```bash
pip install -r serving/requirements.txt
```

(The service imports the `PhoBERT` package from the repo root; `settings.py` adds
the repo root to `sys.path`, so it works regardless of where you launch uvicorn.)

## 2. Point at your model

By default it loads the `PhoBERT_cnn_crf_seed_43/` model directory from the repo
root (a PhoBERT + CNN + CRF model). A `.zip` of a model works too and is unpacked
once into `.model_cache/`. Override with an env var to use a different directory
or zip:

```bash
# Windows PowerShell
$env:NER_MODEL_PATH = "D:\NER_for_address\PhoBERT_cnn_crf_seed_43"
# macOS/Linux
export NER_MODEL_PATH=/path/to/model_dir_or.zip
```

> First load also needs the base `vinai/phobert-base` (from the HF cache or
> online). Other env vars: `NER_EXTRACT_DIR`, `NER_MAX_BATCH` (default 1000).

## 3. Run

```bash
uvicorn serving.app:app --reload
# or, from inside the serving/ folder:
cd serving && uvicorn app:app --reload
```

Open the interactive docs: **http://localhost:8000/docs**

## Endpoints

| Method | Path | Body | Purpose |
|--------|------|------|---------|
| GET  | `/`                   | — | service info |
| GET  | `/health`             | — | readiness + model labels |
| POST | `/predict`            | `{"text": "..."}` | tag one address |
| POST | `/batch_predict`      | `{"texts": ["...", "..."]}` | tag a list |
| POST | `/batch_predict/file` | multipart file | tag a `.txt`/`.csv` upload |

### `POST /predict`

Request:
```json
{ "text": "123 đường lê lợi, phường bến nghé, quận 1, tp hcm" }
```
Response (`start`/`end` are **character offsets** in the input):
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

`label` is one of the model's entity types: `HOUSE_NO, ALLEY, STREET, AREA,
BUILDING, WARD, DISTRICT, CITY`.

### `POST /batch_predict/file`

Upload a file:
- `.txt` — one address per line.
- `.csv` — an `address` column if present, otherwise the first column.

Returns `{ "count": N, "results": [ <PredictResponse>, ... ] }`.

## Testing via Swagger UI

1. Start the server and open **http://localhost:8000/docs**.
2. Expand **POST `/predict`** → **Try it out** → edit the JSON → **Execute**.
   The response body, status code, and a ready-made `curl` are shown.
3. For **`/batch_predict`**, send a list of texts the same way.
4. For **`/batch_predict/file`**, click **Choose File**, pick a `.txt`/`.csv`,
   and **Execute**.
5. Try the error cases: an empty `text`, a malformed body, or a `.json` upload —
   you'll get `422` / `400` with a clear `detail` message.

`curl` equivalents:
```bash
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"123 đường lê lợi, quận 1, tp hcm"}'

curl -X POST localhost:8000/batch_predict/file -F "file=@addresses.txt"
```

## Error handling

| Situation | Status | Source |
|-----------|--------|--------|
| empty / missing `text`, empty `texts` | `422` | Pydantic validation |
| malformed JSON | `422` | FastAPI |
| bad/empty/wrong-type upload | `400` | `files.parse_upload` |
| batch too large | `413` | route guard (`NER_MAX_BATCH`) |
| prediction failure | `500` | route `_run` wrapper |
| model not loaded | `503` | route guard |