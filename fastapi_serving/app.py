# -*- coding: utf-8 -*-
"""FastAPI app for serving the PhoBERT address-NER model.

Endpoints
---------
    GET  /              service info
    GET  /health        readiness + loaded model labels
    POST /predict       tag a single text (JSON)
    POST /batch_predict tag many texts (JSON list) or an uploaded .txt/.csv file

Run it (from the repo root or this folder):

    pip install -r serving/requirements.txt
    uvicorn serving.app:app --reload          # or, from inside serving/: uvicorn app:app --reload

Interactive docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from . import files, model_service, settings
from .schemas import (BatchPredictResponse, BatchTextRequest, HealthResponse,
                      PredictRequest, PredictResponse)


# --------------------------------------------------------------------------- #
# Lifespan: load the model ONCE at startup, keep it in memory for all requests.
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] loading model from {settings.MODEL_PATH} ...")
    try:
        svc = model_service.load_service(settings.MODEL_PATH)
        print(f"[startup] model ready ({len(svc.labels)} labels) "
              f"from {svc.model_dir}")
    except Exception as exc:                    # don't crash silently
        # Leave the service unloaded; /health reports it and routes return 503.
        print(f"[startup] FAILED to load model: {exc}")
    yield
    print("[shutdown] bye")


app = FastAPI(
    title="PhoBERT Address NER — Serving API",
    description="A minimal FastAPI service that tags Vietnamese addresses with a "
                "trained PhoBERT NER model. See /docs to try it.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _service() -> model_service.NERService:
    """Return the loaded service or raise 503 if the model failed to load."""
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="model is not loaded")
    return model_service.get_service()


def _run(fn, *args):
    """Run an inference call, turning unexpected failures into HTTP 500."""
    try:
        return fn(*args)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"prediction failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", tags=["meta"])
def root():
    return {"service": "PhoBERT Address NER", "docs": "/docs",
            "endpoints": ["/health", "/predict", "/batch_predict"]}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    loaded = model_service.is_loaded()
    svc = model_service.get_service() if loaded else None
    return HealthResponse(
        status="ok" if loaded else "model_not_loaded",
        model_loaded=loaded,
        model_dir=svc.model_dir if svc else None,
        labels=svc.labels if svc else None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["predict"],
          responses={503: {"description": "Model not loaded"}})
def predict(request: PredictRequest):
    """Tag a single address string and return its entities (with char offsets)."""
    svc = _service()
    return _run(svc.predict, request.text)


@app.post("/batch_predict", response_model=BatchPredictResponse, tags=["predict"])
def batch_predict(request: BatchTextRequest):
    """Tag a JSON list of texts. (For file uploads, use ``/batch_predict/file``.)"""
    svc = _service()
    if len(request.texts) > settings.MAX_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"too many texts ({len(request.texts)} > "
                   f"{settings.MAX_BATCH}); split into smaller batches")
    results = _run(svc.predict_batch, request.texts)
    return BatchPredictResponse(count=len(results), results=results)


@app.post("/batch_predict/file", response_model=BatchPredictResponse,
          tags=["predict"])
async def batch_predict_file(file: UploadFile = File(...)):
    """Tag every line of an uploaded ``.txt`` (one text/line) or ``.csv``
    (an ``address`` column, else the first column)."""
    svc = _service()
    try:
        content = await file.read()
        texts = files.parse_upload(file.filename, content)
    except ValueError as exc:                   # bad/empty/wrong-type file
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if len(texts) > settings.MAX_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"file has too many lines ({len(texts)} > {settings.MAX_BATCH})")
    results = _run(svc.predict_batch, texts)
    return BatchPredictResponse(count=len(results), results=results)
