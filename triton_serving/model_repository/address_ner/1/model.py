# -*- coding: utf-8 -*-
"""Triton Python-backend model that serves the PhoBERT address-NER model.

This is the Triton counterpart to the FastAPI service in ``serving/``. It exists
so the two serving stacks can be compared head-to-head on the *same* model with
the *same* pre/post-processing. To guarantee that, this backend deliberately
**reuses** ``serving.model_service.NERService`` (segmentation -> PhoBERT ->
softmax decode -> token->char alignment). A given address therefore yields a
byte-for-byte identical prediction from either server; the only thing that
differs is the transport:

    FastAPI  ->  uvicorn (Python ASGI worker)
    Triton   ->  C++ inference core + dynamic batching + N model instances

Wire format
-----------
    input   TEXT    TYPE_STRING, dims [1]   one (or a per-request batch of)
                                            address string(s)
    output  RESULT  TYPE_STRING, dims [1]   the prediction as a JSON string,
                                            one per input text

The JSON payload matches the FastAPI ``/predict`` response exactly::

    {"text": ...,
     "entities": [{"text","label","start","end","confidence"}, ...],
     "latency_ms": ...}
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils


def _find_repo_root() -> str:
    """Locate the repository root (the dir containing ``vn_ner/`` + ``serving/``).

    Works both when the whole repo is mounted (local dev / ``-v $PWD:/workspace``)
    and when only the model repository is mounted (then set ``NER_REPO_ROOT``).
    """
    env = os.environ.get("NER_REPO_ROOT")
    if env and (Path(env) / "vn_ner").is_dir():
        return str(Path(env).resolve())
    for parent in Path(__file__).resolve().parents:      # walk up from this file
        if (parent / "vn_ner").is_dir() and (parent / "serving").is_dir():
            return str(parent)
    if (Path("/workspace") / "vn_ner").is_dir():          # common docker mount
        return "/workspace"
    raise RuntimeError(
        "could not locate the repo root (a dir with vn_ner/ and serving/). "
        "Set the NER_REPO_ROOT environment variable to the repository path.")


class TritonPythonModel:
    """Loads the model once (``initialize``) and serves predictions (``execute``)."""

    def initialize(self, args):
        repo_root = _find_repo_root()
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # If NER_MODEL_PATH isn't set, prefer the default model dir at the repo
        # root (a ready directory -> fast startup, no unzip). If it's absent,
        # serving/settings.py's own default still applies.
        if not os.environ.get("NER_MODEL_PATH"):
            default_dir = Path(repo_root) / "PhoBERT_cnn_crf_seed_43"
            if (default_dir / "config.json").exists():
                os.environ["NER_MODEL_PATH"] = str(default_dir)

        # Reuse the SAME service the FastAPI app uses -> identical predictions.
        from serving import model_service, settings
        self._svc = model_service.load_service(settings.MODEL_PATH)

        # Resolve output dtype from the parsed config (TYPE_STRING -> np.object_).
        model_config = json.loads(args["model_config"])
        out_cfg = pb_utils.get_output_config_by_name(model_config, "RESULT")
        self._out_dtype = pb_utils.triton_string_to_numpy(out_cfg["data_type"])

        print(f"[triton/address_ner] model ready "
              f"({len(self._svc.labels)} labels) from {self._svc.model_dir}",
              flush=True)

    def execute(self, requests):
        """Run inference for a list of requests, returning one response each.

        Triton's dynamic batcher merges many concurrent requests into a single
        ``execute`` call. To exploit that, we gather every text across every
        request into ONE list, run a single batched forward
        (``NERService.predict_batch`` -> one padded model pass), then scatter the
        predictions back to their originating requests. This is the step that lets
        Triton's batching amortise GPU cost — a per-request loop could not.

        A failure in one request is returned as that request's error and does not
        take down the server or the other requests in the batch.
        """
        # 1) gather: flatten all texts, remember each request's slice (or None on
        #    a parse error so we can emit that request's error later).
        all_texts, counts = [], []
        for request in requests:
            try:
                in_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
                texts = self._to_texts(in_tensor.as_numpy())
                counts.append(len(texts))
                all_texts.extend(t if t.strip() else "" for t in texts)
            except Exception:
                counts.append(None)

        # 2) one batched forward for the whole execute() call.
        try:
            preds = self._svc.predict_batch(all_texts) if all_texts else []
        except Exception as exc:                         # whole-batch failure
            err = pb_utils.TritonError(f"prediction failed: {exc}")
            return [pb_utils.InferenceResponse(output_tensors=[], error=err)
                    for _ in requests]

        # 3) scatter: hand each request back its own slice of predictions.
        responses, cursor = [], 0
        for cnt in counts:
            if cnt is None:                              # this request failed to parse
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError("prediction failed: bad TEXT input")))
                continue
            chunk = preds[cursor:cursor + cnt]
            cursor += cnt
            payload = [json.dumps(p, ensure_ascii=False).encode("utf-8")
                       for p in chunk]
            out = pb_utils.Tensor(
                "RESULT",
                np.array(payload, dtype=self._out_dtype).reshape(len(payload), 1))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses

    @staticmethod
    def _to_texts(arr):
        """Flatten a TYPE_STRING input tensor (shape [N,1]) into a ``list[str]``."""
        texts = []
        for item in arr.reshape(-1).tolist():
            texts.append(item.decode("utf-8") if isinstance(item, bytes) else item)
        return texts

    def finalize(self):
        self._svc = None
