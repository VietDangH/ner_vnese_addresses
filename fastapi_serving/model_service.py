# -*- coding: utf-8 -*-
"""Model service: load the trained NER model once and serve predictions.

The model is loaded a single time (at app startup, via ``load_service``) and kept
in memory as a module-level singleton; every request reuses it. Routes obtain it
with ``get_service`` and never touch the model internals directly — so swapping
in a different model later only means changing this file.

The underlying model (a PhoBERT NER model from the ``vn_ner`` package) returns
*token-level* spans; this service converts them to *character offsets* in the
original input so the API response matches the requested shape.
"""

from typing import List, Optional

from . import settings


# --------------------------------------------------------------------------- #
# Token -> character offset alignment
# --------------------------------------------------------------------------- #
def _token_char_spans(text: str, tokens: List[str]):
    """Best-effort map of each (segmented, lower-cased) token to a ``(start, end)``
    char span in the original ``text``.

    PhoBERT input is word-segmented and lower-cased, so we locate each token's
    sub-word pieces (split on ``_``) in a lower-cased copy of the text, scanning
    forward. Returns one ``(start, end)`` per token, or ``None`` if a token could
    not be located (rare; the caller degrades gracefully)."""
    low = text.lower()
    spans, cursor = [], 0
    for tok in tokens:
        pieces = [p for p in tok.split("_") if p]
        if not pieces:
            spans.append(None)
            continue
        start = low.find(pieces[0], cursor)
        if start == -1:                       # fall back to a global search
            start = low.find(pieces[0])
        if start == -1:
            spans.append(None)
            continue
        end = start + len(pieces[0])
        for p in pieces[1:]:
            nxt = low.find(p, end)
            if nxt == -1:
                break
            end = nxt + len(p)
        spans.append((start, end))
        cursor = end
    return spans


def _entity_char_span(spans, start_tok: int, end_tok: int):
    """Char span for an entity covering tokens ``[start_tok, end_tok)``."""
    located = [s for s in spans[start_tok:end_tok] if s is not None]
    if not located:
        return None
    return located[0][0], located[-1][1]


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class NERService:
    """Thin wrapper around the trained model with a stable inference API."""

    def __init__(self, model_path: Optional[str] = None):
        # Imported lazily so importing this module is cheap and import errors in
        # the (heavy) model stack surface clearly at startup.
        from vn_ner import Predictor

        self.model_dir = settings.resolve_model_dir(model_path)
        self.predictor = Predictor.from_pretrained(self.model_dir)
        self.labels = list(self.predictor.label_list)

    def _align(self, text: str, res: dict) -> dict:
        """Convert one predictor result (token spans) to char-offset entities."""
        spans = _token_char_spans(text, res["tokens"])
        entities = []
        for e in res["entities"]:
            cs = _entity_char_span(spans, e["start"], e["end"])
            if cs is not None:
                start, end = cs
                surface = text[start:end]
            else:                              # alignment failed: keep model text
                start, end, surface = -1, -1, e["text"]
            entities.append({
                "text": surface,
                "label": e["type"],
                "start": start,
                "end": end,
                "confidence": e.get("confidence"),
            })
        return {"text": text, "entities": entities,
                "latency_ms": res.get("latency_ms")}

    def predict(self, text: str) -> dict:
        """Tag one string -> ``{text, entities[...], latency_ms}`` (char offsets)."""
        return self._align(text, self.predictor.predict(text))

    def predict_batch(self, texts: List[str]) -> List[dict]:
        """Tag many strings with ONE batched model forward pass (via
        :meth:`vn_ner.Predictor.predict_batch`). Empty/blank strings yield an empty
        entity list rather than failing the whole batch. This is the method a
        batched server (Triton's dynamic batcher) drives to amortise GPU cost."""
        clean = [t if (t and t.strip()) else "" for t in texts]
        return [self._align(text, res)
                for text, res in zip(clean, self.predictor.predict_batch(clean))]


# --------------------------------------------------------------------------- #
# Singleton management (the model is loaded once, at startup)
# --------------------------------------------------------------------------- #
_service: Optional[NERService] = None


def load_service(model_path: Optional[str] = None) -> NERService:
    """Instantiate the singleton service (called once from the app lifespan)."""
    global _service
    _service = NERService(model_path)
    return _service


def get_service() -> NERService:
    """Return the loaded service, or raise if it isn't ready."""
    if _service is None:
        raise RuntimeError("model service is not loaded")
    return _service


def is_loaded() -> bool:
    return _service is not None
