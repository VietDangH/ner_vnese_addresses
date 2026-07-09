# -*- coding: utf-8 -*-
"""Inference pipeline: load a trained model and tag new addresses.

Typical use::

    from vn_ner import Predictor
    p = Predictor.from_pretrained("ner_model")
    p.predict("123 đường lê lợi, phường bến nghé, quận 1, tp hcm")

returns a dict with per-token tags, per-token confidence scores, an overall
sentence confidence, a per-call latency, and grouped entities, e.g.::

    {
      "tokens":   ["123", "đường", "lê_lợi", ...],
      "tags":     ["B-HOUSE_NO", "B-STREET", "I-STREET", ...],
      "token_confidences": [0.99, 0.98, 0.97, ...],   # P(tag|x) (softmax / CRF marginal)
      "confidence": 0.97,        # mean over tokens (sentence-level confidence)
      "min_confidence": 0.81,    # weakest token (useful to flag for review)
      "latency_ms": 12.3,        # wall-clock time of this predict() call
      "entities": [{"type": "HOUSE_NO", "text": "123", "confidence": 0.99,
                    "start": 0, "end": 1}, ...]
    }

It works for any trained architecture (softmax / CRF / +BiLSTM / ...): the model
is rebuilt from the saved ``config.json`` and the confidence is whatever the
head provides.
"""

import time
from typing import Dict, List, Optional

import torch

from .data import Collator
from .models import load_trained
from .segment import segment


class Predictor:
    def __init__(self, model, tokenizer, label_list, max_length, device):
        self.model = model
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_length = max_length
        self.device = device
        self.collate = Collator(tokenizer, self.label2id, max_length)
        self.model.eval()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(cls, model_dir: str, device: Optional[str] = None):
        """Load model weights + tokenizer + label maps from a training output dir."""
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, tokenizer, label_list, _id2label, cfg = load_trained(model_dir, dev)
        return cls(model, tokenizer, label_list, cfg.get("max_length", 256), dev)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _decode(self, tokens: List[str]):
        """Decode tokens -> ``(tags, confidences)`` (confidence per token)."""
        if not tokens:
            return [], []
        batch = self.collate([(tokens, ["O"] * len(tokens))])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        preds, confs = self.model.decode_with_confidence(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            word_positions=batch["word_positions"],
            word_mask=batch["word_mask"],
        )
        n = int(batch["word_mask"].sum().item())
        tags = [self.label_list[t] for t in preds[0][:n]]
        return tags, confs[0][:n]

    @torch.no_grad()
    def _decode_batch(self, token_lists: List[List[str]]):
        """Decode many token-lists in ONE model forward pass.

        Returns a list of ``(tags, confidences)`` aligned 1:1 with ``token_lists``.
        Empty token-lists are kept out of the padded batch (so a blank input never
        distorts the tensor) and map to ``([], [])``.
        """
        results = [([], []) for _ in token_lists]
        indexed = [(i, toks) for i, toks in enumerate(token_lists) if toks]
        if not indexed:
            return results
        batch = self.collate([(toks, ["O"] * len(toks)) for _, toks in indexed])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        preds, confs = self.model.decode_with_confidence(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            word_positions=batch["word_positions"],
            word_mask=batch["word_mask"],
        )
        # heads already trim each row to its real word count (see models/heads.py)
        for (orig_i, _toks), p, c in zip(indexed, preds, confs):
            results[orig_i] = ([self.label_list[t] for t in p], [float(x) for x in c])
        return results

    def predict_tokens(self, tokens: List[str]) -> List[str]:
        """Tag an already word-segmented list of tokens -> list of BIO labels."""
        tags, _ = self._decode(tokens)
        return tags

    def predict(self, text: str, lower: bool = True) -> Dict:
        """Tag a raw address string. Segments the text first, then decodes.

        Carries per-token / sentence-level confidence and ``latency_ms`` (the
        wall-clock time of this call: segmentation + model inference).
        """
        t0 = time.perf_counter()
        tokens = segment(text, lower=lower)
        tags, confidences = self._decode(tokens)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        overall = float(sum(confidences) / len(confidences)) if confidences else 0.0
        return {
            "text": text,
            "tokens": tokens,
            "tags": tags,
            "token_confidences": [round(float(c), 4) for c in confidences],
            "confidence": round(overall, 4),
            "min_confidence": round(float(min(confidences)), 4) if confidences else 0.0,
            "latency_ms": round(latency_ms, 3),
            "entities": group_entities(tokens, tags, confidences),
        }

    def predict_batch(self, texts: List[str], lower: bool = True) -> List[Dict]:
        """Tag many raw strings with ONE batched model forward pass.

        Segmentation stays per-text (it calls an external segmenter), but the
        expensive encoder + head forward runs once over the whole padded batch.
        This is what lets a batched server (e.g. Triton's dynamic batcher) amortise
        GPU cost across concurrent requests. Each item has the same shape as
        :meth:`predict`; ``latency_ms`` is the amortised per-item wall time.
        """
        t0 = time.perf_counter()
        token_lists = [segment(t, lower=lower) if (t and t.strip()) else []
                       for t in texts]
        decoded = self._decode_batch(token_lists)
        per_ms = round((time.perf_counter() - t0) * 1000.0 / max(len(texts), 1), 3)

        results: List[Dict] = []
        for text, tokens, (tags, confidences) in zip(texts, token_lists, decoded):
            overall = float(sum(confidences) / len(confidences)) if confidences else 0.0
            results.append({
                "text": text,
                "tokens": tokens,
                "tags": tags,
                "token_confidences": [round(float(c), 4) for c in confidences],
                "confidence": round(overall, 4),
                "min_confidence": round(float(min(confidences)), 4) if confidences else 0.0,
                "latency_ms": per_ms,
                "entities": group_entities(tokens, tags, confidences),
            })
        return results


def group_entities(tokens: List[str], tags: List[str],
                   confidences: Optional[List[float]] = None) -> List[Dict]:
    """Merge BIO tags into entity spans with surface text (``_`` -> space).

    If ``confidences`` is given, each entity also gets a ``confidence`` field
    (the mean of its tokens' confidence scores).
    """
    entities = []
    cur_type, cur_start, cur_toks, cur_confs = None, None, [], []

    def flush(end):
        if cur_type is not None:
            text = " ".join(cur_toks).replace("_", " ")
            ent = {"type": cur_type, "text": text, "start": cur_start, "end": end}
            if confidences is not None and cur_confs:
                ent["confidence"] = round(sum(cur_confs) / len(cur_confs), 4)
            entities.append(ent)

    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        conf = confidences[i] if confidences is not None else None
        if tag == "O":
            flush(i)
            cur_type, cur_start, cur_toks, cur_confs = None, None, [], []
            continue
        prefix, _, etype = tag.partition("-")
        if prefix == "B" or etype != cur_type:
            flush(i)
            cur_type, cur_start, cur_toks = etype, i, [tok]
            cur_confs = [conf] if conf is not None else []
        else:                               # I- continuing same type
            cur_toks.append(tok)
            if conf is not None:
                cur_confs.append(conf)
    flush(len(tokens))
    return entities
