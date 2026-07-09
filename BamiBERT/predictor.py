# -*- coding: utf-8 -*-
"""Inference pipeline: load a trained model and tag new addresses.

Typical use::

    from bami_ner import Predictor
    p = Predictor.from_pretrained("bami_ner_model")
    p.predict("123 đường lê lợi, phường bến nghé, quận 1, tp hcm")

Unlike the PhoBERT pipeline, there is **no word-segmentation step**: BamiBERT
consumes raw text, so the input is only split on whitespace/punctuation (see
``rawtext.raw_tokenize``).

Returns a dict with per-token tags, per-token confidence scores, an overall
sentence confidence, a per-call latency, and grouped entities, e.g.::

    {
      "tokens":   ["123", "đường", "lê", "lợi", ...],
      "tags":     ["B-HOUSE_NO", "B-STREET", "I-STREET", "I-STREET", ...],
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
from .rawtext import raw_tokenize


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

    def predict_tokens(self, tokens: List[str]) -> List[str]:
        """Tag an already-tokenized list of tokens -> list of BIO labels."""
        tags, _ = self._decode(tokens)
        return tags

    def predict(self, text: str, lower: bool = True) -> Dict:
        """Tag a raw address string. Splits the text first, then decodes.

        Carries per-token / sentence-level confidence and ``latency_ms`` (the
        wall-clock time of this call: tokenization + model inference).
        """
        t0 = time.perf_counter()
        tokens = raw_tokenize(text, lower=lower)
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
        return [self.predict(t, lower=lower) for t in texts]


def group_entities(tokens: List[str], tags: List[str],
                   confidences: Optional[List[float]] = None) -> List[Dict]:
    """Merge BIO tags into entity spans with surface text.

    If ``confidences`` is given, each entity also gets a ``confidence`` field
    (the mean of its tokens' confidence scores).
    """
    entities = []
    cur_type, cur_start, cur_toks, cur_confs = None, None, [], []

    def flush(end):
        if cur_type is not None:
            text = " ".join(cur_toks)
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
