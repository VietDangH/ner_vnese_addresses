# -*- coding: utf-8 -*-
"""Detailed performance reporting.

``seqeval`` gives entity-level (span-level) precision / recall / F1, which is
the metric people actually quote for NER -- a span counts as correct only if
its boundaries *and* type both match. On top of that we add:

  * micro / macro / weighted F1,
  * per-entity-type breakdown,
  * token-level accuracy,
  * a token-level confusion matrix (which tags get confused for which),

and we can dump the whole thing to JSON + a human-readable text report.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def _to_native(obj):
    """Recursively convert numpy scalars (int64/float64) to Python types so the
    metrics dict is JSON-serializable -- seqeval reports use numpy under the hood."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            return obj
    return obj


def _safe(fn, *a, **k):
    try:
        return float(fn(*a, **k))
    except Exception:
        return 0.0


def compute_metrics(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
    """Entity-level + token-level metrics as a plain dict (JSON-friendly)."""
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0, digits=4
    )
    # token-level accuracy across all tokens
    flat_true = [t for seq in y_true for t in seq]
    flat_pred = [p for seq in y_pred for p in seq]
    token_correct = sum(t == p for t, p in zip(flat_true, flat_pred))
    token_total = max(len(flat_true), 1)

    # sentence-level accuracy: a sentence counts as correct only if EVERY token
    # in it is predicted correctly (exact-match of the whole label sequence).
    sent_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    n_sent = max(len(y_true), 1)

    return _to_native({
        "entity": {
            "precision": _safe(precision_score, y_true, y_pred, zero_division=0),
            "recall": _safe(recall_score, y_true, y_pred, zero_division=0),
            "f1": _safe(f1_score, y_true, y_pred, zero_division=0),
        },
        "token_accuracy": token_correct / token_total,
        "sentence_accuracy": sent_correct / n_sent,
        "n_correct_sentences": sent_correct,
        "per_type": report,
        "n_sentences": len(y_true),
        "n_tokens": token_total,
    })


def token_confusion_matrix(y_true: List[List[str]], y_pred: List[List[str]]):
    """Return ``(labels, matrix)`` -- a token-level confusion matrix.

    ``matrix[i][j]`` = number of tokens whose gold tag is ``labels[i]`` and
    predicted tag is ``labels[j]``. Pure-Python so it needs no sklearn.
    """
    labels = sorted({t for seq in y_true for t in seq} |
                    {p for seq in y_pred for p in seq})
    index = {l: i for i, l in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for ts, ps in zip(y_true, y_pred):
        for t, p in zip(ts, ps):
            matrix[index[t]][index[p]] += 1
    return labels, matrix


def format_report(metrics: Dict, y_true=None, y_pred=None,
                  title: str = "Performance") -> str:
    """Build a pretty, multi-section text report."""
    lines = []
    lines.append("=" * 64)
    lines.append(title)
    lines.append("=" * 64)
    ent = metrics["entity"]
    lines.append(f"Sentences               : {metrics['n_sentences']}")
    lines.append(f"Tokens                  : {metrics['n_tokens']}")
    lines.append("")
    lines.append("Entity-level (span) — strict BIO matching")
    lines.append(f"  Precision : {ent['precision']:.4f}")
    lines.append(f"  Recall    : {ent['recall']:.4f}")
    lines.append(f"  F1        : {ent['f1']:.4f}")
    lines.append(f"Token-level accuracy    : {metrics['token_accuracy']:.4f}")
    sc = metrics.get("n_correct_sentences", 0)
    lines.append(f"Sentence-level accuracy : {metrics['sentence_accuracy']:.4f}"
                 f"  ({sc}/{metrics['n_sentences']} sentences fully correct)")
    lines.append("")

    # Per-entity-type table (skip the aggregate rows that seqeval adds).
    lines.append("Per-entity-type breakdown")
    lines.append(f"  {'type':<16}{'precision':>10}{'recall':>10}"
                 f"{'f1':>10}{'support':>10}")
    lines.append("  " + "-" * 56)
    per_type = metrics["per_type"]
    aggregates = {"micro avg", "macro avg", "weighted avg", "accuracy"}
    for name, vals in per_type.items():
        if name in aggregates or not isinstance(vals, dict):
            continue
        lines.append(f"  {name:<16}{vals['precision']:>10.4f}"
                     f"{vals['recall']:>10.4f}{vals['f1-score']:>10.4f}"
                     f"{int(vals['support']):>10}")
    lines.append("  " + "-" * 56)
    for agg in ("micro avg", "macro avg", "weighted avg"):
        if agg in per_type:
            v = per_type[agg]
            lines.append(f"  {agg:<16}{v['precision']:>10.4f}"
                         f"{v['recall']:>10.4f}{v['f1-score']:>10.4f}"
                         f"{int(v['support']):>10}")

    # Optional token-level confusion matrix.
    if y_true is not None and y_pred is not None:
        labels, matrix = token_confusion_matrix(y_true, y_pred)
        lines.append("")
        lines.append("Token-level confusion matrix (rows = gold, cols = pred)")
        short = [l.replace("B-", "B").replace("I-", "I") for l in labels]
        header = " " * 10 + "".join(f"{s[:8]:>9}" for s in short)
        lines.append(header)
        for i, row in enumerate(matrix):
            lines.append(f"{short[i][:9]:>9} " + "".join(f"{c:>9}" for c in row))

    lines.append("=" * 64)
    return "\n".join(lines)


def save_report(metrics: Dict, out_dir, y_true=None, y_pred=None,
                title: str = "Test performance", prefix: str = "test"):
    """Write ``<prefix>_metrics.json`` and ``<prefix>_report.txt`` to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = format_report(metrics, y_true, y_pred, title=title)
    (out_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(_to_native(metrics), ensure_ascii=False, indent=2,
                   default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    (out_dir / f"{prefix}_report.txt").write_text(text, encoding="utf-8")
    return text
