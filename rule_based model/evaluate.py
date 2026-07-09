# -*- coding: utf-8 -*-
"""Evaluation entry point for the rule-based baseline.

Scores the rule-based tagger on a CoNLL file and writes the **same** detailed
report as the neural pipeline (``vn_ner.evaluate`` / BamiBERT): entity-level
(span) P/R/F1, per-type breakdown, token + sentence accuracy, and a token-level
confusion matrix — because it reuses an identical metrics implementation
(:mod:`rule_ner.metrics`). Output files ``<prefix>_report.txt`` /
``<prefix>_metrics.json`` are drop-in comparable to the model packages'.

The baseline runs on the **raw, un-segmented** CoNLL (one syllable per token) so
it has no dependency on a word segmenter. ``Cleaned_Data_new_raw.conll`` is the
raw counterpart of the segmented ``Cleaned_Data_new.conll`` (same 6820 sentences,
same order), so scoring the same ``--split`` / ``--seed`` yields the test set that
corresponds one-to-one to a model trained on the segmented file::

    # score the held-out test split of the raw data (seed 41)
    python -m rule_ner.evaluate --data Cleaned_Data_new_raw.conll --split test --seed 41

    # or score a pre-split held-out file directly
    python -m rule_ner.evaluate --data raw_test.conll --split all
"""

import argparse
import sys
from pathlib import Path

from .data import dedupe, read_conll, split_sentences
from .metrics import compute_metrics, save_report
from .tagger import RuleBasedNER

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the rule-based address NER baseline")
    p.add_argument("--data", default="Cleaned_Data_new_raw.conll",
                   help="raw (un-segmented) CoNLL file to score — one syllable per token")
    p.add_argument("--split", choices=["all", "test", "val", "train"],
                   default="test",
                   help="'all' scores the whole file; the others re-create the "
                        "same internal split as vn_ner.split (dedupe + shuffle by seed)")
    p.add_argument("--gazetteer", default=None,
                   help="path to gazetteer.csv (auto-discovered at repo root if omitted)")
    p.add_argument("--out_dir", default="rule_model",
                   help="where to write <prefix>_report.txt / <prefix>_metrics.json")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=41,
                   help="split seed; 41 matches .model_cache/softmax_model_seed_41")
    p.add_argument("--report_prefix", default="eval")
    return p.parse_args()


def main():
    a = parse_args()

    model = RuleBasedNER(gazetteer_path=a.gazetteer)
    print(f"Loaded rule-based tagger "
          f"(gazetteer: {len(model.gaz.wards)} wards / "
          f"{len(model.gaz.districts)} districts / {len(model.gaz.cities)} cities)")

    sentences = read_conll(a.data)
    if a.split == "all":
        data = sentences
    else:
        deduped = dedupe(sentences)
        train_d, val_d, test_d = split_sentences(
            deduped, a.val_ratio, a.test_ratio, a.seed)
        data = {"train": train_d, "val": val_d, "test": test_d}[a.split]
    print(f"Scoring {len(data)} sentences ({a.split} of {a.data})")

    y_true = [labels for _, labels in data]
    y_pred = [model.predict_tokens(tokens) for tokens, _ in data]

    metrics = compute_metrics(y_true, y_pred)
    report = save_report(
        metrics, a.out_dir, y_true, y_pred,
        title=f"Rule-based baseline on '{a.split}' of {a.data}",
        prefix=a.report_prefix,
    )
    print(report)
    print(f"\nReport saved to {Path(a.out_dir).resolve()} "
          f"({a.report_prefix}_report.txt / {a.report_prefix}_metrics.json)")


if __name__ == "__main__":
    main()
