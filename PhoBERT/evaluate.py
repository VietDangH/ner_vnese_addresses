# -*- coding: utf-8 -*-
"""Evaluation entry point: the single place metrics are computed.

Scores a trained model on a CoNLL file and writes a detailed report (entity-level
P/R/F1, per-type breakdown, token + sentence accuracy, confusion matrix). By
default it scores the **whole** ``--data`` file — point it at your held-out
``test.conll``:

    python -m vn_ner.evaluate --model_dir ner_model --data data_split/test.conll

It reloads whatever architecture was trained (softmax / CRF / +BiLSTM / ...) from
the saved ``config.json`` automatically.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import Collator, NERDataset, read_conll, split_sentences
from .metrics import compute_metrics, save_report
from .models import load_trained
from .trainer import run_inference

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained PhoBERT NER model")
    p.add_argument("--model_dir", default="ner_model")
    p.add_argument("--data", default="data_split/test.conll")
    p.add_argument("--split", choices=["all", "test", "val", "train"],
                   default="all",
                   help="'all' scores the whole file; the others re-create an "
                        "internal split of --data (legacy single-file mode)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_prefix", default="eval")
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")

    model, tokenizer, label_list, id2label, mcfg = load_trained(a.model_dir, device)
    label2id = {l: i for i, l in enumerate(label_list)}
    print(f"Loaded {mcfg.get('head', 'crf')} model "
          f"(top_encoder={mcfg.get('top_encoder', 'none')}) from {a.model_dir}")

    sentences = read_conll(a.data)
    if a.split == "all":
        data = sentences
    else:
        seen, deduped = set(), []
        for s in sentences:
            key = (tuple(s[0]), tuple(s[1]))
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        train_d, val_d, test_d = split_sentences(
            deduped, a.val_ratio, a.test_ratio, a.seed)
        data = {"train": train_d, "val": val_d, "test": test_d}[a.split]
    print(f"Scoring {len(data)} sentences ({a.split} of {a.data})")

    collate = Collator(tokenizer, label2id, mcfg.get("max_length", 256))
    loader = DataLoader(NERDataset(data), batch_size=a.batch_size,
                        shuffle=False, collate_fn=collate)

    y_true, y_pred, loss = run_inference(model, loader, id2label, device, use_amp)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = loss
    report = save_report(metrics, a.model_dir, y_true, y_pred,
                         title=f"Performance on '{a.split}' of {a.data}",
                         prefix=a.report_prefix)
    print(report)
    print(f"\nReport saved to {Path(a.model_dir).resolve()} "
          f"({a.report_prefix}_report.txt / {a.report_prefix}_metrics.json)")


if __name__ == "__main__":
    main()
