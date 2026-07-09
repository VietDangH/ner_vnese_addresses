# -*- coding: utf-8 -*-
"""Noise-augment ANY CoNLL file (decoupled from splitting).

Because this is its own step, you can augment whichever file you want — usually
the train split, but nothing stops you from building a *noisy test set* to probe
robustness. Keep val/test clean for honest evaluation unless you mean to stress-test.

    # append one noisy copy per sentence (file grows ~2x)
    python -m vn_ner.augment --data data_split/train.conll \
        --out data_split/train_aug.conll

    # replace each sentence by a noisy version (same size) — e.g. a hard test set
    python -m vn_ner.augment --data data_split/test.conll \
        --out data_split/test_noisy.conll --mode replace

Noise types (each with its own probability): keyword abbreviation
(``quận``->``q``), prefix drop (``quận 12``->``12``), comma drop, diacritics
removal (``nghệ an``->``nghe an``), and optional char typos. See ``noise.py``.
"""

import argparse
import random
import sys

from .data import read_conll, write_conll
from .noise import (EXCLUDE_CITIES, augment_sentences, make_noisy_copy, merge_cfg)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def augment_file(in_path, out_path, cfg, mode="append", seed=42,
                 exclude_metro=True):
    """Read ``in_path``, augment, write ``out_path``. Returns a stats dict."""
    sentences = read_conll(in_path)
    exclude = EXCLUDE_CITIES if exclude_metro else set()

    if mode == "append":
        noisy, stats = augment_sentences(sentences, cfg, seed=seed, exclude=exclude)
        out = list(sentences) + noisy
        stats = {"input": len(sentences), "added": len(noisy),
                 "output": len(out), **stats}
    elif mode == "replace":
        rng = random.Random(seed)
        out = [make_noisy_copy(s, cfg, rng) for s in sentences]
        stats = {"input": len(sentences), "replaced": len(out), "output": len(out)}
    else:
        raise ValueError(f"unknown mode '{mode}' (use 'append' or 'replace')")

    write_conll(out_path, out)
    return stats


def parse_args():
    p = argparse.ArgumentParser(description="Noise-augment a CoNLL file")
    p.add_argument("--data", required=True, help="input CoNLL file")
    p.add_argument("--out", required=True, help="output CoNLL file")
    p.add_argument("--mode", choices=["append", "replace"], default="append",
                   help="append noisy copies, or replace each sentence by a noisy one")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_exclude_metro", action="store_true",
                   help="also augment HCM / Hà Nội addresses (append mode)")
    p.add_argument("--deaccent_prob", type=float, default=0.12)
    p.add_argument("--drop_prefix_prob", type=float, default=0.10)
    p.add_argument("--drop_comma_prob", type=float, default=0.15)
    p.add_argument("--abbrev_scale", type=float, default=1.0)
    p.add_argument("--typo_prob", type=float, default=0.0)
    return p.parse_args()


def main():
    a = parse_args()
    cfg = merge_cfg({
        "deaccent_prob": a.deaccent_prob,
        "drop_prefix_prob": a.drop_prefix_prob,
        "drop_comma_prob": a.drop_comma_prob,
        "abbrev_scale": a.abbrev_scale,
        "typo_prob": a.typo_prob,
    })
    stats = augment_file(a.data, a.out, cfg, mode=a.mode, seed=a.seed,
                         exclude_metro=not a.no_exclude_metro)
    print(f"[✓] {a.mode}: {a.data} -> {a.out}")
    print(f"    {stats}")


if __name__ == "__main__":
    main()
