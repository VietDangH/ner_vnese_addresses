# -*- coding: utf-8 -*-
"""Data layer for the rule-based baseline — CoNLL IO + split.

This is a **torch-free** mirror of :mod:`vn_ner.data` so the rule-based baseline
runs with no deep-learning dependencies (only ``seqeval`` is needed, for the
metrics). ``read_conll`` / ``split_sentences`` are byte-for-byte compatible with
the PhoBERT/BamiBERT pipeline, so pointing this baseline at the same file with
the same ``seed`` / ``val_ratio`` / ``test_ratio`` yields the **exact same test
split** — the two models are therefore directly comparable.

CoNLL-2003 style, one token per line::

    thành_phố -X- _ B-DISTRICT

i.e. ``<token> -X- _ <BIO-label>``; a blank line separates sentences.
"""

import random
from typing import List, Tuple

Sentence = Tuple[List[str], List[str]]      # (tokens, labels)


def read_conll(path: str) -> List[Sentence]:
    """Parse a CoNLL file into a list of ``(tokens, labels)`` sentences.

    Blank lines delimit sentences; ``-DOCSTART-`` lines are ignored. The token
    is the first whitespace-separated field and the label is the last.
    """
    sentences: List[Sentence] = []
    tokens: List[str] = []
    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
                continue
            parts = line.split()
            token, label = parts[0], parts[-1]
            if token == "-DOCSTART-":
                continue
            tokens.append(token)
            labels.append(label)
    if tokens:
        sentences.append((tokens, labels))
    return sentences


def write_conll(path: str, sentences: List[Sentence]):
    """Write ``(tokens, labels)`` sentences in the CoNLL format
    ``<token> -X- _ <label>``, blank line between sentences."""
    with open(path, "w", encoding="utf-8") as f:
        for tokens, labels in sentences:
            for tok, lab in zip(tokens, labels):
                f.write(f"{tok} -X- _ {lab}\n")
            f.write("\n")


def dedupe(sentences: List[Sentence]) -> List[Sentence]:
    """Drop exact duplicate sentences (same tokens *and* labels), keeping order.

    Mirrors the de-duplication that ``vn_ner.split`` performs before splitting,
    so the leak-safe train/val/test partition matches the neural pipeline."""
    seen, out = set(), []
    for s in sentences:
        key = (tuple(s[0]), tuple(s[1]))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def split_sentences(sentences: List[Sentence], val_ratio: float,
                    test_ratio: float, seed: int):
    """Shuffle once with ``seed`` and carve out test / val / train slices.

    Identical logic to :func:`vn_ner.data.split_sentences`."""
    data = list(sentences)
    random.Random(seed).shuffle(data)
    n = len(data)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = data[:n_test]
    val = data[n_test:n_test + n_val]
    train = data[n_test + n_val:]
    return train, val, test
