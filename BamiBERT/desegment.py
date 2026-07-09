# -*- coding: utf-8 -*-
"""Inverse word segmentation for a CoNLL NER dataset (standalone).

PhoBERT needs word-segmented input (compound words joined with ``_``, e.g.
``thành_phố``); BamiBERT works on RAW text. This script turns a segmented CoNLL
file into a raw one by splitting every ``_``-joined token back into syllables and
expanding its BIO label so entity spans are preserved exactly:

    thành_phố  B-DISTRICT   ->   thành  B-DISTRICT
                                  phố    I-DISTRICT
    phổ_yên    I-DISTRICT   ->   phổ    I-DISTRICT
                                  yên    I-DISTRICT
    ,          O            ->   ,      O

Rule: a ``B-X`` word keeps ``B-X`` on its first syllable and ``I-X`` on the rest;
an ``I-X`` word puts ``I-X`` on every syllable; ``O`` stays ``O``. This is the
exact inverse of merging syllables into a word, so no entity boundary moves.

Usage:
    python desegment.py --data Cleaned_Data_new.conll --out Cleaned_Data_new_raw.conll

The script self-verifies: after de-segmentation, every entity (its type and
surface text) must be identical to the source, or it aborts.
"""

import argparse
import sys
from typing import List, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

Sentence = Tuple[List[str], List[str]]      # (tokens, labels)


def read_conll(path: str) -> List[Sentence]:
    """CoNLL-2003 style: ``<token> -X- _ <BIO-label>``, blank line between sentences."""
    sentences, tokens, labels = [], [], []
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
    with open(path, "w", encoding="utf-8") as f:
        for tokens, labels in sentences:
            for tok, lab in zip(tokens, labels):
                f.write(f"{tok} -X- _ {lab}\n")
            f.write("\n")


def desegment_sentence(tokens: List[str], labels: List[str]) -> Sentence:
    """Split ``_``-joined words into syllables, expanding BIO labels correctly.

    A token that is just ``_`` or has no syllables after splitting (rare data
    artifacts like ``sh1_02`` where ``_`` is standalone) is kept unchanged.
    """
    new_tokens, new_labels = [], []
    for tok, lab in zip(tokens, labels):
        pieces = [p for p in tok.split("_") if p]   # drop empties from stray '_'
        if len(pieces) <= 1:                        # plain token (or literal '_')
            new_tokens.append(pieces[0] if pieces else tok)
            new_labels.append(lab)
            continue
        for i, piece in enumerate(pieces):
            new_tokens.append(piece)
            if lab == "O":
                new_labels.append("O")
            elif i == 0:
                new_labels.append(lab)              # keep B-X / I-X on 1st syllable
            else:
                new_labels.append("I-" + lab[2:])   # continuation inside the word
    return new_tokens, new_labels


def check_bio(sentences: List[Sentence]) -> int:
    """Count BIO violations (an I-X not preceded by B-X/I-X of the same type)."""
    bad = 0
    for _, labels in sentences:
        prev = "O"
        for lab in labels:
            if lab.startswith("I-") and prev[2:] != lab[2:]:
                bad += 1
            prev = lab if lab != "O" else "O"
    return bad


def entity_spans_text(tokens: List[str], labels: List[str]):
    """Entities as ``(type, surface text)`` — must be identical before/after."""
    spans, cur_type, cur_toks = [], None, []
    for tok, lab in zip(tokens, labels):
        tok = " ".join(p for p in tok.split("_") if p) or tok
        if lab.startswith("B-"):
            if cur_type:
                spans.append((cur_type, " ".join(cur_toks)))
            cur_type, cur_toks = lab[2:], [tok]
        elif lab.startswith("I-") and cur_type == lab[2:]:
            cur_toks.append(tok)
        else:
            if cur_type:
                spans.append((cur_type, " ".join(cur_toks)))
            cur_type, cur_toks = (lab[2:], [tok]) if lab.startswith("I-") else (None, [])
    if cur_type:
        spans.append((cur_type, " ".join(cur_toks)))
    return spans


def desegment_file(in_path: str, out_path: str):
    segmented = read_conll(in_path)
    print(f"[*] Loaded {len(segmented)} sentences from {in_path}")
    print(f"[*] BIO violations in source: {check_bio(segmented)}")

    raw = [desegment_sentence(t, l) for t, l in segmented]
    print(f"[*] BIO violations after de-segmentation: {check_bio(raw)}")

    # Strong invariant: every entity (type + surface text) is unchanged.
    for (st, sl), (rt, rl) in zip(segmented, raw):
        assert len(rt) == len(rl), "token/label length mismatch!"
        assert entity_spans_text(st, sl) == entity_spans_text(rt, rl), (
            f"entity spans changed!\n  seg: {list(zip(st, sl))}\n"
            f"  raw: {list(zip(rt, rl))}")
    print("[*] Verified: all entity spans identical before/after de-segmentation.")

    n_seg = sum(len(t) for t, _ in segmented)
    n_raw = sum(len(t) for t, _ in raw)
    print(f"[*] Tokens: {n_seg} segmented words -> {n_raw} raw syllables")

    write_conll(out_path, raw)
    print(f"[OK] Wrote de-segmented file -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Inverse word segmentation of a CoNLL file (for BamiBERT)")
    p.add_argument("--data", default="Cleaned_Data_new.conll",
                   help="input segmented CoNLL file")
    p.add_argument("--out", default="Cleaned_Data_new_raw.conll",
                   help="output raw (de-segmented) CoNLL file")
    return p.parse_args()


def main():
    a = parse_args()
    desegment_file(a.data, a.out)


if __name__ == "__main__":
    main()
