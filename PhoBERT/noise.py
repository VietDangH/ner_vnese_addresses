# -*- coding: utf-8 -*-
"""Noise augmentation for address sentences (in-package version).

Mirrors the standalone ``augment_noise.py`` but operates on this package's
sentence representation -- ``(tokens, labels)`` parallel lists -- so the trainer
can augment the TRAIN split in-process (split first, noise only train -> no
leakage into val/test).

Noise types (each with its own probability):
  * abbreviate keyword  : ngõ->ng. , quận->q , xã->x , thành_phố->tp ...
  * drop prefix         : 'quận 12' -> '12'  (BIO re-fixed)
  * drop comma          : remove stray ','
  * remove diacritics   : 'nghệ an' -> 'nghe an'  (per sentence)
  * char typo           : light char perturbation (off by default)

Addresses whose CITY is a big metro (HCM / Hà Nội forms) are skipped so we don't
over-augment already-dominant classes.
"""

import random
import unicodedata
from typing import Dict, List, Tuple

Sentence = Tuple[List[str], List[str]]

# keyword -> (abbreviation variants, probability)
ABBREV: Dict[str, Tuple[List[str], float]] = {
    "ngõ": (["ng.", "ngo"], 0.30),
    "hẻm": (["h.", "hem"], 0.25),
    "kiệt": (["k."], 0.20),
    "ngách": (["ng."], 0.15),
    "đường": (["đ.", "d."], 0.25),
    "phố": (["p."], 0.10),
    "phường": (["p.", "p"], 0.30),
    "xã": (["x.", "x"], 0.30),
    "thị_trấn": (["tt.", "tt"], 0.30),
    "quận": (["q.", "q"], 0.30),
    "huyện": (["h.", "h"], 0.25),
    "thị_xã": (["tx.", "tx"], 0.30),
    "thành_phố": (["tp.", "tp", "t.p"], 0.30),
    "tỉnh": (["t.", "t"], 0.20),
}

NOISE_DEFAULTS = {
    "abbrev_scale": 1.0,
    "drop_prefix_prob": 0.10,
    "drop_comma_prob": 0.15,
    "deaccent_prob": 0.12,
    "typo_prob": 0.0,
}

HEAD_KEYWORDS = {
    "phường", "xã", "thị_trấn", "quận", "huyện", "thị_xã", "thành_phố",
    "tỉnh", "đường", "phố", "ngõ", "hẻm", "kiệt", "ngách",
}
DROP_LABELS = {"WARD", "DISTRICT", "CITY", "STREET", "ALLEY"}

EXCLUDE_CITIES = {"tp . hồ_chí_minh", "hà_nội", "hồ_chí_minh", "hcm"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def de_accent(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _spans(labels: List[str]):
    spans, i, n = [], 0, len(labels)
    while i < n:
        tag = labels[i]
        if tag.startswith("B-"):
            lab = tag[2:]; j = i + 1
            while j < n and labels[j] == f"I-{lab}":
                j += 1
            spans.append((lab, i, j)); i = j
        else:
            i += 1
    return spans


def _fix_bio(labels: List[str]) -> List[str]:
    out, prev = [], None
    for tag in labels:
        if tag.startswith("I-"):
            lab = tag[2:]
            if prev != lab:
                tag = f"B-{lab}"
            prev = lab
        elif tag.startswith("B-"):
            prev = tag[2:]
        else:
            prev = None
        out.append(tag)
    return out


def city_texts(tokens, labels) -> List[str]:
    return [" ".join(tokens[s:e]).lower().strip()
            for lab, s, e in _spans(labels) if lab == "CITY"]


def is_excluded(tokens, labels, exclude) -> bool:
    return any(c in exclude for c in city_texts(tokens, labels))


# --------------------------------------------------------------------------- #
# Noise ops: (tokens, labels, cfg, rng) -> (tokens, labels)
# --------------------------------------------------------------------------- #
def op_abbreviate(tokens, labels, cfg, rng):
    out = []
    for tok in tokens:
        key = tok.lower()
        if key in ABBREV:
            variants, prob = ABBREV[key]
            if rng.random() < min(prob * cfg["abbrev_scale"], 1.0):
                tok = rng.choice(variants).strip()
        out.append(tok)
    return out, labels


def op_drop_prefix(tokens, labels, cfg, rng):
    drop = set()
    for lab, s, e in _spans(labels):
        if (lab in DROP_LABELS and e - s >= 2
                and tokens[s].lower() in HEAD_KEYWORDS
                and rng.random() < cfg["drop_prefix_prob"]):
            drop.add(s)
    if not drop:
        return tokens, labels
    toks = [t for i, t in enumerate(tokens) if i not in drop]
    labs = [l for i, l in enumerate(labels) if i not in drop]
    return toks, _fix_bio(labs)


def op_drop_comma(tokens, labels, cfg, rng):
    toks, labs = [], []
    for t, l in zip(tokens, labels):
        if l == "O" and t in {",", ";"} and rng.random() < cfg["drop_comma_prob"]:
            continue
        toks.append(t); labs.append(l)
    return toks, labs


def op_typo(tokens, labels, cfg, rng):
    if cfg["typo_prob"] <= 0:
        return tokens, labels
    out = []
    for t in tokens:
        if len(t) >= 4 and t.isalpha() and rng.random() < cfg["typo_prob"]:
            i = rng.randrange(len(t) - 1)
            ch = list(t)
            if rng.random() < 0.5:
                ch[i], ch[i + 1] = ch[i + 1], ch[i]
            else:
                del ch[i]
            t = "".join(ch)
        out.append(t)
    return out, labels


def op_deaccent(tokens, labels, cfg, rng):
    if rng.random() >= cfg["deaccent_prob"]:
        return tokens, labels
    return [de_accent(t) for t in tokens], labels


_PIPELINE = [op_drop_prefix, op_abbreviate, op_drop_comma, op_typo, op_deaccent]


def make_noisy_copy(sentence: Sentence, cfg, rng) -> Sentence:
    """Return a noised copy of ``(tokens, labels)`` (original untouched)."""
    tokens, labels = list(sentence[0]), list(sentence[1])
    for op in _PIPELINE:
        tokens, labels = op(tokens, labels, cfg, rng)
    return tokens, labels


# --------------------------------------------------------------------------- #
# Dataset-level augmentation (used by the trainer on the TRAIN split only)
# --------------------------------------------------------------------------- #
def merge_cfg(overrides: dict = None) -> dict:
    cfg = dict(NOISE_DEFAULTS)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def augment_sentences(sentences: List[Sentence], cfg: dict, seed: int = 42,
                      exclude=None, skip_unchanged: bool = True):
    """Make one noisy copy per (non-excluded) sentence. Returns (copies, stats).

    A copy is dropped if it is unchanged (skip_unchanged) or if noise turned its
    CITY into an excluded metro (so we never inject extra HCM / Hà Nội samples)."""
    exclude = EXCLUDE_CITIES if exclude is None else exclude
    rng = random.Random(seed)
    out, n_excluded, n_unchanged = [], 0, 0
    for tokens, labels in sentences:
        if is_excluded(tokens, labels, exclude):
            n_excluded += 1
            continue
        nt, nl = make_noisy_copy((tokens, labels), cfg, rng)
        if is_excluded(nt, nl, exclude):
            n_excluded += 1
            continue
        if skip_unchanged and nt == list(tokens) and nl == list(labels):
            n_unchanged += 1
            continue
        out.append((nt, nl))
    return out, {"added": len(out), "excluded": n_excluded,
                 "unchanged": n_unchanged}
