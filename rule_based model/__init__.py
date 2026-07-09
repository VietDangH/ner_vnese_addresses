# -*- coding: utf-8 -*-
"""rule_ner: a rule-based Vietnamese-address NER baseline.

A transparent, training-free counterpart to the PhoBERT (``vn_ner``) and
BamiBERT (``bami_ner``) models. It runs on the **raw, un-segmented** data (one
syllable per token — no word segmenter dependency) and tags the same 8 entity
types (``HOUSE_NO, ALLEY, STREET, AREA, WARD, DISTRICT, BUILDING, CITY``) using
phrase-aware trigger words, regexes, a gazetteer, and the hierarchical structure
of addresses. It reports metrics through the *same* code path so the numbers are
directly comparable.

    python -m rule_ner.evaluate --data Cleaned_Data_new_raw.conll --split test --seed 41
    python -m rule_ner.predict  --text "204 / 9 hải phòng , huyện cát hải , hải phòng"

Public API
----------
    from rule_ner import RuleBasedNER, GazetteerIndex
"""

from .gazetteer import GazetteerIndex
from .tagger import RuleBasedNER
from .data import read_conll, write_conll, split_sentences, dedupe

__all__ = [
    "RuleBasedNER",
    "GazetteerIndex",
    "read_conll",
    "write_conll",
    "split_sentences",
    "dedupe",
]

__version__ = "1.0.0"
