# -*- coding: utf-8 -*-
"""CLI: tag a single address with the rule-based baseline.

    python -m rule_ner.predict --text "204 / 9 hải phòng , huyện cát hải , hải phòng"

Input is expected to be raw (un-segmented) — one Vietnamese syllable per
whitespace token, exactly as in the raw CoNLL data. The tagger needs no word
segmenter; multi-syllable triggers like ``thành phố`` are matched as phrases.
(Pre-segmented ``thành_phố`` tokens also work.)"""

import argparse
import sys

from .tagger import RuleBasedNER

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Tag one address with the rule-based baseline")
    p.add_argument("--text", required=True, help="a pre-segmented address string")
    p.add_argument("--gazetteer", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    model = RuleBasedNER(gazetteer_path=a.gazetteer)
    r = model.predict(a.text)

    width = max((len(t) for t in r["tokens"]), default=0)
    print("Token / label")
    print("-" * (width + 16))
    for tok, lab in zip(r["tokens"], r["labels"]):
        print(f"  {tok:<{width}}  {lab}")

    print("\nEntities")
    for e in r["entities"]:
        print(f"  {e['type']:<10} {e['text']}")


if __name__ == "__main__":
    main()
