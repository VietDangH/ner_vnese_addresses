"""Dataset 2 - vietnamese-address-parser (Kaggle).

A raw, noisy set with duplicated tail segments. Beyond the shared pass it needs
**structural repair**: split on newlines / pipes into segments, strip trailing
country tokens (`vietnam|vn|...`), collapse semantic duplicates via an
abbreviation dictionary ("TP. Hồ Chí Minh" == "hcm", "Bà Rịa-Vũng Tàu" == "brvt"),
and drop corrupt (`#ERROR!`) or degenerate single-token rows.

    load `Norm Address`  ->  structural repair (clean_address)  ->  normalize
    ->  drop #ERROR! / single-token  ->  de-dup  ->  EDA  ->  segment
    ->  number fix-ups  ->  data2_final.csv

`clean_address` is ported verbatim from Data_Cleaning_EDA.ipynb (it is specific
to this dataset, so it lives here rather than in common.py).

Run from the repo root (`_organized/`):

    python -m data_processing.preprocess_vietnamese_address_parser
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def clean_address(text: str) -> str:
    """Structural repair for the vietnamese-address-parser rows.

    (ported from Data_Cleaning_EDA.ipynb :: clean_address_df2)
    """
    # Newlines -> pipe separators.
    text = re.sub(r"\\n|\n", " | ", str(text))
    parts = [p.strip() for p in text.split("|") if p.strip()]

    # Strip a trailing country token from each segment; drop bare country parts.
    cleaned = []
    for p in parts:
        p = re.sub(r"(?i)\b(vietnam|việt nam|viet nam|vn)\s*$", "", p).strip()
        if p and p.lower() not in ("vietnam", "việt nam", "viet nam", "vn"):
            cleaned.append(p)
    parts = cleaned

    mapping = {
        "tp. hồ chí minh": ["hcm", "ho chi minh", "hồ chí minh", "tp hcm", "tphcm"],
        "hồ chí minh": ["hcm", "ho chi minh", "tp. hồ chí minh", "tp hcm", "tphcm"],
        "hà nội": ["hn", "ha noi"],
        "hn": ["hà nội", "ha noi"],
        "bà rịa vũng tàu": ["brvt", "vũng tàu", "vung tau",
                            "bà rịa - vũng tàu", "bà rịa-vũng tàu"],
        "thừa thiên huế": ["huế", "hue", "tth", "tt huế"],
        "đà nẵng": ["đn", "dn", "da nang"],
    }

    def strong_normalize(s: str) -> str:
        s = s.lower()
        for prefix in ("tp.", "tp ", "thành phố ", "tỉnh "):
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
        s = re.sub(r"[^\w\s]", "", s)   # drop punctuation
        s = re.sub(r"\s+", "", s)       # drop spaces (catch "bà rịa-vũng tàu" etc.)
        return s

    final_parts = []
    for p in parts:
        norm_p = strong_normalize(p)
        left = "".join(strong_normalize(x) for x in final_parts)
        is_dup = norm_p in left
        if not is_dup:
            for key, values in mapping.items():
                norm_key = strong_normalize(key)
                if norm_key in norm_p or norm_p in norm_key:
                    if any(strong_normalize(v) in left for v in values) or norm_key in left:
                        is_dup = True
                        break
        if not is_dup:
            final_parts.append(p)
    return " | ".join(final_parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join(C.RAW_DIR,
                    "vietnamese-address-parser_dataset.csv"))
    ap.add_argument("--column", default="Norm Address")
    ap.add_argument("--output", default=os.path.join(C.PROC_DIR, "data2_final.csv"))
    ap.add_argument("--min-tokens", type=int, default=2,
                    help="drop rows with fewer than this many tokens (default 2)")
    ap.add_argument("--no-segment", action="store_true")
    ap.add_argument("--no-eda", action="store_true")
    args = ap.parse_args()

    print(f"[Dataset 2 · vietnamese-address-parser] {args.input}")
    s = C.read_column(args.input, args.column)
    s = s.apply(clean_address)                       # structural repair
    s = C.normalize_series(s)                         # shared normalize
    # Drop corrupt / degenerate rows.
    before = len(s)
    s = s[~s.str.contains("#error", case=False, na=False)]
    s = s[s.apply(lambda x: len(x.split()) >= args.min_tokens)]
    print(f"  [filter] {before} -> {len(s)} rows "
          f"(dropped #ERROR! and < {args.min_tokens}-token rows)")
    s = C.deduplicate(s.reset_index(drop=True), "vap")
    C.finalize(s, args.output, segment=not args.no_segment,
               eda=not args.no_eda, name="data2_vap", eda_dir=C.EDA_DIR)


if __name__ == "__main__":
    main()
