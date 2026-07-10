"""Dataset 1 - Vietnam House Rent (Kaggle, vanviethieuanh).

Well-formatted but heavily Đà Nẵng-skewed. Per the report we rebalance it
geographically by **ward-level tail replacement**: each address is truncated at
its ward and given a random ward/district/city cluster drawn from the gazetteer
(biased toward rarer provinces), which spreads coverage across the country while
keeping the house/street/area prefix.

    load `address`  ->  normalize + de-dup  ->  ward-tail rebalancing
    ->  EDA  ->  VnCoreNLP segmentation  ->  number fix-ups  ->  data1_final.csv

Run from the repo root (`_organized/`):

    python -m data_processing.preprocess_house_rent
    python -m data_processing.preprocess_house_rent --no-rebalance --no-segment
"""

import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# Ward head-keywords (lower-cased, since rebalancing runs after normalization).
_WARD_RE = re.compile(r"^\s*(phường|xã|thị trấn|p\.?|x\.?|tt\.?)\b")


def _truncate_prefix(address: str) -> str:
    """Return the house/street/area part before the ward (comma-delimited).

    If no ward keyword is found, returns the whole address unchanged so it can be
    left as-is by the caller.
    """
    parts = [p.strip() for p in address.split(",")]
    for i, p in enumerate(parts):
        if _WARD_RE.match(p):
            return ", ".join(parts[:i]).strip()          # prefix before ward
    return None                                           # no ward -> not found


def rebalance(series, gazetteer_path, rare_bias=1.0, exclude=None, seed=42):
    """Replace each address's admin tail with a rare-biased gazetteer cluster."""
    rng = random.Random(seed)
    rows = C.load_gazetteer(gazetteer_path, exclude)
    weights = C.build_weights(rows, rare_bias)
    print(f"  [rebalance] {len(rows)} gazetteer clusters "
          f"(rare-bias={rare_bias}); replacing admin tails")

    out, kept_prefix, no_ward = [], 0, 0
    for addr in series:
        prefix = _truncate_prefix(addr)
        row = rng.choices(rows, weights=weights, k=1)[0]
        tail = C.normalize_text(f"{row['ward']}, {row['district']}, {row['city']}")
        if prefix is None:          # no ward keyword -> keep original head, add tail
            no_ward += 1
            head = addr
        else:
            head = prefix
            kept_prefix += 1
        out.append(f"{head}, {tail}" if head else tail)
    print(f"  [rebalance] kept prefix on {kept_prefix} rows; "
          f"{no_ward} rows had no detectable ward (whole head kept)")
    import pandas as pd
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join(C.RAW_DIR,
                    "Vietnam House Rent Dataset.csv"))
    ap.add_argument("--column", default="address")
    ap.add_argument("--output", default=os.path.join(C.PROC_DIR, "data1_final.csv"))
    ap.add_argument("--gazetteer", default=C.GAZETTEER)
    ap.add_argument("--rare-bias", type=float, default=1.0,
                    help="0 = uniform over provinces; higher prefers rarer ones")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="province keys to hold out of the draw (default: the big 4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-rebalance", action="store_true",
                    help="skip geographic rebalancing (shared cleaning only)")
    ap.add_argument("--no-segment", action="store_true")
    ap.add_argument("--no-eda", action="store_true")
    args = ap.parse_args()

    print(f"[Dataset 1 · House Rent] {args.input}")
    s = C.read_column(args.input, args.column)
    s = C.normalize_series(s)
    s = C.deduplicate(s, "house_rent")
    if not args.no_rebalance:
        exclude = set(args.exclude) if args.exclude is not None else None
        s = rebalance(s, args.gazetteer, args.rare_bias, exclude, args.seed)
        s = C.normalize_series(s)
    C.finalize(s, args.output, segment=not args.no_segment,
               eda=not args.no_eda, name="data1_house_rent", eda_dir=C.EDA_DIR)


if __name__ == "__main__":
    main()
