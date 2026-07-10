"""Dataset 3 - "Xe dù" HCMC delivery-logistics set (Kaggle, hunhquanglc).

Highly irregular shorthand entries. Each record carries a `senderAddress` and a
`receiverAddress`; both are independent, full addresses, so we **stack** them into
one address-per-row stream (the report's "concatenating each record's sender and
receiver address fields") and then apply the shared pass.

    load `senderAddress` + `receiverAddress`  ->  stack into one stream
    ->  normalize + de-dup  ->  EDA  ->  segment  ->  number fix-ups
    ->  data3_final.csv

Note: per the report the Xe dù set gets only the shared pass (no country-token
stripping); trailing "vietnam" / "ho chi minh city" tokens are therefore left in
place. Pass --strip-country to drop a trailing comma-segment if you want them
removed.

Run from the repo root (`_organized/`):

    python -m data_processing.preprocess_xe_du
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def _drop_last_segment(text: str) -> str:
    parts = [p for p in str(text).split(",")]
    return ", ".join(parts[:-1]).strip() if len(parts) > 1 else str(text).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join(C.RAW_DIR, "data3.csv"))
    ap.add_argument("--sender", default="senderAddress")
    ap.add_argument("--receiver", default="receiverAddress")
    ap.add_argument("--output", default=os.path.join(C.PROC_DIR, "data3_final.csv"))
    ap.add_argument("--strip-country", action="store_true",
                    help="drop the trailing comma-segment (usually the country)")
    ap.add_argument("--no-segment", action="store_true")
    ap.add_argument("--no-eda", action="store_true")
    args = ap.parse_args()

    print(f"[Dataset 3 · Xe dù] {args.input}")
    df = pd.read_csv(args.input)
    for col in (args.sender, args.receiver):
        if col not in df.columns:
            raise SystemExit(f"[!] column {col!r} not in {args.input}")
    # Stack sender + receiver into one address-per-row stream.
    s = pd.concat([df[args.sender], df[args.receiver]], ignore_index=True)
    print(f"  [stack] {len(df)} records -> {len(s)} addresses "
          f"(sender + receiver)")
    if args.strip_country:
        s = s.apply(_drop_last_segment)
    s = C.normalize_series(s)
    s = C.deduplicate(s, "xe_du")
    C.finalize(s, args.output, segment=not args.no_segment,
               eda=not args.no_eda, name="data3_xe_du", eda_dir=C.EDA_DIR)


if __name__ == "__main__":
    main()
