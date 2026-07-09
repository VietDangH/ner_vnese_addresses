# -*- coding: utf-8 -*-
"""Inference entry point: tag addresses with a trained model.

    # single string
    python -m bami_ner.predict --model_dir bami_ner_model \
        --text "123 đường lê lợi, phường bến nghé, quận 1, tp hcm"

    # a file with one address per line, write JSONL results
    python -m bami_ner.predict --model_dir bami_ner_model \
        --input_file addresses.txt --output_file out.jsonl

    # interactive REPL
    python -m bami_ner.predict --model_dir bami_ner_model --interactive

BamiBERT consumes raw text, so there is no word-segmentation step here.
"""

import argparse
import json
import sys

from .predictor import Predictor

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _print_result(res):
    print("\nText  :", res["text"])
    print(f"Confidence: {res['confidence']:.4f} (mean) | "
          f"{res['min_confidence']:.4f} (min) | latency {res['latency_ms']:.2f} ms")
    print("Tokens:")
    for tok, tag, c in zip(res["tokens"], res["tags"], res["token_confidences"]):
        print(f"    {tok:<20} {tag:<12} conf={c:.4f}")
    print("Entities:")
    for e in res["entities"]:
        conf = e.get("confidence")
        suffix = f"  (conf={conf:.4f})" if conf is not None else ""
        print(f"    {e['type']:<10} -> {e['text']}{suffix}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict address entities with a BamiBERT NER model")
    p.add_argument("--model_dir", default="bami_ner_model")
    p.add_argument("--text", help="a single address string to tag")
    p.add_argument("--input_file", help="file with one address per line")
    p.add_argument("--output_file", help="write JSONL predictions here")
    p.add_argument("--interactive", action="store_true", help="type addresses in a REPL")
    p.add_argument("--no_lower", action="store_true",
                   help="do not lower-case input (training data is lower-cased)")
    return p.parse_args()


def main():
    a = parse_args()
    lower = not a.no_lower
    print(f"Loading model from {a.model_dir} ...")
    predictor = Predictor.from_pretrained(a.model_dir)
    print("Loaded. (BamiBERT: raw text, no word segmentation)")

    if a.text:
        _print_result(predictor.predict(a.text, lower=lower))

    if a.input_file:
        with open(a.input_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        results = predictor.predict_batch(lines, lower=lower)
        if a.output_file:
            with open(a.output_file, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Wrote {len(results)} predictions to {a.output_file}")
        else:
            for r in results:
                _print_result(r)

    if a.interactive:
        print("Interactive mode — type an address and press Enter (Ctrl-C to quit).")
        try:
            while True:
                text = input("\naddress> ").strip()
                if text:
                    _print_result(predictor.predict(text, lower=lower))
        except (KeyboardInterrupt, EOFError):
            print("\nbye")

    if not (a.text or a.input_file or a.interactive):
        print("Nothing to do. Pass --text, --input_file, or --interactive.")


if __name__ == "__main__":
    main()
