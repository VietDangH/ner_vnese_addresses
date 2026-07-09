# -*- coding: utf-8 -*-
"""Split a clean CoNLL file into train / val / test files (leak-safe).

This step does **one** thing: de-duplicate and split. Noise augmentation is a
separate step (``bami_ner.augment``) so you can augment any file you like — not
just the training set — and keep val/test clean.

    python -m bami_ner.split --data Cleaned_Data_new_raw.conll --out_dir data_split

Writes ``<out_dir>/{train,val,test}.conll`` (all clean). Then, optionally:

    python -m bami_ner.augment --data data_split/train.conll \
        --out data_split/train_aug.conll
    python -m bami_ner.train --train_data data_split/train_aug.conll \
        --val_data data_split/val.conll --head crf
    python -m bami_ner.evaluate --model_dir bami_ner_model --data data_split/test.conll
"""

import argparse
import os
import sys

from .data import read_conll, write_conll, split_sentences

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Split a clean CoNLL file into train/val/test (no augmentation)")
    p.add_argument("--data", default="Cleaned_Data_new_raw.conll")
    p.add_argument("--out_dir", default="data_split")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = parse_args()
    sentences = read_conll(a.data)
    print(f"[*] Loaded {len(sentences)} sentences from {a.data}")

    # de-duplicate so an identical address can't land in two splits
    seen, deduped = set(), []
    for s in sentences:
        key = (tuple(s[0]), tuple(s[1]))
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    if len(deduped) != len(sentences):
        print(f"[*] Removed {len(sentences) - len(deduped)} duplicate sentences")

    train, val, test = split_sentences(deduped, a.val_ratio, a.test_ratio, a.seed)
    print(f"[*] Split -> train {len(train)} | val {len(val)} | test {len(test)}")

    os.makedirs(a.out_dir, exist_ok=True)
    paths = {n: os.path.join(a.out_dir, f"{n}.conll")
             for n in ("train", "val", "test")}
    write_conll(paths["train"], train)
    write_conll(paths["val"], val)
    write_conll(paths["test"], test)

    print("\n[OK] Wrote:")
    for n in ("train", "val", "test"):
        print(f"    {paths[n]}")
    print("\n[i] (optional) augment the train file:")
    print(f"    python -m bami_ner.augment --data {paths['train']} "
          f"--out {a.out_dir}/train_aug.conll")
    print("[i] Train:")
    print(f"    python -m bami_ner.train --train_data {paths['train']} "
          f"--val_data {paths['val']}")
    print("[i] Evaluate:")
    print(f"    python -m bami_ner.evaluate --model_dir bami_ner_model "
          f"--data {paths['test']}")


if __name__ == "__main__":
    main()
