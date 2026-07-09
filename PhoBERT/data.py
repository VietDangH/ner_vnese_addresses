# -*- coding: utf-8 -*-
"""Data layer: read CoNLL, build the label vocabulary, split, and collate.

The training file is CoNLL-2003 style -- one token per line, a blank line
between sentences, and each line shaped like::

    thành_phố -X- _ B-DISTRICT

i.e. ``<token> -X- _ <BIO-label>``. Tokens are already word-segmented (compound
words joined with ``_``), which is exactly the surface form PhoBERT expects.

PhoBERT splits each (already segmented) word into one or more BPE sub-words. We
feed the hidden state of the *first* sub-word of every word into the classifier,
so the head runs over a clean, word-level sequence aligned 1:1 with the gold
labels.
"""

import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

Sentence = Tuple[List[str], List[str]]      # (tokens, labels)


# --------------------------------------------------------------------------- #
# Reading / label vocabulary / splitting
# --------------------------------------------------------------------------- #
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
    """Write ``(tokens, labels)`` sentences back out in the CoNLL format
    ``<token> -X- _ <label>``, blank line between sentences."""
    with open(path, "w", encoding="utf-8") as f:
        for tokens, labels in sentences:
            for tok, lab in zip(tokens, labels):
                f.write(f"{tok} -X- _ {lab}\n")
            f.write("\n")


def build_label_list(sentences: List[Sentence]) -> List[str]:
    """Deterministic label list with ``O`` pinned to index 0."""
    labels = set()
    for _, labs in sentences:
        labels.update(labs)
    labels.discard("O")
    return ["O"] + sorted(labels)


def split_sentences(sentences: List[Sentence], val_ratio: float,
                    test_ratio: float, seed: int):
    """Shuffle once with ``seed`` and carve out test / val / train slices."""
    data = list(sentences)
    random.Random(seed).shuffle(data)
    n = len(data)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = data[:n_test]
    val = data[n_test:n_test + n_val]
    train = data[n_test + n_val:]
    return train, val, test


# --------------------------------------------------------------------------- #
# Dataset + Collator
# --------------------------------------------------------------------------- #
class NERDataset(Dataset):
    def __init__(self, sentences: List[Sentence]):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]


class Collator:
    """Turn a batch of ``(tokens, labels)`` into padded model tensors.

    For every word we store the index of its first BPE sub-word so the model can
    ``gather`` one vector per word. Output keys:
        input_ids, attention_mask  -> for the PhoBERT encoder (sub-word level)
        word_positions              -> first sub-word index of each word
        word_mask                   -> True for real words (left-aligned)
        labels                      -> gold tag id per word
    """

    def __init__(self, tokenizer, label2id: Dict[str, int], max_length: int):
        self.tok = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id
        self.unk = tokenizer.unk_token_id

    def encode_one(self, tokens: List[str], labels: List[str]):
        input_ids = [self.bos]
        first_pos: List[int] = []
        label_ids: List[int] = []
        for tok, lab in zip(tokens, labels):
            pieces = self.tok.convert_tokens_to_ids(self.tok.tokenize(tok))
            if not pieces:
                pieces = [self.unk]
            if len(input_ids) + len(pieces) > self.max_length - 1:   # reserve eos
                break
            first_pos.append(len(input_ids))
            input_ids.extend(pieces)
            label_ids.append(self.label2id.get(lab, 0))
        input_ids.append(self.eos)
        # Guard: a sentence must keep at least one word for the word mask.
        if not first_pos:
            first_pos = [1]
            label_ids = [0]
            input_ids = [self.bos, self.unk, self.eos]
        return input_ids, first_pos, label_ids

    def __call__(self, batch: List[Sentence]):
        enc = [self.encode_one(t, l) for t, l in batch]
        max_sub = max(len(e[0]) for e in enc)
        max_word = max(len(e[1]) for e in enc)

        input_ids, attn_mask, word_pos, word_mask, labels = [], [], [], [], []
        for ids, fpos, labs in enc:
            pad_sub = max_sub - len(ids)
            input_ids.append(ids + [self.pad] * pad_sub)
            attn_mask.append([1] * len(ids) + [0] * pad_sub)
            pad_word = max_word - len(fpos)
            word_pos.append(fpos + [0] * pad_word)
            word_mask.append([1] * len(fpos) + [0] * pad_word)
            labels.append(labs + [0] * pad_word)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "word_positions": torch.tensor(word_pos, dtype=torch.long),
            "word_mask": torch.tensor(word_mask, dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
