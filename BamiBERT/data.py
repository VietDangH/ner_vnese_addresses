# -*- coding: utf-8 -*-
"""Data layer: read CoNLL, build the label vocabulary, split, and collate.

The training file is CoNLL-2003 style -- one token per line, a blank line
between sentences, and each line shaped like::

    thành -X- _ B-DISTRICT

i.e. ``<token> -X- _ <BIO-label>``.

Unlike the PhoBERT pipeline (``vn_ner``), BamiBERT operates on **raw**
(un-segmented) text: the tokens here are individual syllables, not ``_``-joined
compound words. If your CoNLL is word-segmented, run ``desegment.py`` first to
split the compounds back into syllables (BIO labels are expanded accordingly).

BamiBERT's tokenizer is a byte-level BPE, which is whitespace-sensitive:
``"phố"`` and ``" phố"`` tokenize differently. To reproduce how a token would be
tokenized inside a running sentence, every token except the first is encoded
**with a leading space**. We then feed the hidden state of the *first* sub-word
of each token into the classifier, so the head runs over a clean, token-level
sequence aligned 1:1 with the gold labels. (This whitespace handling is
auto-detected, so pointing the collator at a non-byte-level tokenizer like
PhoBERT still works.)
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

    For every token we store the index of its first BPE sub-word so the model
    can ``gather`` one vector per token. Output keys:
        input_ids, attention_mask  -> for the BamiBERT encoder (sub-word level)
        word_positions              -> first sub-word index of each token
        word_mask                   -> True for real tokens (left-aligned)
        labels                      -> gold tag id per token

    ``add_prefix_space`` is auto-detected: for a byte-level BPE tokenizer (like
    BamiBERT) every non-first token is encoded with a leading space so its
    sub-words match how it would appear mid-sentence.
    """

    def __init__(self, tokenizer, label2id: Dict[str, int], max_length: int):
        self.tok = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        # RoBERTa-style special tokens: prefer cls/sep, fall back to bos/eos.
        self.bos = (tokenizer.cls_token_id if tokenizer.cls_token_id is not None
                    else tokenizer.bos_token_id)
        self.eos = (tokenizer.sep_token_id if tokenizer.sep_token_id is not None
                    else tokenizer.eos_token_id)
        self.pad = tokenizer.pad_token_id
        if self.pad is None:
            self.pad = self.eos if self.eos is not None else 0
        self.unk = tokenizer.unk_token_id
        self.add_prefix_space = self._detect_prefix_space(tokenizer)

    @staticmethod
    def _detect_prefix_space(tok) -> bool:
        """True if a leading space changes the sub-words (byte-level BPE)."""
        try:
            probe = "phường"
            no_space = tok(probe, add_special_tokens=False)["input_ids"]
            with_space = tok(" " + probe, add_special_tokens=False)["input_ids"]
            return no_space != with_space
        except Exception:
            return True   # BamiBERT is byte-level; default to the safe case

    def _encode_token(self, tok: str, first: bool) -> List[int]:
        text = tok if (first or not self.add_prefix_space) else " " + tok
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def encode_one(self, tokens: List[str], labels: List[str]):
        input_ids = [self.bos] if self.bos is not None else []
        first_pos: List[int] = []
        label_ids: List[int] = []
        reserve = 1 if self.eos is not None else 0
        for tok, lab in zip(tokens, labels):
            pieces = self._encode_token(tok, first=not first_pos)
            if not pieces:
                pieces = [self.unk] if self.unk is not None else []
                if not pieces:
                    continue
            if len(input_ids) + len(pieces) > self.max_length - reserve:
                break
            first_pos.append(len(input_ids))
            input_ids.extend(pieces)
            label_ids.append(self.label2id.get(lab, 0))
        if self.eos is not None:
            input_ids.append(self.eos)
        # Guard: a sentence must keep at least one token for the token mask.
        if not first_pos:
            input_ids = [x for x in (self.bos, self.unk, self.eos) if x is not None]
            first_pos = [1 if self.bos is not None else 0]
            label_ids = [0]
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
