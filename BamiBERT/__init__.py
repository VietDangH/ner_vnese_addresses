# -*- coding: utf-8 -*-
"""
bami_ner: Vietnamese Address NER on BamiBERT, with a composable architecture.

A sibling of ``vn_ner`` (PhoBERT). The architecture, heads (CRF / softmax) and
top encoders (none / BiLSTM / CNN) are identical and fully reusable — only two
things differ, both isolated here:

  * tokenizer loading — BamiBERT needs a ``PreTrainedTokenizerFast`` fallback
    (see ``bami_ner.tokenization``);
  * sub-word encoding — BamiBERT is a byte-level BPE, so tokens are encoded with
    a leading space (auto-detected in ``bami_ner.data.Collator``), and it runs
    on RAW (un-segmented) text.

One package, pluggable parts:
  * head        — "crf" (with CRF) or "softmax" (without); add your own.
  * top_encoder — "none" / "bilstm" / "cnn" on top of BamiBERT; add your own.

The pipeline is split into single-purpose steps:
    python -m bami_ner.split      # clean file  -> train/val/test (no augmentation)
    python -m bami_ner.augment    # augment ANY file (decoupled from splitting)
    python -m bami_ner.train      # train (choose --head / --top_encoder); no test eval
    python -m bami_ner.evaluate   # the ONLY place metrics are computed
    python -m bami_ner.predict    # tag new addresses (raw text, no segmentation)
    python -m bami_ner.benchmark  # prediction-time comparison

Raw input: if your CoNLL is word-segmented, run ``desegment.py`` (outside this
package) first to turn compounds back into syllables.

Public API
----------
    from bami_ner import build_model, NERModel, load_trained, Predictor, read_conll
"""

from .models import NERModel, build_model, load_trained
from .data import read_conll, write_conll, build_label_list, split_sentences
from .predictor import Predictor
from .tokenization import load_tokenizer

__all__ = [
    "NERModel",
    "build_model",
    "load_trained",
    "Predictor",
    "load_tokenizer",
    "read_conll",
    "write_conll",
    "build_label_list",
    "split_sentences",
]

__version__ = "1.0.0"
