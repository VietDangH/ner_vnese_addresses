# -*- coding: utf-8 -*-
"""
vn_ner: Vietnamese Address NER on PhoBERT, with a composable architecture.

One package, pluggable parts:
  * head        — "crf" (with CRF) or "softmax" (without); add your own.
  * top_encoder — "none" / "bilstm" / "cnn" on top of PhoBERT; add your own.

The pipeline is split into single-purpose steps:
    python -m vn_ner.split      # clean file  -> train/val/test (no augmentation)
    python -m vn_ner.augment    # augment ANY file (decoupled from splitting)
    python -m vn_ner.train      # train (choose --head / --top_encoder); no test eval
    python -m vn_ner.evaluate   # the ONLY place metrics are computed
    python -m vn_ner.predict    # tag new addresses
    python -m vn_ner.benchmark  # prediction-time comparison

Public API
----------
    from vn_ner import build_model, NERModel, load_trained, Predictor, read_conll
"""

from .models import NERModel, build_model, load_trained
from .data import read_conll, write_conll, build_label_list, split_sentences
from .predictor import Predictor

__all__ = [
    "NERModel",
    "build_model",
    "load_trained",
    "Predictor",
    "read_conll",
    "write_conll",
    "build_label_list",
    "split_sentences",
]

__version__ = "2.0.0"
