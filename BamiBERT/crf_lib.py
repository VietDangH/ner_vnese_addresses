# -*- coding: utf-8 -*-
"""Loader for the linear-chain CRF layer (the well-tested ``pytorch-crf`` package).

``from torchcrf import CRF`` exposes ``forward`` -> log-likelihood and
``decode`` -> Viterbi tags, with ``transitions[i, j]`` = score of going from tag
``i`` to ``j`` (the convention ``model.sequence_marginals`` relies on for the
per-token confidence scores).

``pytorch-crf`` is not bundled with Colab, so we auto-install it on first use.
"""

import importlib
import subprocess
import sys


def _ensure_pytorch_crf():
    try:
        importlib.import_module("torchcrf")
    except ImportError:
        print("[setup] installing pytorch-crf ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "pytorch-crf"])


def build_crf(num_tags: int):
    """Return a batch-first ``torchcrf.CRF`` with ``num_tags`` states."""
    _ensure_pytorch_crf()
    from torchcrf import CRF
    return CRF(num_tags, batch_first=True)
