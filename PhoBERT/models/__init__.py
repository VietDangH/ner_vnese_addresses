# -*- coding: utf-8 -*-
"""Model factory + a single loader used everywhere a trained model is reloaded.

    from vn_ner.models import build_model, load_trained, HEAD_REGISTRY, ENCODER_REGISTRY
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .base import NERModel
from .encoders import ENCODER_REGISTRY, build_encoder, register_encoder
from .heads import HEAD_REGISTRY, build_head, register_head

__all__ = [
    "NERModel", "build_model", "load_trained",
    "HEAD_REGISTRY", "ENCODER_REGISTRY",
    "build_head", "build_encoder", "register_head", "register_encoder",
]


def build_model(model_name: str, num_labels: int, head: str = "crf",
                top_encoder: str = "none", dropout: float = 0.1,
                encoder_kwargs: dict = None, head_kwargs: dict = None) -> NERModel:
    """Construct an (untrained) :class:`NERModel` with the chosen head/top layer."""
    return NERModel(model_name, num_labels, head=head, top_encoder=top_encoder,
                    dropout=dropout, encoder_kwargs=encoder_kwargs,
                    head_kwargs=head_kwargs)


def load_trained(model_dir, device):
    """Rebuild the *exact* trained architecture from ``config.json`` and load it.

    Returns ``(model, tokenizer, label_list, id2label, cfg)``. The architecture
    choices (head, top_encoder and their kwargs) are read back from the saved
    config, so softmax/CRF/BiLSTM/... models all reload through this one path.
    """
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    label_list = [id2label[i] for i in range(len(id2label))]

    tok_path = model_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_path) if tok_path.exists() else cfg["model_name"])

    model = build_model(
        cfg["model_name"], len(label_list),
        head=cfg.get("head", "crf"),
        top_encoder=cfg.get("top_encoder", "none"),
        encoder_kwargs=cfg.get("encoder_kwargs"),
        head_kwargs=cfg.get("head_kwargs"),
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.to(device)
    return model, tokenizer, label_list, id2label, cfg
