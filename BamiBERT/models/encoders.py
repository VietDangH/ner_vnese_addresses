# -*- coding: utf-8 -*-
"""Optional *top encoders* — extra contextual layers stacked on the BamiBERT
token features before the head.

This is the axis along which you "easily add other layers like LSTM, CNN". Each
top encoder maps token features ``(B, W, in_dim)`` -> ``(B, W, out_dim)`` and
exposes its ``out_dim`` so the head can size its projection accordingly.

    top_encoder="none"    -> identity (BamiBERT features straight into the head)
    top_encoder="bilstm"  -> a BiLSTM over the token sequence
    top_encoder="cnn"     -> a stack of 1-D convolutions over the token sequence

### Adding your own top encoder

Subclass ``nn.Module``, set ``self.out_dim``, implement ``forward(x, mask)``,
then register it::

    @register_encoder("mylayer")
    class MyLayer(nn.Module):
        def __init__(self, in_dim, **kw):
            super().__init__(); self.out_dim = in_dim
        def forward(self, x, mask): ...

It is then selectable everywhere via ``--top_encoder mylayer``.
"""

import torch
import torch.nn as nn

ENCODER_REGISTRY = {}


def register_encoder(name):
    def deco(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return deco


@register_encoder("none")
class Identity(nn.Module):
    """Pass BamiBERT token features straight through to the head."""

    def __init__(self, in_dim: int, **kw):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x, mask):
        return x


@register_encoder("bilstm")
class BiLSTM(nn.Module):
    """A (multi-layer) bidirectional LSTM over the token sequence."""

    def __init__(self, in_dim: int, hidden: int = 256, num_layers: int = 1,
                 dropout: float = 0.0, **kw):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out_dim = hidden * 2

    def forward(self, x, mask):
        # pack so the LSTM ignores padded tokens, then pad back to W.
        lengths = mask.sum(dim=1).clamp(min=1).to("cpu")
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.size(1))
        return out


@register_encoder("cnn")
class CNN(nn.Module):
    """A stack of width-preserving 1-D convolutions over the token sequence."""

    def __init__(self, in_dim: int, channels: int = 256, kernel_size: int = 3,
                 num_layers: int = 1, dropout: float = 0.0, **kw):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(num_layers):
            layers += [nn.Conv1d(d, channels, kernel_size,
                                 padding=kernel_size // 2), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            d = channels
        self.net = nn.Sequential(*layers)
        self.out_dim = channels

    def forward(self, x, mask):
        h = self.net(x.transpose(1, 2)).transpose(1, 2)   # (B, W, C)
        return h * mask.unsqueeze(-1).to(h.dtype)          # zero padded tokens


def build_encoder(name: str, in_dim: int, **kwargs) -> nn.Module:
    """Instantiate a registered top encoder by name."""
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"unknown top_encoder '{name}'; available: {sorted(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[name](in_dim, **kwargs)
