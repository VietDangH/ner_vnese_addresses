# -*- coding: utf-8 -*-
"""The composable NER model: encoder -> optional top layer -> head.

    sub-word ids ─► BamiBERT ─► last_hidden_state (B, T, H)
                                 │  gather first sub-word per token
                                 ▼
                         token features (B, W, H)
                                 │  optional top encoder (none / BiLSTM / CNN)
                                 ▼
                         features (B, W, H')
                                 │  head (softmax / CRF / ...)
                                 ▼
                         loss (train) / tags (infer)

The model is architecture-agnostic: the *top encoder* and the *head* are chosen
by name (see ``encoders.py`` / ``heads.py``). It exposes a single, stable
interface the rest of the pipeline depends on, regardless of those choices:

    forward(..., labels=None, decode=False) -> {"loss"?, "preds"?}
    decode_with_confidence(...)             -> (preds, confidences)
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from .encoders import build_encoder
from .heads import build_head


class NERModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int, head: str = "crf",
                 top_encoder: str = "none", dropout: float = 0.1,
                 encoder_kwargs: dict = None, head_kwargs: dict = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.top = build_encoder(top_encoder, hidden, **(encoder_kwargs or {}))
        self.head = build_head(head, self.top.out_dim, num_labels,
                               **(head_kwargs or {}))
        self.num_labels = num_labels
        self.head_name = head
        self.top_name = top_encoder

    def _features(self, input_ids, attention_mask, word_positions, word_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(out.last_hidden_state)                # (B, T, H)
        h = seq.size(-1)
        idx = word_positions.unsqueeze(-1).expand(-1, -1, h)     # (B, W, H)
        word_repr = torch.gather(seq, 1, idx)                    # (B, W, H)
        return self.top(word_repr, word_mask)                    # (B, W, H')

    def forward(self, input_ids, attention_mask, word_positions, word_mask,
                labels=None, decode=False):
        feats = self._features(input_ids, attention_mask, word_positions, word_mask)
        result = {}
        if labels is not None:
            result["loss"] = self.head.loss(feats, labels, word_mask)
        if decode:
            result["preds"] = self.head.decode(feats, word_mask)
        return result

    @torch.no_grad()
    def decode_with_confidence(self, input_ids, attention_mask, word_positions,
                               word_mask):
        """Decode and attach a per-token confidence ``P(predicted_tag | x)``.

        The confidence source depends on the head: softmax probability for the
        softmax head, CRF marginal (forward-backward) for the CRF head. Returns
        ``(preds, confidences)`` — two parallel lists, one entry per sentence,
        one value per token.
        """
        feats = self._features(input_ids, attention_mask, word_positions, word_mask)
        return self.head.decode_with_confidence(feats, word_mask)
