# -*- coding: utf-8 -*-
"""Decoder *heads* — the part that turns per-word features into tag predictions.

A head owns the projection from feature space to the label space **and** the
training loss + decoding for that label space. Swapping the head is how you
choose "with or without CRF":

    head="softmax"  -> per-token linear + cross-entropy, argmax decoding
    head="crf"      -> linear emissions + linear-chain CRF (pytorch-crf),
                       sequence NLL loss, Viterbi decoding

Every head implements the same three methods so the rest of the pipeline never
needs to know which one is in use::

    loss(feats, labels, mask)              -> scalar tensor
    decode(feats, mask)                    -> List[List[int]]   (trimmed to words)
    decode_with_confidence(feats, mask)    -> (preds, confidences)

`feats` is `(B, W, in_dim)`, `mask` is `(B, W)` bool (True = real word), `labels`
is `(B, W)` long. Outputs are Python lists with padding removed.

### Adding your own head

Subclass ``nn.Module``, implement the three methods above, then register it::

    @register_head("myhead")
    class MyHead(nn.Module):
        def __init__(self, in_dim, num_labels, **kw): ...

It is then selectable everywhere via ``--head myhead``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..crf_lib import build_crf

HEAD_REGISTRY = {}


def register_head(name):
    def deco(cls):
        HEAD_REGISTRY[name] = cls
        return cls
    return deco


def _trim(tensor, mask):
    """(B, W) tensor -> list of per-sentence lists, padded words removed."""
    lengths = mask.sum(dim=1).tolist()
    return [tensor[i, :int(n)].tolist() for i, n in enumerate(lengths)]


# --------------------------------------------------------------------------- #
# Softmax head (no CRF)
# --------------------------------------------------------------------------- #
@register_head("softmax")
class SoftmaxHead(nn.Module):
    """Per-token linear classifier with softmax cross-entropy.

    Scores each word independently, so it is cheap at inference but can emit
    globally inconsistent BIO sequences (e.g. ``I-CITY`` after ``B-WARD``)."""

    def __init__(self, in_dim: int, num_labels: int, **kw):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_labels)
        self.num_labels = num_labels

    def loss(self, feats, labels, mask):
        logits = self.classifier(feats)
        active = mask.reshape(-1)
        return F.cross_entropy(
            logits.reshape(-1, self.num_labels)[active], labels.reshape(-1)[active])

    def decode(self, feats, mask):
        return _trim(self.classifier(feats).argmax(dim=-1), mask)

    @torch.no_grad()
    def decode_with_confidence(self, feats, mask):
        probs = torch.softmax(self.classifier(feats).float(), dim=-1)
        conf, preds = probs.max(dim=-1)
        return _trim(preds, mask), _trim(conf, mask)


# --------------------------------------------------------------------------- #
# CRF head
# --------------------------------------------------------------------------- #
@register_head("crf")
class CRFHead(nn.Module):
    """Linear emissions + a linear-chain CRF (``pytorch-crf``).

    Models label-transition structure, so the BIO output is globally consistent
    (illegal transitions are forbidden). Slightly slower at inference (Viterbi)."""

    def __init__(self, in_dim: int, num_labels: int, **kw):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_labels)
        self.crf = build_crf(num_labels)
        self.num_labels = num_labels

    def _emissions(self, feats):
        # CRF math (logsumexp / Viterbi) is unstable in fp16, so emit fp32 even
        # under AMP autocast.
        return self.classifier(feats).float()

    def loss(self, feats, labels, mask):
        # CRF returns log-likelihood; the loss is its negative.
        return -self.crf(self._emissions(feats), labels, mask=mask, reduction="mean")

    def decode(self, feats, mask):
        return self.crf.decode(self._emissions(feats), mask=mask)

    @torch.no_grad()
    def decode_with_confidence(self, feats, mask):
        emissions = self._emissions(feats)
        preds = self.crf.decode(emissions, mask=mask)
        st, et, tr = (self.crf.start_transitions, self.crf.end_transitions,
                      self.crf.transitions)
        lengths = mask.sum(dim=1).tolist()
        confidences = []
        for b, n in enumerate(lengths):
            n = int(n)
            probs, _ = sequence_marginals(emissions[b, :n], st, et, tr)
            confidences.append([probs[i, preds[b][i]].item() for i in range(n)])
        return preds, confidences


@torch.no_grad()
def sequence_marginals(emissions, start_transitions, end_transitions, transitions):
    """Forward-backward token marginals for ONE (unpadded) sequence.

    Returns ``(probs, log_Z)`` where ``probs[t, k] = P(y_t = k | x)``; the
    confidence of the predicted tag at position ``t`` is ``probs[t, pred_t]``.
    ``transitions[i, j]`` is the score of going from tag ``i`` to ``j`` (the
    ``pytorch-crf`` convention). All math is float32 for numerical stability.
    """
    emissions = emissions.float()
    start_transitions = start_transitions.float()
    end_transitions = end_transitions.float()
    transitions = transitions.float()
    seq_len, num_tags = emissions.shape

    alpha = emissions.new_empty(seq_len, num_tags)
    alpha[0] = start_transitions + emissions[0]
    for t in range(1, seq_len):
        alpha[t] = emissions[t] + torch.logsumexp(
            alpha[t - 1].unsqueeze(1) + transitions, dim=0)
    log_Z = torch.logsumexp(alpha[-1] + end_transitions, dim=0)

    beta = emissions.new_empty(seq_len, num_tags)
    beta[-1] = end_transitions
    for t in range(seq_len - 2, -1, -1):
        beta[t] = torch.logsumexp(
            transitions + (emissions[t + 1] + beta[t + 1]).unsqueeze(0), dim=1)

    probs = torch.exp(alpha + beta - log_Z)
    return probs, log_Z


def build_head(name: str, in_dim: int, num_labels: int, **kwargs) -> nn.Module:
    """Instantiate a registered head by name."""
    if name not in HEAD_REGISTRY:
        raise ValueError(
            f"unknown head '{name}'; available: {sorted(HEAD_REGISTRY)}")
    return HEAD_REGISTRY[name](in_dim, num_labels, **kwargs)
