# -*- coding: utf-8 -*-
"""Tokenizer loading for BamiBERT.

BamiBERT ships a byte-level BPE ``tokenizer.json`` but its
``tokenizer_config.json`` declares ``XLMRobertaTokenizer`` — on some
``transformers`` versions ``AutoTokenizer`` tries to build a SentencePiece
Unigram model from that and crashes. ``PreTrainedTokenizerFast`` loads
``tokenizer.json`` directly and always works, so we fall back to it.

This is the single place tokenizers are loaded, so training, evaluation and
prediction all agree on how the encoder's input is tokenized.
"""

from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(model_name_or_path: str):
    """Load a tokenizer for ``model_name_or_path`` (HF id or a saved dir).

    Tries ``AutoTokenizer`` first (works for PhoBERT and for tokenizers we saved
    ourselves), then falls back to ``PreTrainedTokenizerFast`` which reads
    ``tokenizer.json`` directly (needed for the BamiBERT repo).
    """
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception as e:  # noqa: BLE001  (any construction error -> fall back)
        print(f"[tokenizer] AutoTokenizer failed ({type(e).__name__}); "
              f"falling back to PreTrainedTokenizerFast")
        return PreTrainedTokenizerFast.from_pretrained(model_name_or_path)
