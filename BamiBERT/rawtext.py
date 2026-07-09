# -*- coding: utf-8 -*-
"""Raw-text tokenization for BamiBERT inference.

BamiBERT works directly on raw text, so — unlike the PhoBERT pipeline — there is
**no word-segmentation step** here. We only split on whitespace and peel
standalone punctuation (``, / ( ) ; .``) into their own tokens, matching how the
CoNLL training data is tokenized (one syllable / punctuation mark per token).
"""

import re


def raw_tokenize(text: str, lower: bool = True) -> list:
    """Split raw address text into syllable / punctuation tokens.

    This is the inference-time counterpart of the tokens in the CoNLL data:
    punctuation the data keeps as its own token is split out so it aligns with
    how the model was trained.
    """
    text = text.strip()
    if not text:
        return []
    # pad standalone punctuation so it becomes its own token
    text = re.sub(r"([,/();])", r" \1 ", text)
    tokens = text.split()
    if lower:
        tokens = [t.lower() for t in tokens]
    return tokens
