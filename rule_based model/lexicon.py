# -*- coding: utf-8 -*-
"""Trigger lexicons and regexes — the hand-crafted knowledge of the baseline.

Vietnamese addresses are strongly *marked*: almost every field is introduced by
a small, closed set of **trigger words** (``đường`` for a street, ``phường``/``xã``
for a ward, ``quận``/``huyện`` for a district, ``tỉnh`` for a province, ``ngõ``/
``hẻm`` for an alley, ``tổ``/``thôn``/``ấp`` for a sub-area, ``số`` for a house
number, …). These triggers are the backbone of the rule model.

The baseline runs on the **raw, un-segmented** data (one Vietnamese *syllable*
per token — no word segmenter in the loop), so many triggers are **multi-token**:
``thành phố`` (2 tokens), ``thị trấn``, ``khu phố``, ``chung cư``, ``quốc lộ``,
``tổ dân phố`` (3 tokens), … :func:`match_trigger` therefore does a longest-phrase
match at a position. Each trigger is also registered in its underscore-joined form
(``thành_phố``) so the very same code still works if fed pre-segmented tokens.
"""

import re

# --------------------------------------------------------------------------- #
# Entity-type codes used by the trigger table. "CITY_OR_DISTRICT" is the shared
# ``thành phố`` / ``tp`` prefix, resolved to CITY vs DISTRICT by the tagger.
# --------------------------------------------------------------------------- #
STREET, WARD, DISTRICT, CITY = "STREET", "WARD", "DISTRICT", "CITY"
CITY_OR_DISTRICT = "CITY_OR_DISTRICT"
ALLEY, AREA, BUILDING, HOUSE_NO = "ALLEY", "AREA", "BUILDING", "HOUSE_NO"

# --------------------------------------------------------------------------- #
# Trigger phrases per type. A phrase is a space-separated run of raw syllables;
# multi-syllable phrases are what a compound word looks like *without* the
# segmenter (``"thành phố"``). The name/number that follows a trigger is absorbed
# as the I- continuation of the same type.
# --------------------------------------------------------------------------- #
_TRIGGER_TABLE = {
    STREET: [
        "đường", "duong", "phố", "pho", "ql", "tl", "đt",
        "quốc lộ", "đại lộ", "tỉnh lộ", "hương lộ", "vành đai", "đường tỉnh",
    ],
    WARD: [
        "phường", "phuong", "xã", "xa", "p",
        "thị trấn", "tt",
    ],
    DISTRICT: [
        "quận", "quan", "huyện", "huyen", "q",
        "thị xã", "tx", "đặc khu",
    ],
    CITY: [
        "tỉnh", "tinh",
    ],
    CITY_OR_DISTRICT: [
        "thành phố", "tp", "tp.",
    ],
    ALLEY: [
        "ngõ", "ngo", "ngách", "ngach", "hẻm", "hem", "kiệt", "kiet",
        "ng", "ng.", "k", "h", "hxh",
    ],
    AREA: [
        "thôn", "thon", "ấp", "ap", "tổ", "to", "khu", "xóm", "xom",
        "khóm", "khom", "khối", "khoi", "đội", "doi", "làng", "lang",
        "bản", "ban", "buôn", "buon", "sóc", "soc", "cụm", "cum",
        "lô", "lo", "tờ", "tdp", "kđt", "kdt", "kdc",
        "khu phố", "tổ dân phố", "khu đô thị", "khu dân cư", "khu tái định cư",
    ],
    BUILDING: [
        "cc", "toà", "tòa", "toa", "tầng", "tang", "block", "lầu", "lau",
        "tháp", "thap", "landmark", "vinhomes", "vinhome", "keangnam",
        "chung cư", "toà nhà", "tòa nhà", "căn hộ", "bệnh viện", "times city",
    ],
    HOUSE_NO: [
        "số", "so", "thửa", "thua", "phòng", "phong", "căn", "can",
        "nhà", "nha", "kiot", "sạp", "sap", "quầy", "quay",
    ],
}

# Flat single-token sets still used by the low-level numeric/guard logic (all
# alley + house triggers are single-syllable, so these stay simple sets).
ALLEY_TRIGGERS = {p for p in _TRIGGER_TABLE[ALLEY] if " " not in p}
HOUSE_TRIGGERS = {p for p in _TRIGGER_TABLE[HOUSE_NO] if " " not in p}

# Punctuation-only tokens.
SEPARATORS = {",", ";", "|", "-", "–", ".", "/"}


# --------------------------------------------------------------------------- #
# Longest-phrase trigger matcher
# --------------------------------------------------------------------------- #
def _build_index():
    """first-token → list of (phrase_tuple, type), longest phrase first."""
    index = {}
    for typ, phrases in _TRIGGER_TABLE.items():
        for phrase in phrases:
            toks = tuple(phrase.split())
            variants = {toks}
            if len(toks) > 1:                       # also match the segmented form
                variants.add((phrase.replace(" ", "_"),))
            for tup in variants:
                index.setdefault(tup[0], []).append((tup, typ))
    for first in index:
        index[first].sort(key=lambda pt: len(pt[0]), reverse=True)
    return index


_INDEX = _build_index()


def match_trigger(tokens, i):
    """Longest trigger phrase starting at ``tokens[i]``.

    Returns ``(type, phrase_len)`` for the longest matching phrase, or
    ``(None, 0)``. Case-insensitive. No context guards are applied here — the
    tagger layers those on top (e.g. the bare ``p`` / ``q`` / ``k`` triggers)."""
    cands = _INDEX.get(tokens[i].lower())
    if not cands:
        return None, 0
    n = len(tokens)
    for tup, typ in cands:                          # longest first
        L = len(tup)
        if i + L <= n and tuple(t.lower() for t in tokens[i:i + L]) == tup:
            return typ, L
    return None, 0


# --------------------------------------------------------------------------- #
# Regexes for numeric constituents (house numbers, alley numbers)
# --------------------------------------------------------------------------- #

# 12, 12a, 05, 123b — a number with an optional single trailing letter.
RE_NUMBER = re.compile(r"^\d+[a-zA-Z]?$", re.UNICODE)

# 12/3, 12/3b — number-slash-number kept as one token (rare in raw data).
RE_NUMSLASH = re.compile(r"^\d+[a-zA-Z]?(?:/\d+[a-zA-Z]?)+$", re.UNICODE)


def is_number(tok: str) -> bool:
    """True for a bare house/alley number token (``12``, ``12a``, ``12/3``)."""
    return bool(RE_NUMBER.match(tok) or RE_NUMSLASH.match(tok))
