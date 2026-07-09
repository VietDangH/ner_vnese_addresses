# -*- coding: utf-8 -*-
"""Rule-based Vietnamese-address NER tagger (the baseline model).

The tagger turns a list of **raw, un-segmented** tokens (one Vietnamese syllable
per token — no word segmenter in the loop) into BIO tags for the 8 address entity
types: ``HOUSE_NO, ALLEY, STREET, AREA, WARD, DISTRICT, BUILDING, CITY``.
Multi-syllable markers such as ``thành phố`` / ``thị trấn`` / ``khu phố`` are
matched as phrases; pre-segmented ``thành_phố`` tokens also work.

It combines four families of rules, in priority order:

1. **Field structure.**  Commas (``,`` / ``;`` / ``|``) split an address into
   fields; the first field carries the house-number / alley / street, later
   fields the administrative units, ordered *specific → general*.

2. **Leading-numeric parsing.**  At the very front we parse the
   ``<alley>/<house-no>`` block with a small grammar (``k 13``, ``204 / 9``,
   ``ng. 43 / 1a`` …) driven by :mod:`.lexicon` regexes.

3. **Trigger words.**  A closed set of markers introduces each field —
   ``đường``→STREET, ``phường``/``xã``→WARD, ``quận``/``huyện``→DISTRICT,
   ``tỉnh``→CITY, ``ngõ``/``hẻm``→ALLEY, ``tổ``/``thôn``/``ấp``→AREA,
   ``chung_cư``→BUILDING, ``số``→HOUSE_NO. The trigger opens a ``B-`` span and
   the name/number that follows is absorbed as ``I-``.

4. **Gazetteer + hierarchy.**  When a name has no trigger, we look it up in the
   gazetteer (:mod:`.gazetteer`). The shared ``thành_phố`` / ``tp`` prefix is
   resolved to CITY vs DISTRICT using the gazetteer and the field's position
   (the last field is the province/city).

The result is a transparent, dependency-light baseline: no training, fully
deterministic, and directly comparable to the neural models on the same split.
"""

from typing import Dict, List, Optional, Tuple

from .gazetteer import GazetteerIndex, normalise
from .lexicon import (
    ALLEY_TRIGGERS,
    HOUSE_TRIGGERS,
    is_number,
    match_trigger,
)

FIELD_SEP = {",", ";", "|"}            # split an address into fields
IN_FIELD_BREAK = {"/"}                 # a boundary *inside* a field (never a name char)
# single-letter / bare-abbrev triggers that need a guard before firing
_ALLEY_GUARDED = {"k", "h", "ng", "ng."}
_WARD_GUARDED = {"p"}
_DISTRICT_GUARDED = {"q"}
# stop the greedy name-consumer when we hit a trigger for one of these types
_NAME_STOP_TYPES = {"STREET", "WARD", "DISTRICT", "CITY", "AREA", "BUILDING",
                    "ALLEY", "HOUSE_NO"}


class RuleBasedNER:
    """Deterministic address tagger. ``predict_tokens`` is the core entry point."""

    def __init__(self, gazetteer: Optional[GazetteerIndex] = None,
                 gazetteer_path: Optional[str] = None):
        self.gaz = gazetteer or GazetteerIndex.load(gazetteer_path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict_tokens(self, tokens: List[str]) -> List[str]:
        """Tag a list of segmented tokens → list of BIO labels (same length)."""
        n = len(tokens)
        labels = ["O"] * n
        if n == 0:
            return labels

        # 1) carve the token stream into comma-separated fields
        fields: List[List[int]] = []
        cur: List[int] = []
        for i, tok in enumerate(tokens):
            if tok in FIELD_SEP:
                if cur:
                    fields.append(cur)
                cur = []
            else:
                cur.append(i)
        if cur:
            fields.append(cur)

        # 2) tag each field, telling it whether it is the first / last field
        for fpos, field in enumerate(fields):
            self._tag_field(
                tokens, field, labels,
                is_first=(fpos == 0),
                is_last=(fpos == len(fields) - 1),
            )

        return self._repair_bio(labels)

    def predict(self, text: str) -> Dict:
        """Tag a raw (pre-segmented, whitespace-joined) string → entities dict.

        Mirrors the shape of the neural ``Predictor.predict`` output enough to be
        a drop-in for quick demos: ``{"tokens", "labels", "entities"}``."""
        tokens = text.split()
        labels = self.predict_tokens(tokens)
        return {
            "tokens": tokens,
            "labels": labels,
            "entities": self.extract_entities(tokens, labels),
        }

    @staticmethod
    def extract_entities(tokens: List[str], labels: List[str]) -> List[Dict]:
        """Collapse BIO labels into ``[{type, text, start, end}]`` spans."""
        ents, cur = [], None
        for i, lab in enumerate(labels):
            if lab.startswith("B-"):
                if cur:
                    ents.append(cur)
                cur = {"type": lab[2:], "start": i, "end": i + 1,
                       "tokens": [tokens[i]]}
            elif lab.startswith("I-") and cur and cur["type"] == lab[2:]:
                cur["end"] = i + 1
                cur["tokens"].append(tokens[i])
            else:
                if cur:
                    ents.append(cur)
                cur = None
        if cur:
            ents.append(cur)
        for e in ents:
            e["text"] = " ".join(e.pop("tokens"))
        return ents

    # ------------------------------------------------------------------ #
    # Field tagging
    # ------------------------------------------------------------------ #
    def _tag_field(self, tokens: List[str], field: List[int],
                   labels: List[str], is_first: bool, is_last: bool):
        # work on a local view of the field, then write labels back by index
        ft = [tokens[i] for i in field]
        fl = ["O"] * len(ft)

        # (A) leading <alley>/<house-no> numeric block
        j = self._tag_leading_numeric(ft, fl)

        # (B) scan the remainder
        residual: List[int] = []
        street_seen = any(l.endswith("STREET") for l in fl)

        def flush_residual():
            nonlocal street_seen
            if not residual:
                return
            has_alpha = any(not is_number(ft[k]) and ft[k] not in IN_FIELD_BREAK
                            for k in residual)
            # a bare, trigger-less alpha run at the very FRONT of the address is
            # almost always a street name written without "đường".
            if is_first and not street_seen and has_alpha:
                first = True
                for k in residual:
                    if ft[k] in IN_FIELD_BREAK:
                        fl[k] = "O"
                        continue
                    fl[k] = "B-STREET" if first else "I-STREET"
                    first = False
                street_seen = True
            residual.clear()

        n = len(ft)
        while j < n:
            tok = ft[j]
            if tok in IN_FIELD_BREAK:
                flush_residual()
                fl[j] = "O"
                j += 1
                continue

            typ, consumed = self._match_trigger(ft, j, is_last)
            if typ:
                flush_residual()
                self._assign(ft, fl, j, consumed, typ)
                if typ == "STREET":
                    street_seen = True
                j += consumed
                continue

            gz = self._gazetteer_match(ft, j, is_first, is_last)
            if gz:
                span, level = gz
                flush_residual()
                self._assign(ft, fl, j, span, level)
                j += span
                continue

            residual.append(j)
            j += 1
        flush_residual()

        for local_i, global_i in enumerate(field):
            labels[global_i] = fl[local_i]

    # ------------------------------------------------------------------ #
    # (A) leading alley / house-number block
    # ------------------------------------------------------------------ #
    def _tag_leading_numeric(self, ft: List[str], fl: List[str]) -> int:
        """Tag a leading ``<alley>/<house-no>`` block; return where it ends."""
        n = len(ft)
        if n == 0:
            return 0
        head = ft[0]
        if not (is_number(head) or head in ALLEY_TRIGGERS or head in HOUSE_TRIGGERS):
            return 0

        # collect the block: numbers, '/', and alley/house trigger words
        i = 0
        block: List[int] = []
        while i < n:
            t = ft[i]
            if t in IN_FIELD_BREAK or is_number(t) or t in ALLEY_TRIGGERS \
                    or t in HOUSE_TRIGGERS:
                block.append(i)
                i += 1
            else:
                break
        if not block:
            return 0

        # split into '/'-separated segments
        segments: List[List[int]] = []
        cur: List[int] = []
        for k in block:
            if ft[k] in IN_FIELD_BREAK:
                fl[k] = "O"
                if cur:
                    segments.append(cur)
                    cur = []
            else:
                cur.append(k)
        if cur:
            segments.append(cur)
        segments = [s for s in segments if s]
        if not segments:
            return i

        last_idx = len(segments) - 1
        for si, seg in enumerate(segments):
            head_tok = ft[seg[0]]
            if head_tok in HOUSE_TRIGGERS:
                typ = "HOUSE_NO"
            elif head_tok in ALLEY_TRIGGERS:
                typ = "ALLEY"
            elif len(segments) == 1:
                typ = "HOUSE_NO"              # a single bare number is the house no
            elif si == last_idx:
                typ = "HOUSE_NO"             # last segment before the street
            else:
                typ = "ALLEY"                # earlier segments are the alley
            for pos, k in enumerate(seg):
                fl[k] = ("B-" if pos == 0 else "I-") + typ
        return i

    # ------------------------------------------------------------------ #
    # (C) trigger-word matching  (multi-token / phrase aware)
    # ------------------------------------------------------------------ #
    def _trigger_at(self, ft: List[str], k: int) -> Tuple[Optional[str], int]:
        """Longest guarded trigger phrase at ``ft[k]`` → ``(type, phrase_len)``.

        Wraps :func:`lexicon.match_trigger` and applies the context guards for the
        ambiguous single-letter / abbreviation triggers (``k``/``h``/``ng`` need a
        following number; ``p``/``q`` need a following name)."""
        typ, L = match_trigger(ft, k)
        if typ is None:
            return None, 0
        head = ft[k].lower()
        if typ == "ALLEY" and L == 1 and head in _ALLEY_GUARDED \
                and not self._next_is_number(ft, k + 1):
            return None, 0
        if typ == "WARD" and L == 1 and head in _WARD_GUARDED \
                and not self._next_is_namey(ft, k + 1):
            return None, 0
        if typ == "DISTRICT" and L == 1 and head in _DISTRICT_GUARDED \
                and not self._next_is_namey(ft, k + 1):
            return None, 0
        return typ, L

    def _match_trigger(self, ft: List[str], j: int, is_last: bool
                       ) -> Tuple[Optional[str], int]:
        """If ``ft[j]`` opens a trigger phrase, return ``(type, span_len)`` covering
        the trigger + the name/number it introduces; else ``(None, 0)``."""
        typ, L = self._trigger_at(ft, j)
        if typ is None:
            return None, 0

        if typ in ("STREET", "WARD", "DISTRICT", "CITY", "AREA", "BUILDING",
                   "CITY_OR_DISTRICT"):
            cur = "CITY" if typ == "CITY_OR_DISTRICT" else typ
            end = self._consume_name(ft, j + L, cur, allow_numbers=True)
            if typ == "CITY_OR_DISTRICT":
                name = normalise(" ".join(ft[j + L:end]))
                return self._resolve_city_district(name, is_last), end - j
            return typ, end - j
        if typ == "ALLEY":
            end = self._consume_number_run(ft, j + L)
            return "ALLEY", end - j
        # HOUSE_NO
        end = self._consume_number_run(ft, j + L)
        if end == j + L:                            # nothing numeric followed
            end = self._consume_name(ft, j + L, "HOUSE_NO", allow_numbers=True)
        return "HOUSE_NO", end - j

    def _consume_name(self, ft: List[str], start: int, cur_type: str,
                      allow_numbers: bool) -> int:
        """Greedily absorb the name after a trigger: alpha tokens (and ``.``/``-``,
        optionally numbers), stopping at a ``/`` break or *another* type's trigger.

        A trigger of the **same** ``cur_type`` (e.g. ``đt`` inside ``đường đt 605``)
        is absorbed, so repeated same-type markers stay one span. Multi-token
        triggers (``thị trấn``, ``khu phố``) are detected as phrases."""
        k = start
        n = len(ft)
        while k < n:
            tok = ft[k]
            if tok in IN_FIELD_BREAK:
                break
            trig, _ = self._trigger_at(ft, k)
            if trig is not None:
                trig = "CITY" if trig == "CITY_OR_DISTRICT" else trig
                if trig in _NAME_STOP_TYPES and trig != cur_type:
                    break
            if is_number(tok) and not allow_numbers:
                break
            k += 1
        return k

    def _consume_number_run(self, ft: List[str], start: int) -> int:
        """Absorb a run of number-ish tokens (and inner ``.``/``-``) after an
        alley/house trigger, e.g. ``13``, ``266 h 14``... stopping at ``/``."""
        k = start
        n = len(ft)
        while k < n:
            tok = ft[k]
            if tok in IN_FIELD_BREAK:
                break
            if is_number(tok) or tok in {".", "-"} or tok in ALLEY_TRIGGERS:
                k += 1
            else:
                break
        return k

    # ------------------------------------------------------------------ #
    # (D) gazetteer matching + level resolution
    # ------------------------------------------------------------------ #
    def _gazetteer_match(self, ft: List[str], j: int, is_first: bool,
                         is_last: bool) -> Optional[Tuple[int, str]]:
        """Bare-name gazetteer lookup with precision guards. Returns
        ``(span_len, 'WARD'|'DISTRICT'|'CITY')`` or ``None``."""
        # protect the front-of-address street: skip bare matches in a first field
        # that is not the whole address.
        if is_first and not is_last:
            return None
        m = self.gaz.longest_match(ft, j)
        if not m:
            return None
        span, levels = m
        name = normalise(" ".join(ft[j:j + span]))
        # reject short single-syllable matches (e.g. "an", "phú") unless final
        if " " not in name and span < 2 and not is_last:
            return None
        return span, self._choose_level(levels, is_last)

    @staticmethod
    def _choose_level(levels, is_last: bool) -> str:
        if len(levels) == 1:
            return next(iter(levels))
        order = ("CITY", "DISTRICT", "WARD") if is_last \
            else ("DISTRICT", "WARD", "CITY")
        for lv in order:
            if lv in levels:
                return lv
        return "CITY"

    def _resolve_city_district(self, name: str, is_last: bool) -> str:
        """Resolve the shared ``thành_phố`` / ``tp`` prefix to CITY vs DISTRICT."""
        in_city = name in self.gaz.cities
        in_dist = name in self.gaz.districts
        if in_city and not in_dist:
            return "CITY"
        if in_dist and not in_city:
            return "DISTRICT"
        if in_city:
            return "CITY"
        if in_dist:
            return "DISTRICT"
        return "CITY" if is_last else "DISTRICT"

    # ------------------------------------------------------------------ #
    # small helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _next_is_namey(ft: List[str], k: int) -> bool:
        """True if ``ft[k]`` looks like the start of a place name (alpha token or
        an abbreviation dot), used to guard the bare ``p`` / ``q`` triggers."""
        if k >= len(ft):
            return False
        tok = ft[k]
        if tok in IN_FIELD_BREAK or tok in FIELD_SEP:
            return False
        if tok in {".", "-"}:                       # "p . dinh cong"
            return True
        if is_number(tok):
            return False
        # not the start of some other trigger phrase (match_trigger has no guards,
        # so this call cannot recurse back into _trigger_at)
        return match_trigger(ft, k)[0] is None

    @staticmethod
    def _next_is_number(ft: List[str], k: int) -> bool:
        return k < len(ft) and is_number(ft[k])

    @staticmethod
    def _assign(ft: List[str], fl: List[str], start: int, length: int, typ: str):
        """Write ``B-typ``/``I-typ`` over ``[start, start+length)``; a ``/`` inside
        the span is kept as ``O`` and the next real token restarts the ``I-`` run."""
        first = True
        for k in range(start, min(start + length, len(fl))):
            if ft[k] in IN_FIELD_BREAK:
                fl[k] = "O"
                continue
            fl[k] = ("B-" if first else "I-") + typ
            first = False

    @staticmethod
    def _repair_bio(labels: List[str]) -> List[str]:
        """Promote any dangling ``I-X`` (no matching open span) to ``B-X`` so the
        output is always valid BIO."""
        out = []
        prev_type = None
        for lab in labels:
            if lab == "O":
                out.append("O")
                prev_type = None
                continue
            tag, typ = lab[0], lab[2:]
            if tag == "I" and prev_type != typ:
                out.append("B-" + typ)
            else:
                out.append(lab)
            prev_type = typ
        return out
