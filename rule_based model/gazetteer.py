# -*- coding: utf-8 -*-
"""Gazetteer index — dictionary lookup for ward / district / city names.

The gazetteer (``gazetteer.csv`` with columns ``ward,district,city``) is the
authoritative list of Vietnamese administrative units. We turn it into fast
membership sets so the tagger can:

  * recognise a place **name** even when its trigger word is missing
    (e.g. a bare ``"nghệ_an"`` at the end of an address → CITY), and
  * **disambiguate** the shared ``thành_phố`` / ``tp`` prefix, which can mean a
    province-level CITY or a district-level DISTRICT depending on the name that
    follows.

Matching is done on a **normalised** surface form so that the segmentation used
by the CoNLL data (compound words joined with ``_``, all lower-case, tone marks
intact) lines up with the CSV regardless of how either side was tokenised:

    "Thành phố Hà Nội"  ─normalise─►  "ha noi"
    "hà_nội"            ─normalise─►  "ha noi"

Because a name can span several tokens, the index exposes
:meth:`GazetteerIndex.longest_match`, which greedily matches the **longest** run
of tokens starting at a given position and reports which admin levels it could
be (a name like ``"hà_nội"`` is both a city and, historically, other things —
the tagger resolves the level using address position + hierarchy).
"""

import csv
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Administrative prefixes to peel off a raw gazetteer cell before indexing the
# bare name. (The CSV stores e.g. "Phường Trúc Bạch" / "Quận Ba Đình".)
_PREFIXES = [
    "thành phố", "thanh pho", "tỉnh", "tinh",
    "quận", "quan", "huyện", "huyen", "thị xã", "thi xa",
    "phường", "phuong", "xã", "xa", "thị trấn", "thi tran",
    "tp", "tx", "tt", "q", "p", "h",
]

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lower-case, replace ``_`` with space, strip punctuation, collapse spaces.

    Tone marks are **kept** (Vietnamese addresses depend on them); only layout
    noise is removed. The result is the canonical key used on both sides of a
    lookup."""
    text = text.lower().replace("_", " ")
    # keep letters/digits/space; drop commas, dots, slashes, etc.
    text = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text)
    return _WS.sub(" ", text).strip()


def _strip_prefix(name: str) -> str:
    """Remove a leading admin prefix (``phường``/``quận``/``tp`` …) if present."""
    n = normalise(name)
    for pref in _PREFIXES:
        if n == pref:
            return ""
        if n.startswith(pref + " "):
            return n[len(pref) + 1:]
    return n


class GazetteerIndex:
    """Membership sets + hierarchy maps built from the gazetteer CSV."""

    def __init__(self, wards: Set[str], districts: Set[str], cities: Set[str],
                 district_to_city: Dict[str, Set[str]],
                 ward_to_district: Dict[str, Set[str]]):
        self.wards = wards
        self.districts = districts
        self.cities = cities
        self.district_to_city = district_to_city
        self.ward_to_district = ward_to_district
        # longest name (in tokens) at each level — bounds the greedy match window
        self.max_span = 1
        for s in (wards, districts, cities):
            for name in s:
                self.max_span = max(self.max_span, len(name.split()))

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def from_csv(cls, path: str) -> "GazetteerIndex":
        wards: Set[str] = set()
        districts: Set[str] = set()
        cities: Set[str] = set()
        d2c: Dict[str, Set[str]] = {}
        w2d: Dict[str, Set[str]] = {}

        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                w = _strip_prefix(row.get("ward", "") or "")
                d = _strip_prefix(row.get("district", "") or "")
                c = _strip_prefix(row.get("city", "") or "")
                if w:
                    wards.add(w)
                if d:
                    districts.add(d)
                if c:
                    cities.add(c)
                if d and c:
                    d2c.setdefault(d, set()).add(c)
                if w and d:
                    w2d.setdefault(w, set()).add(d)
        return cls(wards, districts, cities, d2c, w2d)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "GazetteerIndex":
        """Load from ``path`` or auto-discover ``gazetteer.csv`` at the repo root."""
        if path is None:
            here = Path(__file__).resolve().parent
            for cand in (here.parent / "gazetteer.csv", here / "gazetteer.csv"):
                if cand.exists():
                    path = str(cand)
                    break
            if path is None:                       # pragma: no cover
                raise FileNotFoundError("gazetteer.csv not found; pass an explicit path")
        return cls.from_csv(path)

    # ---- lookup ---------------------------------------------------------- #
    def levels_of(self, name: str) -> Set[str]:
        """Which admin levels a (normalised) name belongs to: subset of
        ``{'WARD','DISTRICT','CITY'}``."""
        out = set()
        if name in self.wards:
            out.add("WARD")
        if name in self.districts:
            out.add("DISTRICT")
        if name in self.cities:
            out.add("CITY")
        return out

    def longest_match(self, tokens: List[str], start: int
                      ) -> Optional[Tuple[int, Set[str]]]:
        """Greedy longest gazetteer match beginning at ``tokens[start]``.

        Returns ``(span_len, levels)`` for the longest run of tokens that
        normalises to a known ward/district/city name, or ``None``. ``levels``
        tells the caller which admin level(s) that name can be so the tagger can
        pick one using address position + hierarchy.
        """
        best: Optional[Tuple[int, Set[str]]] = None
        upper = min(len(tokens), start + self.max_span)
        for end in range(start + 1, upper + 1):
            cand = normalise(" ".join(tokens[start:end]))
            if not cand:
                continue
            levels = self.levels_of(cand)
            if levels:
                best = (end - start, levels)       # keep extending → longest wins
        return best
