# -*- coding: utf-8 -*-
"""
Generate ~500 synthetic CoNLL sentences, each containing at least one BUILDING
entity, written the way humans actually write Vietnamese addresses.

Strategy
--------
* Real component spans (BUILDING / STREET / AREA / ALLEY / HOUSE_NO) are
  extracted from Cleaned_Data.conll and RECOMBINED, so every token is something
  a human really wrote -- only the *combination* is new.
* Admin parts (WARD / DISTRICT / CITY) come from gazetteer_segmented.csv so the
  ward really belongs to that district / city (geographically coherent).
* BUILDING is additionally enriched with composed "head + name + suffix/block"
  spans to widen the open vocabulary the model struggles with.
* Field order, separators ( ; , - | space ), keyword prefixes (phường/p, quận/q,
  tỉnh/tp ...) and the trailing "( landmark )" pattern all mimic the real file.

Output: Generated_Building_Data.conll  (same 4-column format as Cleaned_Data.conll)

Deterministic: a fixed RNG seed is used so re-running yields the same file.
"""

import csv
import random
import re
from collections import defaultdict

SEED = 20260630
N_SENTENCES = 500
SRC = "Cleaned_Data.conll"
GAZ = "Used/gazetteer_segmented.csv"
OUT = "Generated_Building_Data.conll"

rng = random.Random(SEED)


# --------------------------------------------------------------------------- #
# 1. Read real spans from the existing CoNLL file
# --------------------------------------------------------------------------- #
def read_sentences(path):
    sents, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            parts = line.split()
            if len(parts) >= 4:
                cur.append((parts[0], parts[-1]))
    if cur:
        sents.append(cur)
    return sents


def extract_spans(sents):
    """Return {LABEL: [ [tok, tok, ...], ... ]} of real entity spans."""
    pools = defaultdict(list)
    for sent in sents:
        cur_label, cur_toks = None, []
        for tok, tag in sent:
            if tag.startswith("B-"):
                if cur_label:
                    pools[cur_label].append(cur_toks)
                cur_label, cur_toks = tag[2:], [tok]
            elif tag.startswith("I-") and cur_label == tag[2:]:
                cur_toks.append(tok)
            else:
                if cur_label:
                    pools[cur_label].append(cur_toks)
                cur_label, cur_toks = None, []
        if cur_label:
            pools[cur_label].append(cur_toks)
    # de-duplicate
    for lab in pools:
        seen, uniq = set(), []
        for s in pools[lab]:
            key = " ".join(s)
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        pools[lab] = uniq
    return pools


# --------------------------------------------------------------------------- #
# 2. Composed BUILDING vocabulary (to widen the open-vocab variety)
# --------------------------------------------------------------------------- #
BUILD_HEADS = [
    ["chung_cư"], ["cc"], ["toà"], ["toà", "nhà"], ["tòa", "nhà"], ["block"],
    ["lô"], ["căn_hộ"], ["villa"], ["biệt_thự"], ["cao_ốc"], ["tháp"],
    ["tttm"], ["trung_tâm", "thương_mại"], ["nhà_hàng"], ["khách_sạn"], ["ks"],
    ["cty"], ["công_ty"], ["cư_xá"],
]
# brand-like proper names (single tokens or short multi-token, segmented form)
BUILD_NAMES = [
    ["sunrise"], ["sunview"], ["sunshine"], ["riverside"], ["opal"], ["jasmine"],
    ["camellia"], ["safira"], ["citihome"], ["vista"], ["lexington"], ["masteri"],
    ["estella"], ["centana"], ["topaz"], ["ruby"], ["saigon", "pearl"],
    ["thảo", "điền", "pearl"], ["the", "manor"], ["the", "view"], ["golden", "king"],
    ["green", "park"], ["sky", "center"], ["bình_minh"], ["thắng_lợi"], ["hoà_bình"],
    ["phú_mỹ"], ["an_bình"], ["hoàng_anh"], ["gia_phước"], ["new", "city"],
    ["happy", "valley"], ["river", "gate"], ["sky", "garden"], ["era", "town"],
]
BUILD_SUFFIX = [
    [], [], [], ["tower"], ["plaza"], ["residence"], ["apartment"],
    ["building"], ["complex"], ["center"], ["towers"],
]
BLOCK_TAILS = [
    [], [], ["block", "a"], ["block", "b"], ["block", "c"], ["block", "d"],
    ["lô", "a"], ["lô", "b"], ["tháp", "t1"], ["tháp", "t2"], ["a1"], ["b2"], ["ct2"],
]


def make_building(real_pool):
    """Either reuse a real BUILDING span or compose a fresh one."""
    if rng.random() < 0.45 and real_pool:
        return list(rng.choice(real_pool))
    span = list(rng.choice(BUILD_HEADS))
    span += list(rng.choice(BUILD_NAMES))
    if rng.random() < 0.4:
        span += list(rng.choice(BUILD_SUFFIX))
    return span


# --------------------------------------------------------------------------- #
# 3. Admin spans from the gazetteer (coherent ward/district/city)
# --------------------------------------------------------------------------- #
def load_triples(path):
    triples = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            w, d, c = row.get("ward"), row.get("district"), row.get("city")
            if w and d and c:
                triples.append((w.strip(), d.strip(), c.strip()))
    return triples


CENTRAL_CITIES = {"hà_nội", "hồ_chí_minh", "hải_phòng", "đà_nẵng", "cần_thơ"}
CITY_ABBR = {"hà_nội": "hn", "hồ_chí_minh": "hcm"}

WARD_KW = [["phường"], ["p"], ["p."], ["xã"], ["x"], ["thị_trấn"], ["tt"], []]
DIST_KW = [["quận"], ["q"], ["q."], ["huyện"], ["h"], ["thị_xã"], ["tx"],
           ["thành_phố"], ["tp"], []]

# keywords that may already be embedded in a gazetteer name -> don't double them
EMBEDDED_KW = {"phường", "xã", "thị_trấn", "quận", "huyện", "thị_xã",
               "thành_phố", "tp", "tỉnh", "p", "q", "h", "x", "tt", "tx"}


def toks(name):
    """gazetteer name -> token list (spaces split, underscores kept)."""
    return name.split(" ")


def has_kw(name):
    return name.split(" ")[0] in EMBEDDED_KW


def ward_span(name):
    if has_kw(name):
        return toks(name)
    pre = rng.choice(WARD_KW)
    # numeric ward names are almost always 'phường'/'p'
    if name.isdigit():
        pre = rng.choice([["phường"], ["p"], ["p."]])
    return pre + toks(name)


def dist_span(name):
    if has_kw(name):
        return toks(name)
    pre = rng.choice(DIST_KW)
    return pre + toks(name)


def city_span(name):
    if has_kw(name):
        return toks(name)
    r = rng.random()
    if name in CITY_ABBR and r < 0.25:
        return [CITY_ABBR[name]]
    if r < 0.55:
        pre = ["thành_phố"] if name in CENTRAL_CITIES else ["tỉnh"]
        if rng.random() < 0.5:
            pre = ["tp"] if name in CENTRAL_CITIES else ["tỉnh"]
        return pre + toks(name)
    return toks(name)


# --------------------------------------------------------------------------- #
# 4. Sentence templates
# --------------------------------------------------------------------------- #
SEPARATORS = [";", ",", "-", "|"]


def build_sentence(pools, triples):
    w, d, c = rng.choice(triples)
    bld = make_building(pools["BUILDING"])

    # optional leading detail fields (all real spans)
    parts = []  # each part = list of (token, label) ; separator inserted between

    def span(toks_, label):
        return [(t, ("B-" if i == 0 else "I-") + label) for i, t in enumerate(toks_)]

    template = rng.choices(
        ["bld_first", "house_bld", "bld_street", "area_bld", "landmark", "full"],
        weights=[26, 20, 18, 12, 12, 12], k=1,
    )[0]

    if template == "bld_first":
        parts.append(span(bld, "BUILDING"))
        if rng.random() < 0.4 and pools["STREET"]:
            parts.append(span(rng.choice(pools["STREET"]), "STREET"))
        parts.append(span(ward_span(w), "WARD"))
        parts.append(span(dist_span(d), "DISTRICT"))
        parts.append(span(city_span(c), "CITY"))

    elif template == "house_bld":
        parts.append(span(rng.choice(pools["HOUSE_NO"]), "HOUSE_NO"))
        parts.append(span(bld, "BUILDING"))
        parts.append(span(ward_span(w), "WARD"))
        parts.append(span(dist_span(d), "DISTRICT"))
        parts.append(span(city_span(c), "CITY"))

    elif template == "bld_street":
        parts.append(span(bld, "BUILDING"))
        if rng.random() < 0.6 and pools["HOUSE_NO"]:
            parts.append(span(rng.choice(pools["HOUSE_NO"]), "HOUSE_NO"))
        parts.append(span(rng.choice(pools["STREET"]), "STREET"))
        parts.append(span(ward_span(w), "WARD"))
        parts.append(span(dist_span(d), "DISTRICT"))
        parts.append(span(city_span(c), "CITY"))

    elif template == "area_bld":
        parts.append(span(rng.choice(pools["AREA"]), "AREA"))
        parts.append(span(bld, "BUILDING"))
        parts.append(span(ward_span(w), "WARD"))
        parts.append(span(dist_span(d), "DISTRICT"))
        parts.append(span(city_span(c), "CITY"))

    elif template == "full":
        if rng.random() < 0.5 and pools["ALLEY"]:
            parts.append(span(rng.choice(pools["ALLEY"]), "ALLEY"))
        parts.append(span(rng.choice(pools["HOUSE_NO"]), "HOUSE_NO"))
        parts.append(span(bld, "BUILDING"))
        parts.append(span(rng.choice(pools["STREET"]), "STREET"))
        parts.append(span(ward_span(w), "WARD"))
        parts.append(span(dist_span(d), "DISTRICT"))
        parts.append(span(city_span(c), "CITY"))

    else:  # landmark: building in trailing parentheses
        parts.append(span(rng.choice(pools["HOUSE_NO"]), "HOUSE_NO"))
        parts.append(span(rng.choice(pools["STREET"]), "STREET"))
        parts.append(span(ward_span(w), "WARD"))
        parts.append(span(dist_span(d), "DISTRICT"))
        parts.append(span(city_span(c), "CITY"))

    # ---- render parts with separators ----
    sep = rng.choice(SEPARATORS)
    out = []
    if template == "house_bld" and rng.random() < 0.5:
        # join HOUSE_NO + BUILDING with a space (no separator) like "căn_hộ 3020 toà s"
        glue_first = True
    else:
        glue_first = False

    for i, part in enumerate(parts):
        if i > 0:
            if glue_first and i == 1:
                pass  # no separator token, just concatenate
            else:
                out.append((sep, "O"))
        out.extend(part)

    if template == "landmark":
        out.append(("(", "O"))
        if rng.random() < 0.5:
            out.append((rng.choice(["gần", "đối_diện", "cạnh", "kế"]), "O"))
        out.extend([(t, ("B-" if i == 0 else "I-") + "BUILDING")
                    for i, t in enumerate(bld)])
        out.append((")", "O"))

    return out


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main():
    sents = read_sentences(SRC)
    pools = extract_spans(sents)
    triples = load_triples(GAZ)
    # drop the 4 over-represented provinces a bit for balance
    common = {"hà_nội", "hồ_chí_minh", "thanh_hoá", "nghệ_an"}
    rare = [t for t in triples if t[2] not in common]

    lines, seen = [], set()
    attempts = 0
    while len(lines) < N_SENTENCES and attempts < N_SENTENCES * 20:
        attempts += 1
        tri = triples if rng.random() < 0.5 else rare
        sent = build_sentence(pools, tri)
        key = " ".join(t for t, _ in sent)
        if key in seen:
            continue
        seen.add(key)
        lines.append(sent)

    with open(OUT, "w", encoding="utf-8") as f:
        for sent in lines:
            for tok, tag in sent:
                f.write(f"{tok} -X- _ {tag}\n")
            f.write("\n")

    n_bld = sum(1 for s in lines for _, t in s if t == "B-BUILDING")
    print(f"Wrote {len(lines)} sentences to {OUT}")
    print(f"BUILDING entities: {n_bld}")
    print(f"Real BUILDING spans available: {len(pools['BUILDING'])}")


if __name__ == "__main__":
    main()
