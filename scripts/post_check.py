import sys
import os
import re
import argparse
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional


# python post_check.py --show 603

DEFAULT_FILE = "Cleaned_Data.conll"
OUT_DIR = "review_lists"

ALL_CHECKS = [
    ("malformed_tag", "ERROR"),
    ("I_at_start", "ERROR"),
    ("O_to_I", "ERROR"),
    ("label_jump", "ERROR"),
    ("punct_as_entity", "WARN"),
    ("keyword_label_mismatch", "WARN"),
    ("city_not_in_gazetteer", "WARN"),
    ("duplicate_admin", "WARN"),
    ("admin_order", "WARN"),
    ("missing_city", "WARN"),
    ("text_label_conflict", "WARN"),
]


# ============================================================
# 0. PARSER
# ============================================================

def parse_conll_file(path: str) -> List[List[Tuple[str, str]]]:
    sentences = []
    current = []
    skipped = []
    with open(path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if line.strip() == "":
                if current:
                    sentences.append(current)
                    current = []
                continue
            cols = line.split()
            if len(cols) < 2:
                skipped.append((line_no, line))
                continue
            current.append((cols[0], cols[-1]))
    if current:
        sentences.append(current)
    return sentences


# ============================================================
# helpers
# ============================================================

def get_label(tag: str) -> str:
    if tag == "O":
        return "O"
    if "-" in tag:
        return tag.split("-", 1)[1]
    return tag


def get_prefix(tag: str) -> str:
    if tag == "O":
        return "O"
    if tag.startswith("B-"):
        return "B"
    if tag.startswith("I-"):
        return "I"
    return "?"


_TONE_FOLD = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}


def norm_text(s: str) -> str:
    s = s.lower()
    for a, b in _TONE_FOLD.items():
        s = s.replace(a, b)
    s = s.replace("_", " ").replace("-", " ")
    return " ".join(s.split())


def render_sentence(sentence: List[Tuple[str, str]], hi: Optional[int] = None) -> str:
    parts = []
    for i, (tok, tag) in enumerate(sentence):
        s = f"{tok}/{tag}"
        if hi is not None and i == hi:
            s = f"**{s}**"
        parts.append(s)
    return " ".join(parts)


def make_issue(sent_idx, token_idx, severity, check, detail) -> Dict:
    return {"sent_idx": sent_idx, "token_idx": token_idx,
            "severity": severity, "check": check, "detail": detail}


def extract_entities(sentence: List[Tuple[str, str]], sent_idx: int = None) -> List[Dict]:
    entities = []
    toks, label, start = [], None, None

    def flush():
        nonlocal toks, label, start
        if toks and label:
            entities.append({
                "label": label, "text": " ".join(toks), "tokens": list(toks),
                "start": start, "end": start + len(toks) - 1, "sent_idx": sent_idx,
            })
        toks, label, start = [], None, None

    for i, (tok, tag) in enumerate(sentence):
        prefix, lab = get_prefix(tag), get_label(tag)
        if prefix == "B":
            flush(); toks, label, start = [tok], lab, i
        elif prefix == "I":
            if label == lab:
                toks.append(tok)
            else:                    
                flush(); toks, label, start = [tok], lab, i
        else:                    
            flush()
    flush()
    return entities


# ============================================================
# 1. BIO STRUCTURAL CHECK  (ERROR)
# ============================================================

def check_bio(sentence, sent_idx) -> List[Dict]:
    issues = []
    tags = [t for _, t in sentence]
    for i, tag in enumerate(tags):
        prefix, label = get_prefix(tag), get_label(tag)
        if prefix == "?":
            issues.append(make_issue(sent_idx, i, "ERROR", "malformed_tag",
                f"Tag '{tag}' violates format B-xxx/I-xxx/O"))
            continue
        if prefix == "I":
            if i == 0:
                issues.append(make_issue(sent_idx, i, "ERROR", "I_at_start",
                    f"'{tag}' at the beginning without B"))
                continue
            prev = tags[i - 1]
            pprefix, plabel = get_prefix(prev), get_label(prev)
            if pprefix in ("O", "?"):
                issues.append(make_issue(sent_idx, i, "ERROR", "O_to_I",
                    f"'{prev}' -> '{tag}': I after O, miss B-"))
            elif plabel != label:
                issues.append(make_issue(sent_idx, i, "ERROR", "label_jump",
                    f"'{prev}' -> '{tag}': different classes"))
    return issues


# ============================================================
# 2. KEYWORD vs LABEL  (WARN)
# ============================================================
HEAD_KEYWORDS = {
    "phường": "WARD", "xã": "WARD", "thị_trấn": "WARD",
    "quận": "DISTRICT", "huyện": "DISTRICT", "thị_xã": "DISTRICT",
    "tỉnh": "CITY",
    "đường": "STREET",
    "ngõ": "ALLEY", "hẻm": "ALLEY", "kiệt": "ALLEY", "ngách": "ALLEY",
    "thôn": "AREA", "xóm": "AREA", "ấp": "AREA", "khu_phố": "AREA",
    "chung_cư": "BUILDING", "tòa_nhà": "BUILDING",
}


def check_keyword_label(entities) -> List[Dict]:
    issues = []
    for e in entities:
        head = e["tokens"][0].lower()
        expected = HEAD_KEYWORDS.get(head)
        if expected and expected != e["label"]:
            issues.append(make_issue(e["sent_idx"], e["start"], "WARN",
                "keyword_label_mismatch",
                f"Entity '{e['text']}' was [{e['label']}] but start with "
                f"'{head}' (expect: [{expected}])"))
    return issues


# ============================================================
# 3. PUNCTUATION TAGGED AS ENTITY  (WARN)
# ============================================================
SEPARATORS = set(",;:()[]{}\"'`")


def check_punctuation(sentence, sent_idx) -> List[Dict]:
    issues = []
    for i, (tok, tag) in enumerate(sentence):
        if tag != "O" and tok and all(ch in SEPARATORS for ch in tok):
            issues.append(make_issue(sent_idx, i, "WARN", "punct_as_entity",
                f"Token '{tok}' was '{tag}' (expect: O)"))
    return issues

# ============================================================
# 5,6,7. Administrative-level check  (WARN)
# ============================================================

UNIQUE_ADMIN = ("WARD", "DISTRICT", "CITY")   
ADMIN_ORDER = {"WARD": 0, "DISTRICT": 1, "CITY": 2} 

def check_admin_per_sentence(entities, sent_idx) -> List[Dict]:
    issues = []
    by_label = defaultdict(list)
    for e in entities:
        by_label[e["label"]].append(e)

    # 5. Administrative-level duplication
    for lab in UNIQUE_ADMIN:
        if len(by_label.get(lab, [])) > 1:
            vals = " | ".join(x["text"] for x in by_label[lab])
            issues.append(make_issue(sent_idx, None, "WARN", "duplicate_admin",
                f"Having {len(by_label[lab])} entities [{lab}] in an address: {vals}"))

    # 6. Order: WARD -> DISTRICT -> CITY
    present = [(ADMIN_ORDER[lab], by_label[lab][0]["start"], lab)
               for lab in ADMIN_ORDER if by_label.get(lab)]
    present.sort(key=lambda x: x[1])             
    rank_seq = [r for r, _, _ in present]
    if rank_seq != sorted(rank_seq):
        seq = " -> ".join(lab for _, _, lab in present)
        issues.append(make_issue(sent_idx, None, "WARN", "admin_order",
            f"Order: {seq} "
            f"(Expect: (WARD -> DISTRICT -> CITY)"))

    # 7. missing CITY
    if not by_label.get("CITY"):
        issues.append(make_issue(sent_idx, None, "WARN", "missing_city",
            "No CITY"))
    return issues


# ============================================================
# 8. TEXT-LABEL CONFLICT (WARN)
# ============================================================
_NUMERIC_KEY = re.compile(r"^[\d\s/.\-]+$")


def check_text_label_conflicts(all_entities) -> Tuple[List[Dict], Dict]:
    text2labels = defaultdict(Counter)
    text2sents = defaultdict(set)
    for e in all_entities:
        key = norm_text(e["text"])
        if len(key) < 2 or _NUMERIC_KEY.match(key):
            continue
        text2labels[key][e["label"]] += 1
        text2sents[key].add(e["sent_idx"])

    conflicts = {}
    issues = []
    for text, counter in text2labels.items():
        if len(counter) > 1:
            conflicts[text] = counter
            dist = ", ".join(f"{lab}={n}" for lab, n in counter.most_common())
            issues.append(make_issue(min(text2sents[text]), None, "WARN",
                "text_label_conflict",
                f"'{text}' was labelled by multiple classes: {dist}"))
    return issues, conflicts

def repair_bio(sentence) -> Tuple[List[Tuple[str, str]], int]:
    fixed = []
    n_changed = 0
    prev_label, prev_prefix = None, "O"
    for i, (tok, tag) in enumerate(sentence):
        new_tag = tag
        m = re.search(r"(B|I)-([A-Za-z_]+)$", tag)
        if get_prefix(tag) == "?" and m:
            new_tag = f"{m.group(1)}-{m.group(2)}"
        prefix, label = get_prefix(new_tag), get_label(new_tag)
        if prefix == "I":
            if prev_prefix in ("O", "?") or prev_label != label:
                new_tag = f"B-{label}"
                prefix = "B"
        if new_tag != tag:
            n_changed += 1
        fixed.append((tok, new_tag))
        prev_label, prev_prefix = label, prefix
    return fixed, n_changed

def write_fixed_file(sentences, in_path) -> str:
    base, ext = os.path.splitext(in_path)
    out_path = f"{base}_bioFixed{ext or '.txt'}"
    total_changed = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            fixed, n = repair_bio(sent)
            total_changed += n
            for tok, tag in fixed:
                f.write(f"{tok} -X- _ {tag}\n")
            f.write("\n")
    print(f"[fix-bio] Fixed {total_changed} tag -> write: {out_path}")
    return out_path


# ============================================================
# EXPORT
# ============================================================

def export(sentences, issues, conflicts, stats, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # all_issues.tsv
    tsv = os.path.join(out_dir, "all_issues.tsv")
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("sent_idx\tseverity\tcheck\ttoken_idx\tdetail\n")
        for it in sorted(issues, key=lambda x: (x["severity"] != "ERROR",
                                                x["check"], x["sent_idx"])):
            f.write(f"{it['sent_idx']}\t{it['severity']}\t{it['check']}\t"
                    f"{'' if it['token_idx'] is None else it['token_idx']}\t"
                    f"{it['detail']}\n")
    print(f"  -> {tsv} ({len(issues)} issue)")

    # errors_bio.txt
    bio_path = os.path.join(out_dir, "errors_bio.txt")
    bio_issues = [it for it in issues if it["severity"] == "ERROR"]
    with open(bio_path, "w", encoding="utf-8") as f:
        if not bio_issues:
            f.write("No structural BIO errors.\n")
        for it in bio_issues:
            f.write(f"[Sentence #{it['sent_idx']}] {it['check']}: {it['detail']}\n")
            f.write(f"  {render_sentence(sentences[it['sent_idx']], it['token_idx'])}\n\n")
    print(f"  -> {bio_path} ({len(bio_issues)} errors)")

    # warn_<type>.txt
    warns = defaultdict(list)
    for it in issues:
        if it["severity"] == "WARN" and it["check"] != "text_label_conflict":
            warns[it["check"]].append(it)
    for check, items in warns.items():
        p = os.path.join(out_dir, f"warn_{check}.txt")
        with open(p, "w", encoding="utf-8") as f:
            for it in items:
                f.write(f"[Sentence #{it['sent_idx']}] {it['detail']}\n")
                if it["token_idx"] is not None:
                    f.write(f"  {render_sentence(sentences[it['sent_idx']], it['token_idx'])}\n")
                f.write("\n")
        print(f"  -> {p} ({len(items)})")

    # text_label_conflicts.txt
    if conflicts:
        p = os.path.join(out_dir, "text_label_conflicts.txt")
        with open(p, "w", encoding="utf-8") as f:
            ranked = sorted(conflicts.items(),
                            key=lambda kv: -sum(kv[1].values()))
            for text, counter in ranked:
                dist = ", ".join(f"{lab}={n}" for lab, n in counter.most_common())
                f.write(f"'{text}'  ->  {dist}\n")
        print(f"  -> {p} ({len(conflicts)} conflicting strings)")

    # summary.txt
    p = os.path.join(out_dir, "summary.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(stats)
    print(f"  -> {p}")


def build_summary(sentences, all_entities, issues) -> str:
    lines = []
    n_tok = sum(len(s) for s in sentences)
    lines.append(f"Sentences       : {len(sentences)}")
    lines.append(f"Tokens          : {n_tok}")
    lines.append(f"Entities        : {len(all_entities)}")
    lines.append("")
    lines.append("Entity distribution by class:")
    cnt = Counter(e["label"] for e in all_entities)
    for lab, n in cnt.most_common():
        lines.append(f"  {lab:<10} {n}")
    lines.append("")
    lines.append("Issues by severity:")
    for sev, n in Counter(it["severity"] for it in issues).most_common():
        lines.append(f"  {sev:<6} {n}")
    lines.append("")
    lines.append("Issues by check:")
    for check, n in Counter(it["check"] for it in issues).most_common():
        lines.append(f"  {check:<24} {n}")
    return "\n".join(lines)

def show_sentences(sentences, indices, all_entities, conflicts):
    """Print details of sentence(s) by index — matching the #N shown in the report (base-0):
    token/tag list, extracted entities, and every issue detected in that sentence
    (including the cross-dataset 'same text, multiple labels' error)."""
    n = len(sentences)
    by_sent = defaultdict(list)
    for e in all_entities:
        by_sent[e["sent_idx"]].append(e)

    for idx in indices:
        print("=" * 72)
        if idx < 0 or idx >= n:
            print(f"[!] Sentence #{idx} does not exist (file has {n} sentences, valid 0..{n - 1})\n")
            continue
        sent = sentences[idx]
        ents = by_sent.get(idx, [])
        print(f"Sentence #{idx}   ({len(sent)} tokens, {len(ents)} entities)")
        print("-" * 72)
        for i, (tok, tag) in enumerate(sent):
            mark = "   <-- unknown tag" if get_prefix(tag) == "?" else ""
            print(f"  {i:>3}  {tok:<24} {tag}{mark}")
        print("-" * 72)
        print("  One line :", render_sentence(sent))
        print("  Entities :")
        for e in ents:
            print(f"     [{e['label']:<9}] {e['text']}")

        # collect every issue for this sentence only
        issues = check_bio(sent, idx) + check_punctuation(sent, idx)
        issues += check_keyword_label(ents) + check_city_gazetteer(ents)
        issues += check_admin_per_sentence(ents, idx)
        for e in ents:                       # cross-dataset error, looked up per entity
            key = norm_text(e["text"])
            if key in conflicts:
                dist = ", ".join(f"{lab}={c}" for lab, c in conflicts[key].most_common())
                issues.append(make_issue(idx, e["start"], "WARN",
                    "text_label_conflict",
                    f"'{key}' is labelled across the whole dataset as: {dist}"))

        print("  Issues   :" + (" (none)" if not issues else ""))
        for it in issues:
            loc = "" if it["token_idx"] is None else f" @token#{it['token_idx']}"
            print(f"     [{it['severity']}] {it['check']}{loc}: {it['detail']}")
        print()


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Advanced post-check for the address NER dataset")
    ap.add_argument("path", nargs="?", default=DEFAULT_FILE,
                    help="path to the CoNLL file (default: %(default)s)")
    ap.add_argument("--fix-bio", action="store_true",
                    help="auto-fix structural BIO errors (safe) -> write file *_bioFixed.txt")
    ap.add_argument("--show", nargs="+", type=int, metavar="N",
                    help="only print details of sentence(s) #N then exit, e.g.: --show 603")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(f"[*] Reading dataset: {args.path}")
    sentences = parse_conll_file(args.path)
    print(f"[*] {len(sentences)} sentences, {sum(len(s) for s in sentences)} tokens\n")

    all_entities = []
    for idx, sent in enumerate(sentences):
        all_entities.extend(extract_entities(sent, sent_idx=idx))

    if args.show:
        _, conflicts = check_text_label_conflicts(all_entities)
        show_sentences(sentences, args.show, all_entities, conflicts)
        return

    issues = []
    for idx, sent in enumerate(sentences):
        ents = [e for e in all_entities if e["sent_idx"] == idx]
        issues += check_bio(sent, idx)
        issues += check_punctuation(sent, idx)
        issues += check_keyword_label(ents)
        issues += check_admin_per_sentence(ents, idx)
    conflict_issues, conflicts = check_text_label_conflicts(all_entities)
    issues += conflict_issues

    print("[*] Check results:")
    by_check = Counter(it["check"] for it in issues)
    by_sev = Counter(it["severity"] for it in issues)
    for sev in ("ERROR", "WARN"):
        print(f"    {sev}: {by_sev.get(sev, 0)}")
    for check, sev in ALL_CHECKS:                
        n = by_check.get(check, 0)
        status = "OK (0)" if n == 0 else str(n)
        print(f"      - [{sev:<5}] {check:<24} {status}")

    stats = build_summary(sentences, all_entities, issues)
    print("\n[*] Exporting review_lists_v2/ ...")
    export(sentences, issues, conflicts, stats)

    if args.fix_bio:
        print()
        write_fixed_file(sentences, args.path)

    print("\n[✓] Done.")

main()
