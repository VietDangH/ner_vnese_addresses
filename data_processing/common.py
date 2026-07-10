# -*- coding: utf-8 -*-
"""Shared cleaning / EDA / segmentation utilities for the four address datasets.

Per-dataset pipeline (see the report's *Dataset and Features* section)::

    load raw
      -> dataset-specific structural cleaning        (per script)
      -> shared normalization                        (normalize_series)
         (NFC, emoji removal, lower-case, whitespace collapse, exact de-dup)
      -> VnCoreNLP word segmentation                 (segment_series)
      -> address-specific regex number fix-ups       (post_process_address_numbers)
      -> save  data<N>_final.csv

This stops *before* the label-time "augmentation as features" step
(keyword abbreviation `quận`->`q.`, diacritic drop, ...), which is applied later
to the train/test splits by ``vn_ner.augment`` / ``bami_ner.augment``.

Every function here is dataset-agnostic; the four ``preprocess_*`` / ``crawl_*``
scripts import from this module so the cleaning + EDA logic lives in one place.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd

# --------------------------------------------------------------------------- #
# stdout: make Vietnamese printable on a Windows (cp1252) console
# --------------------------------------------------------------------------- #
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------------------------------------------------------------------------- #
# 1. Shared text normalization
# --------------------------------------------------------------------------- #
# Optional: use the `emoji` package if available, else a regex fallback.
try:
    import emoji as _emoji  # type: ignore

    def _strip_emoji(text: str) -> str:
        return _emoji.replace_emoji(text, replace="")
except Exception:  # pragma: no cover - fallback path
    _EMOJI_RE = re.compile(
        "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
        "\U00002190-\U000021FF\U00002B00-\U00002BFF\U0000FE00-\U0000FE0F]",
        flags=re.UNICODE,
    )

    def _strip_emoji(text: str) -> str:
        return _EMOJI_RE.sub("", text)


def normalize_text(text) -> str:
    """Unicode NFC -> strip emoji -> lower-case -> collapse whitespace.

    The single normalization pass applied to *every* source (report §Dataset).
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _strip_emoji(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_series(s: pd.Series) -> pd.Series:
    return s.apply(normalize_text)


def deduplicate(s: pd.Series, name: str = "") -> pd.Series:
    """Exact de-duplication, dropping empties, with a short report."""
    before = len(s)
    s = s[s.astype(str).str.strip() != ""]
    dups = int(s.duplicated().sum())
    s = s.drop_duplicates().reset_index(drop=True)
    print(f"  [dedup{f' {name}' if name else ''}] {before} -> {len(s)} "
          f"rows ({dups} exact duplicates removed)")
    return s


# --------------------------------------------------------------------------- #
# 2. Address-specific regex number fix-ups
#    (ported verbatim from Data_Segmentation.ipynb :: post_process_address_numbers)
# --------------------------------------------------------------------------- #
def post_process_address_numbers(text: str) -> str:
    """Normalize house-number spacing so tokens align with the label scheme.

    Examples: ``894H``->``894 H``, ``K894``->``K 894``, ``7/4``->``7 / 4``,
    ``5 7``->``57``, ``457 A``->``457A``.
    """
    if not isinstance(text, str) or not text:
        return ""
    # Split a K/H/N/P/Q letter glued to a digit, on either side.
    text = re.sub(r"(\d)([KkHhNnPpQq])", r"\1 \2", text)
    text = re.sub(r"([KkHhNnPpQq])(\d)", r"\1 \2", text)
    # Space around a slash between two digits: "7/4" -> "7 / 4".
    text = re.sub(r"(\d)\s*[\/\\]\s*(\d)", r"\1 / \2", text)
    # Re-merge digits split by whitespace: "5 7" -> "57".
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    # Re-join a lone A-G letter suffix to its number: "457 A" -> "457A".
    text = re.sub(r"(?<=\d)\s+([A-Ga-g])\b", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# 3. Word segmentation (VnCoreNLP RDRsegmenter, with graceful fallbacks)
# --------------------------------------------------------------------------- #
# Self-contained so this package does not depend on `vn_ner` (and hence torch).
# Backend priority mirrors vn_ner.segment: VnCoreNLP -> underthesea -> pyvi ->
# whitespace no-op. Override the model dir with the VNCORENLP_DIR env var.
_SEGMENTER = None
_BACKEND = None
_SAVE_DIR = os.environ.get(
    "VNCORENLP_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vncorenlp"),
)
_rdrsegmenter = None


def _load_segmenter():
    global _SEGMENTER, _BACKEND, _rdrsegmenter
    if _BACKEND is not None:
        return
    # 1) VnCoreNLP - matches PhoBERT's training-time segmentation.
    try:
        import py_vncorenlp  # type: ignore

        save_dir = os.path.abspath(_SAVE_DIR)
        os.makedirs(save_dir, exist_ok=True)
        seg_model = os.path.join(save_dir, "models", "wordsegmenter",
                                 "wordsegmenter.rdr")
        if not os.path.exists(seg_model):
            py_vncorenlp.download_model(save_dir=save_dir)
        if _rdrsegmenter is None:
            cwd = os.getcwd()
            try:
                _rdrsegmenter = py_vncorenlp.VnCoreNLP(
                    annotators=["wseg"], save_dir=save_dir)
            finally:
                os.chdir(cwd)
        _SEGMENTER = lambda t: " ".join(_rdrsegmenter.word_segment(t))
        _BACKEND = "vncorenlp"
        return
    except Exception:
        pass
    # 2) underthesea
    try:
        from underthesea import word_tokenize  # type: ignore
        _SEGMENTER = lambda t: word_tokenize(t, format="text")
        _BACKEND = "underthesea"
        return
    except Exception:
        pass
    # 3) pyvi
    try:
        from pyvi import ViTokenizer  # type: ignore
        _SEGMENTER = lambda t: ViTokenizer.tokenize(t)
        _BACKEND = "pyvi"
        return
    except Exception:
        pass
    # 4) whitespace no-op
    _SEGMENTER, _BACKEND = (lambda t: t), "whitespace"


def segment_backend() -> str:
    _load_segmenter()
    return _BACKEND


def segment_text(text: str) -> str:
    """Word-segment one address into a space-joined string (compounds use ``_``)."""
    _load_segmenter()
    text = (text or "").strip()
    if not text:
        return ""
    seg = _SEGMENTER(text)
    # Pad standalone punctuation so it becomes its own token, matching the
    # CoNLL tokenisation the models are trained on.
    seg = re.sub(r"([,/();])", r" \1 ", seg)
    return re.sub(r"\s+", " ", seg).strip()


def segment_series(s: pd.Series, enabled: bool = True) -> pd.Series:
    if not enabled:
        return s
    print(f"  [segment] backend = {segment_backend()}")
    return s.apply(segment_text)


# --------------------------------------------------------------------------- #
# 4. Gazetteer (ward / district / city) + rare-province-biased sampler
#    Shared by the House Rent geographic rebalancing.
# --------------------------------------------------------------------------- #
def _norm_key(s: str) -> str:
    """Loose key for comparing province names (lower, strip prefix/punct/space)."""
    s = (s or "").lower()
    for pre in ("thành phố ", "tp. ", "tp ", "tỉnh ", "t. "):
        if s.startswith(pre):
            s = s[len(pre):]
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", "", s)


# Big provinces to hold out of the rebalancing draw (mirror
# augment_replace_components.py), matched on the loose key above.
DEFAULT_EXCLUDE = {"hanoi", "hànội", "hồchíminh", "hochiminh", "thanhhóa",
                   "thanhhoa", "nghệan", "nghean"}


def load_gazetteer(path: str, exclude: set | None = None):
    """Return a list of {'ward','district','city'} rows, dropping excluded cities."""
    exclude = DEFAULT_EXCLUDE if exclude is None else exclude
    g = pd.read_csv(path).dropna(subset=["ward", "district", "city"])
    rows = []
    for ward, district, city in zip(g["ward"], g["district"], g["city"]):
        if _norm_key(city) in exclude:
            continue
        rows.append({"ward": str(ward).strip(),
                     "district": str(district).strip(),
                     "city": str(city).strip()})
    if not rows:
        raise SystemExit("[!] Gazetteer empty after applying --exclude.")
    return rows


def build_weights(rows, bias: float = 1.0):
    """weight(row) = freq(city) ** -(1+bias)  ->  P(province) ∝ freq ** -bias.

    bias=0 -> uniform over provinces; larger -> rarer provinces strongly preferred.
    """
    freq = Counter(r["city"] for r in rows)
    exp = -(1.0 + bias)
    return [freq[r["city"]] ** exp for r in rows]


# --------------------------------------------------------------------------- #
# 5. EDA (shared) - token-length stats + city distribution + short/long peek
# --------------------------------------------------------------------------- #
try:
    import matplotlib
    matplotlib.use("Agg")  # headless: save figures, never block
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:  # pragma: no cover
    _HAS_PLT = False


def _token_counts(s: pd.Series) -> pd.Series:
    return s.astype(str).apply(lambda x: len(x.split()))


def run_eda(s: pd.Series, name: str, out_dir: str | None = None,
            short_max: int = 5, long_min: int = 20) -> dict:
    """Print token-length stats + city distribution and (if possible) save a
    histogram PNG. Returns a small summary dict."""
    tc = _token_counts(s)
    stats = {
        "rows": int(len(s)),
        "tokens_mean": round(float(tc.mean()), 2) if len(s) else 0.0,
        "tokens_min": int(tc.min()) if len(s) else 0,
        "tokens_max": int(tc.max()) if len(s) else 0,
        "short_le_%d" % short_max: int((tc <= short_max).sum()),
        "long_gt_%d" % long_min: int((tc > long_min).sum()),
    }
    print(f"  [eda {name}] rows={stats['rows']} "
          f"tokens(mean/min/max)={stats['tokens_mean']}/{stats['tokens_min']}/"
          f"{stats['tokens_max']} "
          f"short(<= {short_max})={stats['short_le_%d' % short_max]} "
          f"long(> {long_min})={stats['long_gt_%d' % long_min]}")

    # Last-two-token "city" tail distribution (cheap geographic signal).
    tail = s.astype(str).apply(lambda x: " ".join(x.split()[-2:]) if x.split() else "")
    top = tail.value_counts().head(10)
    print(f"  [eda {name}] top-10 trailing city tails:")
    for city, n in top.items():
        print(f"      {n:5d}  {city}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        top.to_csv(os.path.join(out_dir, f"{name}_city_tail_top.csv"),
                   header=["count"])
        if _HAS_PLT:
            plt.figure(figsize=(9, 4.5))
            plt.hist(tc, bins=30, color="#4C78A8", edgecolor="white")
            plt.title(f"{name}: token-count distribution")
            plt.xlabel("tokens per address"); plt.ylabel("frequency")
            plt.tight_layout()
            fig_path = os.path.join(out_dir, f"{name}_token_hist.png")
            plt.savefig(fig_path, dpi=110); plt.close()
            print(f"  [eda {name}] histogram -> {fig_path}")
    return stats


# --------------------------------------------------------------------------- #
# 6. IO helpers + default project paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)                       # _organized/
DATA_DIR = os.path.join(REPO_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw_data")            # the four raw sources
PROC_DIR = os.path.join(DATA_DIR, "processed")          # *_final.csv outputs
EDA_DIR = os.path.join(PROC_DIR, "eda")                 # histograms + tail CSVs
GAZETTEER = os.path.join(DATA_DIR, "gazetteer.csv")



def read_column(csv_path: str, column: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise SystemExit(f"[!] column {column!r} not in {csv_path} "
                         f"(have: {list(df.columns)})")
    return df[column]


def save_series(s: pd.Series, out_path: str, header: str = "address"):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    s.to_frame(name=header).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [save] {len(s)} rows -> {out_path}")


def finalize(s: pd.Series, out_path: str, segment: bool = True,
             eda: bool = True, name: str = "data", eda_dir: str | None = None):
    """Shared tail of every pipeline: normalize numbers -> (segment) -> save,
    with an EDA pass on the cleaned (pre-segmentation) text."""
    if eda:
        run_eda(s, name=name, out_dir=eda_dir)
    s = segment_series(s, enabled=segment)
    s = s.apply(post_process_address_numbers)
    s = deduplicate(s, name=f"{name} final")
    save_series(s, out_path)
    return s
