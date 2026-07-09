"""Vietnamese word segmentation used to prepare input for the PhoBERT model.

Primary backend is VnCoreNLP (``wseg``) via ``py_vncorenlp`` — the same
segmenter PhoBERT was trained against — with graceful fallbacks to
``underthesea`` / ``pyvi`` / whitespace when it (or Java) is unavailable.

Colab setup (run once, before importing this module)::

    !apt-get install -y openjdk-17-jdk-headless -qq   # Java for VnCoreNLP
    !pip install py_vncorenlp

The model is downloaded on first use into ``/content/vncorenlp`` (override
with the ``VNCORENLP_DIR`` environment variable).
"""

import os
import re

_SEGMENTER = None
_BACKEND = None

# Where VnCoreNLP keeps its model files. Default matches Colab; override via env.
SAVE_DIR = os.environ.get("VNCORENLP_DIR", "/content/vncorenlp")

# Module-level singleton: the JVM behind VnCoreNLP can only be started once per
# process, so we keep the instance around and reuse it across calls/cell re-runs.
rdrsegmenter = None


def _load_vncorenlp():
    """Initialise the VnCoreNLP word segmenter (returns a callable or raises)."""
    global rdrsegmenter

    import py_vncorenlp  # type: ignore

    save_dir = os.path.abspath(SAVE_DIR)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Only download if the segmenter model isn't present yet (avoids re-download
    # and survives a half-created directory).
    seg_model = os.path.join(save_dir, "models", "wordsegmenter",
                             "wordsegmenter.rdr")
    if not os.path.exists(seg_model):
        py_vncorenlp.download_model(save_dir=save_dir)

    # VnCoreNLP chdir()s into save_dir during init (it loads the model via a
    # relative path); restore the cwd afterwards so the rest of the program
    # keeps working with relative paths. Guard against re-init in notebooks.
    if rdrsegmenter is None:
        cwd = os.getcwd()
        try:
            rdrsegmenter = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"], save_dir=save_dir
            )
        finally:
            os.chdir(cwd)

    def seg(text):
        # word_segment returns a list of segmented sentences; rejoin into one
        # space-separated string (compound words already joined with "_").
        sentences = rdrsegmenter.word_segment(text)
        return " ".join(sentences)

    return seg


def _load_segmenter():
    global _SEGMENTER, _BACKEND
    if _BACKEND is not None:
        return
    # 1) VnCoreNLP — preferred, matches PhoBERT training-time segmentation.
    try:
        _SEGMENTER, _BACKEND = _load_vncorenlp(), "vncorenlp"
        return
    except Exception:
        pass
    # 2) underthesea
    try:
        from underthesea import word_tokenize  # type: ignore

        def seg(text):
            return word_tokenize(text, format="text")

        _SEGMENTER, _BACKEND = seg, "underthesea"
        return
    except Exception:
        pass
    # 3) pyvi
    try:
        from pyvi import ViTokenizer  # type: ignore

        _SEGMENTER, _BACKEND = (lambda t: ViTokenizer.tokenize(t)), "pyvi"
        return
    except Exception:
        pass
    # 4) whitespace (no-op)
    _SEGMENTER, _BACKEND = (lambda t: t), "whitespace"


def get_backend() -> str:
    _load_segmenter()
    return _BACKEND


def segment(text: str, lower: bool = True) -> list:
    """Return a list of word-segmented tokens (compound words joined with ``_``).

    Punctuation that the CoNLL data keeps as its own token (``,`` ``/`` ``.``)
    is split out so it aligns with how the model was trained.
    """
    _load_segmenter()
    text = text.strip()
    if not text:
        return []
    # Segment on the original (cased) text — capitalisation helps segmentation —
    # then lowercase the resulting tokens if requested.
    if "_" in text and " " in text and _BACKEND == "whitespace":
        # already segmented? trust it.
        segmented = text
    else:
        segmented = _SEGMENTER(text)
    # pad standalone punctuation so it becomes its own token
    segmented = re.sub(r"([,/();])", r" \1 ", segmented)
    tokens = segmented.split()
    if lower:
        tokens = [t.lower() for t in tokens]
    return tokens
