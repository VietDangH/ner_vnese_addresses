# -*- coding: utf-8 -*-
"""Configuration + model-path resolution for the serving app.

Everything here is overridable via environment variables so the same code runs
locally and on a server without edits:

    NER_MODEL_PATH   path to a trained model dir OR a .zip of one
                     (default: <repo>/PhoBERT_cnn_crf_seed_43)
    NER_EXTRACT_DIR  where a .zip model is unpacked (default: <repo>/.model_cache)
    NER_MAX_BATCH    max sentences accepted by /batch_predict (default: 1000)

Note: the default model has a CRF head, so the serving env needs the CRF library
(``pip install pytorch-crf``); it is already listed in ``requirements.txt``.
"""

import os
import pathlib
import sys
import zipfile

# Repo root = parent of this `serving/` folder. Put it on sys.path so the trained
# model package (`vn_ner`) is importable no matter what directory uvicorn is
# launched from (e.g. `cd serving && uvicorn app:app`).
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = os.environ.get(
    "NER_MODEL_PATH", str(REPO_ROOT / "PhoBERT_cnn_crf_seed_43"))
EXTRACT_DIR = os.environ.get("NER_EXTRACT_DIR", str(REPO_ROOT / ".model_cache"))
MAX_BATCH = int(os.environ.get("NER_MAX_BATCH", "1000"))

# Accepted upload types for /batch_predict file mode.
ALLOWED_UPLOAD_EXT = {".txt", ".csv"}


def _find_model_dir(root: pathlib.Path) -> str:
    """Return the directory under ``root`` that holds ``config.json``.

    Accepts either ``root`` itself or a single nested folder (as produced by
    zipping a model directory)."""
    root = pathlib.Path(root)
    if (root / "config.json").exists():
        return str(root)
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "config.json").exists():
            return str(child)
    raise FileNotFoundError(f"no model (config.json) found under {root}")


def resolve_model_dir(path: str = None) -> str:
    """Resolve a model dir from a directory or a ``.zip`` (extracted once).

    Raises ``FileNotFoundError`` if the path is missing or has no model inside.
    """
    path = path or MODEL_PATH
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"model path does not exist: {p}")

    if p.is_dir():
        return _find_model_dir(p)

    if p.suffix.lower() == ".zip":
        out = pathlib.Path(EXTRACT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        try:                                  # already extracted? reuse it.
            return _find_model_dir(out)
        except FileNotFoundError:
            pass
        with zipfile.ZipFile(p) as z:
            z.extractall(out)
        return _find_model_dir(out)

    raise FileNotFoundError(
        f"unsupported model path '{p}' (expected a directory or a .zip)")
