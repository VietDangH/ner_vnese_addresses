# -*- coding: utf-8 -*-
"""Parse an uploaded .txt / .csv file into a list of input texts.

Kept separate from the routes so the parsing rules are easy to test and extend.
Raises ``ValueError`` with a human-readable message on bad input; the route turns
that into an HTTP 400.
"""

import csv
import io
import pathlib
from typing import List

from . import settings


def parse_upload(filename: str, content: bytes) -> List[str]:
    """Return the list of non-empty texts contained in an uploaded file.

    * ``.txt`` -> one text per line.
    * ``.csv`` -> the ``address`` column if present, else the first column.
    """
    ext = pathlib.Path(filename or "").suffix.lower()
    if ext not in settings.ALLOWED_UPLOAD_EXT:
        raise ValueError(
            f"unsupported file type '{ext or filename}'; "
            f"allowed: {sorted(settings.ALLOWED_UPLOAD_EXT)}")

    try:
        text = content.decode("utf-8-sig")    # tolerate a BOM
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8 text: {exc}") from exc

    if ext == ".txt":
        texts = [ln.strip() for ln in text.splitlines()]
    else:                                      # .csv
        texts = _parse_csv(text)

    texts = [t for t in texts if t]
    if not texts:
        raise ValueError("no non-empty lines found in the uploaded file")
    return texts


def _parse_csv(text: str) -> List[str]:
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if r]
    if not rows:
        return []
    header = [h.strip().lower() for h in rows[0]]
    if "address" in header:
        col = header.index("address")
        body = rows[1:]                        # skip the header row
    else:
        col = 0                                # no header: use the first column
        body = rows
    return [row[col].strip() for row in body if len(row) > col]
