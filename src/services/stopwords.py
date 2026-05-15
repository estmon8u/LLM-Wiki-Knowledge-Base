"""Shared English stopwords loaded from a bundled word list.

The data file (``src/data/english_stopwords.txt``) ships with the repo so no
network download or NLTK corpus is required at runtime.
"""

from __future__ import annotations

from pathlib import Path

_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "english_stopwords.txt"


def _load_stopwords(path: Path = _DATA_FILE) -> frozenset[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Stopword file is missing or unreadable: {path}") from exc

    words = frozenset(word.strip().casefold() for word in lines if word.strip())
    if not words:
        raise RuntimeError(f"Stopword file is empty: {path}")
    return words


STOPWORDS: frozenset[str] = _load_stopwords()
