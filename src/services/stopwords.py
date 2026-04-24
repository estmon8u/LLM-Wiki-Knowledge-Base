"""Shared English stopwords loaded from a bundled word list.

The data file (``src/data/english_stopwords.txt``) ships with the repo so no
network download or NLTK corpus is required at runtime.
"""

from __future__ import annotations

from pathlib import Path

_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "english_stopwords.txt"

STOPWORDS: frozenset[str] = frozenset(
    word for word in _DATA_FILE.read_text(encoding="utf-8").splitlines() if word.strip()
)
