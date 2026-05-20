"""Lexical retrieval helpers for the WikiGraphRAG backend.

By default this module uses :mod:`bm25s` when available because it gives a
modern, persistent BM25 implementation. If :mod:`bm25s` is missing (for
example when the ``wikigraph`` extra is not installed), the helpers fall
back to a tiny in-memory BM25 implementation that still produces sensible
ranked scores for testing and lightweight runs.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

try:
    import bm25s

    _BM25S_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    bm25s = None
    _BM25S_AVAILABLE = False

from graphwiki_kb.services.stopwords import STOPWORDS

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Return lowercased alphanumeric tokens, dropping stopwords."""
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token and token not in STOPWORDS
    ]


@dataclass
class LexicalDocument:
    """A single document indexed by :class:`LexicalIndex`."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LexicalHit:
    """A retrieval hit returned from :meth:`LexicalIndex.search`."""

    doc_id: str
    score: float
    metadata: dict[str, Any]


class LexicalIndex:
    """A small BM25-style retriever with optional :mod:`bm25s` backend.

    The class behaves identically regardless of which backend is in use.

    Args:
        prefer_simple: When ``True``, the pure-python BM25 implementation is
            used even if :mod:`bm25s` is installed. This is what the
            ``wikigraph.lexical_backend: simple`` config setting selects.
    """

    def __init__(self, *, prefer_simple: bool = False) -> None:
        self._documents: list[LexicalDocument] = []
        self._tokenized: list[list[str]] = []
        self._bm25s_retriever: Any | None = None
        self._fitted = False
        self._doc_frequencies: Counter[str] = Counter()
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._k1: float = 1.5
        self._b: float = 0.75
        self._prefer_simple: bool = prefer_simple

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def add(self, doc: LexicalDocument) -> None:
        """Stage a document for indexing. Call :meth:`fit` after all adds."""
        if self._fitted:
            raise RuntimeError("LexicalIndex is already fitted; rebuild instead.")
        self._documents.append(doc)
        tokens = tokenize(doc.text)
        self._tokenized.append(tokens)
        self._doc_lengths.append(len(tokens))
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            self._doc_frequencies[token] += 1

    def fit(self) -> None:
        """Build the underlying index. Safe to call once."""
        if self._fitted:
            return
        if not self._documents:
            self._fitted = True
            return
        total_length = sum(self._doc_lengths)
        self._avg_doc_length = total_length / len(self._documents)
        if _BM25S_AVAILABLE and not self._prefer_simple:
            retriever = bm25s.BM25()
            retriever.index(self._tokenized, show_progress=False)
            self._bm25s_retriever = retriever
        self._fitted = True

    # ------------------------------------------------------------------ #
    # Retrieval                                                          #
    # ------------------------------------------------------------------ #

    def search(self, query: str, *, limit: int = 10) -> list[LexicalHit]:
        """Return up to ``limit`` ranked hits for ``query``."""
        if not self._fitted:
            self.fit()
        if not self._documents:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        if self._bm25s_retriever is not None:
            return self._search_bm25s(query_tokens, limit=limit)
        return self._search_pure_python(query_tokens, limit=limit)

    def _search_bm25s(self, tokens: list[str], *, limit: int) -> list[LexicalHit]:
        results, scores = self._bm25s_retriever.retrieve(  # type: ignore[union-attr]
            [tokens],
            k=min(limit, len(self._documents)),
            show_progress=False,
        )
        hits: list[LexicalHit] = []
        for idx, score in zip(results[0], scores[0], strict=False):
            doc = self._documents[int(idx)]
            hits.append(
                LexicalHit(
                    doc_id=doc.doc_id,
                    score=float(score),
                    metadata=doc.metadata,
                )
            )
        return hits

    def _search_pure_python(self, tokens: list[str], *, limit: int) -> list[LexicalHit]:
        n_docs = len(self._documents)
        scores: list[tuple[float, int]] = []
        for doc_index, doc_tokens in enumerate(self._tokenized):
            if not doc_tokens:
                continue
            doc_length = self._doc_lengths[doc_index]
            term_counts = Counter(doc_tokens)
            score = 0.0
            for token in tokens:
                tf = term_counts.get(token, 0)
                if tf == 0:
                    continue
                df = self._doc_frequencies.get(token, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                norm = (
                    1 - self._b + self._b * (doc_length / (self._avg_doc_length or 1))
                )
                score += idf * (tf * (self._k1 + 1)) / (tf + self._k1 * norm)
            if score > 0:
                scores.append((score, doc_index))
        scores.sort(key=lambda item: item[0], reverse=True)
        hits: list[LexicalHit] = []
        for score, doc_index in scores[:limit]:
            doc = self._documents[doc_index]
            hits.append(
                LexicalHit(doc_id=doc.doc_id, score=float(score), metadata=doc.metadata)
            )
        return hits

    @property
    def backend(self) -> str:
        """Return ``"bm25s"`` when the optional dependency is loaded.

        When :mod:`bm25s` is available but ``prefer_simple=True``, the
        pure-python fallback is used and this property returns
        ``"simple"`` so the choice is observable.
        """
        if self._bm25s_retriever is not None:
            return "bm25s"
        return "simple" if self._prefer_simple else "pure-python-bm25"

    @property
    def doc_count(self) -> int:
        """Number of staged documents."""
        return len(self._documents)
