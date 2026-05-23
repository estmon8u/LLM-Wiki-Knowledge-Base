"""Embedding provider abstractions for the LightRAG-style backend.

This module defines a tiny :class:`EmbeddingProvider` protocol plus two
concrete implementations:

* :class:`BM25SparseEmbeddingProvider` — a deterministic, provider-free,
  BM25-inspired sparse embedder used as the local fallback. It produces
  vectors in a stable feature space derived from the corpus vocabulary
  and is sufficient for tests and small projects.
* :class:`HashingEmbeddingProvider` — a feature-hashed bag-of-words
  embedder. Smaller, faster, and useful when the corpus is large enough
  that BM25 vocabulary tracking is wasteful.

Real provider-backed embedders (OpenAI, Gemini, etc.) can implement the
protocol later without touching the rest of the LightRAG pipeline.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [tok.casefold() for tok in _TOKEN.findall(text)]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal interface every LightRAG embedding provider must implement."""

    model_name: str

    @property
    def dimension(self) -> int:
        """Vector dimension produced by :meth:`embed_texts`."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text."""
        ...


class HashingEmbeddingProvider:
    """Feature-hashed bag-of-words embedder.

    Useful as a no-vocabulary fallback. Output vectors are L2-normalized
    so cosine similarity behaves predictably.
    """

    def __init__(self, *, dimension: int = 512) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.model_name = "hashing-bow"
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        """Vector dimension produced by :meth:`embed_texts`."""
        return self._dimension

    def _hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` into ``self.dimension``-vectors."""
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dimension
            for tok in _tokenize(text):
                vec[self._hash(tok)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class BM25SparseEmbeddingProvider:
    """BM25-inspired sparse embedder backed by an explicit vocabulary.

    Build a provider with :meth:`fit` over a corpus, then call
    :meth:`embed_texts` for both index documents and queries. The
    resulting vectors are dense over the discovered vocabulary; cosine
    similarity over them approximates BM25 ranking well enough for the
    tens-of-documents capstone scale.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.model_name = "bm25-sparse"
        self.k1 = float(k1)
        self.b = float(b)
        self._vocab: dict[str, int] = {}
        self._idf: list[float] = []
        self._avg_dl: float = 1.0

    @property
    def dimension(self) -> int:
        """Return the size of the learned vocabulary."""
        return len(self._vocab)

    def fit(self, corpus: Iterable[str]) -> None:
        """Learn the vocabulary and IDF weights from ``corpus``.

        Always call this before :meth:`embed_texts`. Re-fitting resets
        the learned state.
        """
        docs = [_tokenize(t) for t in corpus]
        doc_freq: Counter[str] = Counter()
        total_len = 0
        for tokens in docs:
            total_len += len(tokens)
            for tok in set(tokens):
                doc_freq[tok] += 1
        vocab_tokens = sorted(doc_freq)
        if not vocab_tokens:
            # Corpus is empty or contains no tokens. Keep a sentinel
            # vocab slot so subsequent embed_texts calls don't raise.
            self._vocab = {"__empty__": 0}
            self._idf = [0.0]
            self._avg_dl = 1.0
            return
        self._vocab = {tok: idx for idx, tok in enumerate(vocab_tokens)}
        n = max(1, len(docs))
        self._idf = [
            math.log((n - doc_freq[tok] + 0.5) / (doc_freq[tok] + 0.5) + 1.0)
            for tok in vocab_tokens
        ]
        self._avg_dl = total_len / n if n else 1.0
        if self._avg_dl <= 0:
            self._avg_dl = 1.0

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` using the learned vocabulary and IDF."""
        if not self._vocab:
            raise RuntimeError("BM25SparseEmbeddingProvider.fit() must be called first")
        vectors: list[list[float]] = []
        dim = len(self._vocab)
        for text in texts:
            tokens = _tokenize(text)
            counts = Counter(tok for tok in tokens if tok in self._vocab)
            dl = max(1, len(tokens))
            length_norm = 1 - self.b + self.b * (dl / self._avg_dl)
            vec = [0.0] * dim
            for tok, tf in counts.items():
                idx = self._vocab[tok]
                idf = self._idf[idx]
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * length_norm
                vec[idx] = idf * (num / den) if den else 0.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


def default_embedding_provider(
    *,
    corpus: Sequence[str] | None = None,
) -> EmbeddingProvider:
    """Return a sensible default provider for offline / fallback use.

    Builds a :class:`BM25SparseEmbeddingProvider` when ``corpus`` is
    supplied; otherwise returns a :class:`HashingEmbeddingProvider`.
    """
    if corpus is not None:
        provider = BM25SparseEmbeddingProvider()
        provider.fit(corpus)
        return provider
    return HashingEmbeddingProvider()
