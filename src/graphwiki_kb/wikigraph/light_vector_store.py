"""Minimal vector store for the LightRAG-style backend.

This module intentionally avoids FAISS / Milvus / Qdrant — a local
NumPy-backed cosine top-K is sufficient for the capstone corpus
(see project recommendation §21). It also degrades gracefully when
NumPy is unavailable by falling back to pure-Python dot products.

The store is **dense or sparse agnostic**: vectors are stored as
plain ``list[float]``. The embedding provider chooses the dimension.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

try:  # pragma: no cover - numpy is in the dep tree
    import numpy as _np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - defensive
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


@dataclass(frozen=True)
class VectorHit:
    """A scored vector hit returned by :meth:`LightVectorStore.search`."""

    id: str
    score: float
    rank: int


class LightVectorStore:
    """A tiny in-memory cosine vector index keyed by stable ids.

    Vectors must all share the same dimension. Empty queries return an
    empty result. The store never raises for legitimate-but-poor matches
    (e.g. when a query has no overlap with any stored item) — callers
    can decide what to do with low-score hits.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def ids(self) -> tuple[str, ...]:
        """Return the stored ids in insertion order."""
        return tuple(self._ids)

    @property
    def dimension(self) -> int:
        """Return the vector dimension (or 0 when empty)."""
        if not self._vectors:
            return 0
        return len(self._vectors[0])

    def add(self, item_id: str, vector: Sequence[float]) -> None:
        """Append a single vector to the store."""
        vec = [float(v) for v in vector]
        if self._vectors and len(vec) != len(self._vectors[0]):
            raise ValueError(
                "Vector dimension mismatch: store has "
                f"{len(self._vectors[0])}, got {len(vec)}"
            )
        self._ids.append(item_id)
        self._vectors.append(vec)

    def add_many(self, items: Sequence[tuple[str, Sequence[float]]]) -> None:
        """Append multiple ``(id, vector)`` pairs."""
        for item_id, vector in items:
            self.add(item_id, vector)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 10,
    ) -> list[VectorHit]:
        """Return the ``top_k`` highest cosine-similarity hits."""
        if not self._vectors or top_k <= 0:
            return []
        q = [float(v) for v in query_vector]
        if not q:
            return []
        if len(q) != len(self._vectors[0]):
            raise ValueError(
                "Query dimension mismatch: store has "
                f"{len(self._vectors[0])}, got {len(q)}"
            )

        if _HAS_NUMPY:
            assert _np is not None
            matrix = _np.asarray(self._vectors, dtype=_np.float32)
            query = _np.asarray(q, dtype=_np.float32)
            mat_norms = _np.linalg.norm(matrix, axis=1) + 1e-12
            query_norm = float(_np.linalg.norm(query)) + 1e-12
            scores = (matrix @ query) / (mat_norms * query_norm)
            order = _np.argsort(-scores)[: max(top_k, 0)]
            return [
                VectorHit(
                    id=self._ids[int(idx)],
                    score=float(scores[int(idx)]),
                    rank=rank,
                )
                for rank, idx in enumerate(order)
            ]

        # Pure-Python fallback.
        query_norm = math.sqrt(sum(x * x for x in q)) + 1e-12
        scored: list[tuple[float, str]] = []
        for vec, item_id in zip(self._vectors, self._ids, strict=True):
            dot = sum(a * b for a, b in zip(vec, q, strict=True))
            norm = math.sqrt(sum(x * x for x in vec)) + 1e-12
            scored.append((dot / (norm * query_norm), item_id))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:top_k]
        return [
            VectorHit(id=item_id, score=float(score), rank=rank)
            for rank, (score, item_id) in enumerate(scored)
        ]
