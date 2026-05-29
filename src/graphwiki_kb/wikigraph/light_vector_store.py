"""Local vector store for LightRAG entity/relation/chunk embeddings.

For a 20-30 document capstone corpus a JSON-backed store with cosine top-k is
plenty (and stays fully inspectable). Vectors are L2-normalized on construction
so retrieval is a single dot product. NumPy is used when available for speed,
with a deterministic pure-python fallback so the module imports on a base
install.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphwiki_kb.services.project_service import atomic_write_text


def _get_numpy() -> Any | None:
    try:
        import numpy as np

        return np
    except Exception:  # pragma: no cover - exercised via monkeypatch
        return None


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def cosine_top_k(
    query_vector: list[float],
    vectors: list[list[float]],
    k: int,
    *,
    numpy_module: Any | None = None,
) -> list[tuple[int, float]]:
    """Return up to ``k`` ``(row_index, cosine_score)`` pairs, best first.

    ``vectors`` are assumed already L2-normalized (as stored by
    :class:`LightVectorStore`). The query vector is normalized here.
    """
    if not vectors or k <= 0:
        return []
    query = _normalize(query_vector)
    np = numpy_module if numpy_module is not None else _get_numpy()
    if np is not None:
        matrix = np.asarray(vectors, dtype=float)
        query_arr = np.asarray(query, dtype=float)
        scores = matrix @ query_arr
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order]
    scored = [
        (index, sum(a * b for a, b in zip(row, query, strict=False)))
        for index, row in enumerate(vectors)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [(index, float(score)) for index, score in scored[:k]]


@dataclass
class LightVectorStore:
    """An in-memory, JSON-persistable store of normalized embedding vectors."""

    model: str
    dimension: int
    ids: list[str]
    vectors: list[list[float]]

    @classmethod
    def from_embeddings(
        cls,
        ids: list[str],
        raw_vectors: list[list[float]],
        *,
        model: str,
        dimension: int = 0,
    ) -> LightVectorStore:
        """Build a store, validating shapes and L2-normalizing each vector."""
        if len(ids) != len(raw_vectors):
            raise ValueError(
                f"ids/vectors length mismatch ({len(ids)} != {len(raw_vectors)})"
            )
        resolved_dim = dimension or (len(raw_vectors[0]) if raw_vectors else 0)
        normalized: list[list[float]] = []
        for vector in raw_vectors:
            if resolved_dim and len(vector) != resolved_dim:
                raise ValueError(
                    "embedding dimension mismatch: expected "
                    f"{resolved_dim}, got {len(vector)}"
                )
            normalized.append(_normalize([float(value) for value in vector]))
        return cls(
            model=model,
            dimension=resolved_dim,
            ids=list(ids),
            vectors=normalized,
        )

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query_vector: list[float], k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(id, cosine_score)`` pairs, best first."""
        hits = cosine_top_k(query_vector, self.vectors, k)
        return [(self.ids[index], score) for index, score in hits]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the store."""
        return {
            "model": self.model,
            "dimension": self.dimension,
            "ids": self.ids,
            "vectors": self.vectors,
        }

    def save(self, path: Path) -> None:
        """Persist the store as JSON at ``path`` (atomic)."""
        atomic_write_text(path, json.dumps(self.to_payload(), indent=2))

    @classmethod
    def load(cls, path: Path) -> LightVectorStore | None:
        """Load a store from JSON, returning ``None`` when missing/corrupt."""
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return cls(
            model=str(payload.get("model", "")),
            dimension=int(payload.get("dimension", 0)),
            ids=[str(item) for item in payload.get("ids", [])],
            vectors=[
                [float(value) for value in row] for row in payload.get("vectors", [])
            ],
        )
