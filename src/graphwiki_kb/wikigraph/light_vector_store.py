"""Local vector storage and cosine search for LightRAG profiles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from graphwiki_kb.services.project_service import atomic_write_text
from graphwiki_kb.wikigraph.lexical_index import LexicalDocument, LexicalIndex


def cosine_top_k(
    query_vector: list[float],
    matrix: list[list[float]],
    k: int,
) -> list[tuple[int, float]]:
    """Return top-``k`` (index, score) pairs by cosine similarity."""
    if not matrix or k <= 0:
        return []
    scores: list[tuple[int, float]] = []
    q_norm = _l2_norm(query_vector)
    if q_norm == 0.0:
        return []
    for idx, row in enumerate(matrix):
        r_norm = _l2_norm(row)
        if r_norm == 0.0:
            continue
        dot = sum(a * b for a, b in zip(query_vector, row, strict=False))
        scores.append((idx, dot / (q_norm * r_norm)))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:k]


def _l2_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


@dataclass
class VectorStoreMeta:
    """Metadata for a persisted vector matrix."""

    ids: list[str]
    dimension: int
    model: str
    backend: str


class LightVectorStore:
    """Persist vectors as JSON lists (MVP) with optional BM25 fallback."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def matrix_file(self) -> Path:
        return self.root / "vectors.json"

    @property
    def meta_file(self) -> Path:
        return self.root / "meta.json"

    def save(
        self,
        *,
        ids: list[str],
        vectors: list[list[float]],
        model: str,
        backend: str = "embedding",
    ) -> None:
        dimension = len(vectors[0]) if vectors else 0
        meta = VectorStoreMeta(
            ids=ids,
            dimension=dimension,
            model=model,
            backend=backend,
        )
        atomic_write_text(self.meta_file, json.dumps(meta.__dict__, indent=2))
        atomic_write_text(self.matrix_file, json.dumps(vectors, indent=0))

    def load(self) -> tuple[VectorStoreMeta, list[list[float]]]:
        meta_payload = json.loads(self.meta_file.read_text(encoding="utf-8"))
        meta = VectorStoreMeta(
            ids=list(meta_payload.get("ids", [])),
            dimension=int(meta_payload.get("dimension", 0)),
            model=str(meta_payload.get("model", "")),
            backend=str(meta_payload.get("backend", "")),
        )
        vectors = json.loads(self.matrix_file.read_text(encoding="utf-8"))
        return meta, vectors

    def exists(self) -> bool:
        return self.meta_file.exists() and self.matrix_file.exists()


@dataclass
class HybridRetriever:
    """Vector search with BM25 fallback when vectors are absent."""

    ids: list[str]
    texts: list[str]
    vectors: list[list[float]] | None = None
    lexical: LexicalIndex | None = None
    backend_label: str = "lightrag"

    def __post_init__(self) -> None:
        if self.lexical is None and self.texts:
            self.lexical = LexicalIndex(prefer_simple=True)
            for doc_id, text in zip(self.ids, self.texts, strict=False):
                self.lexical.add(LexicalDocument(doc_id=doc_id, text=text))
            self.lexical.fit()

    def search(
        self,
        query: str,
        *,
        query_vector: list[float] | None = None,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        if query_vector is not None and self.vectors:
            hits = cosine_top_k(query_vector, self.vectors, k)
            return [
                (self.ids[idx], score) for idx, score in hits if idx < len(self.ids)
            ]
        if self.lexical is not None:
            lexical_hits = self.lexical.search(query, limit=k)
            return [(hit.doc_id, hit.score) for hit in lexical_hits]
        return []
