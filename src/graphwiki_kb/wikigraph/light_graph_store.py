"""Persistent JSON store for the LightRAG-style WikiGraphRAG index.

Everything is plain JSON so the index stays fully inspectable:

```
graph/wikigraph/lightrag/
  index.json                 # metadata + counts
  chunks.json
  entities.json
  relations.json
  entity_vectors.json        # optional (omitted in BM25 fallback)
  relation_vectors.json      # optional
  chunk_vectors.json         # optional
  source_contributions.json
  build_manifest.json
  extraction_cache/<key>.json
```
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphwiki_kb.services.project_service import atomic_write_text
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphIndex,
    RelationProfile,
)
from graphwiki_kb.wikigraph.light_vector_store import LightVectorStore


@dataclass
class LightGraphStorePaths:
    """Filesystem layout for persisted LightRAG artifacts."""

    root: Path

    @property
    def index_file(self) -> Path:
        return self.root / "index.json"

    @property
    def chunks_file(self) -> Path:
        return self.root / "chunks.json"

    @property
    def entities_file(self) -> Path:
        return self.root / "entities.json"

    @property
    def relations_file(self) -> Path:
        return self.root / "relations.json"

    @property
    def entity_vectors_file(self) -> Path:
        return self.root / "entity_vectors.json"

    @property
    def relation_vectors_file(self) -> Path:
        return self.root / "relation_vectors.json"

    @property
    def chunk_vectors_file(self) -> Path:
        return self.root / "chunk_vectors.json"

    @property
    def source_contributions_file(self) -> Path:
        return self.root / "source_contributions.json"

    @property
    def build_manifest_file(self) -> Path:
        return self.root / "build_manifest.json"

    @property
    def extraction_cache_dir(self) -> Path:
        return self.root / "extraction_cache"


class LightGraphStore:
    """Loads/persists the LightRAG index and its vector stores."""

    def __init__(self, paths: LightGraphStorePaths) -> None:
        self.paths = paths

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def exists(self) -> bool:
        """Return ``True`` when a persisted index is present."""
        return self.paths.index_file.exists() and self.paths.entities_file.exists()

    def save(
        self,
        index: LightGraphIndex,
        *,
        entity_vectors: LightVectorStore | None = None,
        relation_vectors: LightVectorStore | None = None,
        chunk_vectors: LightVectorStore | None = None,
        source_contributions: dict[str, Any] | None = None,
        build_manifest: dict[str, Any] | None = None,
    ) -> list[str]:
        """Persist the index and return the list of written file paths."""
        self.paths.root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "built_at": index.built_at,
            "schema_version": index.schema_version,
            "tier": index.tier,
            "embedding_model": index.embedding_model,
            "embedding_dimension": index.embedding_dimension,
            "provider_identity": index.provider_identity,
            "embedding_identity": index.embedding_identity,
            "extraction_prompt_hash": index.extraction_prompt_hash,
            "source_hashes": index.source_hashes,
            "chunk_count": index.chunk_count,
            "entity_count": index.entity_count,
            "relation_count": index.relation_count,
        }
        written: list[str] = []

        def _write(path: Path, payload: Any) -> None:
            atomic_write_text(path, json.dumps(payload, indent=2, default=str))
            written.append(str(path))

        _write(self.paths.index_file, metadata)
        _write(self.paths.chunks_file, [chunk.model_dump() for chunk in index.chunks])
        _write(
            self.paths.entities_file,
            [entity.model_dump() for entity in index.entities],
        )
        _write(
            self.paths.relations_file,
            [relation.model_dump() for relation in index.relations],
        )
        if entity_vectors is not None:
            entity_vectors.save(self.paths.entity_vectors_file)
            written.append(str(self.paths.entity_vectors_file))
        if relation_vectors is not None:
            relation_vectors.save(self.paths.relation_vectors_file)
            written.append(str(self.paths.relation_vectors_file))
        if chunk_vectors is not None:
            chunk_vectors.save(self.paths.chunk_vectors_file)
            written.append(str(self.paths.chunk_vectors_file))
        _write(self.paths.source_contributions_file, source_contributions or {})
        _write(self.paths.build_manifest_file, build_manifest or {})
        return written

    def load(self) -> LightGraphIndex | None:
        """Load the persisted index, returning ``None`` if missing/corrupt."""
        if not self.exists():
            return None
        try:
            metadata = json.loads(self.paths.index_file.read_text(encoding="utf-8"))
            chunks_raw = json.loads(self.paths.chunks_file.read_text(encoding="utf-8"))
            entities_raw = json.loads(
                self.paths.entities_file.read_text(encoding="utf-8")
            )
            relations_raw = json.loads(
                self.paths.relations_file.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return LightGraphIndex(
                built_at=str(metadata.get("built_at", "")),
                chunks=[LightChunk.model_validate(item) for item in chunks_raw],
                entities=[EntityProfile.model_validate(item) for item in entities_raw],
                relations=[
                    RelationProfile.model_validate(item) for item in relations_raw
                ],
                source_hashes=dict(metadata.get("source_hashes", {})),
                extraction_prompt_hash=str(metadata.get("extraction_prompt_hash", "")),
                embedding_model=str(metadata.get("embedding_model", "")),
                embedding_dimension=int(metadata.get("embedding_dimension", 0)),
                provider_identity=metadata.get("provider_identity"),
                embedding_identity=metadata.get("embedding_identity"),
                tier=str(metadata.get("tier", "fallback")),
                schema_version=int(metadata.get("schema_version", 1)),
            )
        except ValueError:
            return None

    def load_build_manifest(self) -> dict[str, Any] | None:
        """Load the build manifest dict, or ``None`` when missing/corrupt."""
        path = self.paths.build_manifest_file
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def load_source_contributions(self) -> dict[str, Any]:
        """Load the source-contribution map (empty dict when missing)."""
        path = self.paths.source_contributions_file
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def load_entity_vectors(self) -> LightVectorStore | None:
        """Load the entity vector store (``None`` when absent)."""
        return LightVectorStore.load(self.paths.entity_vectors_file)

    def load_relation_vectors(self) -> LightVectorStore | None:
        """Load the relation vector store (``None`` when absent)."""
        return LightVectorStore.load(self.paths.relation_vectors_file)

    def load_chunk_vectors(self) -> LightVectorStore | None:
        """Load the chunk vector store (``None`` when absent)."""
        return LightVectorStore.load(self.paths.chunk_vectors_file)
