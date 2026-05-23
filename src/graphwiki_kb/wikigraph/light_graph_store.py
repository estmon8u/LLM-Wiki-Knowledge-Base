"""Persistence for the LightRAG-style WikiGraphRAG index.

Files written under ``graph/wikigraph/lightrag/``:

* ``index.json``          — top-level pointer including manifest.
* ``chunks.json``         — list of :class:`LightChunk`.
* ``entities.json``       — list of :class:`EntityProfile`.
* ``relations.json``      — list of :class:`RelationProfile`.
* ``contributions.json``  — list of :class:`SourceContribution`.
* ``entity_vectors.json`` — entity embedding vectors (id, vector).
* ``relation_vectors.json`` — relation embedding vectors (id, vector).
* ``chunk_vectors.json``    — optional chunk embedding vectors.
* ``build_manifest.json`` — freshness/digest material.

The store deliberately keeps the format JSON-only for the capstone
scale; switching to ``.npy`` later is a localized change. See
project recommendation §21.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from graphwiki_kb.services.project_service import atomic_write_text
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphBuildManifest,
    LightGraphIndex,
    RelationProfile,
    SourceContribution,
)


@dataclass(frozen=True)
class LightGraphStorePaths:
    """Filesystem locations used by :class:`LightGraphStore`."""

    root: Path

    @property
    def index_file(self) -> Path:
        """Top-level pointer + manifest mirror."""
        return self.root / "index.json"

    @property
    def chunks_file(self) -> Path:
        """Persisted chunk list."""
        return self.root / "chunks.json"

    @property
    def entities_file(self) -> Path:
        """Persisted entity profile list."""
        return self.root / "entities.json"

    @property
    def relations_file(self) -> Path:
        """Persisted relation profile list."""
        return self.root / "relations.json"

    @property
    def contributions_file(self) -> Path:
        """Per-source contribution rows."""
        return self.root / "contributions.json"

    @property
    def manifest_file(self) -> Path:
        """Freshness manifest."""
        return self.root / "build_manifest.json"

    @property
    def entity_vectors_file(self) -> Path:
        """Persisted entity embedding vectors."""
        return self.root / "entity_vectors.json"

    @property
    def relation_vectors_file(self) -> Path:
        """Persisted relation embedding vectors."""
        return self.root / "relation_vectors.json"

    @property
    def chunk_vectors_file(self) -> Path:
        """Persisted chunk embedding vectors (optional)."""
        return self.root / "chunk_vectors.json"

    @property
    def extraction_cache_dir(self) -> Path:
        """Per-chunk extraction result cache directory."""
        return self.root / "extraction_cache"


@dataclass
class LightGraphStore:
    """File-backed store for a :class:`LightGraphIndex`."""

    paths: LightGraphStorePaths

    def exists(self) -> bool:
        """Return True when an index has been written before."""
        return self.paths.index_file.exists() and self.paths.manifest_file.exists()

    def load(self) -> LightGraphIndex | None:
        """Load the persisted index, or return ``None`` when absent/invalid."""
        if not self.exists():
            return None
        try:
            chunks = self._read_models(self.paths.chunks_file, LightChunk)
            entities = self._read_models(self.paths.entities_file, EntityProfile)
            relations = self._read_models(self.paths.relations_file, RelationProfile)
            contributions = self._read_models(
                self.paths.contributions_file, SourceContribution
            )
            manifest_payload = json.loads(
                self.paths.manifest_file.read_text(encoding="utf-8")
            )
            manifest = LightGraphBuildManifest.model_validate(manifest_payload)
        except (OSError, ValueError):
            return None
        return LightGraphIndex(
            built_at=manifest.built_at,
            chunks=chunks,
            entities=entities,
            relations=relations,
            contributions=contributions,
            manifest=manifest,
        )

    def save(
        self,
        index: LightGraphIndex,
        *,
        entity_vectors: list[tuple[str, list[float]]] | None = None,
        relation_vectors: list[tuple[str, list[float]]] | None = None,
        chunk_vectors: list[tuple[str, list[float]]] | None = None,
    ) -> list[str]:
        """Persist ``index`` + optional vector arrays to disk.

        Returns the list of relative file paths that were (re)written.
        """
        self.paths.root.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        atomic_write_text(
            self.paths.chunks_file,
            json.dumps(
                [c.model_dump() for c in index.chunks], indent=2, sort_keys=True
            ),
        )
        written.append(self._rel(self.paths.chunks_file))
        atomic_write_text(
            self.paths.entities_file,
            json.dumps(
                [e.model_dump() for e in index.entities], indent=2, sort_keys=True
            ),
        )
        written.append(self._rel(self.paths.entities_file))
        atomic_write_text(
            self.paths.relations_file,
            json.dumps(
                [r.model_dump() for r in index.relations], indent=2, sort_keys=True
            ),
        )
        written.append(self._rel(self.paths.relations_file))
        atomic_write_text(
            self.paths.contributions_file,
            json.dumps(
                [c.model_dump() for c in index.contributions], indent=2, sort_keys=True
            ),
        )
        written.append(self._rel(self.paths.contributions_file))
        atomic_write_text(
            self.paths.manifest_file,
            json.dumps(index.manifest.model_dump(), indent=2, sort_keys=True),
        )
        written.append(self._rel(self.paths.manifest_file))
        atomic_write_text(
            self.paths.index_file,
            json.dumps(
                {
                    "built_at": index.built_at,
                    "chunk_count": index.chunk_count,
                    "entity_count": index.entity_count,
                    "relation_count": index.relation_count,
                    "source_count": len(index.contributions),
                    "manifest": index.manifest.model_dump(),
                },
                indent=2,
                sort_keys=True,
            ),
        )
        written.append(self._rel(self.paths.index_file))

        if entity_vectors is not None:
            atomic_write_text(
                self.paths.entity_vectors_file,
                json.dumps(
                    {
                        "vectors": [
                            {"id": vid, "vector": vec} for vid, vec in entity_vectors
                        ]
                    },
                    indent=2,
                ),
            )
            written.append(self._rel(self.paths.entity_vectors_file))
        if relation_vectors is not None:
            atomic_write_text(
                self.paths.relation_vectors_file,
                json.dumps(
                    {
                        "vectors": [
                            {"id": vid, "vector": vec} for vid, vec in relation_vectors
                        ]
                    },
                    indent=2,
                ),
            )
            written.append(self._rel(self.paths.relation_vectors_file))
        if chunk_vectors is not None:
            atomic_write_text(
                self.paths.chunk_vectors_file,
                json.dumps(
                    {
                        "vectors": [
                            {"id": vid, "vector": vec} for vid, vec in chunk_vectors
                        ]
                    },
                    indent=2,
                ),
            )
            written.append(self._rel(self.paths.chunk_vectors_file))
        return written

    def load_vectors(self, kind: str) -> list[tuple[str, list[float]]]:
        """Load a persisted vector array (``entity`` / ``relation`` / ``chunk``)."""
        path = {
            "entity": self.paths.entity_vectors_file,
            "relation": self.paths.relation_vectors_file,
            "chunk": self.paths.chunk_vectors_file,
        }.get(kind)
        if path is None or not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        vectors = payload.get("vectors") or []
        return [
            (str(entry["id"]), [float(v) for v in entry["vector"]])
            for entry in vectors
            if isinstance(entry, dict) and "id" in entry and "vector" in entry
        ]

    def _read_models(self, path: Path, model_cls: type[BaseModel]) -> list:
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []
        return [model_cls.model_validate(item) for item in payload]

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.paths.root.parent.parent.parent))
        except ValueError:
            return str(path)


def serialize_vectors(
    profiles: Iterable[object],
    vectors: list[list[float]],
    *,
    id_attr: str = "id",
) -> list[tuple[str, list[float]]]:
    """Pair profile ids with their embedding vectors for persistence."""
    paired: list[tuple[str, list[float]]] = []
    for profile, vector in zip(profiles, vectors, strict=False):
        paired.append((str(getattr(profile, id_attr)), list(vector)))
    return paired
