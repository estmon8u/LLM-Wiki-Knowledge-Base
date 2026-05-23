"""Persistent JSON store for the LightRAG-style WikiGraphRAG backend."""

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


@dataclass
class LightGraphStorePaths:
    """Filesystem layout under ``graph/wikigraph/lightrag/``."""

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
    def build_manifest_file(self) -> Path:
        return self.root / "build_manifest.json"

    @property
    def source_contributions_file(self) -> Path:
        return self.root / "source_contributions.json"

    @property
    def extraction_cache_dir(self) -> Path:
        return self.root / "extraction_cache"

    @property
    def entity_vectors_dir(self) -> Path:
        return self.root / "vectors" / "entities"

    @property
    def relation_vectors_dir(self) -> Path:
        return self.root / "vectors" / "relations"

    @property
    def chunk_vectors_dir(self) -> Path:
        return self.root / "vectors" / "chunks"


class LightGraphStore:
    """Load/save LightGraph index artifacts."""

    def __init__(self, paths: LightGraphStorePaths) -> None:
        self.paths = paths

    def exists(self) -> bool:
        return self.paths.index_file.exists()

    def load_or_none(self) -> LightGraphIndex | None:
        if not self.exists():
            return None
        return self.load()

    def load(self) -> LightGraphIndex:
        payload = json.loads(self.paths.index_file.read_text(encoding="utf-8"))
        chunks = [
            LightChunk.model_validate(item)
            for item in json.loads(self.paths.chunks_file.read_text(encoding="utf-8"))
        ]
        entities = [
            EntityProfile.model_validate(item)
            for item in json.loads(self.paths.entities_file.read_text(encoding="utf-8"))
        ]
        relations = [
            RelationProfile.model_validate(item)
            for item in json.loads(
                self.paths.relations_file.read_text(encoding="utf-8")
            )
        ]
        return LightGraphIndex(
            built_at=str(payload.get("built_at", "")),
            chunks=chunks,
            entities=entities,
            relations=relations,
            source_hashes=dict(payload.get("source_hashes", {})),
            extraction_prompt_hash=str(payload.get("extraction_prompt_hash", "")),
            embedding_model=str(payload.get("embedding_model", "")),
            embedding_dimension=int(payload.get("embedding_dimension", 0)),
            provider_identity=payload.get("provider_identity"),
            chunk_count=int(payload.get("chunk_count", len(chunks))),
            entity_count=int(payload.get("entity_count", len(entities))),
            relation_count=int(payload.get("relation_count", len(relations))),
        )

    def save(
        self,
        index: LightGraphIndex,
        *,
        build_manifest: dict[str, Any] | None = None,
        source_contributions: dict[str, Any] | None = None,
    ) -> list[str]:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        index_payload = index.model_dump(
            include={
                "built_at",
                "source_hashes",
                "extraction_prompt_hash",
                "embedding_model",
                "embedding_dimension",
                "provider_identity",
                "chunk_count",
                "entity_count",
                "relation_count",
            }
        )
        atomic_write_text(
            self.paths.index_file,
            json.dumps(index_payload, indent=2, default=str),
        )
        written.append(self.paths.index_file.name)
        atomic_write_text(
            self.paths.chunks_file,
            json.dumps([c.model_dump() for c in index.chunks], indent=2, default=str),
        )
        written.append(self.paths.chunks_file.name)
        atomic_write_text(
            self.paths.entities_file,
            json.dumps([e.model_dump() for e in index.entities], indent=2, default=str),
        )
        written.append(self.paths.entities_file.name)
        atomic_write_text(
            self.paths.relations_file,
            json.dumps(
                [r.model_dump() for r in index.relations], indent=2, default=str
            ),
        )
        written.append(self.paths.relations_file.name)
        if build_manifest is not None:
            atomic_write_text(
                self.paths.build_manifest_file,
                json.dumps(build_manifest, indent=2, default=str),
            )
            written.append(self.paths.build_manifest_file.name)
        if source_contributions is not None:
            atomic_write_text(
                self.paths.source_contributions_file,
                json.dumps(source_contributions, indent=2, default=str),
            )
            written.append(self.paths.source_contributions_file.name)
        return written

    def load_build_manifest(self) -> dict[str, Any] | None:
        if not self.paths.build_manifest_file.exists():
            return None
        return json.loads(self.paths.build_manifest_file.read_text(encoding="utf-8"))

    def load_source_contributions(self) -> dict[str, Any]:
        if not self.paths.source_contributions_file.exists():
            return {}
        return json.loads(
            self.paths.source_contributions_file.read_text(encoding="utf-8")
        )
