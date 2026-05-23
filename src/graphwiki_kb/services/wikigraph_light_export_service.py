"""Export human-readable LightRAG graph cards to the wiki."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
    utc_now_iso,
)
from graphwiki_kb.wikigraph.light_chunker import chunk_citation_ref
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
)
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    RelationProfile,
)


class WikiGraphLightExportService:
    """Write entity/relation/chunk cards under ``wiki/wikigraph/``."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.store_paths = LightGraphStorePaths(
            paths.graph_dir / "wikigraph" / "lightrag"
        )

    def export_cards(self) -> list[str]:
        store = LightGraphStore(self.store_paths)
        index = store.load_or_none()
        if index is None:
            raise FileNotFoundError(
                "LightRAG WikiGraph index is not built. Run `kb update` first."
            )
        timestamp = utc_now_iso()
        written: list[str] = []
        base = self.paths.wiki_dir / "wikigraph"
        for sub in ("entities", "relations", "sources", "diagnostics"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        for entity in index.entities:
            written.append(self._write_entity_card(base, entity, timestamp))
        for relation in index.relations:
            written.append(self._write_relation_card(base, relation, index, timestamp))
        for chunk in index.chunks:
            written.append(self._write_chunk_summary(base, chunk, timestamp))
        manifest = store.load_build_manifest() or {}
        missing = manifest.get("missing_sources", [])
        if missing:
            written.append(self._write_stale_sources(base, missing, timestamp))
        written.sort()
        return written

    def _write_entity_card(
        self, base: Path, entity: EntityProfile, timestamp: str
    ) -> str:
        filename = f"{slugify(entity.canonical_name)}.md"
        rel = f"wiki/wikigraph/entities/{filename}"
        path = self.paths.root / rel
        aliases = "\n".join(f"  - {alias}" for alias in entity.aliases[:12])
        chunk_refs = "\n".join(f"  - {chunk_id}" for chunk_id in entity.chunk_ids[:12])
        body = (
            "---\n"
            "kind: wikigraph_entity\n"
            "engine: wikigraph-lightrag\n"
            f"entity_id: {entity.id}\n"
            f"entity_type: {entity.type}\n"
            + (f"aliases:\n{aliases}\n" if aliases else "")
            + "source_ids:\n"
            + "\n".join(f"  - {sid}" for sid in entity.source_ids[:12])
            + "\n"
            + (f"chunk_refs:\n{chunk_refs}\n" if chunk_refs else "")
            + f"updated_at: {timestamp}\n"
            "---\n\n"
            f"# {entity.canonical_name}\n\n"
            "## Summary\n\n"
            f"{entity.description}\n\n"
            "## Profile\n\n"
            f"{entity.profile_text}\n"
        )
        atomic_write_text(path, body)
        return rel

    def _write_relation_card(
        self,
        base: Path,
        relation: RelationProfile,
        index: Any,
        timestamp: str,
    ) -> str:
        entities = {entity.id: entity for entity in index.entities}
        source = entities.get(relation.source_entity_id)
        target = entities.get(relation.target_entity_id)
        title = (
            f"{source.canonical_name if source else relation.source_entity_id} "
            f"{relation.relation_type} "
            f"{target.canonical_name if target else relation.target_entity_id}"
        )
        filename = f"{slugify(title)}.md"
        rel = f"wiki/wikigraph/relations/{filename}"
        path = self.paths.root / rel
        keywords = "\n".join(f"  - {kw}" for kw in relation.keywords[:12])
        body = (
            "---\n"
            "kind: wikigraph_relation\n"
            "engine: wikigraph-lightrag\n"
            f"relation_id: {relation.id}\n"
            f"relation_type: {relation.relation_type}\n"
            + (f"keywords:\n{keywords}\n" if keywords else "")
            + f"updated_at: {timestamp}\n"
            "---\n\n"
            f"# {title}\n\n"
            "## Description\n\n"
            f"{relation.description}\n"
        )
        atomic_write_text(path, body)
        return rel

    def _write_chunk_summary(
        self, base: Path, chunk: LightChunk, timestamp: str
    ) -> str:
        filename = f"{chunk.source_slug}-chunks.md"
        rel = f"wiki/wikigraph/sources/{filename}"
        path = self.paths.root / rel
        if path.exists():
            return rel
        body = (
            "---\n"
            "kind: wikigraph_source_chunks\n"
            "engine: wikigraph-lightrag\n"
            f"source_id: {chunk.source_id}\n"
            f"updated_at: {timestamp}\n"
            "---\n\n"
            f"# {chunk.source_slug} chunks\n\n"
            f"- {chunk_citation_ref(chunk)}\n"
        )
        atomic_write_text(path, body)
        return rel

    def _write_stale_sources(
        self, base: Path, missing: list[dict[str, Any]], timestamp: str
    ) -> str:
        rel = "wiki/wikigraph/diagnostics/stale-sources.md"
        path = self.paths.root / rel
        lines = ["# Stale / missing sources", ""]
        for item in missing:
            lines.append(
                f"- `{item.get('source_id', '')}` status={item.get('status', 'missing')} "
                f"requires_review={item.get('requires_review', True)}"
            )
        lines.append(f"\n_updated: {timestamp}_\n")
        atomic_write_text(path, "\n".join(lines))
        return rel
