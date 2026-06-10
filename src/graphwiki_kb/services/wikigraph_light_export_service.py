"""Export human-readable LightRAG graph cards into ``wiki/wikigraph/``.

This keeps the project "wiki-first" in the human sense: the LightRAG index is a
retrieval engine, but its entities, relations, source chunks, and diagnostics
are mirrored as inspectable Markdown cards that flow into the Obsidian vault
export like any other wiki page.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from slugify import slugify

from graphwiki_kb.services.project_service import ProjectPaths, atomic_write_text
from graphwiki_kb.wikigraph.light_graph_store import LightGraphStore
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphIndex,
    RelationProfile,
)

ARTIFACT_FILENAME_STEM_LIMIT = 88


def _yaml_list(values: list[str]) -> str:
    if not values:
        return " []"
    return "\n" + "\n".join(f"  - {value}" for value in values)


def _artifact_slug(value: str, *, fallback: str) -> str:
    stem = slugify(str(value)) or fallback
    if len(stem) <= ARTIFACT_FILENAME_STEM_LIMIT:
        return stem
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:10]
    prefix = stem[:ARTIFACT_FILENAME_STEM_LIMIT].rstrip("-")
    return f"{prefix}-{digest}"


def _relation_slug(value: str, relation_id: str) -> str:
    stem = slugify(str(value)) or "relation"
    suffix = hashlib.sha1(relation_id.encode("utf-8")).hexdigest()[:8]
    stem_limit = ARTIFACT_FILENAME_STEM_LIMIT - len(suffix) - 1
    if len(stem) > stem_limit:
        stem = stem[:stem_limit].rstrip("-")
    return f"{stem}-{suffix}"


@dataclass
class WikiGraphLightExportService:
    """Writes LightRAG graph cards under ``wiki/wikigraph/``."""

    paths: ProjectPaths
    store: LightGraphStore
    base_subdir: str = "wikigraph"

    def export(self) -> list[str]:
        """Export all cards; returns relative paths in deterministic order.

        Raises:
            FileNotFoundError: when the LightRAG index has not been built.
        """
        index = self.store.load()
        if index is None:
            raise FileNotFoundError(
                "LightRAG index is not built. Run `kb update "
                "--wikigraph-mode lightrag` first."
            )
        base = self.paths.wiki_dir / self.base_subdir
        for sub in ("entities", "relations", "sources", "diagnostics"):
            artifact_dir = base / sub
            artifact_dir.mkdir(parents=True, exist_ok=True)
            for stale in artifact_dir.glob("*.md"):
                stale.unlink()

        chunk_by_id = {chunk.id: chunk for chunk in index.chunks}
        name_by_id = {entity.id: entity.canonical_name for entity in index.entities}
        relation_slug_by_id = {
            relation.id: self._relation_slug(relation, name_by_id)
            for relation in index.relations
        }

        written: list[str] = []
        for entity in index.entities:
            written.append(
                self._write_entity(
                    base, entity, chunk_by_id, relation_slug_by_id, index
                )
            )
        for relation in index.relations:
            written.append(
                self._write_relation(base, relation, name_by_id, chunk_by_id)
            )
        written.extend(self._write_sources(base, index))
        written.append(self._write_index(base, index))
        written.extend(self._write_diagnostics(base, index))
        written.sort()
        return written

    # ------------------------------------------------------------------ #

    def _rel(self, path) -> str:
        return path.relative_to(self.paths.root).as_posix()

    @staticmethod
    def _relation_slug(relation: RelationProfile, name_by_id: dict[str, str]) -> str:
        src = name_by_id.get(relation.source_entity_id, relation.source_entity_id)
        tgt = name_by_id.get(relation.target_entity_id, relation.target_entity_id)
        return _relation_slug(
            f"{src}-{relation.relation_type}-{tgt}",
            relation.id,
        )

    def _chunk_refs(
        self, chunk_ids: list[str], chunk_by_id: dict[str, LightChunk]
    ) -> list[str]:
        refs: list[str] = []
        for chunk_id in chunk_ids:
            chunk = chunk_by_id.get(chunk_id)
            if chunk is not None and chunk.source_ref not in refs:
                refs.append(chunk.source_ref)
        return refs

    def _write_entity(
        self,
        base,
        entity: EntityProfile,
        chunk_by_id: dict[str, LightChunk],
        relation_slug_by_id: dict[str, str],
        index: LightGraphIndex,
    ) -> str:
        filename = f"{_artifact_slug(entity.canonical_name, fallback=entity.id)}.md"
        path = base / "entities" / filename
        chunk_refs = self._chunk_refs(entity.chunk_ids, chunk_by_id)
        evidence_rows = "\n".join(
            f"| {ref.split('#', 1)[0]} | {ref} |" for ref in chunk_refs
        )
        relation_links = "\n".join(
            f"- [[wikigraph/relations/{relation_slug_by_id[rid]}]]"
            for rid in entity.relation_ids
            if rid in relation_slug_by_id
        )
        body = (
            "---\n"
            "kind: wikigraph_entity\n"
            "engine: wikigraph-lightrag\n"
            "generated: true\n"
            f"entity_id: {entity.id}\n"
            f"entity_type: {entity.type}\n"
            f"aliases:{_yaml_list(entity.aliases)}\n"
            f"source_ids:{_yaml_list(entity.source_ids)}\n"
            f"chunk_refs:{_yaml_list(chunk_refs)}\n"
            f"updated_at: {entity.updated_at}\n"
            "---\n\n"
            f"# {entity.canonical_name}\n\n"
            "## Summary\n\n"
            f"{entity.description or entity.profile_text or 'No summary.'}\n\n"
            "## Evidence\n\n"
            "| Source | Chunk |\n|---|---|\n"
            f"{evidence_rows or '| _none_ | |'}\n\n"
            "## Relations\n\n"
            f"{relation_links or '_none_'}\n"
        )
        atomic_write_text(path, body)
        return self._rel(path)

    def _write_relation(
        self,
        base,
        relation: RelationProfile,
        name_by_id: dict[str, str],
        chunk_by_id: dict[str, LightChunk],
    ) -> str:
        src = name_by_id.get(relation.source_entity_id, relation.source_entity_id)
        tgt = name_by_id.get(relation.target_entity_id, relation.target_entity_id)
        filename = f"{self._relation_slug(relation, name_by_id)}.md"
        path = base / "relations" / filename
        chunk_refs = self._chunk_refs(relation.chunk_ids, chunk_by_id)
        evidence = "\n".join(f"- {ref}" for ref in chunk_refs) or "_none_"
        body = (
            "---\n"
            "kind: wikigraph_relation\n"
            "engine: wikigraph-lightrag\n"
            "generated: true\n"
            f"relation_id: {relation.id}\n"
            f"source_entity: {src}\n"
            f"target_entity: {tgt}\n"
            f"relation_type: {relation.relation_type}\n"
            f"keywords:{_yaml_list(relation.keywords)}\n"
            f"source_ids:{_yaml_list(relation.source_ids)}\n"
            f"updated_at: {relation.updated_at}\n"
            "---\n\n"
            f"# {src} {relation.relation_type} {tgt}\n\n"
            "## Description\n\n"
            f"{relation.description or 'No description.'}\n\n"
            "## Evidence\n\n"
            f"{evidence}\n"
        )
        atomic_write_text(path, body)
        return self._rel(path)

    def _write_sources(self, base, index: LightGraphIndex) -> list[str]:
        by_slug: dict[str, list[LightChunk]] = {}
        for chunk in index.chunks:
            by_slug.setdefault(chunk.source_slug, []).append(chunk)
        written: list[str] = []
        for slug in sorted(by_slug):
            chunks = sorted(by_slug[slug], key=lambda c: c.chunk_index)
            filename = (
                f"{_artifact_slug(f'{slug}-chunks', fallback='source-chunks')}.md"
            )
            path = base / "sources" / filename
            lines = [
                "---",
                "kind: wikigraph_source_chunks",
                "engine: wikigraph-lightrag",
                "generated: true",
                f"source_slug: {slug}",
                "---",
                "",
                f"# Source chunks: {slug}",
                "",
            ]
            for chunk in chunks:
                lines.append(f"## chunk-{chunk.chunk_index} (`{chunk.source_ref}`)")
                lines.append("")
                lines.append(chunk.text.strip())
                lines.append("")
            atomic_write_text(path, "\n".join(lines).rstrip() + "\n")
            written.append(self._rel(path))
        return written

    def _write_index(self, base, index: LightGraphIndex) -> str:
        path = base / "index.md"
        body = (
            "---\n"
            "kind: wikigraph_index\n"
            "engine: wikigraph-lightrag\n"
            "generated: true\n"
            f"built_at: {index.built_at}\n"
            f"tier: {index.tier}\n"
            "---\n\n"
            "# WikiGraphRAG (LightRAG) Index\n\n"
            f"- Built at: {index.built_at}\n"
            f"- Tier: {index.tier}\n"
            f"- Chunks: {index.chunk_count}\n"
            f"- Entities: {index.entity_count}\n"
            f"- Relations: {index.relation_count}\n"
            f"- Embedding model: {index.embedding_model or 'BM25 fallback'}\n"
        )
        atomic_write_text(path, body)
        return self._rel(path)

    def _write_diagnostics(self, base, index: LightGraphIndex) -> list[str]:
        # Quality diagnostics: profiles lacking source chunks.
        orphan_entities = [e.canonical_name for e in index.entities if not e.chunk_ids]
        orphan_relations = [r.id for r in index.relations if not r.chunk_ids]
        warn_path = base / "diagnostics" / "extraction-warnings.md"
        warn_lines = [
            "---",
            "kind: wikigraph_diagnostics",
            "engine: wikigraph-lightrag",
            "generated: true",
            "---",
            "",
            "# Extraction warnings",
            "",
            "## Entities without source chunks",
            "",
            *(f"- {name}" for name in orphan_entities),
            *([] if orphan_entities else ["_none_"]),
            "",
            "## Relations without source chunks",
            "",
            *(f"- {rid}" for rid in orphan_relations),
            *([] if orphan_relations else ["_none_"]),
            "",
        ]
        atomic_write_text(warn_path, "\n".join(warn_lines))

        manifest = self.store.load_build_manifest() or {}
        missing = manifest.get("missing_sources", [])
        stale_path = base / "diagnostics" / "stale-sources.md"
        stale_lines = [
            "---",
            "kind: wikigraph_diagnostics",
            "engine: wikigraph-lightrag",
            "generated: true",
            "---",
            "",
            "# Stale / missing sources (require review)",
            "",
            *(
                f"- {item.get('source_id')} (status: {item.get('status')})"
                for item in missing
            ),
            *([] if missing else ["_none_"]),
            "",
        ]
        atomic_write_text(stale_path, "\n".join(stale_lines))
        return [self._rel(warn_path), self._rel(stale_path)]
