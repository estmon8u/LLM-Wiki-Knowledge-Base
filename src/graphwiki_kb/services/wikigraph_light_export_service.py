"""Export human-readable LightRAG graph cards to ``wiki/wikigraph/``.

This is the wiki-artifact-layer half of the LightRAG-style backend:
every canonical :class:`EntityProfile` / :class:`RelationProfile` is
rendered as a small inspectable Markdown card under
``wiki/wikigraph/{entities,relations,sources,diagnostics}/`` so the
wiki remains the human-readable artifact layer (project recommendation
§16) even after the retrieval engine migrates to a vector-first
backend.

Cards always carry ``engine: wikigraph-lightrag`` and ``generated:
true`` in their frontmatter so they are trivially distinguishable from
hand-maintained wiki pages and easy to exclude from future LightGraph
builds.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
    utc_now_iso,
)
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
)
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphIndex,
    RelationProfile,
    SourceContribution,
)


@dataclass
class WikiGraphLightExportService:
    """Render LightGraph profiles as inspectable wiki artifacts.

    Attributes:
        paths: Project paths used to resolve the wiki output directory.
        store: Optional pre-built store; defaults to the canonical
            ``graph/wikigraph/lightrag`` path on first use.
    """

    paths: ProjectPaths
    store: LightGraphStore | None = field(default=None)

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = LightGraphStore(
                LightGraphStorePaths(self.paths.graph_dir / "wikigraph" / "lightrag")
            )

    def export_cards(self, *, index: LightGraphIndex | None = None) -> list[str]:
        """Write entity/relation/source/diagnostic cards.

        Returns the relative paths written, sorted for deterministic IO.
        Raises :class:`FileNotFoundError` when the LightGraph index has
        not been built yet and the caller did not provide one.
        """
        if index is None:
            assert self.store is not None
            index = self.store.load()
        if index is None:
            raise FileNotFoundError(
                "LightRAG WikiGraph index is not built. Set "
                "`wikigraph.mode: lightrag` and run `kb update` first."
            )
        timestamp = utc_now_iso()
        base = self.paths.wiki_dir / "wikigraph"
        for sub in ("entities", "relations", "sources", "diagnostics"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        entities_by_id = {entity.id: entity for entity in index.entities}
        for entity in index.entities:
            written.append(self._write_entity_card(entity, timestamp))
        for relation in index.relations:
            written.append(
                self._write_relation_card(relation, entities_by_id, timestamp)
            )
        for slug, chunk_list in _group_chunks_by_slug(index.chunks).items():
            written.append(self._write_source_chunks_card(slug, chunk_list, timestamp))
        missing = [c for c in index.contributions if c.status == "missing"]
        if missing:
            written.append(self._write_stale_sources(missing, timestamp))
        written.sort()
        return written

    # ----------------------------------------------------------------- #
    # Card writers                                                       #
    # ----------------------------------------------------------------- #

    def _write_entity_card(self, entity: EntityProfile, timestamp: str) -> str:
        filename = f"{slugify(entity.canonical_name)}.md"
        rel = f"wiki/wikigraph/entities/{filename}"
        path = self.paths.root / rel
        aliases_block = "\n".join(f"  - {alias}" for alias in entity.aliases[:12])
        sources_block = "\n".join(f"  - {sid}" for sid in entity.source_ids[:12])
        chunk_refs_block = "\n".join(f"  - {cid}" for cid in entity.chunk_ids[:12])
        relation_refs_block = "\n".join(
            f"  - {rid}" for rid in entity.relation_ids[:12]
        )
        body = (
            "---\n"
            "kind: wikigraph_entity\n"
            "engine: wikigraph-lightrag\n"
            "generated: true\n"
            f"entity_id: {entity.id}\n"
            f"entity_type: {entity.type}\n"
            + (f"aliases:\n{aliases_block}\n" if aliases_block else "")
            + (f"source_ids:\n{sources_block}\n" if sources_block else "")
            + (f"chunk_refs:\n{chunk_refs_block}\n" if chunk_refs_block else "")
            + (
                f"relation_refs:\n{relation_refs_block}\n"
                if relation_refs_block
                else ""
            )
            + f"updated_at: {timestamp}\n"
            "---\n\n"
            f"# {entity.canonical_name}\n\n"
            "## Summary\n\n"
            f"{entity.description or 'Entity profile generated by WikiGraphRAG (LightRAG).'}\n\n"
            "## Profile\n\n"
            f"{entity.profile_text or '_no profile text generated_'}\n"
        )
        atomic_write_text(path, body)
        return rel

    def _write_relation_card(
        self,
        relation: RelationProfile,
        entities_by_id: dict[str, EntityProfile],
        timestamp: str,
    ) -> str:
        source = entities_by_id.get(relation.source_entity_id)
        target = entities_by_id.get(relation.target_entity_id)
        source_name = source.canonical_name if source else relation.source_entity_id
        target_name = target.canonical_name if target else relation.target_entity_id
        title = f"{source_name} {relation.relation_type} {target_name}"
        filename = f"{slugify(title)}.md"
        rel = f"wiki/wikigraph/relations/{filename}"
        path = self.paths.root / rel
        keywords_block = "\n".join(f"  - {kw}" for kw in relation.keywords[:12])
        sources_block = "\n".join(f"  - {sid}" for sid in relation.source_ids[:12])
        chunks_block = "\n".join(f"  - {cid}" for cid in relation.chunk_ids[:12])
        body = (
            "---\n"
            "kind: wikigraph_relation\n"
            "engine: wikigraph-lightrag\n"
            "generated: true\n"
            f"relation_id: {relation.id}\n"
            f"relation_type: {relation.relation_type}\n"
            f"source_entity: {source_name}\n"
            f"target_entity: {target_name}\n"
            + (f"keywords:\n{keywords_block}\n" if keywords_block else "")
            + (f"source_ids:\n{sources_block}\n" if sources_block else "")
            + (f"chunk_refs:\n{chunks_block}\n" if chunks_block else "")
            + f"updated_at: {timestamp}\n"
            "---\n\n"
            f"# {title}\n\n"
            "## Description\n\n"
            f"{relation.description or 'Relation profile generated by WikiGraphRAG (LightRAG).'}\n\n"
            "## Profile\n\n"
            f"{relation.profile_text or '_no profile text generated_'}\n"
        )
        atomic_write_text(path, body)
        return rel

    def _write_source_chunks_card(
        self, slug: str, chunks: list[LightChunk], timestamp: str
    ) -> str:
        rel = f"wiki/wikigraph/sources/{slug}-chunks.md"
        path = self.paths.root / rel
        title = chunks[0].source_title or slug
        source_id = chunks[0].source_id
        body_lines = [
            "---",
            "kind: wikigraph_source_chunks",
            "engine: wikigraph-lightrag",
            "generated: true",
            f"source_id: {source_id}",
            f"source_slug: {slug}",
            f"chunk_count: {len(chunks)}",
            f"updated_at: {timestamp}",
            "---",
            "",
            f"# {title} — chunks",
            "",
        ]
        for chunk in chunks:
            anchor = (
                f"{chunk.compiled_page_path or chunk.normalized_path}"
                f"#chunk-{chunk.chunk_index}"
            )
            preview = chunk.text.strip().splitlines()[0][:200] if chunk.text else ""
            body_lines.append(
                f"- `{chunk.id}` → `{anchor}` ({chunk.token_count} tokens)"
                + (f" — {preview}" if preview else "")
            )
        body_lines.append("")
        atomic_write_text(path, "\n".join(body_lines))
        return rel

    def _write_stale_sources(
        self, missing: list[SourceContribution], timestamp: str
    ) -> str:
        rel = "wiki/wikigraph/diagnostics/stale-sources.md"
        path = self.paths.root / rel
        lines = [
            "---",
            "kind: wikigraph_diagnostic",
            "engine: wikigraph-lightrag",
            "generated: true",
            f"missing_source_count: {len(missing)}",
            f"updated_at: {timestamp}",
            "---",
            "",
            "# Stale / missing sources",
            "",
            (
                "Sources known to a previous LightGraph build but absent "
                "from the current manifest. These are flagged for review "
                "rather than silently purged."
            ),
            "",
        ]
        for item in sorted(missing, key=lambda c: c.source_id):
            lines.append(
                f"- `{item.source_id}` status={item.status} "
                f"requires_review={str(item.requires_review).lower()}"
            )
        lines.append("")
        atomic_write_text(path, "\n".join(lines))
        return rel


def _group_chunks_by_slug(chunks: list[LightChunk]) -> dict[str, list[LightChunk]]:
    grouped: dict[str, list[LightChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.source_slug].append(chunk)
    for chunk_list in grouped.values():
        chunk_list.sort(key=lambda c: c.chunk_index)
    return dict(grouped)
