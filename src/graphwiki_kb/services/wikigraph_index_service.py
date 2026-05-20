"""Service layer for building and inspecting the WikiGraphRAG index."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphwiki_kb.services.config_service import (
    WikiGraphRuntimeConfig,
    resolve_wikigraph_config,
)
from graphwiki_kb.services.project_service import ProjectPaths
from graphwiki_kb.wikigraph.graph_store import (
    WikiGraphStore,
    WikiGraphStorePaths,
)
from graphwiki_kb.wikigraph.index_builder import (
    BuildOptions,
    build_wikigraph_index,
)
from graphwiki_kb.wikigraph.models import (
    WikiGraphBuildReport,
    WikiGraphIndex,
    WikiGraphNode,
)


@dataclass
class WikiGraphIndexService:
    """Builds and persists the wiki graph index for a project."""

    paths: ProjectPaths
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._store = WikiGraphStore(
            WikiGraphStorePaths(self.paths.graph_dir / "wikigraph")
        )

    @property
    def runtime_config(self) -> WikiGraphRuntimeConfig:
        """Resolve the WikiGraphRAG runtime config from project config."""
        return resolve_wikigraph_config(self.config)

    @property
    def store(self) -> WikiGraphStore:
        """Underlying :class:`WikiGraphStore`."""
        return self._store

    @property
    def store_root(self) -> Path:
        """The on-disk directory where wikigraph artifacts are written."""
        return self._store.paths.root

    def build(
        self,
        *,
        include_graphrag_export_pages: bool | None = None,
        chunk_char_limit: int | None = None,
    ) -> WikiGraphBuildReport:
        """Build the index from the maintained wiki and persist it.

        When ``include_graphrag_export_pages`` or ``chunk_char_limit`` is
        ``None``, the value is read from the resolved
        :class:`WikiGraphRuntimeConfig` (or the package default when no
        config was supplied).
        """
        try:
            runtime = self.runtime_config
        except ValueError:
            runtime = resolve_wikigraph_config({})
        effective_include = (
            include_graphrag_export_pages
            if include_graphrag_export_pages is not None
            else runtime.include_graphrag_export_pages
        )
        effective_chunk = (
            chunk_char_limit
            if chunk_char_limit is not None
            else runtime.chunk_char_limit
        )
        index = build_wikigraph_index(
            self.paths,
            options=BuildOptions(
                chunk_char_limit=effective_chunk,
                include_graphrag_export_pages=effective_include,
                fuzzy_entity_match_threshold=runtime.fuzzy_entity_match_threshold,
            ),
        )
        written = self._store.save(index)
        warnings: list[str] = []
        if index.source_count == 0:
            warnings.append("no source pages found under wiki/sources")
        return WikiGraphBuildReport(
            built_at=index.built_at,
            node_count=len(index.nodes),
            edge_count=len(index.edges),
            chunk_count=index.chunk_count,
            entity_count=index.entity_count,
            community_count=len(index.communities),
            source_count=index.source_count,
            include_graphrag_export_pages=index.include_graphrag_export_pages,
            artifacts=written,
            warnings=warnings,
        )

    def load(self) -> WikiGraphIndex | None:
        """Load the persisted index from disk."""
        return self._store.load()

    SUPPORTED_ARTIFACT_TYPES: tuple[str, ...] = ("entities", "communities", "chunks")

    def export_artifacts(self, *, types: tuple[str, ...] | None = None) -> list[str]:
        """Write generated wiki artifact pages under ``wiki/wikigraph/``.

        Produces one markdown card per ``entity``, ``community``, and
        ``chunk`` node in the persisted index by default. Every card
        carries ``generated: true`` and ``retrieval_backend: wikigraph`` in
        its frontmatter so it is easy to filter from other tooling, and
        the directory ``wiki/wikigraph/`` is explicitly excluded from the
        default index build (see ``BuildOptions``) so generated cards
        cannot feed back into the next graph build.

        Args:
            types: Optional subset of ``{"entities", "communities",
                "chunks"}`` to write. Unknown types raise ``ValueError``.

        Returns:
            The list of relative paths written, in deterministic order.

        Raises:
            FileNotFoundError: When the WikiGraphRAG index has not been
                built yet.
            ValueError: When ``types`` contains an unknown value.
        """
        if types is not None:
            unknown = [t for t in types if t not in self.SUPPORTED_ARTIFACT_TYPES]
            if unknown:
                raise ValueError(
                    "Unknown wikigraph artifact type(s): "
                    + ", ".join(sorted(set(unknown)))
                )
        selected = tuple(types) if types else self.SUPPORTED_ARTIFACT_TYPES

        index = self.load()
        if index is None:
            raise FileNotFoundError(
                "WikiGraphRAG index is not built. Run `kb update` first."
            )

        from graphwiki_kb.services.project_service import (
            slugify,
            utc_now_iso,
        )

        base = self.paths.wiki_dir / "wikigraph"
        for subdir in selected:
            (base / subdir).mkdir(parents=True, exist_ok=True)
        timestamp = utc_now_iso()
        written: list[str] = []
        for node in index.nodes:
            if node.kind == "entity" and "entities" in selected:
                rel = self._write_entity_card(base, node, timestamp, slugify)
                written.append(rel)
            elif node.kind == "community" and "communities" in selected:
                rel = self._write_community_card(base, node, timestamp, slugify)
                written.append(rel)
            elif node.kind == "chunk" and "chunks" in selected:
                rel = self._write_chunk_card(base, node, timestamp, slugify)
                written.append(rel)
        written.sort()
        return written

    def _write_entity_card(
        self,
        base: Path,
        node: WikiGraphNode,
        timestamp: str,
        slug: Any,
    ) -> str:
        from graphwiki_kb.services.project_service import atomic_write_text

        slugify_fn = slug
        filename = f"{slugify_fn(node.title)}.md"
        rel = f"wiki/wikigraph/entities/{filename}"
        path = self.paths.root / rel
        sources_block = "\n".join(f"  - {sid}" for sid in node.source_ids[:8])
        aliases_block = "\n".join(f"  - {alias}" for alias in node.aliases[:8])
        body = (
            "---\n"
            f'title: "{node.title}"\n'
            "type: wikigraph_entity\n"
            "generated: true\n"
            "retrieval_backend: wikigraph\n"
            f'generated_at: "{timestamp}"\n'
            "confidence: medium\n"
            + (f"aliases:\n{aliases_block}\n" if aliases_block else "")
            + (f"source_ids:\n{sources_block}\n" if sources_block else "")
            + "---\n\n"
            f"# {node.title}\n\n"
            f"{node.text or 'Entity surface form generated by WikiGraphRAG.'}\n"
        )
        atomic_write_text(path, body)
        return rel

    def _write_community_card(
        self,
        base: Path,
        node: WikiGraphNode,
        timestamp: str,
        slug: Any,
    ) -> str:
        from graphwiki_kb.services.project_service import atomic_write_text

        slugify_fn = slug
        filename = f"{slugify_fn(node.id)}.md"
        rel = f"wiki/wikigraph/communities/{filename}"
        path = self.paths.root / rel
        top_entities = node.metadata.get("top_entities") or []
        top_block = "\n".join(f"- {item}" for item in top_entities[:10])
        body = (
            "---\n"
            f'title: "{node.title}"\n'
            "type: wikigraph_community\n"
            "generated: true\n"
            "retrieval_backend: wikigraph\n"
            f'generated_at: "{timestamp}"\n'
            f"community_id: {node.id}\n"
            "---\n\n"
            f"# {node.title}\n\n"
            f"{node.text or 'Community summary generated by WikiGraphRAG.'}\n\n"
            f"## Top Entities\n\n{top_block or '_no annotated entities_'}\n"
        )
        atomic_write_text(path, body)
        return rel

    def _write_chunk_card(
        self,
        base: Path,
        node: WikiGraphNode,
        timestamp: str,
        slug: Any,
    ) -> str:
        from graphwiki_kb.services.project_service import atomic_write_text

        slugify_fn = slug
        filename = f"{slugify_fn(node.id)}.md"
        rel = f"wiki/wikigraph/chunks/{filename}"
        path = self.paths.root / rel
        body = (
            "---\n"
            f'title: "{node.title}"\n'
            "type: wikigraph_chunk\n"
            "generated: true\n"
            "retrieval_backend: wikigraph\n"
            f'generated_at: "{timestamp}"\n'
            f"source_path: {node.path or ''}\n"
            f"chunk_index: {node.metadata.get('chunk_index', '')}\n"
            "---\n\n"
            f"# {node.title}\n\n"
            f"{node.text}\n"
        )
        atomic_write_text(path, body)
        return rel

    def status(self) -> dict[str, object]:
        """Return a quick-look status payload for ``kb wikigraph status``."""
        if not self._store.exists():
            return {
                "initialized": False,
                "index_path": str(self.store_root),
                "message": (
                    "Run `kb update` to materialize the WikiGraphRAG index "
                    "(enabled by default; pass `--no-wikigraph` to skip)."
                ),
            }
        index = self.load()
        if index is None:
            return {
                "initialized": True,
                "index_path": str(self.store_root),
                "readable": False,
                "message": "Index files exist but failed to load.",
            }
        return {
            "initialized": True,
            "index_path": str(self.store_root),
            "readable": True,
            "built_at": index.built_at,
            "node_count": len(index.nodes),
            "edge_count": len(index.edges),
            "chunk_count": index.chunk_count,
            "entity_count": index.entity_count,
            "community_count": len(index.communities),
            "source_count": index.source_count,
            "include_graphrag_export_pages": index.include_graphrag_export_pages,
        }
