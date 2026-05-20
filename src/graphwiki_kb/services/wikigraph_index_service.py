"""Service layer for building and inspecting the WikiGraphRAG index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
)


@dataclass
class WikiGraphIndexService:
    """Builds and persists the wiki graph index for a project."""

    paths: ProjectPaths

    def __post_init__(self) -> None:
        self._store = WikiGraphStore(
            WikiGraphStorePaths(self.paths.graph_dir / "wikigraph")
        )

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
        include_graphrag_export_pages: bool = False,
        chunk_char_limit: int = 1200,
    ) -> WikiGraphBuildReport:
        """Build the index from the maintained wiki and persist it."""
        index = build_wikigraph_index(
            self.paths,
            options=BuildOptions(
                chunk_char_limit=chunk_char_limit,
                include_graphrag_export_pages=include_graphrag_export_pages,
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
