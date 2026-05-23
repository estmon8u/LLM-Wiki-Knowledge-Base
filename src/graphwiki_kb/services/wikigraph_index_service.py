"""Service layer for building and inspecting the WikiGraphRAG index."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphwiki_kb.services.config_service import (
    WikiGraphRuntimeConfig,
    resolve_wikigraph_config,
)
from graphwiki_kb.services.embedding_service import (
    build_embedding_provider,
    resolve_embedding_config,
)
from graphwiki_kb.services.manifest_service import ManifestService
from graphwiki_kb.services.project_service import ProjectPaths
from graphwiki_kb.wikigraph.graph_store import (
    WikiGraphStore,
    WikiGraphStorePaths,
)
from graphwiki_kb.wikigraph.index_builder import (
    BuildOptions,
    build_wikigraph_index,
)
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
)
from graphwiki_kb.wikigraph.light_index_builder import (
    LightGraphConfig,
    build_lightgraph_index,
)
from graphwiki_kb.wikigraph.light_models import LightGraphBuildReport
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
    manifest_service: ManifestService | None = None

    def __post_init__(self) -> None:
        self._classic_store = WikiGraphStore(
            WikiGraphStorePaths(self.paths.graph_dir / "wikigraph")
        )
        self._light_store_paths = LightGraphStorePaths(
            self.paths.graph_dir / "wikigraph" / "lightrag"
        )
        self._light_store = LightGraphStore(self._light_store_paths)

    @property
    def mode(self) -> str:
        try:
            return self.runtime_config.mode
        except ValueError:
            return "classic"

    @property
    def _store(self) -> WikiGraphStore:
        return self._classic_store

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
        include_normalized_text_units: bool | None = None,
        mode: str | None = None,
        provider: Any | None = None,
    ) -> WikiGraphBuildReport | LightGraphBuildReport:
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
        effective_include_text_units = (
            include_normalized_text_units
            if include_normalized_text_units is not None
            else runtime.include_normalized_text_units
        )
        effective_mode = mode or self.mode
        if effective_mode == "lightrag":
            return self._build_lightrag(provider=provider)

        sources = (
            self.manifest_service.list_sources()
            if self.manifest_service is not None
            else []
        )
        index = build_wikigraph_index(
            self.paths,
            sources=sources,
            options=BuildOptions(
                chunk_char_limit=effective_chunk,
                include_graphrag_export_pages=effective_include,
                fuzzy_entity_match_threshold=runtime.fuzzy_entity_match_threshold,
                include_normalized_text_units=effective_include_text_units,
                text_unit_char_limit=runtime.text_unit_char_limit,
                text_unit_overlap_chars=runtime.text_unit_overlap_chars,
                text_unit_min_chars=runtime.text_unit_min_chars,
                text_unit_source=runtime.text_unit_source,
                text_unit_entity_mode=runtime.text_unit_entity_mode,
            ),
        )
        written = self._store.save(index)
        warnings: list[str] = []
        if index.source_count == 0:
            warnings.append("no source pages found under wiki/sources")
        if effective_include_text_units and index.text_unit_count == 0:
            warnings.append(
                "WikiGraphRAG was configured to include normalized TextUnits "
                "but the manifest produced no usable normalized text "
                "(check raw/normalized/ paths and `text_unit_source`)"
            )
        return WikiGraphBuildReport(
            built_at=index.built_at,
            node_count=len(index.nodes),
            edge_count=len(index.edges),
            chunk_count=index.chunk_count,
            text_unit_count=index.text_unit_count,
            document_count=index.document_count,
            entity_count=index.entity_count,
            community_count=len(index.communities),
            source_count=index.source_count,
            include_graphrag_export_pages=index.include_graphrag_export_pages,
            include_normalized_text_units=index.include_normalized_text_units,
            artifacts=written,
            warnings=warnings,
        )

    def _build_lightrag(self, *, provider: Any | None = None) -> LightGraphBuildReport:
        runtime = self.runtime_config
        light = runtime.lightrag
        sources = (
            self.manifest_service.list_sources()
            if self.manifest_service is not None
            else []
        )
        embedding_runtime = resolve_embedding_config(self.config)
        embedding_provider = build_embedding_provider(embedding_runtime)
        previous = self._light_store.load_or_none()
        report = build_lightgraph_index(
            self.paths,
            sources,
            config=LightGraphConfig(
                chunk_token_size=light.chunk_token_size,
                chunk_overlap_tokens=light.chunk_overlap_tokens,
                entity_extract_max_gleaning=light.entity_extract_max_gleaning,
                entity_types=light.entity_types,
                relation_types=light.relation_types,
                embedding_runtime=embedding_runtime,
                local_fallback=light.local_fallback,
            ),
            provider=provider,
            embedding_provider=embedding_provider,
            previous_index=previous,
        )
        if runtime.export_generated_artifacts:
            from graphwiki_kb.services.wikigraph_light_export_service import (
                WikiGraphLightExportService,
            )

            exporter = WikiGraphLightExportService(self.paths)
            try:
                report.artifacts.extend(exporter.export_cards())
            except Exception as exc:
                report.warnings.append(f"artifact export failed: {exc}")
        return report

    def load(self) -> WikiGraphIndex | None:
        """Load the persisted classic index from disk."""
        if self.mode == "lightrag":
            return None
        return self._store.load()

    def load_light(self) -> Any | None:
        """Load the persisted LightRAG index when active."""
        if self.mode != "lightrag":
            return None
        return self._light_store.load_or_none()

    SUPPORTED_ARTIFACT_TYPES: tuple[str, ...] = (
        "entities",
        "communities",
        "chunks",
        "text_units",
    )

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
            elif node.kind == "text_unit" and "text_units" in selected:
                rel = self._write_text_unit_card(base, node, timestamp, slugify)
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

    def _write_text_unit_card(
        self,
        base: Path,
        node: WikiGraphNode,
        timestamp: str,
        slug: Any,
    ) -> str:
        from graphwiki_kb.services.project_service import atomic_write_text

        slugify_fn = slug
        filename = f"{slugify_fn(node.id)}.md"
        rel = f"wiki/wikigraph/text_units/{filename}"
        path = self.paths.root / rel
        metadata = node.metadata or {}
        body = (
            "---\n"
            f'title: "{node.title}"\n'
            "type: wikigraph_text_unit\n"
            "generated: true\n"
            "retrieval_backend: wikigraph\n"
            f'generated_at: "{timestamp}"\n'
            f"source_path: {node.path or ''}\n"
            f"source_id: {metadata.get('source_id', '')}\n"
            f"unit_index: {metadata.get('unit_index', '')}\n"
            f"start_char: {metadata.get('start_char', '')}\n"
            f"end_char: {metadata.get('end_char', '')}\n"
            "---\n\n"
            f"# {node.title}\n\n"
            f"{node.text}\n"
        )
        atomic_write_text(path, body)
        return rel

    def status(self) -> dict[str, object]:
        """Return a quick-look status payload for ``kb wikigraph status``."""
        if self.mode == "lightrag":
            return self._light_status()
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
            "text_unit_count": index.text_unit_count,
            "document_count": index.document_count,
            "entity_count": index.entity_count,
            "community_count": len(index.communities),
            "source_count": index.source_count,
            "include_graphrag_export_pages": index.include_graphrag_export_pages,
            "include_normalized_text_units": index.include_normalized_text_units,
        }

    def _light_status(self) -> dict[str, object]:
        if not self._light_store.exists():
            return {
                "mode": "lightrag",
                "initialized": False,
                "index_path": str(self._light_store_paths.root),
                "message": "Run `kb update` to build the LightRAG-style WikiGraph index.",
            }
        index = self._light_store.load_or_none()
        if index is None:
            return {
                "mode": "lightrag",
                "initialized": True,
                "readable": False,
                "index_path": str(self._light_store_paths.root),
            }
        manifest = self._light_store.load_build_manifest() or {}
        stale = [
            item.get("source_id", "")
            for item in manifest.get("missing_sources", [])
            if isinstance(item, dict)
        ]
        return {
            "mode": "lightrag",
            "initialized": True,
            "fresh": not stale,
            "built_at": index.built_at,
            "source_count": len(index.source_hashes),
            "chunk_count": index.chunk_count,
            "entity_count": index.entity_count,
            "relation_count": index.relation_count,
            "embedding_model": index.embedding_model,
            "provider_required": True,
            "provider_ready": bool(index.provider_identity),
            "stale_reasons": [f"missing_source:{sid}" for sid in stale if sid],
        }
