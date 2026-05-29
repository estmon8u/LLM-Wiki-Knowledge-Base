"""Service layer for building and inspecting the WikiGraphRAG index."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphwiki_kb.providers import build_lazy_provider, resolve_provider_settings
from graphwiki_kb.services.config_service import (
    WikiGraphRuntimeConfig,
    resolve_embeddings_config,
    resolve_wikigraph_config,
)
from graphwiki_kb.services.embedding_service import build_embedding_provider
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
from graphwiki_kb.wikigraph.light_index_builder import build_lightgraph_index
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
        include_normalized_text_units: bool | None = None,
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
        if runtime.mode == "lightrag":
            return self._build_lightrag(runtime)
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

    def lightrag_store(self) -> LightGraphStore:
        """Return the LightRAG store under ``graph/wikigraph/lightrag``."""
        return LightGraphStore(
            LightGraphStorePaths(self.paths.graph_dir / "wikigraph" / "lightrag")
        )

    def build_lightrag_report(self) -> LightGraphBuildReport | None:
        """Return the most recent LightRAG build report, if any."""
        return getattr(self, "_last_light_report", None)

    def _build_lightrag(self, runtime: WikiGraphRuntimeConfig) -> WikiGraphBuildReport:
        config = self.config or {}
        sources = (
            self.manifest_service.list_sources()
            if self.manifest_service is not None
            else []
        )
        # Extraction tier is opt-in: only build/pass an LLM provider when
        # `wikigraph.lightrag.extraction.extractor == "llm"`. Otherwise the
        # deterministic (provider-free) extractor runs, so `kb update
        # --wikigraph-mode lightrag` never makes surprise LLM calls by default.
        use_llm_extractor = runtime.lightrag.extraction_mode == "llm"
        provider = build_lazy_provider(config) if use_llm_extractor else None
        embedding_provider = build_embedding_provider(config)
        identity = "deterministic"
        if use_llm_extractor:
            resolved = resolve_provider_settings(config)
            if resolved is not None:
                name, provider_cfg = resolved
                identity = f"{name}:{provider_cfg.get('model', '')}"
        store = self.lightrag_store()
        previous_index = store.load()
        report = build_lightgraph_index(
            self.paths.root,
            sources,
            store=store,
            lightrag_config=runtime.lightrag,
            embeddings_config=resolve_embeddings_config(config),
            provider=provider,
            embedding_provider=embedding_provider,
            provider_identity=identity,
            previous_index=previous_index,
            previous_entity_vectors=store.load_entity_vectors(),
            previous_relation_vectors=store.load_relation_vectors(),
        )
        self._last_light_report = report
        return _lightrag_to_build_report(report)

    def load(self) -> WikiGraphIndex | None:
        """Load the persisted index from disk."""
        return self._store.load()

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
        try:
            runtime = self.runtime_config
        except ValueError:
            runtime = resolve_wikigraph_config({})
        if runtime.mode == "lightrag":
            from graphwiki_kb.services.wikigraph_light_export_service import (
                WikiGraphLightExportService,
            )

            return WikiGraphLightExportService(
                paths=self.paths, store=self.lightrag_store()
            ).export()
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

    def _lightrag_status(self, runtime: WikiGraphRuntimeConfig) -> dict[str, object]:
        from graphwiki_kb.providers import resolve_provider_settings
        from graphwiki_kb.services.config_service import resolve_embeddings_config
        from graphwiki_kb.wikigraph.light_extractor import (
            ExtractionConfig,
            extraction_prompt_hash,
        )

        config = self.config or {}
        store = self.lightrag_store()
        sources = (
            self.manifest_service.list_sources()
            if self.manifest_service is not None
            else []
        )
        current = {source.source_id: source.content_hash for source in sources}
        provider_ready = resolve_provider_settings(config) is not None
        provider_required = runtime.lightrag.embeddings_required_for_strict
        if not store.exists():
            return {
                "mode": "lightrag",
                "initialized": False,
                "fresh": False,
                "source_count": len(sources),
                "provider_required": provider_required,
                "provider_ready": provider_ready,
                "stale_reasons": ["index not built"],
            }
        index = store.load()
        manifest = store.load_build_manifest() or {}
        previous = dict(manifest.get("source_hashes", {}))
        stale_reasons: list[str] = []
        new = [sid for sid in current if sid not in previous]
        changed = [
            sid for sid in current if sid in previous and previous[sid] != current[sid]
        ]
        missing = [sid for sid in previous if sid not in current]
        if new:
            stale_reasons.append(f"{len(new)} new source(s) not yet indexed")
        if changed:
            stale_reasons.append(f"{len(changed)} changed source(s)")
        if missing:
            stale_reasons.append(f"{len(missing)} missing source(s) require review")
        current_prompt = extraction_prompt_hash(
            ExtractionConfig(
                entity_types=tuple(runtime.lightrag.entity_types),
                relation_types=tuple(runtime.lightrag.relation_types),
                max_gleaning=runtime.lightrag.entity_extract_max_gleaning,
            )
        )
        if (
            manifest.get("extraction_prompt_hash")
            and manifest["extraction_prompt_hash"] != current_prompt
        ):
            stale_reasons.append("extraction prompt changed")
        embeddings_cfg = resolve_embeddings_config(config)
        if (
            index is not None
            and index.embedding_model
            and index.embedding_model != embeddings_cfg.model
        ):
            stale_reasons.append("embedding model changed")
        return {
            "mode": "lightrag",
            "initialized": True,
            "fresh": not stale_reasons,
            "built_at": index.built_at if index else "",
            "tier": index.tier if index else "",
            "source_count": len(sources),
            "chunk_count": index.chunk_count if index else 0,
            "entity_count": index.entity_count if index else 0,
            "relation_count": index.relation_count if index else 0,
            "embedding_model": (index.embedding_model if index else "")
            or "bm25-fallback",
            "provider_required": provider_required,
            "provider_ready": provider_ready,
            "stale_reasons": stale_reasons,
        }

    def status(self) -> dict[str, object]:
        """Return a quick-look status payload for ``kb wikigraph status``."""
        try:
            runtime = self.runtime_config
        except ValueError:
            runtime = resolve_wikigraph_config({})
        if runtime.mode == "lightrag":
            return self._lightrag_status(runtime)
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


def _lightrag_to_build_report(
    report: LightGraphBuildReport,
) -> WikiGraphBuildReport:
    """Adapt a LightRAG build report to the classic WikiGraphBuildReport shape."""
    return WikiGraphBuildReport(
        built_at=report.built_at,
        node_count=report.entity_count,
        edge_count=report.relation_count,
        chunk_count=0,
        text_unit_count=report.chunk_count,
        document_count=report.source_count,
        entity_count=report.entity_count,
        community_count=0,
        source_count=report.source_count,
        include_graphrag_export_pages=False,
        include_normalized_text_units=True,
        artifacts=report.artifacts,
        warnings=[f"lightrag tier: {report.tier}", *report.warnings],
    )
