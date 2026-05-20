"""Coordinates source ingest, compile, concept refresh, search, and GraphRAG sync."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.compile_service import CompileResult, CompileService
from graphwiki_kb.services.concept_service import (
    ConceptGenerationResult,
    ConceptService,
)
from graphwiki_kb.services.config_service import (
    concept_generation_enabled,
    concept_provider_backed_enabled,
    resolve_graph_config,
    resolve_wikigraph_config,
)
from graphwiki_kb.services.graphrag_defaults import env_file_has_key
from graphwiki_kb.services.graphrag_sync_service import GraphRAGSyncResult
from graphwiki_kb.services.graphrag_wiki_export_service import GraphRAGWikiExportResult
from graphwiki_kb.services.ingest_service import IngestService
from graphwiki_kb.services.search_service import SearchService
from graphwiki_kb.wikigraph.index_builder import WikiGraphBuildResult

GRAPH_INDEX_METHODS = ("auto", "standard", "fast", "standard-update", "fast-update")


@dataclass
class UpdateOptions:
    """Options accepted by the high-level update workflow."""

    source_paths: tuple[Path, ...] = ()
    force: bool = False
    resume: bool = False
    no_graph: bool = False
    graph_only: bool = False
    allow_partial: bool = False
    concepts: bool | None = None
    graph_method: str = "auto"
    no_wikigraph: bool = False
    wikigraph_include_graphrag_export_pages: bool = False
    export_wikigraph_artifacts: bool = False


@dataclass
class IngestSummary:
    """Short CLI-facing summary for one ingest input."""

    path: Path
    is_dir: bool
    created_count: int = 0
    message: str = ""


@dataclass
class UpdateResult:
    """Combined result of a full or graph-only update run."""

    ingest_summaries: list[IngestSummary] = field(default_factory=list)
    compile_result: CompileResult | None = None
    concept_result: ConceptGenerationResult | None = None
    concepts_skipped: bool = False
    concepts_skip_reason: str = ""
    search_refreshed: bool = False
    search_warning: str = ""
    graph_result: GraphUpdateResult | None = None
    wikigraph_result: WikiGraphUpdateResult | None = None

    @property
    def ok(self) -> bool:
        """Return true when update produced legacy or GraphRAG work."""
        return (
            self.compile_result is not None
            or self.graph_result is not None
            or self.wikigraph_result is not None
        )


@dataclass
class WikiGraphUpdateResult:
    """WikiGraphRAG index build result for update runs."""

    skipped: bool = False
    skip_reason: str = ""
    build: WikiGraphBuildResult | None = None
    exported_artifacts: tuple[str, ...] = ()
    warning: str = ""


@dataclass
class GraphUpdateResult:
    """GraphRAG-specific result for update preflight, indexing, and export."""

    skipped: bool = False
    skip_reason: str = ""
    initialized: bool = False
    preflight_result: GraphRAGSyncResult | None = None
    sync_result: GraphRAGSyncResult | None = None
    export_result: GraphRAGWikiExportResult | None = None
    active_output_dir: str | None = None
    warning: str = ""


CompileProgressCallback = Callable[[RawSourceRecord], None]
CompileProgressFactory = Callable[
    [int], AbstractContextManager[CompileProgressCallback]
]


class UpdateService:
    """Runs the project update workflow across all managed subsystems."""

    def __init__(
        self,
        *,
        ingest_service: IngestService,
        compile_service: CompileService,
        concept_service: ConceptService,
        search_service: SearchService,
        config: dict[str, Any],
        graphrag_workspace_service: Any | None = None,
        graphrag_sync_service: Any | None = None,
        graphrag_wiki_export_service: Any | None = None,
        wikigraph_index_service: Any | None = None,
    ) -> None:
        self._ingest = ingest_service
        self._compile = compile_service
        self._concepts = concept_service
        self._search = search_service
        self._config = config
        self._graphrag_workspace = graphrag_workspace_service
        self._graphrag_sync = graphrag_sync_service
        self._graphrag_wiki_export = graphrag_wiki_export_service
        self._wikigraph_index = wikigraph_index_service

    def preflight(self) -> None:
        """Raise if provider is missing or broken."""
        provider_name = self._config.get("provider", {}).get("name")
        if not provider_name:
            raise UpdatePreflightError(
                "Provider is not configured, so the KB cannot be updated yet.\n"
                "Next: add a provider section to kb.config.yaml and set the "
                "matching API key environment variable."
            )
        actual_provider = getattr(self._compile, "provider", None)
        if actual_provider is not None and hasattr(actual_provider, "ensure_available"):
            actual_provider.ensure_available()

    def run(
        self,
        options: UpdateOptions,
        *,
        ingest_progress: Callable[[Path], None] | None = None,
        compile_progress: CompileProgressCallback | None = None,
        compile_progress_factory: CompileProgressFactory | None = None,
        graph_status_callback: Callable[[str], None] | None = None,
    ) -> UpdateResult:
        """Run source ingest, compile, concept/search refresh, and GraphRAG sync."""
        if options.force and options.resume:
            raise ValueError("--resume cannot be combined with --force.")
        if options.no_graph and options.graph_only:
            raise ValueError("--graph-only cannot be combined with --no-graph.")
        if options.graph_only and options.source_paths:
            raise ValueError("--graph-only cannot ingest new source paths.")
        if options.graph_only and options.resume:
            raise ValueError("--graph-only cannot resume legacy compile runs.")
        if options.graph_method not in GRAPH_INDEX_METHODS:
            supported = ", ".join(GRAPH_INDEX_METHODS)
            raise ValueError(
                f"Unsupported GraphRAG index method '{options.graph_method}'. "
                f"Use one of: {supported}."
            )

        result = UpdateResult()
        if options.graph_only:
            result.graph_result = self._run_graph_sync(
                options, status_callback=graph_status_callback
            )
            result.wikigraph_result = self._run_wikigraph_update(
                options, status_callback=graph_status_callback
            )
            return result

        self.preflight()

        # Ingest phase
        for source_path in options.source_paths:
            summary = self._ingest_one(source_path, progress=ingest_progress)
            result.ingest_summaries.append(summary)

        # Compile phase — plan AFTER ingestion so new sources are included.
        plan = self._compile.plan(force=options.force, resume=options.resume)

        if compile_progress_factory is not None:
            with compile_progress_factory(plan.pending_count) as progress_cb:
                result.compile_result = self._compile.compile(
                    force=options.force,
                    resume=options.resume,
                    progress_callback=progress_cb,
                )
        else:
            result.compile_result = self._compile.compile(
                force=options.force,
                resume=options.resume,
                progress_callback=compile_progress,
            )

        # Legacy LLM-wiki concept pages are opt-in now that GraphRAG is the
        # default cross-document retrieval layer.
        if self._should_generate_concepts(options):
            result.concept_result = self._concepts.generate(
                use_provider=concept_provider_backed_enabled(self._config)
            )
            # Compile writes the index before concepts are regenerated. Refresh it
            # here so wiki/index.md and wiki/_index.json reflect the current set.
            self._compile.refresh_index()
        else:
            result.concepts_skipped = True
            result.concepts_skip_reason = (
                "disabled by default; set concepts.enabled: true or pass "
                "--concepts to refresh legacy concept pages"
            )
            result.concept_result = self._concepts.remove_generated_pages()
            if (
                result.concept_result.removed_paths
                or result.concept_result.updated_source_paths
            ):
                self._compile.refresh_index()

        # Search refresh
        result.search_refreshed = self._search.refresh(force=True)
        if not result.search_refreshed:
            result.search_warning = (
                "Search index refresh skipped because SQLite FTS5 is unavailable; "
                "`kb find` will scan markdown files."
            )

        result.graph_result = self._run_graph_sync(
            options, status_callback=graph_status_callback
        )
        result.wikigraph_result = self._run_wikigraph_update(
            options, status_callback=graph_status_callback
        )

        return result

    def _run_wikigraph_update(
        self,
        options: UpdateOptions,
        *,
        status_callback: Callable[[str], None] | None = None,
    ) -> WikiGraphUpdateResult:
        runtime = resolve_wikigraph_config(self._config)
        if options.no_wikigraph:
            return WikiGraphUpdateResult(
                skipped=True, skip_reason="--no-wikigraph requested."
            )
        if not runtime.enabled:
            return WikiGraphUpdateResult(
                skipped=True,
                skip_reason="wikigraph.enabled is false in kb.config.yaml.",
            )
        if self._wikigraph_index is None:
            return WikiGraphUpdateResult(
                skipped=True, skip_reason="WikiGraphRAG services unavailable."
            )
        try:
            if status_callback is not None:
                status_callback("building WikiGraphRAG index")
            build = self._wikigraph_index.build(
                include_graphrag_export_pages=(
                    options.wikigraph_include_graphrag_export_pages
                    or runtime.include_graphrag_export_pages
                )
            )
            exported: tuple[str, ...] = ()
            if options.export_wikigraph_artifacts:
                if status_callback is not None:
                    status_callback("exporting WikiGraphRAG artifacts")
                exported = tuple(self._wikigraph_index.export_artifacts())
            return WikiGraphUpdateResult(build=build, exported_artifacts=exported)
        except ImportError as exc:
            message = (
                f"WikiGraphRAG index skipped: {exc}. "
                "Install extras with: poetry install -E wikigraph"
            )
            if options.allow_partial:
                return WikiGraphUpdateResult(skipped=True, warning=message)
            raise ValueError(message) from exc
        except ValueError as exc:
            message = str(exc)
            if "No wiki pages found" in message:
                return WikiGraphUpdateResult(skipped=True, skip_reason=message)
            if options.allow_partial:
                return WikiGraphUpdateResult(skipped=True, warning=message)
            raise ValueError(message) from exc
        except Exception as exc:
            message = f"WikiGraphRAG index build failed: {exc}"
            if options.allow_partial:
                return WikiGraphUpdateResult(skipped=True, warning=message)
            raise ValueError(message) from exc

    # ------------------------------------------------------------------

    def _should_generate_concepts(self, options: UpdateOptions) -> bool:
        if options.concepts is not None:
            return options.concepts
        return concept_generation_enabled(self._config)

    def _ingest_one(
        self,
        source_path: Path,
        *,
        progress: Callable[[Path], None] | None = None,
    ) -> IngestSummary:
        if source_path.is_dir():
            dir_result = self._ingest.ingest_directory(
                source_path,
                progress_callback=progress,
            )
            return IngestSummary(
                path=source_path,
                is_dir=True,
                created_count=dir_result.created_count,
            )
        else:
            file_result = self._ingest.ingest_path(source_path)
            return IngestSummary(
                path=source_path,
                is_dir=False,
                created_count=1 if file_result.created else 0,
                message=(
                    ""
                    if file_result.created
                    else f"Already present: {source_path.name}"
                ),
            )

    def _run_graph_sync(
        self,
        options: UpdateOptions,
        *,
        status_callback: Callable[[str], None] | None = None,
    ) -> GraphUpdateResult:
        if options.no_graph:
            return GraphUpdateResult(skipped=True, skip_reason="--no-graph requested.")
        if not (
            self._graphrag_workspace
            and self._graphrag_sync
            and self._graphrag_wiki_export
        ):
            return GraphUpdateResult(
                skipped=True, skip_reason="Graph services unavailable."
            )
        if not isinstance(self._config.get("graph"), dict):
            return GraphUpdateResult(
                skipped=True, skip_reason="Graph config not configured."
            )

        result = GraphUpdateResult()
        if not self._graphrag_workspace.is_initialized():
            self._graphrag_workspace.ensure_workspace()
            result.initialized = True

        try:
            preflight = self._graphrag_sync.sync(
                method=options.graph_method,
                force=options.force,
                dry_run=True,
                preview_only=True,
                allow_missing_sources=not options.graph_only,
            )
        except Exception as exc:
            message = f"Graph preflight failed: {exc}"
            if options.allow_partial:
                return GraphUpdateResult(skipped=True, warning=message)
            raise ValueError(message) from exc
        result.preflight_result = preflight
        if preflight.input_sync.skipped_sources:
            result.warning = _missing_source_warning(
                preflight.input_sync.skipped_sources
            )

        decision = preflight.decision
        if decision.action != "index" or decision.method is None:
            result.skipped = True
            result.skip_reason = decision.reason
            if decision.output_state == "complete":
                try:
                    result.sync_result = self._graphrag_sync.sync(
                        method=options.graph_method,
                        force=False,
                        dry_run=False,
                        run_index=False,
                        preview_only=False,
                        allow_missing_sources=not options.graph_only,
                        status_callback=status_callback,
                    )
                    if status_callback is not None:
                        status_callback("exporting graph pages")
                    result.export_result = self._graphrag_wiki_export.export_wiki()
                    result.active_output_dir = self._active_graph_output_dir()
                except Exception as exc:
                    message = f"Graph sync/export failed after skipped index: {exc}"
                    if options.allow_partial:
                        result.warning = message
                        return result
                    raise ValueError(message) from exc
            return result

        try:
            missing_keys = self._missing_graph_credentials()
        except ValueError as exc:
            message = f"Graph index skipped because graph config is invalid: {exc}"
            if options.allow_partial:
                result.skipped = True
                result.warning = message
                return result
            raise ValueError(message) from exc
        if missing_keys:
            message = (
                "Graph index skipped because provider credentials are missing: "
                + ", ".join(missing_keys)
            )
            if options.graph_only:
                raise ValueError(message)
            result.skipped = True
            result.warning = message
            return result

        try:
            if status_callback is not None:
                status_callback(f"running {decision.method} graph index")
            result.sync_result = self._graphrag_sync.sync(
                method=decision.method,
                force=options.force,
                dry_run=False,
                run_index=True,
                allow_missing_sources=not options.graph_only,
                status_callback=status_callback,
            )
            if status_callback is not None:
                status_callback("exporting graph pages")
            result.export_result = self._graphrag_wiki_export.export_wiki()
            result.active_output_dir = self._active_graph_output_dir()
        except Exception as exc:
            message = f"Graph index/export failed: {exc}"
            if options.allow_partial:
                result.warning = message
                return result
            raise ValueError(message) from exc
        return result

    def _active_graph_output_dir(self) -> str | None:
        if self._graphrag_sync is None:
            return None
        status_service = getattr(self._graphrag_sync, "status_service", None)
        if status_service is None:
            return None
        active_output_dir = status_service.active_output_dir()
        if active_output_dir is None:
            return None
        try:
            return (
                active_output_dir.resolve()
                .relative_to(self._graphrag_sync.paths.root)
                .as_posix()
            )
        except ValueError:
            return active_output_dir.as_posix()

    def _missing_graph_credentials(self) -> list[str]:
        if self._graphrag_sync is None:
            raise ValueError("Graph sync service unavailable.")
        graph_config = resolve_graph_config(self._config)
        dot_env = self._graphrag_sync.workspace_dir / ".env"
        missing: list[str] = []
        for key in dict.fromkeys(
            (graph_config.api_key_env, graph_config.embedding_api_key_env)
        ):
            if os.environ.get(key) or env_file_has_key(dot_env, key):
                continue
            missing.append(key)
        return missing


def _missing_source_warning(skipped_sources: tuple[str, ...]) -> str:
    count = len(skipped_sources)
    examples = "; ".join(skipped_sources[:3])
    suffix = "" if count <= 3 else f"; {count - 3} more"
    return (
        f"Graph input skipped {count} source(s) with missing normalized artifacts: "
        f"{examples}{suffix}. Run `kb update --force` or re-ingest the source files "
        "to restore them."
    )


class UpdatePreflightError(Exception):
    """Error raised for update preflight failures.

    Attributes:
        See annotated class attributes for stored values.
    """
