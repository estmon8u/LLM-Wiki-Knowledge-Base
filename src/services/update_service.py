"""Update service service behavior for the knowledge-base workflow.

This module belongs to `src.services.update_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""


from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from src.services.config_service import resolve_graph_config
from src.services.compile_service import CompileResult, CompileService
from src.services.concept_service import ConceptGenerationResult, ConceptService
from src.services.graphrag_defaults import env_file_has_key
from src.services.graphrag_sync_service import GraphRAGSyncResult
from src.services.graphrag_wiki_export_service import GraphRAGWikiExportResult
from src.services.ingest_service import IngestResult, IngestService
from src.services.search_service import SearchService


@dataclass
class UpdateOptions:
    """Represents update options behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    source_paths: tuple[Path, ...] = ()
    force: bool = False
    resume: bool = False
    no_graph: bool = False


@dataclass
class IngestSummary:
    """Represents ingest summary behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    path: Path
    is_dir: bool
    created_count: int = 0
    message: str = ""


@dataclass
class UpdateResult:
    """Stores update result data.

    Attributes:
        See annotated class attributes for stored values.
    """

    ingest_summaries: list[IngestSummary] = field(default_factory=list)
    compile_result: Optional[CompileResult] = None
    concept_result: Optional[ConceptGenerationResult] = None
    search_refreshed: bool = False
    graph_result: Optional["GraphUpdateResult"] = None

    @property
    def ok(self) -> bool:
        """Ok.

        Returns:
            bool produced by the operation.
        """
        return self.compile_result is not None


@dataclass
class GraphUpdateResult:
    """Stores graph update result data.

    Attributes:
        See annotated class attributes for stored values.
    """

    skipped: bool = False
    skip_reason: str = ""
    initialized: bool = False
    preflight_result: Optional[GraphRAGSyncResult] = None
    sync_result: Optional[GraphRAGSyncResult] = None
    export_result: Optional[GraphRAGWikiExportResult] = None
    warning: str = ""


class UpdateService:
    """Coordinates update operations.

    Attributes:
        See annotated class attributes for stored values.
    """

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
    ) -> None:
        self._ingest = ingest_service
        self._compile = compile_service
        self._concepts = concept_service
        self._search = search_service
        self._config = config
        self._graphrag_workspace = graphrag_workspace_service
        self._graphrag_sync = graphrag_sync_service
        self._graphrag_wiki_export = graphrag_wiki_export_service

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
        compile_progress: Callable[[str], None] | None = None,
        compile_progress_factory: (
            Callable[[int], contextmanager[Iterator[Callable]]] | None
        ) = None,
        graph_status_callback: Callable[[str], None] | None = None,
    ) -> UpdateResult:
        """Run.

        Args:
            options: Options value used by the operation.
            ingest_progress: Ingest progress value used by the operation.
            compile_progress: Compile progress value used by the operation.
            compile_progress_factory: Compile progress factory value used by the operation.
            graph_status_callback: Graph status callback value used by the operation.

        Returns:
            UpdateResult produced by the operation.
        """
        if options.force and options.resume:
            raise ValueError("--resume cannot be combined with --force.")

        self.preflight()
        result = UpdateResult()

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

        # Concepts phase
        result.concept_result = self._concepts.generate()

        # Compile writes the index before concepts are regenerated. Refresh it here so
        # wiki/index.md and wiki/_index.json reflect the current concept set.
        self._compile.refresh_index()

        # Search refresh
        self._search.refresh(force=True)
        result.search_refreshed = True

        result.graph_result = self._run_graph_sync(
            options, status_callback=graph_status_callback
        )

        return result

    # ------------------------------------------------------------------

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
                message=""
                if file_result.created
                else f"Already present: {source_path.name}",
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
                method="auto",
                force=options.force,
                dry_run=True,
                preview_only=True,
            )
        except Exception as exc:
            return GraphUpdateResult(
                skipped=True,
                warning=f"Graph preflight failed: {exc}",
            )
        result.preflight_result = preflight

        decision = preflight.decision
        if decision.action != "index" or decision.method is None:
            result.skipped = True
            result.skip_reason = decision.reason
            return result

        try:
            missing_keys = self._missing_graph_credentials()
        except ValueError as exc:
            result.skipped = True
            result.warning = (
                f"Graph index skipped because graph config is invalid: {exc}"
            )
            return result
        if missing_keys:
            result.skipped = True
            result.warning = (
                "Graph index skipped because provider credentials are missing: "
                + ", ".join(missing_keys)
            )
            return result

        try:
            if status_callback is not None:
                status_callback(f"running {decision.method} graph index")
            result.sync_result = self._graphrag_sync.sync(
                method=decision.method,
                force=options.force,
                dry_run=False,
                run_index=True,
                status_callback=status_callback,
            )
            if status_callback is not None:
                status_callback("exporting graph pages")
            result.export_result = self._graphrag_wiki_export.export_wiki()
        except Exception as exc:
            result.warning = f"Graph index/export failed: {exc}"
        return result

    def _missing_graph_credentials(self) -> list[str]:
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


class UpdatePreflightError(Exception):
    """Error raised for update preflight failures.

    Attributes:
        See annotated class attributes for stored values.
    """

    pass
