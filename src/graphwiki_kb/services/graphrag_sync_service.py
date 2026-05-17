"""GraphRAG input, runtime, and indexing sync planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from graphwiki_kb.services.file_lock import workspace_lock
from graphwiki_kb.services.graphrag_command_service import (
    GraphRAGCommandError,
    GraphRAGCommandResult,
    GraphRAGCommandService,
)
from graphwiki_kb.services.graphrag_freshness_service import (
    count_source_hash_changes,
    file_digest,
    graph_input_source_hashes,
    graph_runtime_digest,
    source_hashes_from_run,
)
from graphwiki_kb.services.graphrag_input_sync_service import (
    GraphRAGInputSyncResult,
    GraphRAGInputSyncService,
)
from graphwiki_kb.services.graphrag_status_service import (
    GraphRAGIndexRun,
    GraphRAGStatus,
    GraphRAGStatusService,
)
from graphwiki_kb.services.graphrag_workspace_service import GraphRAGWorkspaceService
from graphwiki_kb.services.project_service import ProjectPaths
from graphwiki_kb.services.wiki_priors_service import WikiPriorsService

AUTO_SYNC_METHOD = "auto"
GRAPH_SYNC_METHODS = ("auto", "standard", "fast", "standard-update", "fast-update")
UPDATE_METHODS = {"standard-update", "fast-update"}
FORCED_FULL_METHODS = {
    "auto": "fast",
    "fast-update": "fast",
    "standard-update": "standard",
}


class GraphRAGSyncError(ValueError):
    """Raised when GraphRAG sync cannot decide or run indexing."""


@dataclass(frozen=True)
class GraphRAGSyncDecision:
    """Planner decision describing whether and how the graph should reindex."""

    action: str
    method: str | None
    reason: str
    output_state: str
    input_digest: str
    config_digest: str
    input_changed: bool
    config_changed: bool
    changed_source_count: int | None
    cost_warning: str | None = None
    stale_metadata: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly sync decision."""
        return asdict(self)


@dataclass(frozen=True)
class GraphRAGSyncChangeState:
    """Small, auditable inputs used by the graph sync planner."""

    output_state: str
    input_changed: bool
    config_changed: bool
    changed_source_count: int | None
    stale_metadata: bool


@dataclass(frozen=True)
class GraphRAGSyncResult:
    """Result of syncing GraphRAG inputs and optionally running indexing."""

    input_sync: GraphRAGInputSyncResult
    decision: GraphRAGSyncDecision
    index_run: GraphRAGIndexRun | None = None
    command_result: GraphRAGCommandResult | None = None


class GraphRAGSyncService:
    """Plans and runs GraphRAG input sync, index refreshes, and run recording."""

    def __init__(
        self,
        paths: ProjectPaths,
        workspace_service: GraphRAGWorkspaceService,
        input_sync_service: GraphRAGInputSyncService,
        status_service: GraphRAGStatusService,
        command_service: GraphRAGCommandService,
        wiki_priors_service: WikiPriorsService | None = None,
    ) -> None:
        self.paths = paths
        self.workspace_service = workspace_service
        self.input_sync_service = input_sync_service
        self.status_service = status_service
        self.command_service = command_service
        self.wiki_priors_service = wiki_priors_service
        self.workspace_dir = paths.graph_dir / "graphrag"

    def sync(
        self,
        *,
        method: str = AUTO_SYNC_METHOD,
        force: bool = False,
        dry_run: bool = False,
        cache: bool = True,
        skip_validation: bool = False,
        verbose: bool = False,
        run_index: bool = True,
        preview_only: bool = False,
        allow_missing_sources: bool = False,
        status_callback: Any | None = None,
    ) -> GraphRAGSyncResult:
        """Sync graph input files and run the planned index action when requested."""
        with workspace_lock(self.workspace_dir):
            return self._sync_locked(
                method=method,
                force=force,
                dry_run=dry_run,
                cache=cache,
                skip_validation=skip_validation,
                verbose=verbose,
                run_index=run_index,
                preview_only=preview_only,
                allow_missing_sources=allow_missing_sources,
                status_callback=status_callback,
            )

    def _sync_locked(
        self,
        *,
        method: str,
        force: bool,
        dry_run: bool,
        cache: bool,
        skip_validation: bool,
        verbose: bool,
        run_index: bool,
        preview_only: bool,
        allow_missing_sources: bool,
        status_callback: Any | None,
    ) -> GraphRAGSyncResult:
        wiki_priors_result = (
            self.wiki_priors_service.sync(preview_only=preview_only)
            if self.wiki_priors_service is not None
            else None
        )
        wiki_priors = (
            wiki_priors_result.artifact
            if wiki_priors_result is not None and wiki_priors_result.enabled
            else None
        )
        if self.workspace_service.is_initialized() and not preview_only:
            self.workspace_service.sync_settings(wiki_priors=wiki_priors)

        input_sync = self.input_sync_service.sync(
            preview_only=preview_only,
            allow_missing_sources=allow_missing_sources,
        )
        status = self.status_service.status()
        if preview_only:
            status = replace(
                status,
                input_exists=status.input_exists or input_sync.source_count > 0,
                input_document_count=input_sync.source_count,
            )
        self._require_synced_input(status, allow_planned_input=preview_only)

        input_digest = (
            input_sync.input_digest
            if input_sync.input_digest is not None
            else file_digest(status.input_path)
        )
        settings_text = (
            self.workspace_service.render_settings(wiki_priors=wiki_priors)
            if preview_only and self.workspace_service.is_initialized()
            else None
        )
        extra_prompt_texts: dict[str, str] | None = None
        if preview_only and self.workspace_service.is_initialized():
            prompt_text = self.workspace_service.render_wiki_priors_prompt(
                wiki_priors=wiki_priors,
            )
            if prompt_text is not None:
                extra_prompt_texts = {
                    "prompts/extract_graph_wiki_priors.txt": prompt_text
                }
        config_digest = graph_runtime_digest(
            self.workspace_dir,
            settings_text=settings_text,
            extra_prompt_texts=extra_prompt_texts,
        )
        current_source_hashes = (
            input_sync.source_hashes
            if input_sync.source_hashes is not None
            else graph_input_source_hashes(status.input_path)
        )
        last_successful_run = self.status_service.last_successful_index_run()

        decision = self._decide(
            status=status,
            requested_method=method,
            force=force,
            run_index=run_index,
            input_digest=input_digest,
            config_digest=config_digest,
            current_source_hashes=current_source_hashes,
            last_successful_run=last_successful_run,
        )

        if preview_only or decision.action != "index" or decision.method is None:
            return GraphRAGSyncResult(input_sync=input_sync, decision=decision)

        try:
            command_result = self.command_service.index(
                method=decision.method,
                dry_run=dry_run,
                cache=cache,
                skip_validation=skip_validation,
                verbose=verbose,
                status_callback=status_callback,
            )
        except GraphRAGCommandError as exc:
            if exc.result is not None:
                self.status_service.record_index_run(
                    method=decision.method,
                    dry_run=dry_run,
                    result=exc.result,
                    input_digest=input_digest,
                    config_digest=config_digest,
                    input_source_count=status.input_document_count,
                    source_hashes=current_source_hashes,
                    output_state=decision.output_state,
                )
            else:
                self.status_service.record_index_run(
                    method=decision.method,
                    dry_run=dry_run,
                    result=_failed_before_result(decision.method, str(exc)),
                    input_digest=input_digest,
                    config_digest=config_digest,
                    input_source_count=status.input_document_count,
                    source_hashes=current_source_hashes,
                    output_state=decision.output_state,
                )
            raise

        output_state = decision.output_state
        if not dry_run:
            output_state = graph_output_state(self.status_service.status())

        run = self.status_service.record_index_run(
            method=decision.method,
            dry_run=dry_run,
            result=command_result,
            input_digest=input_digest,
            config_digest=config_digest,
            input_source_count=status.input_document_count,
            source_hashes=current_source_hashes,
            output_state=output_state,
        )
        return GraphRAGSyncResult(
            input_sync=input_sync,
            decision=decision,
            index_run=run,
            command_result=command_result,
        )

    def _decide(
        self,
        *,
        status: GraphRAGStatus,
        requested_method: str,
        force: bool,
        run_index: bool,
        input_digest: str,
        config_digest: str,
        current_source_hashes: dict[str, str],
        last_successful_run: dict[str, Any] | None,
    ) -> GraphRAGSyncDecision:
        change_state = detect_sync_changes(
            status=status,
            input_digest=input_digest,
            config_digest=config_digest,
            current_source_hashes=current_source_hashes,
            last_successful_run=last_successful_run,
        )

        if not run_index:
            return build_sync_decision(
                change_state,
                action="input-only",
                method=None,
                reason="Graph input synced; indexing disabled by --no-index.",
                input_digest=input_digest,
                config_digest=config_digest,
            )

        if status.input_document_count == 0:
            return build_sync_decision(
                change_state,
                action="skip",
                method=None,
                reason=(
                    "Graph input has no documents; add and compile sources before "
                    "indexing."
                ),
                input_digest=input_digest,
                config_digest=config_digest,
            )

        if force:
            method = FORCED_FULL_METHODS.get(requested_method, requested_method)
            return build_sync_decision(
                change_state,
                action="index",
                method=method,
                reason="--force requested a full graph rebuild.",
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=True,
                cost_warning=cost_warning(method),
            )

        if requested_method != AUTO_SYNC_METHOD:
            method = requested_method
            reason = f"Explicit GraphRAG index method requested: {requested_method}."
            if (
                requested_method in UPDATE_METHODS
                and change_state.output_state != "complete"
            ):
                method = FORCED_FULL_METHODS[requested_method]
                reason = (
                    f"Explicit GraphRAG update method {requested_method} requires "
                    f"complete existing output; using full {method} rebuild because "
                    f"graph output is {change_state.output_state}."
                )
            return build_sync_decision(
                change_state,
                action="index",
                method=method,
                reason=reason,
                input_digest=input_digest,
                config_digest=config_digest,
                cost_warning=cost_warning(method),
            )

        if status.last_index_success is False:
            method = (
                "fast-update" if change_state.output_state == "complete" else "fast"
            )
            return build_sync_decision(
                change_state,
                action="index",
                method=method,
                reason="Previous GraphRAG index attempt failed.",
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=(
                    change_state.input_changed
                    or change_state.output_state != "complete"
                ),
                cost_warning=cost_warning(method),
            )

        if change_state.output_state == "missing":
            return build_sync_decision(
                change_state,
                action="index",
                method="fast",
                reason="Graph index output is missing.",
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=True,
                cost_warning=cost_warning("fast"),
            )

        if change_state.output_state == "partial":
            return build_sync_decision(
                change_state,
                action="index",
                method="fast",
                reason="Graph index output is partial or incomplete.",
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=True,
                cost_warning=cost_warning("fast"),
            )

        if change_state.stale_metadata:
            return build_sync_decision(
                change_state,
                action="index",
                method="fast",
                reason="Graph index provenance metadata is missing; rebuilding once.",
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=True,
                config_changed=True,
                cost_warning=cost_warning("fast"),
                stale_metadata=True,
            )

        if change_state.config_changed:
            return build_sync_decision(
                change_state,
                action="index",
                method="fast",
                reason=(
                    "Graph runtime settings, prompts, GraphRAG version, or schema "
                    "changed."
                ),
                input_digest=input_digest,
                config_digest=config_digest,
                config_changed=True,
                cost_warning=cost_warning("fast"),
            )

        if change_state.input_changed:
            return build_sync_decision(
                change_state,
                action="index",
                method="fast-update",
                reason="Normalized source hashes changed.",
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=True,
                config_changed=False,
                cost_warning=cost_warning("fast-update"),
            )

        return build_sync_decision(
            change_state,
            action="skip",
            method=None,
            reason="Graph index is current for the synced sources and runtime config.",
            input_digest=input_digest,
            config_digest=config_digest,
            input_changed=False,
            config_changed=False,
            changed_source_count=0,
        )

    @staticmethod
    def _require_synced_input(
        status: GraphRAGStatus,
        *,
        allow_planned_input: bool = False,
    ) -> None:
        if not status.workspace_initialized:
            raise GraphRAGSyncError(
                "GraphRAG workspace is not initialized. Run `kb init` first."
            )
        if not status.input_exists and not allow_planned_input:
            raise GraphRAGSyncError("GraphRAG input not found. Run `kb update` first.")


def graph_output_state(status: GraphRAGStatus) -> str:
    """Return the coarse GraphRAG output state used by the sync planner."""
    if not status.output_present:
        return "missing"
    if status.output_complete:
        return "complete"
    return "partial"


def detect_sync_changes(
    *,
    status: GraphRAGStatus,
    input_digest: str,
    config_digest: str,
    current_source_hashes: dict[str, str],
    last_successful_run: dict[str, Any] | None,
) -> GraphRAGSyncChangeState:
    """Detect input, settings, output, and metadata changes for planning."""
    output_state = graph_output_state(status)
    if last_successful_run:
        last_input_digest = _optional_str(last_successful_run.get("input_digest"))
        last_config_digest = _optional_str(last_successful_run.get("config_digest"))
        last_source_hashes = source_hashes_from_run(last_successful_run)
        return GraphRAGSyncChangeState(
            output_state=output_state,
            input_changed=last_input_digest != input_digest,
            config_changed=last_config_digest != config_digest,
            changed_source_count=count_source_hash_changes(
                last_source_hashes,
                current_source_hashes,
            ),
            stale_metadata=last_input_digest is None or last_config_digest is None,
        )
    return GraphRAGSyncChangeState(
        output_state=output_state,
        input_changed=False,
        config_changed=False,
        changed_source_count=None,
        stale_metadata=output_state != "missing",
    )


def build_sync_decision(
    change_state: GraphRAGSyncChangeState,
    *,
    action: str,
    method: str | None,
    reason: str,
    input_digest: str,
    config_digest: str,
    input_changed: bool | None = None,
    config_changed: bool | None = None,
    changed_source_count: int | None = None,
    cost_warning: str | None = None,
    stale_metadata: bool | None = None,
) -> GraphRAGSyncDecision:
    """Build a sync decision from a precomputed change-state snapshot."""
    return GraphRAGSyncDecision(
        action=action,
        method=method,
        reason=reason,
        output_state=change_state.output_state,
        input_digest=input_digest,
        config_digest=config_digest,
        input_changed=(
            change_state.input_changed if input_changed is None else input_changed
        ),
        config_changed=(
            change_state.config_changed if config_changed is None else config_changed
        ),
        changed_source_count=(
            change_state.changed_source_count
            if changed_source_count is None
            else changed_source_count
        ),
        cost_warning=cost_warning,
        stale_metadata=(
            change_state.stale_metadata if stale_metadata is None else stale_metadata
        ),
    )


def cost_warning(method: str) -> str:
    """Return the provider-cost warning for a planned index method."""
    if method in UPDATE_METHODS:
        return "Incremental GraphRAG update can incur provider costs."
    return "Full GraphRAG rebuild can incur model and embedding provider costs."


def _failed_before_result(method: str, detail: str) -> GraphRAGCommandResult:
    return GraphRAGCommandResult(
        command=("kb", "internal", "graphrag", "index", "--method", method),
        cwd=Path(),
        returncode=1,
        stdout="",
        stderr=detail,
    )


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
