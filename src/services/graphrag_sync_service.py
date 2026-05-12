from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from src.services.graphrag_command_service import (
    GraphRAGCommandError,
    GraphRAGCommandResult,
    GraphRAGCommandService,
)
from src.services.graphrag_input_sync_service import (
    GraphRAGInputSyncResult,
    GraphRAGInputSyncService,
)
from src.services.graphrag_status_service import (
    GraphRAGIndexRun,
    GraphRAGStatus,
    GraphRAGStatusService,
)
from src.services.graphrag_workspace_service import GraphRAGWorkspaceService
from src.services.project_service import ProjectPaths


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
        return asdict(self)


@dataclass(frozen=True)
class GraphRAGSyncResult:
    input_sync: GraphRAGInputSyncResult
    decision: GraphRAGSyncDecision
    index_run: GraphRAGIndexRun | None = None
    command_result: GraphRAGCommandResult | None = None


class GraphRAGSyncService:
    def __init__(
        self,
        paths: ProjectPaths,
        workspace_service: GraphRAGWorkspaceService,
        input_sync_service: GraphRAGInputSyncService,
        status_service: GraphRAGStatusService,
        command_service: GraphRAGCommandService,
    ) -> None:
        self.paths = paths
        self.workspace_service = workspace_service
        self.input_sync_service = input_sync_service
        self.status_service = status_service
        self.command_service = command_service
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
        status_callback: Any | None = None,
    ) -> GraphRAGSyncResult:
        if self.workspace_service.is_initialized():
            self.workspace_service.sync_settings()

        input_sync = self.input_sync_service.sync()
        status = self.status_service.status()
        self._require_synced_input(status)

        input_digest = file_digest(status.input_path)
        config_digest = graph_runtime_digest(self.workspace_dir)
        current_source_hashes = graph_input_source_hashes(status.input_path)
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
        output_state = graph_output_state(status)
        stale_metadata = False
        config_changed = False
        changed_source_count: int | None = None
        input_changed = False

        if last_successful_run:
            last_input_digest = _optional_str(last_successful_run.get("input_digest"))
            last_config_digest = _optional_str(last_successful_run.get("config_digest"))
            last_source_hashes = _source_hashes_from_run(last_successful_run)
            input_changed = last_input_digest != input_digest
            config_changed = last_config_digest != config_digest
            changed_source_count = count_source_hash_changes(
                last_source_hashes,
                current_source_hashes,
            )
            stale_metadata = last_input_digest is None or last_config_digest is None
        elif output_state != "missing":
            stale_metadata = True

        if not run_index:
            return GraphRAGSyncDecision(
                action="input-only",
                method=None,
                reason="Graph input synced; indexing disabled by --no-index.",
                output_state=output_state,
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=input_changed,
                config_changed=config_changed,
                changed_source_count=changed_source_count,
                stale_metadata=stale_metadata,
            )

        if status.input_document_count == 0:
            return GraphRAGSyncDecision(
                action="skip",
                method=None,
                reason="Graph input has no documents; add and compile sources before indexing.",
                output_state=output_state,
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=input_changed,
                config_changed=config_changed,
                changed_source_count=changed_source_count,
                stale_metadata=stale_metadata,
            )

        if force:
            method = FORCED_FULL_METHODS.get(requested_method, requested_method)
            return GraphRAGSyncDecision(
                action="index",
                method=method,
                reason="--force requested a full graph rebuild.",
                output_state=output_state,
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=True,
                config_changed=config_changed,
                changed_source_count=changed_source_count,
                cost_warning=cost_warning(method),
                stale_metadata=stale_metadata,
            )

        if requested_method != AUTO_SYNC_METHOD:
            return GraphRAGSyncDecision(
                action="index",
                method=requested_method,
                reason=f"Explicit GraphRAG index method requested: {requested_method}.",
                output_state=output_state,
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=input_changed,
                config_changed=config_changed,
                changed_source_count=changed_source_count,
                cost_warning=cost_warning(requested_method),
                stale_metadata=stale_metadata,
            )

        if output_state == "missing":
            return GraphRAGSyncDecision(
                action="index",
                method="fast",
                reason="Graph index output is missing.",
                output_state=output_state,
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=True,
                config_changed=config_changed,
                changed_source_count=changed_source_count,
                cost_warning=cost_warning("fast"),
                stale_metadata=stale_metadata,
            )

        if output_state == "partial":
            return GraphRAGSyncDecision(
                action="index",
                method="fast",
                reason="Graph index output is partial or incomplete.",
                output_state=output_state,
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=True,
                config_changed=config_changed,
                changed_source_count=changed_source_count,
                cost_warning=cost_warning("fast"),
                stale_metadata=stale_metadata,
            )

        if stale_metadata:
            return GraphRAGSyncDecision(
                action="index",
                method="fast",
                reason="Graph index provenance metadata is missing; rebuilding once.",
                output_state=output_state,
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=True,
                config_changed=True,
                changed_source_count=changed_source_count,
                cost_warning=cost_warning("fast"),
                stale_metadata=True,
            )

        if config_changed:
            return GraphRAGSyncDecision(
                action="index",
                method="fast",
                reason="Graph runtime settings or prompts changed.",
                output_state=output_state,
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=input_changed,
                config_changed=True,
                changed_source_count=changed_source_count,
                cost_warning=cost_warning("fast"),
            )

        if input_changed:
            return GraphRAGSyncDecision(
                action="index",
                method="fast-update",
                reason="Normalized source hashes changed.",
                output_state=output_state,
                input_digest=input_digest,
                config_digest=config_digest,
                input_changed=True,
                config_changed=False,
                changed_source_count=changed_source_count,
                cost_warning=cost_warning("fast-update"),
            )

        return GraphRAGSyncDecision(
            action="skip",
            method=None,
            reason="Graph index is current for the synced sources and runtime config.",
            output_state=output_state,
            input_digest=input_digest,
            config_digest=config_digest,
            input_changed=False,
            config_changed=False,
            changed_source_count=0,
        )

    @staticmethod
    def _require_synced_input(status: GraphRAGStatus) -> None:
        if not status.workspace_initialized:
            raise GraphRAGSyncError(
                "GraphRAG workspace is not initialized. Run `kb init` first."
            )
        if not status.input_exists:
            raise GraphRAGSyncError("GraphRAG input not found. Run `kb update` first.")


def graph_output_state(status: GraphRAGStatus) -> str:
    if not status.output_present:
        return "missing"
    required_tables = (
        status.documents_present,
        status.text_units_present,
        status.entities_present,
        status.relationships_present,
        status.communities_present,
        status.community_reports_present,
    )
    if all(required_tables):
        return "complete"
    return "partial"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def graph_runtime_digest(workspace_dir: Path) -> str:
    digest = hashlib.sha256()
    _digest_file(digest, workspace_dir / "settings.yaml", "settings.yaml")
    prompt_dir = workspace_dir / "prompts"
    if prompt_dir.exists():
        for path in sorted(prompt_dir.rglob("*.txt")):
            _digest_file(digest, path, path.relative_to(workspace_dir).as_posix())
    return digest.hexdigest()


def graph_input_source_hashes(input_path: Path) -> dict[str, str]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    records: list[Any]
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = []
        for key in ("sources", "documents"):
            value = payload.get(key)
            if isinstance(value, list):
                records.extend(value)
    else:
        records = []

    source_hashes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        source_id = _optional_str(record.get("source_id") or record.get("id"))
        source_hash = _optional_str(record.get("source_hash"))
        if source_id and source_hash:
            source_hashes[source_id] = source_hash
    return source_hashes


def count_source_hash_changes(
    previous: dict[str, str] | None,
    current: dict[str, str],
) -> int | None:
    if previous is None:
        return None
    changed = 0
    for source_id, source_hash in current.items():
        if previous.get(source_id) != source_hash:
            changed += 1
    removed = set(previous) - set(current)
    return changed + len(removed)


def cost_warning(method: str) -> str:
    if method in UPDATE_METHODS:
        return "Incremental GraphRAG update can incur provider costs."
    return "Full GraphRAG rebuild can incur model and embedding provider costs."


def _digest_file(digest: Any, path: Path, label: str) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    if path.exists():
        digest.update(path.read_bytes())
    digest.update(b"\0")


def _source_hashes_from_run(run: dict[str, Any]) -> dict[str, str] | None:
    payload = run.get("source_hashes")
    if not isinstance(payload, dict):
        return None
    source_hashes: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str):
            source_hashes[key] = value
    return source_hashes


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
