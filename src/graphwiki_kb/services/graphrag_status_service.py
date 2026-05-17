"""GraphRAG status tracking for the knowledge-base workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from graphwiki_kb.services.file_lock import file_lock, workspace_lock
from graphwiki_kb.services.graphrag_command_service import GraphRAGCommandResult
from graphwiki_kb.services.graphrag_freshness_service import evaluate_graph_freshness
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    utc_now_iso,
)

GRAPH_OUTPUT_TABLES: dict[str, tuple[str, ...]] = {
    "documents": ("documents", "create_final_documents"),
    "text_units": ("text_units", "create_final_text_units"),
    "entities": ("entities", "create_final_entities"),
    "relationships": ("relationships", "create_final_relationships"),
    "communities": ("communities", "create_final_communities"),
    "community_reports": ("community_reports", "create_final_community_reports"),
}
VECTOR_STORE_TABLE_HINTS = ("entity", "document", "text_unit", "text-unit")
QUERY_REQUIRED_TABLES: dict[str, tuple[str, ...]] = {
    "basic": ("documents", "text_units"),
    "global": ("communities", "community_reports"),
    "local": ("documents", "text_units", "entities", "relationships"),
    "drift": (
        "documents",
        "text_units",
        "entities",
        "relationships",
        "communities",
        "community_reports",
    ),
}
QUERY_REQUIRES_VECTOR_STORE = {"basic", "local", "drift"}


@dataclass(frozen=True)
class GraphRAGIndexRun:
    """Persisted metadata for one GraphRAG index attempt."""

    run_id: str
    created_at: str
    method: str
    dry_run: bool
    success: bool
    returncode: int
    command: tuple[str, ...]
    stdout_tail: str
    stderr_tail: str
    input_digest: str | None = None
    input_hash: str | None = None
    config_digest: str | None = None
    input_source_count: int | None = None
    source_hashes: dict[str, str] | None = None
    output_state: str | None = None
    active_output_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly run record."""
        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload


@dataclass(frozen=True)
class GraphRAGStatus:
    """Snapshot of GraphRAG workspace, input, output, and freshness state."""

    workspace_dir: Path
    settings_path: Path
    input_path: Path
    output_dir: Path
    workspace_initialized: bool
    input_exists: bool
    input_document_count: int
    output_present: bool
    documents_present: bool
    text_units_present: bool
    entities_present: bool
    relationships_present: bool
    communities_present: bool
    community_reports_present: bool
    last_index_run_id: str | None
    last_index_run_at: str | None
    last_index_method: str | None
    last_index_success: bool | None
    next_action: str
    last_index_input_digest: str | None = None
    last_index_input_hash: str | None = None
    last_index_config_digest: str | None = None
    last_index_input_source_count: int | None = None
    last_index_output_state: str | None = None
    input_updated_at: str | None = None
    output_updated_at: str | None = None
    wiki_export_present: bool = False
    wiki_export_updated_at: str | None = None
    document_count: int | None = None
    text_unit_count: int | None = None
    entity_count: int | None = None
    relationship_count: int | None = None
    community_count: int | None = None
    community_report_count: int | None = None
    active_output_dir: Path | None = None
    vector_store_path: Path | None = None
    vector_store_exists: bool = False
    vector_store_readable: bool = False
    vector_store_state: str = "missing"
    current_input_digest: str | None = None
    current_config_digest: str | None = None
    graph_freshness_state: str = "unknown"
    graph_stale_reasons: tuple[str, ...] = ()

    @property
    def missing_tables(self) -> list[str]:
        """Return required GraphRAG output artifacts missing from active output."""
        missing = [
            name
            for name, present in (
                ("documents", self.documents_present),
                ("text_units", self.text_units_present),
                ("entities", self.entities_present),
                ("relationships", self.relationships_present),
                ("communities", self.communities_present),
                ("community_reports", self.community_reports_present),
            )
            if not present
        ]
        if not self.vector_store_readable:
            missing.append("vector_store")
        return missing

    @property
    def state(self) -> str:
        """Return a normalized machine-readable graph index state."""
        if self.last_index_success is False:
            return "failed"
        if not self.workspace_initialized:
            return "missing"
        if not self.output_present:
            return "missing"
        if not self.output_complete:
            return "partial"
        if iso_timestamp_after(self.input_updated_at, self.output_updated_at):
            return "stale"
        if self.graph_freshness_state in {"stale", "missing-metadata"}:
            return "stale"
        return "complete"

    @property
    def output_complete(self) -> bool:
        """Return True only when all required GraphRAG output artifacts are present."""
        return (
            self.output_present
            and self.documents_present
            and self.text_units_present
            and self.entities_present
            and self.relationships_present
            and self.communities_present
            and self.community_reports_present
            and self.vector_store_readable
        )

    def to_dict(self, project_root: Path) -> dict[str, Any]:
        """Return a JSON-friendly status payload with project-relative paths."""
        payload = asdict(self)
        for key in (
            "workspace_dir",
            "settings_path",
            "input_path",
            "output_dir",
            "active_output_dir",
            "vector_store_path",
        ):
            path = payload[key]
            payload[key] = (
                self._relative_to_project(path, project_root)
                if isinstance(path, Path)
                else None
            )
        payload["output_complete"] = self.output_complete
        payload["missing_tables"] = self.missing_tables
        payload["state"] = self.state
        return payload

    @staticmethod
    def _relative_to_project(path: Path, project_root: Path) -> str:
        try:
            return path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            return path.as_posix()


class GraphRAGStatusService:
    """Builds GraphRAG status snapshots and persists index-run history."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.workspace_dir = paths.graph_dir / "graphrag"
        self.settings_path = self.workspace_dir / "settings.yaml"
        self.input_path = self.workspace_dir / "input" / "sources.json"
        self.output_dir = self.workspace_dir / "output"
        self.runs_file = paths.graph_dir / "runs" / "graph_index_runs.json"

    def status(self) -> GraphRAGStatus:
        """Return the current GraphRAG status under a workspace read lock."""
        with workspace_lock(self.workspace_dir):
            return self._status_unlocked()

    def _status_unlocked(self) -> GraphRAGStatus:
        """Build status while the workspace read lock is held."""
        runs = self._load_runs()
        last_run = runs[-1] if runs else None
        active_output_dir = self._active_output_dir()
        table_paths = self._table_paths(active_output_dir)
        tables = {name: path is not None for name, path in table_paths.items()}
        vector_store_path = self._vector_store_path(active_output_dir)
        vector_store_exists = bool(vector_store_path and vector_store_path.exists())
        vector_store_state = self._vector_store_state(vector_store_path)
        vector_store_readable = vector_store_state == "ready"
        output_complete = all(tables.values()) and vector_store_readable
        table_counts = {
            name: self._table_row_count(path) if path is not None else None
            for name, path in table_paths.items()
        }
        output_present = active_output_dir is not None
        input_document_count = self._input_document_count()
        workspace_initialized = self.settings_path.exists()
        input_exists = self.input_path.exists()
        freshness = evaluate_graph_freshness(
            input_path=self.input_path,
            workspace_dir=self.workspace_dir,
            last_successful_run=self.last_successful_index_run(),
        )
        wiki_export_path = self.paths.wiki_dir / "graph" / "index.md"
        return GraphRAGStatus(
            workspace_dir=self.workspace_dir,
            settings_path=self.settings_path,
            input_path=self.input_path,
            output_dir=self.output_dir,
            workspace_initialized=workspace_initialized,
            input_exists=input_exists,
            input_document_count=input_document_count,
            output_present=output_present,
            documents_present=tables["documents"],
            text_units_present=tables["text_units"],
            entities_present=tables["entities"],
            relationships_present=tables["relationships"],
            communities_present=tables["communities"],
            community_reports_present=tables["community_reports"],
            last_index_run_id=last_run.get("run_id") if last_run else None,
            last_index_run_at=last_run.get("created_at") if last_run else None,
            last_index_method=last_run.get("method") if last_run else None,
            last_index_success=last_run.get("success") if last_run else None,
            next_action=self._next_action(
                workspace_initialized=workspace_initialized,
                input_exists=input_exists,
                input_document_count=input_document_count,
                output_present=output_present,
                output_complete=output_complete,
                vector_store_exists=vector_store_exists,
                vector_store_readable=vector_store_readable,
                freshness_state=freshness.state,
                stale_reasons=freshness.reasons,
                last_run=last_run,
            ),
            last_index_input_digest=last_run.get("input_digest") if last_run else None,
            last_index_input_hash=last_run.get("input_hash") if last_run else None,
            last_index_config_digest=(
                last_run.get("config_digest") if last_run else None
            ),
            last_index_input_source_count=(
                last_run.get("input_source_count") if last_run else None
            ),
            last_index_output_state=last_run.get("output_state") if last_run else None,
            input_updated_at=self._file_mtime_iso(self.input_path),
            output_updated_at=self._latest_parquet_mtime_iso(active_output_dir),
            wiki_export_present=wiki_export_path.exists(),
            wiki_export_updated_at=self._file_mtime_iso(wiki_export_path),
            document_count=table_counts["documents"],
            text_unit_count=table_counts["text_units"],
            entity_count=table_counts["entities"],
            relationship_count=table_counts["relationships"],
            community_count=table_counts["communities"],
            community_report_count=table_counts["community_reports"],
            active_output_dir=active_output_dir,
            vector_store_path=vector_store_path,
            vector_store_exists=vector_store_exists,
            vector_store_readable=vector_store_readable,
            vector_store_state=vector_store_state,
            current_input_digest=freshness.current_input_digest,
            current_config_digest=freshness.current_config_digest,
            graph_freshness_state=freshness.state,
            graph_stale_reasons=freshness.reasons,
        )

    def record_index_run(
        self,
        *,
        method: str,
        dry_run: bool,
        result: GraphRAGCommandResult,
        input_digest: str | None = None,
        config_digest: str | None = None,
        input_source_count: int | None = None,
        source_hashes: dict[str, str] | None = None,
        output_state: str | None = None,
    ) -> GraphRAGIndexRun:
        """Append one GraphRAG index run to the persisted history."""
        created_at = utc_now_iso()
        active_output_dir = None
        if result.returncode == 0 and not dry_run:
            resolved_active_output = self._active_output_dir(prefer_recorded=False)
            if resolved_active_output is not None:
                active_output_dir = self._relative_to_project_root(
                    resolved_active_output
                )
        record = GraphRAGIndexRun(
            run_id=created_at.replace(":", "").replace("+", "Z"),
            created_at=created_at,
            method=method,
            dry_run=dry_run,
            success=result.returncode == 0,
            returncode=result.returncode,
            command=result.command,
            stdout_tail=_tail(result.stdout),
            stderr_tail=_tail(result.stderr),
            input_digest=input_digest,
            input_hash=input_digest,
            config_digest=config_digest,
            input_source_count=input_source_count,
            source_hashes=source_hashes,
            output_state=output_state,
            active_output_dir=active_output_dir,
        )
        with file_lock(self.runs_file):
            runs = self._load_runs()
            runs.append(record.to_dict())
            atomic_write_text(
                self.runs_file,
                json.dumps(runs, indent=2, sort_keys=True) + "\n",
            )
        return record

    def last_successful_index_run(self) -> dict[str, Any] | None:
        """Return the newest successful non-dry-run index record, if any."""
        for run in reversed(self._load_runs()):
            if run.get("success") is True and run.get("dry_run") is False:
                return run
        return None

    def table_path(self, table_name: str) -> Path | None:
        """Return the active output table path for a known GraphRAG table name."""
        tokens = GRAPH_OUTPUT_TABLES.get(table_name)
        if tokens is None:
            return None
        return self._table_path(self._active_output_dir(), *tokens)

    def active_output_dir(self) -> Path | None:
        """Return the authoritative GraphRAG output directory for reads."""
        return self._active_output_dir()

    def _input_document_count(self) -> int:
        if not self.input_path.exists():
            return 0
        try:
            payload = json.loads(self.input_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, dict):
            for key in ("sources", "documents"):
                value = payload.get(key)
                if isinstance(value, list):
                    return len(value)
        return 0

    def _load_settings(self) -> dict[str, Any]:
        if not self.settings_path.exists():
            return {}
        try:
            payload = yaml.safe_load(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _active_output_dir(self, *, prefer_recorded: bool = True) -> Path | None:
        if not self.output_dir.exists():
            return None
        if prefer_recorded:
            last_run = self.last_successful_index_run()
            recorded_output_dir = (
                self._project_relative_path(last_run.get("active_output_dir"))
                if last_run
                else None
            )
            if (
                recorded_output_dir is not None
                and recorded_output_dir.exists()
                and self._output_dir_complete(recorded_output_dir)
            ):
                return recorded_output_dir
        latest_by_parent: dict[Path, float] = {}
        for path in self.output_dir.rglob("*.parquet"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            parent = path.parent
            latest_by_parent[parent] = max(latest_by_parent.get(parent, 0.0), mtime)
        if not latest_by_parent:
            return None
        complete_candidates = {
            parent: mtime
            for parent, mtime in latest_by_parent.items()
            if self._output_dir_complete(parent)
        }
        candidates = complete_candidates or latest_by_parent
        return max(candidates.items(), key=lambda item: (item[1], str(item[0])))[0]

    def _output_dir_complete(self, output_dir: Path) -> bool:
        table_paths = self._table_paths(output_dir)
        return all(table_paths.values()) and (
            self._vector_store_state(self._vector_store_path(output_dir)) == "ready"
        )

    def _table_paths(self, output_dir: Path | None) -> dict[str, Path | None]:
        if output_dir is None or not output_dir.exists():
            return {name: None for name in GRAPH_OUTPUT_TABLES}
        parquet_paths = sorted(output_dir.glob("*.parquet"))
        paths: dict[str, Path | None] = {}
        for name, tokens in GRAPH_OUTPUT_TABLES.items():
            paths[name] = _match_table_path(parquet_paths, tokens)
        return paths

    def _table_path(self, output_dir: Path | None, *tokens: str) -> Path | None:
        if output_dir is None or not output_dir.exists():
            return None
        return _match_table_path(sorted(output_dir.glob("*.parquet")), tokens)

    @staticmethod
    def _table_row_count(path: Path) -> int | None:
        try:
            import pyarrow.lib as arrow_lib
            import pyarrow.parquet as parquet
        except ImportError:
            return None
        try:
            return int(parquet.read_metadata(path).num_rows)
        except (
            OSError,
            TypeError,
            ValueError,
            RuntimeError,
            arrow_lib.ArrowException,
        ):
            return None

    def _vector_store_path(self, active_output_dir: Path | None) -> Path | None:
        candidates: list[Path] = []
        configured = self._configured_vector_store_path()
        if configured is not None:
            candidates.append(configured)
        if active_output_dir is not None:
            candidates.append(active_output_dir / "lancedb")
            if active_output_dir.parent != active_output_dir:
                candidates.append(active_output_dir.parent / "lancedb")
        candidates.append(self.output_dir / "lancedb")

        unique_candidates = list(dict.fromkeys(candidates))
        for candidate in unique_candidates:
            if candidate.exists():
                return candidate
        return unique_candidates[0] if unique_candidates else None

    def _configured_vector_store_path(self) -> Path | None:
        settings = self._load_settings()
        vector_store = settings.get("vector_store", {})
        if not isinstance(vector_store, dict):
            return self.output_dir / "lancedb"
        db_uri = vector_store.get("db_uri") or "output/lancedb"
        if not isinstance(db_uri, str) or not db_uri.strip():
            return self.output_dir / "lancedb"
        path = Path(db_uri)
        if path.is_absolute():
            return path
        return self.workspace_dir / path

    @staticmethod
    def _vector_store_readable(path: Path | None) -> bool:
        return GraphRAGStatusService._vector_store_state(path) == "ready"

    @staticmethod
    def _vector_store_state(path: Path | None) -> str:
        if path is None or not path.exists():
            return "missing"
        try:
            if path.is_file():
                return "ready" if path.stat().st_size > 0 else "unreadable"
            marker = path / "vector-store.marker"
            if marker.exists():
                return "ready" if marker.stat().st_size > 0 else "unreadable"
            lancedb_state = _lancedb_vector_store_state(path)
            if lancedb_state is not None:
                return lancedb_state
            if _looks_like_lancedb_path(path):
                return "ready"
            if next(path.iterdir(), None) is None:
                return "empty"
            return "unreadable"
        except OSError:
            return "unreadable"

    def _latest_parquet_mtime_iso(self, output_dir: Path | None = None) -> str | None:
        output_dir = output_dir or self.output_dir
        if not output_dir.exists():
            return None
        newest = max(
            (path.stat().st_mtime for path in output_dir.rglob("*.parquet")),
            default=None,
        )
        return _timestamp_iso(newest)

    def _relative_to_project_root(self, path: Path) -> str:
        return GraphRAGStatus._relative_to_project(path, self.paths.root)

    def _project_relative_path(self, value: Any) -> Path | None:
        if not isinstance(value, str) or not value.strip():
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        return self.paths.root / path

    @staticmethod
    def _file_mtime_iso(path: Path) -> str | None:
        if not path.exists():
            return None
        try:
            return _timestamp_iso(path.stat().st_mtime)
        except OSError:
            return None

    def _load_runs(self) -> list[dict[str, Any]]:
        if not self.runs_file.exists():
            return []
        try:
            payload = json.loads(self.runs_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    @staticmethod
    def _next_action(
        *,
        workspace_initialized: bool,
        input_exists: bool,
        input_document_count: int,
        output_present: bool,
        output_complete: bool,
        vector_store_exists: bool = True,
        vector_store_readable: bool = True,
        freshness_state: str = "unknown",
        stale_reasons: tuple[str, ...] = (),
        last_run: dict[str, Any] | None = None,
    ) -> str:
        if not workspace_initialized:
            return "Run `kb init`."
        if not input_exists:
            return "Run `kb update`."
        if input_document_count == 0:
            return "Add and compile sources, then run `kb update`."
        if last_run and last_run.get("success") is False:
            return "Fix the last graph index error, then rerun `kb update`."
        if not output_present:
            if last_run and last_run.get("dry_run") is True:
                return "Run `kb update` to build the graph index."
            return "Run `kb update` to sync and build the graph index."
        if not output_complete:
            if not vector_store_exists:
                return "Run `kb update --graph-only` to rebuild the missing graph vector store."
            if not vector_store_readable:
                return "Run `kb update --graph-only` to rebuild the unreadable graph vector store."
            return "Run `kb update` to rebuild incomplete graph index output."
        if freshness_state in {"stale", "missing-metadata"}:
            if stale_reasons:
                return f"Run `kb update --graph-only`: {stale_reasons[0]}"
            return "Run `kb update --graph-only` to refresh the graph index."
        return 'Run `kb ask --method drift "..."` or `kb export`.'


def graph_ready_for_query(status: GraphRAGStatus, *, method: str | None = None) -> bool:
    """Return whether GraphRAG has every artifact required for querying."""
    return (
        status.workspace_initialized
        and status.input_exists
        and status.input_document_count > 0
        and status.output_present
        and not missing_artifacts_for_query(status, method=method)
        and status.graph_freshness_state == "fresh"
        and status.last_index_success is not False
    )


def missing_artifacts_for_query(
    status: GraphRAGStatus,
    *,
    method: str | None = None,
) -> list[str]:
    """Return missing artifacts for a specific GraphRAG query method."""
    if method is None:
        return status.missing_tables
    normalized = method.strip().lower()
    table_names = QUERY_REQUIRED_TABLES.get(normalized, tuple(GRAPH_OUTPUT_TABLES))
    missing: list[str] = []
    for table_name in table_names:
        if not getattr(status, f"{table_name}_present"):
            missing.append(table_name)
    if normalized in QUERY_REQUIRES_VECTOR_STORE and not status.vector_store_readable:
        missing.append("vector_store")
    return missing


def graph_not_ready_message(
    status: GraphRAGStatus,
    *,
    method: str | None = None,
) -> str:
    """Return a precise next-step message for an unqueryable graph."""
    if not status.workspace_initialized:
        return "GraphRAG workspace is not initialized. Run `kb init` first."
    if not status.input_exists:
        return "GraphRAG input not found. Run `kb update` first."
    if status.input_document_count == 0:
        return (
            "GraphRAG input has no documents. Add and compile sources, then run "
            "`kb update`."
        )
    if status.last_index_success is False:
        return "The last GraphRAG index run failed. Re-run `kb update` before asking."
    if not status.output_present:
        return "GraphRAG index output not found. Run `kb update`."
    missing = missing_artifacts_for_query(status, method=method)
    if "vector_store" in missing and not status.vector_store_exists:
        return (
            "GraphRAG vector store not found. Run `kb update --graph-only` to "
            "rebuild the graph index."
        )
    if "vector_store" in missing and not status.vector_store_readable:
        return (
            "GraphRAG vector store is empty or unreadable. Run `kb update --graph-only` "
            "to rebuild the graph index."
        )
    if missing:
        method_detail = f" for `{method}` queries" if method else ""
        return (
            f"GraphRAG index output is incomplete{method_detail}: "
            f"{', '.join(missing)}. Run `kb update --graph-only` to rebuild it."
        )
    if status.graph_freshness_state in {"stale", "missing-metadata"}:
        reason = (
            f" {status.graph_stale_reasons[0]}" if status.graph_stale_reasons else ""
        )
        return f"GraphRAG index is stale.{reason} Run `kb update --graph-only`."
    return f"GraphRAG index is not ready. {status.next_action}"


def _tail(value: str, *, max_chars: int = 2000) -> str:
    return value[-max_chars:]


def _timestamp_iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _match_table_path(paths: list[Path], tokens: tuple[str, ...]) -> Path | None:
    for token in tokens:
        exact_name = f"{token}.parquet"
        for path in paths:
            if path.name == exact_name:
                return path
    normalized_tokens = tuple(token.lower() for token in tokens)
    for path in paths:
        stem = path.stem.lower()
        if any(stem == token or token in stem for token in normalized_tokens):
            return path
    return None


def _lancedb_vector_store_state(path: Path) -> str | None:
    try:
        import lancedb
    except ImportError:
        return None
    try:
        database = lancedb.connect(path)
        table_names = list(database.table_names())
    except Exception:
        return "unreadable"
    if not table_names:
        return "empty"
    normalized_names = tuple(name.casefold() for name in table_names)
    if any(
        hint in table_name
        for table_name in normalized_names
        for hint in VECTOR_STORE_TABLE_HINTS
    ):
        return "ready"
    return "incompatible"


def _looks_like_lancedb_path(path: Path) -> bool:
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    for child in children:
        name = child.name.casefold()
        if child.is_dir() and name.endswith(".lance"):
            return True
        if name in {"_latest.manifest", "_versions"}:
            return True
    return False


def iso_timestamp_after(left: str | None, right: str | None) -> bool:
    """Return whether ISO timestamp *left* is after *right*."""
    if not left or not right:
        return False
    left_dt = _parse_iso_timestamp(left)
    right_dt = _parse_iso_timestamp(right)
    if left_dt is None or right_dt is None:
        return False
    return left_dt > right_dt


def _parse_iso_timestamp(value: str) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
