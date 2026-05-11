from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.services.graphrag_command_service import GraphRAGCommandResult
from src.services.project_service import ProjectPaths, atomic_write_text, utc_now_iso


GRAPH_OUTPUT_TABLES: dict[str, tuple[str, ...]] = {
    "documents": ("documents", "create_final_documents"),
    "text_units": ("text_units", "create_final_text_units"),
    "entities": ("entities", "create_final_entities"),
    "relationships": ("relationships", "create_final_relationships"),
    "communities": ("communities", "create_final_communities"),
    "community_reports": ("community_reports", "create_final_community_reports"),
}


@dataclass(frozen=True)
class GraphRAGIndexRun:
    run_id: str
    created_at: str
    method: str
    dry_run: bool
    success: bool
    returncode: int
    command: tuple[str, ...]
    stdout_tail: str
    stderr_tail: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload


@dataclass(frozen=True)
class GraphRAGStatus:
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

    def to_dict(self, project_root: Path) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("workspace_dir", "settings_path", "input_path", "output_dir"):
            payload[key] = self._relative_to_project(payload[key], project_root)
        return payload

    @staticmethod
    def _relative_to_project(path: Path, project_root: Path) -> str:
        try:
            return path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            return path.as_posix()


class GraphRAGStatusService:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.workspace_dir = paths.graph_dir / "graphrag"
        self.settings_path = self.workspace_dir / "settings.yaml"
        self.input_path = self.workspace_dir / "input" / "sources.json"
        self.output_dir = self.workspace_dir / "output"
        self.runs_file = paths.graph_dir / "runs" / "graph_index_runs.json"

    def status(self) -> GraphRAGStatus:
        runs = self._load_runs()
        last_run = runs[-1] if runs else None
        table_paths = {
            name: self._table_path(*patterns)
            for name, patterns in GRAPH_OUTPUT_TABLES.items()
        }
        tables = {name: path is not None for name, path in table_paths.items()}
        table_counts = {
            name: self._table_row_count(path) if path is not None else None
            for name, path in table_paths.items()
        }
        output_present = self.output_dir.exists() and any(
            self.output_dir.rglob("*.parquet")
        )
        input_document_count = self._input_document_count()
        workspace_initialized = self.settings_path.exists()
        input_exists = self.input_path.exists()
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
            ),
            input_updated_at=self._file_mtime_iso(self.input_path),
            output_updated_at=self._latest_parquet_mtime_iso(),
            wiki_export_present=wiki_export_path.exists(),
            wiki_export_updated_at=self._file_mtime_iso(wiki_export_path),
            document_count=table_counts["documents"],
            text_unit_count=table_counts["text_units"],
            entity_count=table_counts["entities"],
            relationship_count=table_counts["relationships"],
            community_count=table_counts["communities"],
            community_report_count=table_counts["community_reports"],
        )

    def record_index_run(
        self,
        *,
        method: str,
        dry_run: bool,
        result: GraphRAGCommandResult,
    ) -> GraphRAGIndexRun:
        created_at = utc_now_iso()
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
        )
        runs = self._load_runs()
        runs.append(record.to_dict())
        atomic_write_text(
            self.runs_file,
            json.dumps(runs, indent=2, sort_keys=True) + "\n",
        )
        return record

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

    def _table_path(self, *tokens: str) -> Path | None:
        if not self.output_dir.exists():
            return None
        normalized_tokens = tuple(token.lower() for token in tokens)
        for path in self.output_dir.rglob("*.parquet"):
            stem = path.stem.lower()
            if any(stem == token or token in stem for token in normalized_tokens):
                return path
        return None

    def _table_present(self, *tokens: str) -> bool:
        return self._table_path(*tokens) is not None

    @staticmethod
    def _table_row_count(path: Path) -> int | None:
        try:
            import pyarrow.parquet as parquet

            return int(parquet.read_metadata(path).num_rows)
        except Exception:
            return None

    def _latest_parquet_mtime_iso(self) -> str | None:
        if not self.output_dir.exists():
            return None
        newest = max(
            (path.stat().st_mtime for path in self.output_dir.rglob("*.parquet")),
            default=None,
        )
        return _timestamp_iso(newest)

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
    ) -> str:
        if not workspace_initialized:
            return "Run `kb graph init`."
        if not input_exists:
            return "Run `kb graph sync`."
        if input_document_count == 0:
            return "Add and compile sources, then run `kb graph sync`."
        if not output_present:
            return "Run `kb graph index --method fast --dry-run` before a full index."
        return 'Run `kb graph ask --method drift "..."` or `kb graph export-wiki`.'


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
