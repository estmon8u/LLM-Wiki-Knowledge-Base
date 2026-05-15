"""Storage helpers for compile run store.

This module belongs to `graphwiki_kb.storage.compile_run_store` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional
import uuid

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.file_lock import file_lock
from graphwiki_kb.services.project_service import atomic_write_text, utc_now_iso


MAX_COMPILE_RUN_HISTORY = 50


def _default_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "active_run": None,
        "resume_run": None,
        "history": [],
    }


@dataclass
class CompileRunRecord:
    """Stores compile run record data.

    Attributes:
        See annotated class attributes for stored values.
    """

    run_id: str
    status: str
    started_at: str
    finished_at: str = ""
    force: bool = False
    resumed_from_run_id: str = ""
    planned_source_slugs: list[str] = field(default_factory=list)
    completed_source_slugs: list[str] = field(default_factory=list)
    pending_source_slugs: list[str] = field(default_factory=list)
    failed_source_slug: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializes this value to a dictionary.

        Returns:
            dict[str, Any] produced by the operation.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompileRunRecord":
        """Builds an instance from a dictionary payload.

        Args:
            payload: Structured payload being parsed or serialized.

        Returns:
            "CompileRunRecord" produced by the operation.
        """
        return cls(
            run_id=payload["run_id"],
            status=payload["status"],
            started_at=payload["started_at"],
            finished_at=payload.get("finished_at", ""),
            force=bool(payload.get("force", False)),
            resumed_from_run_id=payload.get("resumed_from_run_id", ""),
            planned_source_slugs=list(payload.get("planned_source_slugs", [])),
            completed_source_slugs=list(payload.get("completed_source_slugs", [])),
            pending_source_slugs=list(payload.get("pending_source_slugs", [])),
            failed_source_slug=payload.get("failed_source_slug", ""),
            error=payload.get("error", ""),
        )


class CompileRunStore:
    """Represents compile run store behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file

    def resume_candidate(self) -> Optional[CompileRunRecord]:
        """Resume candidate.

        Returns:
            Optional[CompileRunRecord] produced by the operation.
        """
        with file_lock(self.state_file):
            payload = self._read_payload()
        resume_run = payload.get("resume_run")
        if isinstance(resume_run, dict):
            return CompileRunRecord.from_dict(resume_run)
        active_run = payload.get("active_run")
        if isinstance(active_run, dict) and active_run.get("status") == "running":
            return CompileRunRecord.from_dict(active_run)
        return None

    def active_run(self) -> Optional[CompileRunRecord]:
        """Active run.

        Returns:
            Optional[CompileRunRecord] produced by the operation.
        """
        with file_lock(self.state_file):
            payload = self._read_payload()
        active_run = payload.get("active_run")
        if not isinstance(active_run, dict):
            return None
        return CompileRunRecord.from_dict(active_run)

    def load_history(self) -> list[CompileRunRecord]:
        """Loads history.

        Returns:
            list[CompileRunRecord] produced by the operation.
        """
        with file_lock(self.state_file):
            payload = self._read_payload()
        return [
            CompileRunRecord.from_dict(item)
            for item in payload.get("history", [])
            if isinstance(item, dict)
        ]

    def start_run(
        self,
        pending_sources: list[RawSourceRecord],
        *,
        force: bool,
        resumed_from_run_id: str = "",
    ) -> CompileRunRecord:
        """Start run.

        Args:
            pending_sources: Pending sources value used by the operation.
            force: Force value used by the operation.
            resumed_from_run_id: Resumed from run id value used by the operation.

        Returns:
            CompileRunRecord produced by the operation.
        """
        with file_lock(self.state_file):
            payload = self._normalize_interrupted_active(self._read_payload())
            payload["resume_run"] = None
            record = CompileRunRecord(
                run_id=uuid.uuid4().hex[:12],
                status="running",
                started_at=utc_now_iso(),
                force=force,
                resumed_from_run_id=resumed_from_run_id,
                planned_source_slugs=[source.slug for source in pending_sources],
                pending_source_slugs=[source.slug for source in pending_sources],
            )
            payload["active_run"] = record.to_dict()
            self._write_payload(payload)
            return record

    def mark_source_compiled(self, run_id: str, source: RawSourceRecord) -> None:
        """Mark source compiled.

        Args:
            run_id: Run id value used by the operation.
            source: Source record or path being processed.
        """
        with file_lock(self.state_file):
            payload = self._read_payload()
            active_run = self._require_active(payload, run_id)
            if source.slug not in active_run["completed_source_slugs"]:
                active_run["completed_source_slugs"].append(source.slug)
            active_run["pending_source_slugs"] = [
                slug
                for slug in active_run["pending_source_slugs"]
                if slug != source.slug
            ]
            payload["active_run"] = active_run
            self._write_payload(payload)

    def mark_failed(
        self,
        run_id: str,
        *,
        error: str,
        failed_source: Optional[RawSourceRecord] = None,
    ) -> CompileRunRecord:
        """Mark failed.

        Args:
            run_id: Run id value used by the operation.
            error: Error value used by the operation.
            failed_source: Failed source value used by the operation.

        Returns:
            CompileRunRecord produced by the operation.
        """
        with file_lock(self.state_file):
            payload = self._read_payload()
            active_run = self._require_active(payload, run_id)
            active_run["status"] = "failed"
            active_run["finished_at"] = utc_now_iso()
            active_run["error"] = error
            if failed_source is not None:
                active_run["failed_source_slug"] = failed_source.slug
            record = CompileRunRecord.from_dict(active_run)
            payload["active_run"] = None
            payload["resume_run"] = record.to_dict()
            payload.setdefault("history", []).append(record.to_dict())
            self._write_payload(payload)
            return record

    def mark_completed(self, run_id: str) -> CompileRunRecord:
        """Mark completed.

        Args:
            run_id: Run id value used by the operation.

        Returns:
            CompileRunRecord produced by the operation.
        """
        with file_lock(self.state_file):
            payload = self._read_payload()
            active_run = self._require_active(payload, run_id)
            active_run["status"] = "completed"
            active_run["finished_at"] = utc_now_iso()
            active_run["pending_source_slugs"] = []
            record = CompileRunRecord.from_dict(active_run)
            payload["active_run"] = None
            payload["resume_run"] = None
            payload.setdefault("history", []).append(record.to_dict())
            self._write_payload(payload)
            return record

    def clear_resume_candidate(self) -> None:
        """Clear resume candidate."""
        with file_lock(self.state_file):
            payload = self._read_payload()
            payload["resume_run"] = None
            self._write_payload(payload)

    def _require_active(self, payload: dict[str, Any], run_id: str) -> dict[str, Any]:
        active_run = payload.get("active_run")
        if not isinstance(active_run, dict) or active_run.get("run_id") != run_id:
            raise ValueError(f"Compile run is not active: {run_id}")
        return active_run

    def _normalize_interrupted_active(self, payload: dict[str, Any]) -> dict[str, Any]:
        active_run = payload.get("active_run")
        if not isinstance(active_run, dict) or active_run.get("status") != "running":
            return payload
        archived = dict(active_run)
        archived["status"] = "failed"
        archived["finished_at"] = utc_now_iso()
        archived["error"] = archived.get("error") or "Interrupted before completion."
        payload["active_run"] = None
        payload["resume_run"] = archived
        payload.setdefault("history", []).append(archived)
        return payload

    def _read_payload(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return _default_payload()
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        except OSError:
            return _default_payload()
        except json.JSONDecodeError:
            self._move_corrupt_state_file()
            return _default_payload()
        if not isinstance(payload, dict):
            return _default_payload()
        default = _default_payload()
        for key, value in default.items():
            payload.setdefault(key, value)
        return payload

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload["version"] = 1
        history = payload.get("history", [])
        if isinstance(history, list):
            payload["history"] = history[-MAX_COMPILE_RUN_HISTORY:]
        atomic_write_text(
            self.state_file,
            json.dumps(payload, indent=2, sort_keys=True),
        )

    def _move_corrupt_state_file(self) -> None:
        if not self.state_file.exists():
            return
        stamp = utc_now_iso().replace(":", "").replace("+", "Z")
        corrupt_path = self.state_file.with_name(
            f"{self.state_file.name}.{stamp}.corrupt"
        )
        try:
            self.state_file.replace(corrupt_path)
        except OSError:
            return
