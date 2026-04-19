from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional
import uuid

from src.models.source_models import RawSourceRecord
from src.services.project_service import atomic_write_text, utc_now_iso


def _default_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "active_run": None,
        "resume_run": None,
        "history": [],
    }


@dataclass
class CompileRunRecord:
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
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompileRunRecord":
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
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file

    def resume_candidate(self) -> Optional[CompileRunRecord]:
        payload = self._read_payload()
        resume_run = payload.get("resume_run")
        if isinstance(resume_run, dict):
            return CompileRunRecord.from_dict(resume_run)
        active_run = payload.get("active_run")
        if isinstance(active_run, dict) and active_run.get("status") == "running":
            return CompileRunRecord.from_dict(active_run)
        return None

    def active_run(self) -> Optional[CompileRunRecord]:
        payload = self._read_payload()
        active_run = payload.get("active_run")
        if not isinstance(active_run, dict):
            return None
        return CompileRunRecord.from_dict(active_run)

    def load_history(self) -> list[CompileRunRecord]:
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
        payload = self._read_payload()
        active_run = self._require_active(payload, run_id)
        if source.slug not in active_run["completed_source_slugs"]:
            active_run["completed_source_slugs"].append(source.slug)
        active_run["pending_source_slugs"] = [
            slug for slug in active_run["pending_source_slugs"] if slug != source.slug
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
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.state_file,
            json.dumps(payload, indent=2, sort_keys=True),
        )
