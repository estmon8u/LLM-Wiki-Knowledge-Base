"""Persistence for agent research runs and source recommendations."""

from __future__ import annotations

import json
import re
from pathlib import Path

from graphwiki_kb.agents.models import ResearchRunRecord, SourceRecommendation
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    utc_now_iso,
)

_RUN_ID_PATTERN = re.compile(r"^research[-_](?P<stamp>[\dTZ]+)[-_](?P<slug>.+)$")


def agent_runs_dir(paths: ProjectPaths) -> Path:
    """Directory for agent sessions, research runs, and traces."""
    return paths.graph_dir / "runs" / "agent"


def _latest_pointer_path(runs_dir: Path) -> Path:
    return runs_dir / "latest.json"


def _run_path(runs_dir: Path, run_id: str) -> Path:
    safe_id = run_id.replace("/", "-")
    return runs_dir / f"research-{safe_id}.json"


class SourceRecommendationStore:
    """JSON-backed store for research runs and recommendation lookup."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.runs_dir = agent_runs_dir(paths)

    def ensure_dir(self) -> None:
        """Create the agent runs directory if missing."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, record: ResearchRunRecord) -> Path:
        """Persist a research run and update the latest pointer."""
        self.ensure_dir()
        path = _run_path(self.runs_dir, record.run_id)
        atomic_write_text(path, record.model_dump_json(indent=2))
        if record.recommendations:
            atomic_write_text(
                _latest_pointer_path(self.runs_dir),
                json.dumps({"run_id": record.run_id, "path": path.name}, indent=2),
            )
        return path

    def load_run(self, run_id: str | None = None) -> ResearchRunRecord | None:
        """Load a research run by id or the latest persisted run."""
        self.ensure_dir()
        resolved_id = run_id or self.latest_run_id()
        if resolved_id is None:
            return None
        path = _run_path(self.runs_dir, resolved_id)
        if not path.exists():
            for candidate in sorted(self.runs_dir.glob("research-*.json")):
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                if payload.get("run_id") == resolved_id:
                    path = candidate
                    break
            else:
                return None
        return ResearchRunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def latest_run_id(self) -> str | None:
        """Return the run id from latest.json or the newest research file."""
        pointer = _latest_pointer_path(self.runs_dir)
        if pointer.exists():
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            run_id = payload.get("run_id")
            if isinstance(run_id, str) and run_id:
                return run_id
        records = self.list_runs()
        return records[0].run_id if records else None

    def latest_run_with_recommendations(self) -> ResearchRunRecord | None:
        """Return the newest research run that has at least one recommendation."""
        for record in self.list_runs():
            if record.recommendations:
                return record
        return None

    def list_runs(self) -> list[ResearchRunRecord]:
        """Return research runs sorted newest first."""
        self.ensure_dir()
        records: list[ResearchRunRecord] = []
        for path in self.runs_dir.glob("research-*.json"):
            if path.name == "latest.json":
                continue
            try:
                records.append(
                    ResearchRunRecord.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                )
            except (json.JSONDecodeError, ValueError):
                continue
        records.sort(key=lambda item: item.created_at, reverse=True)
        return records

    def resolve_recommendations(
        self,
        recommendation_ids: list[int],
        *,
        run_id: str | None = None,
    ) -> tuple[ResearchRunRecord, list[SourceRecommendation]]:
        """Resolve recommendation ids from a stored research run."""
        record = self.load_run(run_id)
        if record is None:
            raise ValueError("No research run found. Run research first with kb agent.")
        if not record.recommendations and run_id is None:
            record = self.latest_run_with_recommendations()
        if record is None:
            raise ValueError("No research run found. Run research first with kb agent.")
        by_id = {item.id: item for item in record.recommendations}
        missing = [rid for rid in recommendation_ids if rid not in by_id]
        if missing:
            raise ValueError(
                f"Unknown recommendation id(s): {missing}. "
                f"Valid ids for run {record.run_id}: {sorted(by_id)}"
            )
        return record, [by_id[rid] for rid in recommendation_ids]

    @staticmethod
    def make_run_id(question: str) -> str:
        """Build a stable research run id from a question slug."""
        from graphwiki_kb.services.project_service import slugify

        stamp = utc_now_iso().replace(":", "").replace("-", "")[:15]
        slug = slugify(question)[:48] or "research"
        return f"research_{stamp}_{slug}"
