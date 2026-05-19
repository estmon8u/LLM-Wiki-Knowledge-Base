"""Durable JSON store for research runs and source recommendations.

This module belongs to `graphwiki_kb.services.source_recommendation_store` and
keeps related behavior close to the command, service, model, provider,
storage, script, or test surface that uses it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from graphwiki_kb.agents.models import (
    ResearchRunRecord,
    SourceRecommendation,
)
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
    utc_now_iso,
)

_RUN_FILENAME_PREFIX = "research-"
_RUN_FILENAME_SUFFIX = ".json"
_RUN_ID_TIMESTAMP_RE = re.compile(r"^research_(?P<ts>\d{8}T\d{6}Z)")


class SourceRecommendationStoreError(RuntimeError):
    """Raised for missing runs or malformed run records."""


class SourceRecommendationStore:
    """File-backed store for research runs under ``graph/runs/agent/``."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.directory = paths.graph_dir / "runs" / "agent"
        self.latest_pointer = self.directory / "latest.json"

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------
    def ensure_directory(self) -> None:
        """Create the storage directory if it does not exist."""
        self.directory.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str, *, question: str | None = None) -> Path:
        slug = slugify(question or run_id)
        ts = _extract_timestamp(run_id) or utc_now_iso().replace(":", "").replace(
            "-", ""
        )
        filename = f"{_RUN_FILENAME_PREFIX}{ts}-{slug}{_RUN_FILENAME_SUFFIX}"
        return self.directory / filename

    # ------------------------------------------------------------------
    # Run id helpers
    # ------------------------------------------------------------------
    @staticmethod
    def generate_run_id(question: str, *, created_at: str | None = None) -> str:
        """Build a deterministic, sortable run id for one research question."""
        timestamp = (created_at or utc_now_iso()).replace("-", "").replace(":", "")
        slug = slugify(question)[:48] or "research"
        return f"research_{timestamp}_{slug}"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def save(self, record: ResearchRunRecord) -> Path:
        """Persist a research run as JSON and update the latest pointer."""
        self.ensure_directory()
        path = self._run_path(record.run_id, question=record.question)
        payload = json.loads(record.model_dump_json())
        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=False))
        atomic_write_text(
            self.latest_pointer,
            json.dumps(
                {"run_id": record.run_id, "path": path.name},
                indent=2,
                sort_keys=False,
            ),
        )
        return path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def load(self, run_id: str = "latest") -> ResearchRunRecord:
        """Load a research run by id, or the most recent if id is 'latest'."""
        if run_id == "latest":
            return self._load_latest()
        path = self._find_path_by_run_id(run_id)
        if path is None:
            raise SourceRecommendationStoreError(
                f"No research run with id '{run_id}' was found."
            )
        return self._load_path(path)

    def list_runs(self) -> list[ResearchRunRecord]:
        """Return every persisted research run, oldest-first."""
        if not self.directory.exists():
            return []
        records: list[ResearchRunRecord] = []
        for path in sorted(self.directory.glob(f"{_RUN_FILENAME_PREFIX}*.json")):
            try:
                records.append(self._load_path(path))
            except SourceRecommendationStoreError:
                continue
        return records

    def latest(self) -> ResearchRunRecord | None:
        """Return the most recent persisted research run, or None."""
        try:
            return self._load_latest()
        except SourceRecommendationStoreError:
            return None

    def resolve_recommendations(
        self,
        ids: list[int],
        *,
        run_id: str = "latest",
    ) -> tuple[ResearchRunRecord, list[SourceRecommendation]]:
        """Return (run, recommendations) for the requested IDs."""
        record = self.load(run_id)
        by_id = {rec.id: rec for rec in record.recommendations}
        if not ids:
            return record, list(record.recommendations)
        resolved: list[SourceRecommendation] = []
        missing: list[int] = []
        for rec_id in ids:
            match = by_id.get(rec_id)
            if match is None:
                missing.append(rec_id)
            else:
                resolved.append(match)
        if missing:
            raise SourceRecommendationStoreError(
                f"Recommendation id(s) not found in run {record.run_id}: "
                + ", ".join(str(value) for value in missing)
            )
        return record, resolved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_latest(self) -> ResearchRunRecord:
        if self.latest_pointer.exists():
            try:
                data = json.loads(self.latest_pointer.read_text(encoding="utf-8"))
                pointer_name = data.get("path")
                if pointer_name:
                    candidate = self.directory / pointer_name
                    if candidate.exists():
                        return self._load_path(candidate)
            except (OSError, json.JSONDecodeError):
                pass
        runs = self.list_runs()
        if not runs:
            raise SourceRecommendationStoreError(
                "No research runs have been saved yet. "
                'Run `kb agent "research ..."` first.'
            )
        return runs[-1]

    def _find_path_by_run_id(self, run_id: str) -> Path | None:
        if not self.directory.exists():
            return None
        for path in self.directory.glob(f"{_RUN_FILENAME_PREFIX}*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("run_id") == run_id:
                return path
        return None

    @staticmethod
    def _load_path(path: Path) -> ResearchRunRecord:
        try:
            data: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceRecommendationStoreError(
                f"Unable to read research run at {path}: {exc}"
            ) from exc
        try:
            return ResearchRunRecord.model_validate(data)
        except ValidationError as exc:
            raise SourceRecommendationStoreError(
                f"Research run at {path} is malformed: {exc}"
            ) from exc


def _extract_timestamp(run_id: str) -> str | None:
    match = _RUN_ID_TIMESTAMP_RE.match(run_id)
    return match.group("ts") if match else None
