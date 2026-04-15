"""SQLite-backed persistence for deliberation run artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from src.schemas.runs import RunRecord

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    command       TEXT NOT NULL DEFAULT '',
    timestamp     TEXT NOT NULL,
    model_id      TEXT NOT NULL DEFAULT '',
    prompt_version TEXT NOT NULL DEFAULT '',
    context_hash  TEXT NOT NULL DEFAULT '',
    token_cost    INTEGER NOT NULL DEFAULT 0,
    wall_time_ms  INTEGER NOT NULL DEFAULT 0,
    unresolved    INTEGER NOT NULL DEFAULT 0,
    final_text    TEXT NOT NULL DEFAULT '',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    candidates_json TEXT NOT NULL DEFAULT '[]',
    merged_json   TEXT NOT NULL DEFAULT '{}',
    review_json   TEXT NOT NULL DEFAULT '[]',
    full_record   TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS run_citations (
    run_id      TEXT NOT NULL,
    claim_text  TEXT NOT NULL,
    source_page TEXT NOT NULL DEFAULT '',
    section     TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_run_citations_run_id
    ON run_citations(run_id);

CREATE INDEX IF NOT EXISTS idx_run_citations_source_page
    ON run_citations(source_page);
"""


class RunStore:
    """Persist and query deliberation run artifacts in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA_SQL)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_run(self, record: RunRecord) -> str:
        """Persist a run record.  Returns the run_id."""
        conn = self._connect()

        evidence_json = (
            record.evidence_bundle.model_dump_json() if record.evidence_bundle else "{}"
        )
        candidates_json = json.dumps([c.model_dump() for c in record.candidates])
        merged_json = (
            record.merged_answer.model_dump_json() if record.merged_answer else "{}"
        )
        review_json = json.dumps([f.model_dump() for f in record.review_findings])

        conn.execute(
            """
            INSERT OR REPLACE INTO runs
                (run_id, command, timestamp, model_id, prompt_version,
                 context_hash, token_cost, wall_time_ms, unresolved,
                 final_text, evidence_json, candidates_json,
                 merged_json, review_json, full_record)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                record.command,
                record.timestamp,
                record.model_id,
                record.prompt_version,
                record.context_hash,
                record.token_cost,
                record.wall_time_ms,
                int(record.unresolved_disagreement),
                record.final_text,
                evidence_json,
                candidates_json,
                merged_json,
                review_json,
                record.model_dump_json(),
            ),
        )

        # Persist individual citations for queryability.
        all_claims: list[tuple[str, str, str]] = []
        for candidate in record.candidates:
            for claim in candidate.claims:
                all_claims.append((record.run_id, claim.text, claim.source_page))
        if record.merged_answer:
            for claim in record.merged_answer.accepted_claims:
                all_claims.append((record.run_id, claim.text, claim.source_page))
        for finding in record.review_findings:
            all_claims.append(
                (record.run_id, finding.claim, ",".join(finding.affected_pages))
            )

        if all_claims:
            conn.executemany(
                "INSERT INTO run_citations (run_id, claim_text, source_page) "
                "VALUES (?, ?, ?)",
                all_claims,
            )

        conn.commit()
        return record.run_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Load a single run by ID.  Returns None if not found."""
        conn = self._connect()
        row = conn.execute(
            "SELECT full_record FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return RunRecord.model_validate_json(row[0])

    def list_runs(
        self, *, command: Optional[str] = None, limit: int = 50
    ) -> list[RunRecord]:
        """List recent runs, optionally filtered by command name."""
        conn = self._connect()
        if command:
            rows = conn.execute(
                "SELECT full_record FROM runs WHERE command = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (command, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT full_record FROM runs " "ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [RunRecord.model_validate_json(r[0]) for r in rows]

    def runs_citing_page(self, source_page: str) -> list[str]:
        """Return run IDs that cite a given source page."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT DISTINCT run_id FROM run_citations WHERE source_page = ?",
            (source_page,),
        ).fetchall()
        return [r[0] for r in rows]

    def citation_count(self, run_id: str) -> int:
        """Count citations stored for a given run."""
        conn = self._connect()
        row = conn.execute(
            "SELECT COUNT(*) FROM run_citations WHERE run_id = ?", (run_id,)
        ).fetchone()
        return row[0] if row else 0
