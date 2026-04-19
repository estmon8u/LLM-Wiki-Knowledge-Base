"""SQLite-backed FTS5 search index for compiled wiki content."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from pathlib import Path


_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS indexed_files (
    page_path    TEXT PRIMARY KEY,
    mtime_ns     INTEGER NOT NULL,
    size_bytes   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS page_chunks (
    chunk_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    page_path    TEXT NOT NULL,
    page_type    TEXT NOT NULL DEFAULT '',
    title        TEXT NOT NULL DEFAULT '',
    section      TEXT NOT NULL DEFAULT '',
    chunk_index  INTEGER NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '',
    body         TEXT NOT NULL DEFAULT '',
    UNIQUE(page_path, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_page_chunks_page_path
    ON page_chunks(page_path);

CREATE VIRTUAL TABLE IF NOT EXISTS page_chunks_fts USING fts5(
    title,
    section,
    metadata,
    body,
    content='page_chunks',
    content_rowid='chunk_id',
    tokenize='porter unicode61'
);
"""


class SearchIndexUnavailable(RuntimeError):
    """Raised when the local SQLite build does not support FTS5."""


@dataclass(frozen=True)
class IndexedFileState:
    page_path: str
    mtime_ns: int
    size_bytes: int


@dataclass(frozen=True)
class IndexedChunk:
    page_path: str
    page_type: str
    title: str
    section: str
    chunk_index: int
    metadata: str
    body: str


@dataclass(frozen=True)
class SearchHit:
    page_path: str
    title: str
    section: str
    snippet: str
    score: int


class SearchIndexStore:
    """Persist and query chunked wiki content through SQLite FTS5."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def load_indexed_files(self) -> dict[str, tuple[int, int]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT page_path, mtime_ns, size_bytes FROM indexed_files"
            ).fetchall()
        return {row[0]: (row[1], row[2]) for row in rows}

    def rebuild(
        self,
        file_states: list[IndexedFileState],
        chunks: list[IndexedChunk],
    ) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM indexed_files")
            conn.execute("DELETE FROM page_chunks")

            if file_states:
                conn.executemany(
                    "INSERT INTO indexed_files (page_path, mtime_ns, size_bytes) "
                    "VALUES (?, ?, ?)",
                    [
                        (state.page_path, state.mtime_ns, state.size_bytes)
                        for state in file_states
                    ],
                )

            if chunks:
                conn.executemany(
                    "INSERT INTO page_chunks "
                    "(page_path, page_type, title, section, chunk_index, metadata, body) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            chunk.page_path,
                            chunk.page_type,
                            chunk.title,
                            chunk.section,
                            chunk.chunk_index,
                            chunk.metadata,
                            chunk.body,
                        )
                        for chunk in chunks
                    ],
                )

            conn.execute(
                "INSERT INTO page_chunks_fts(page_chunks_fts) VALUES('rebuild')"
            )
            conn.commit()

    def search(self, match_query: str, *, limit: int) -> list[SearchHit]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    page_chunks.page_path,
                    page_chunks.title,
                    page_chunks.section,
                    snippet(page_chunks_fts, -1, '', '', '...', 24) AS snippet,
                    bm25(page_chunks_fts, 6.0, 3.0, 2.0, 1.0) AS rank
                FROM page_chunks_fts
                JOIN page_chunks ON page_chunks.chunk_id = page_chunks_fts.rowid
                WHERE page_chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()

        return [
            SearchHit(
                page_path=row[0],
                title=row[1],
                section=row[2],
                snippet=row[3] or "",
                score=max(1, int(round(abs(row[4]) * 1000))) if row[4] else 1,
            )
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA_SQL)
        except sqlite3.OperationalError as exc:
            conn.close()
            raise SearchIndexUnavailable(str(exc)) from exc
        return conn
