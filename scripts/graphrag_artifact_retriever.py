"""Read GraphRAG output parquets to provide a real retrieval surface.

The previous evaluation harness compared WikiGraphRAG's full retrieval
pipeline (wiki chunks + source TextUnits + entity-hops) against
GraphRAG's *artifact directory search* (entity/relationship parquet
scan via :class:`GraphRAGFindService`). That was apples-to-oranges:
GraphRAG's real retrieval surfaces ``text_units`` and
``community_reports`` to its local/global/drift search engines, but
those were never exposed through ``kb find``.

This module fixes the asymmetry. It exposes a small BM25 search over
the four GraphRAG output tables --- ``text_units``,
``community_reports``, ``entities``, ``relationships`` --- and returns
results with title, path, and snippet so the evaluation harness can
populate :class:`RetrievalRun.retrieved_text_snippets` symmetrically
with the WikiGraphRAG path.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphwiki_kb.services.graphrag_status_service import GraphRAGStatusService

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
logger = logging.getLogger(__name__)

# Per-table column projections. We read only what we need to score and
# present, so a 100k-row text_units table is still cheap on cold-cache.
_TABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "text_units": (
        "id",
        "human_readable_id",
        "text",
        "document_ids",
    ),
    "community_reports": (
        "id",
        "human_readable_id",
        "title",
        "summary",
        "full_content",
        "community",
        "rank",
    ),
    "entities": (
        "id",
        "human_readable_id",
        "title",
        "name",
        "description",
        "type",
        "community",
    ),
    "relationships": (
        "id",
        "human_readable_id",
        "source",
        "source_title",
        "target",
        "target_title",
        "description",
    ),
}


@dataclass(frozen=True)
class GraphRAGArtifactResult:
    """A single GraphRAG artifact match for the evaluator."""

    kind: str
    title: str
    path: str
    snippet: str
    score: float
    source_ids: tuple[str, ...] = ()


class GraphRAGArtifactRetriever:
    """BM25-style retrieval over GraphRAG parquet outputs.

    Use ``mode="text_units"`` for an apples-to-apples comparison with
    WikiGraphRAG (the default in the de-gamed evaluator). The
    ``"artifact"`` mode preserves the legacy entity/relationship scan
    for backwards-compatible numbers, and ``"both"`` mixes the two by
    score.
    """

    def __init__(
        self,
        status_service: GraphRAGStatusService,
        *,
        mode: str = "text_units",
    ) -> None:
        if mode not in {"text_units", "artifact", "both"}:
            raise ValueError(
                f"Unknown GraphRAG retrieve mode {mode!r}; "
                "expected text_units, artifact, or both."
            )
        self.status_service = status_service
        self.mode = mode

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def search(self, query: str, *, limit: int = 8) -> list[GraphRAGArtifactResult]:
        """Return up to ``limit`` GraphRAG artifact matches for ``query``."""
        terms = _query_terms(query)
        if not terms or limit <= 0:
            return []

        candidates: list[GraphRAGArtifactResult] = []
        tables = self._tables_for_mode()
        for table_name in tables:
            table_path = self.status_service.table_path(table_name)
            if table_path is None:
                continue
            for record in _iter_parquet_records(table_path, table_name):
                result = _record_result(table_name, record, terms)
                if result is not None:
                    candidates.append(result)

        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[:limit]

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _tables_for_mode(self) -> tuple[str, ...]:
        if self.mode == "text_units":
            # Apples-to-apples with WikiGraphRAG: text_unit bodies +
            # community summaries are the real "retrieved context" that
            # GraphRAG feeds to its local/global/drift search engines.
            return ("text_units", "community_reports", "entities", "relationships")
        if self.mode == "artifact":
            # Legacy: only entity/relationship scan, matching the
            # original :class:`GraphRAGFindService` surface.
            return ("entities", "relationships")
        return ("text_units", "community_reports", "entities", "relationships")


# --------------------------------------------------------------------------- #
# Parquet reading                                                             #
# --------------------------------------------------------------------------- #


def _iter_parquet_records(path: Path, table_name: str) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.lib as arrow_lib
        import pyarrow.parquet as parquet
    except ImportError:  # pragma: no cover - optional dependency
        logger.debug(
            "PyArrow not available; cannot read GraphRAG artifacts at %s",
            path,
            exc_info=True,
        )
        return

    requested = _TABLE_FIELDS.get(table_name, ())
    try:
        parquet_file = parquet.ParquetFile(path)
        available = set(parquet_file.schema.names)
        columns = [c for c in requested if c in available] or None
        batches = parquet_file.iter_batches(columns=columns)
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        arrow_lib.ArrowException,
    ) as exc:
        logger.debug("Cannot read GraphRAG parquet %s for evaluator: %s", path, exc)
        return
    try:
        for batch in batches:
            for row in batch.to_pylist():
                yield {str(k): _clean_value(v) for k, v in row.items()}
    except (RuntimeError, arrow_lib.ArrowException) as exc:
        logger.debug("Failed scanning GraphRAG parquet %s for evaluator: %s", path, exc)


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "tolist"):
        return _clean_value(value.tolist())
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_value(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Per-table mapping + scoring                                                 #
# --------------------------------------------------------------------------- #


def _record_result(
    table_name: str,
    record: dict[str, Any],
    terms: list[str],
) -> GraphRAGArtifactResult | None:
    # Snippet cap of 4000 chars matches the WikiGraphRAG path so
    # source-name matching is symmetric across backends; word-boundary
    # matching prevents false positives like "fid" matching "modified".
    snippet_cap = 4000
    if table_name == "text_units":
        text = _first_text(record, "text")
        if not text:
            return None
        artifact_id = _first_text(record, "human_readable_id", "id") or "text_unit"
        title = f"text_unit {artifact_id}"
        path = f"graph://text_units/{artifact_id}"
        snippet = text[:snippet_cap]
        searchable = text.casefold()
        source_ids = _tuple_of_strings(record.get("document_ids"))
    elif table_name == "community_reports":
        title = _first_text(record, "title") or _first_text(
            record, "human_readable_id", "id", default="community"
        )
        summary = _first_text(record, "summary") or _first_text(record, "full_content")
        if not (title or summary):
            return None
        artifact_id = _first_text(record, "human_readable_id", "id") or "community"
        path = f"graph://community_reports/{artifact_id}"
        snippet = (summary or title)[:snippet_cap]
        searchable = _join_text(title, summary).casefold()
        source_ids = ()
    elif table_name == "entities":
        title = _first_text(record, "title", "name", "id", default="entity")
        description = _first_text(record, "description")
        artifact_id = _first_text(record, "human_readable_id", "id") or "entity"
        path = f"graph://entities/{artifact_id}"
        snippet = (description or title)[:snippet_cap]
        searchable = _join_text(
            title,
            description,
            _first_text(record, "type"),
            _first_text(record, "community"),
        ).casefold()
        source_ids = ()
    elif table_name == "relationships":
        source_node = _first_text(record, "source_title", "source")
        target_node = _first_text(record, "target_title", "target")
        title = f"{source_node or '?'} -> {target_node or '?'}"
        description = _first_text(record, "description")
        artifact_id = _first_text(record, "human_readable_id", "id") or "relationship"
        path = f"graph://relationships/{artifact_id}"
        snippet = (description or title)[:snippet_cap]
        searchable = _join_text(source_node, target_node, description).casefold()
        source_ids = ()
    else:
        return None

    score = _score(terms, title=title, searchable=searchable, kind=table_name)
    if score <= 0:
        return None
    return GraphRAGArtifactResult(
        kind=table_name,
        title=title,
        path=path,
        snippet=snippet,
        score=score,
        source_ids=source_ids,
    )


def _score(terms: list[str], *, title: str, searchable: str, kind: str) -> float:
    title_lower = title.casefold()
    score = 0.0
    matched_terms = 0
    for term in terms:
        title_hits = _occurrences(term, title_lower)
        body_hits = _occurrences(term, searchable)
        if title_hits or body_hits:
            matched_terms += 1
        score += 8.0 * title_hits + 1.0 * body_hits
    if matched_terms == 0:
        return 0.0
    # Per-kind small priors so we don't always lose to bare entity
    # titles. Text units and community reports are the long-form
    # context GraphRAG actually uses for answers.
    kind_boost = {
        "community_reports": 1.10,
        "text_units": 1.05,
        "entities": 1.00,
        "relationships": 0.95,
    }.get(kind, 1.0)
    return score * (matched_terms**0.5) * kind_boost


def _occurrences(term: str, text: str) -> int:
    if not text:
        return 0
    if len(term) <= 2:
        # Avoid pathological substring blowups on tiny tokens.
        return sum(1 for _ in re.finditer(rf"\b{re.escape(term)}\b", text))
    return text.count(term)


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #


def _query_terms(query: str) -> list[str]:
    return [m.group(0) for m in _TOKEN_PATTERN.finditer(query.casefold())]


def _first_text(record: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if not isinstance(value, (dict, list, tuple, set)):
            text = str(value).strip()
            if text:
                return text
    return default


def _join_text(*values: str) -> str:
    return " ".join(v for v in values if v)


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    text = str(value).strip()
    return (text,) if text else ()
