"""Direct GraphRAG artifact search for the top-level `kb find` command."""

from __future__ import annotations

import heapq
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from graphwiki_kb.models.wiki_models import SearchResult
from graphwiki_kb.services.graphrag_status_service import GraphRAGStatusService
from graphwiki_kb.services.project_service import ProjectPaths, slugify

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_GRAPH_FIND_TABLES = ("entities", "relationships")
_GRAPH_FIND_COLUMNS = {
    "entities": (
        "id",
        "human_readable_id",
        "title",
        "name",
        "description",
        "summary",
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
        "summary",
    ),
}
logger = logging.getLogger(__name__)


class GraphRAGFindService:
    """Search GraphRAG entity and relationship parquet artifacts directly."""

    def __init__(
        self,
        paths: ProjectPaths,
        status_service: GraphRAGStatusService,
    ) -> None:
        self.paths = paths
        self.status_service = status_service

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        """Return direct graph artifact matches for entity/relationship records."""
        terms = _query_terms(query)
        if not terms or limit <= 0:
            return []

        heap: list[tuple[tuple[float, int, str], int, SearchResult]] = []
        counter = 0
        for table_name in _GRAPH_FIND_TABLES:
            table_path = self.status_service.table_path(table_name)
            if table_path is None:
                continue
            for record in _iter_parquet_records(table_path):
                result = _record_result(table_name, record, terms)
                if result is not None:
                    counter += 1
                    item = (_result_sort_key(result), counter, result)
                    if len(heap) < limit:
                        heapq.heappush(heap, item)
                    elif item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)

        return [
            item[2]
            for item in sorted(
                heap,
                key=lambda item: item[0],
                reverse=True,
            )
        ]


def _record_result(
    table_name: str,
    record: dict[str, Any],
    terms: list[str],
) -> SearchResult | None:
    if table_name == "entities":
        title = _first_text(record, "title", "name", "id", default="Entity")
        description = _first_text(record, "description", "summary")
        artifact_id = _first_text(record, "id", "human_readable_id") or slugify(title)
        path = f"graph://entities/{artifact_id}"
        section = "GraphRAG Entity"
        searchable = _join_search_text(
            title,
            description,
            _first_text(record, "type"),
            _first_text(record, "community"),
        )
    elif table_name == "relationships":
        source = _first_text(record, "source", "source_title", default="source")
        target = _first_text(record, "target", "target_title", default="target")
        title = f"{source} -> {target}"
        description = _first_text(record, "description", "summary")
        artifact_id = _first_text(record, "id", "human_readable_id") or slugify(title)
        path = f"graph://relationships/{artifact_id}"
        section = "GraphRAG Relationship"
        searchable = _join_search_text(source, target, description)
    else:
        return None

    score = _score(terms, title=title, searchable=searchable)
    if score <= 0:
        return None
    return SearchResult(
        title=title,
        path=path,
        score=score,
        snippet=description or _fallback_snippet(record),
        section=section,
        chunk_index=None,
    )


def _read_parquet_records(path: Path) -> list[dict[str, Any]]:
    return list(_iter_parquet_records(path))


def _iter_parquet_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.lib as arrow_lib
        import pyarrow.parquet as parquet
    except ImportError:
        logger.debug(
            "PyArrow is unavailable; skipping GraphRAG artifact search for %s",
            path,
            exc_info=True,
        )
        return

    try:
        parquet_file = parquet.ParquetFile(path)
        available_columns = set(parquet_file.schema.names)
        requested_columns = [
            column for column in _projected_columns(path) if column in available_columns
        ]
        batches = parquet_file.iter_batches(columns=requested_columns or None)
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        arrow_lib.ArrowException,
    ) as exc:
        logger.debug(
            "Unable to read GraphRAG parquet table %s for kb find: %s",
            path,
            exc,
            exc_info=True,
        )
        return
    try:
        for batch in batches:
            for row in batch.to_pylist():
                yield {str(key): _clean_value(value) for key, value in row.items()}
    except (RuntimeError, arrow_lib.ArrowException) as exc:
        logger.debug(
            "Unable to scan GraphRAG parquet table %s for kb find: %s",
            path,
            exc,
            exc_info=True,
        )


def _result_sort_key(result: SearchResult) -> tuple[float, int, str]:
    return (
        result.score,
        1 if result.section == "GraphRAG Entity" else 0,
        result.title.casefold(),
    )


def _projected_columns(path: Path) -> tuple[str, ...]:
    stem = path.stem.lower()
    for table_name, columns in _GRAPH_FIND_COLUMNS.items():
        if table_name in stem:
            return columns
    return tuple(
        sorted(
            {column for columns in _GRAPH_FIND_COLUMNS.values() for column in columns}
        )
    )


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "tolist"):
        return _clean_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): _clean_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_value(item) for item in value]
    return value


def _query_terms(query: str) -> list[str]:
    return [match.group(0) for match in _TOKEN_PATTERN.finditer(query.casefold())]


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


def _join_search_text(*values: str) -> str:
    return " ".join(value for value in values if value).casefold()


def _score(terms: list[str], *, title: str, searchable: str) -> float:
    title_text = title.casefold()
    title_tokens = Counter(_query_terms(title_text))
    searchable_tokens = Counter(_query_terms(searchable))
    score = 0.0
    for term in terms:
        title_matches = _token_match_count(term, title_tokens)
        searchable_matches = _token_match_count(term, searchable_tokens)
        if title_matches:
            score += 10.0 * title_matches
        elif len(term) > 2 and re.search(rf"\b{re.escape(term)}\b", title_text):
            score += 10.0
        score += float(searchable_matches)
    return score


def _token_match_count(term: str, tokens: Counter[str]) -> int:
    if len(term) <= 2:
        return tokens[term]
    exact = tokens[term]
    prefix = sum(
        count
        for token, count in tokens.items()
        if token != term and token.startswith(term)
    )
    return exact + prefix


def _fallback_snippet(record: dict[str, Any]) -> str:
    for key in ("title", "source", "target", "id"):
        value = _first_text(record, key)
        if value:
            return value
    return "GraphRAG artifact match."
