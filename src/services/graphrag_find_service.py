"""Direct GraphRAG artifact search for the top-level `kb find` command."""

from __future__ import annotations

import logging
import math
from pathlib import Path
import re
from typing import Any

from src.models.wiki_models import SearchResult
from src.services.graphrag_status_service import GraphRAGStatusService
from src.services.project_service import ProjectPaths, slugify


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_GRAPH_FIND_TABLES = ("entities", "relationships")
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

        results: list[SearchResult] = []
        for table_name in _GRAPH_FIND_TABLES:
            table_path = self.status_service.table_path(table_name)
            if table_path is None:
                continue
            for record in _read_parquet_records(table_path):
                result = _record_result(table_name, record, terms)
                if result is not None:
                    results.append(result)

        results.sort(
            key=lambda item: (
                item.score,
                1 if item.section == "GraphRAG Entity" else 0,
                item.title.casefold(),
            ),
            reverse=True,
        )
        return results[:limit]


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
    try:
        import pyarrow.lib as arrow_lib
        import pyarrow.parquet as parquet
    except ImportError as exc:
        logger.debug(
            "PyArrow is unavailable; skipping GraphRAG artifact search for %s",
            path,
            exc_info=True,
        )
        return []

    try:
        table = parquet.read_table(path)
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
        return []
    return [
        {str(key): _clean_value(value) for key, value in row.items()}
        for row in table.to_pylist()
    ]


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
    score = 0.0
    for term in terms:
        if term in title_text:
            score += 10.0
        score += float(searchable.count(term))
    return score


def _fallback_snippet(record: dict[str, Any]) -> str:
    for key in ("title", "source", "target", "id"):
        value = _first_text(record, key)
        if value:
            return value
    return "GraphRAG artifact match."
