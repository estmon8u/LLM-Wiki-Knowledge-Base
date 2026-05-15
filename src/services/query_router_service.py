"""Query router service service behavior for the knowledge-base workflow.

This module belongs to `src.services.query_router_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from src.services.graphrag_query_service import GRAPH_QUERY_METHODS
from src.services.graphrag_status_service import GraphRAGStatusService


GRAPH_ASK_METHODS = ("auto", *GRAPH_QUERY_METHODS)
GLOBAL_KEYWORDS = (
    "main theme",
    "main themes",
    "overall",
    "across",
    "patterns",
    "landscape",
    "whole corpus",
)
DRIFT_KEYWORDS = (
    "compare",
    "differ",
    "difference",
    "tradeoff",
    "trade-off",
    "relationship between",
    " versus ",
    " vs ",
    "contrast",
)
DIRECT_LOOKUP_KEYWORDS = (
    "where is",
    "where are",
    "configured",
    "configuration",
    "config",
    "stale",
    "freshness",
    "what should happen",
)
LOCAL_ENTITY_KEYWORDS = (
    "dense passage retrieval",
    "fusion-in-decoder",
    "self-rag",
    "realm",
    "fid",
    "dpr",
)
GENERIC_GRAPH_TERMS = {
    "answer",
    "answers",
    "corpus",
    "data",
    "document",
    "documents",
    "generation",
    "model",
    "models",
    "query",
    "rag",
    "retrieval",
    "source",
    "sources",
}


class QueryRouterError(ValueError):
    """Error raised for query router failures.

    Attributes:
        See annotated class attributes for stored values.
    """

    pass


@dataclass(frozen=True)
class QueryRoute:
    """Represents query route behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    method: str
    planner: str = "heuristic"
    reason: str = "fallback"


class QueryRouterService:
    """Coordinates query router operations.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(self, status_service: GraphRAGStatusService | None = None) -> None:
        self.status_service = status_service
        self._known_terms_cache: tuple[str | None, tuple[str, ...]] | None = None

    def route(self, question: str, *, method: str = "auto") -> QueryRoute:
        """Route.

        Args:
            question: User question to answer from available evidence.
            method: Method value used by the operation.

        Returns:
            QueryRoute produced by the operation.
        """
        normalized_method = method.strip().lower()
        if normalized_method not in GRAPH_ASK_METHODS:
            supported = ", ".join(GRAPH_ASK_METHODS)
            raise QueryRouterError(
                f"Unsupported GraphRAG method '{method}'. Use {supported}."
            )
        if normalized_method != "auto":
            return QueryRoute(
                method=normalized_method,
                reason="explicit method override",
            )

        text = f" {question.casefold()} "
        if any(keyword in text for keyword in DRIFT_KEYWORDS):
            return QueryRoute(method="drift", reason="comparison keyword")
        if any(keyword in text for keyword in GLOBAL_KEYWORDS):
            return QueryRoute(method="global", reason="global corpus keyword")
        if any(keyword in text for keyword in DIRECT_LOOKUP_KEYWORDS):
            return QueryRoute(
                method="basic", reason="direct lookup or maintenance keyword"
            )
        if any(keyword in text for keyword in LOCAL_ENTITY_KEYWORDS):
            return QueryRoute(method="local", reason="known retrieval system keyword")
        if self._mentions_known_graph_term(text):
            return QueryRoute(method="local", reason="known graph entity or document")
        return QueryRoute(method="basic", reason="basic vector baseline fallback")

    def _mentions_known_graph_term(self, normalized_question: str) -> bool:
        if self.status_service is None:
            return False
        for term in self._known_graph_terms():
            if _term_in_question(term, normalized_question):
                return True
        return False

    def _known_graph_terms(self) -> Iterable[str]:
        status = self.status_service.status()
        cache_key = status.output_updated_at or status.last_index_run_id
        if self._known_terms_cache is not None:
            cached_key, cached_terms = self._known_terms_cache
            if cached_key == cache_key:
                return cached_terms
        terms: list[str] = []
        for table_name in ("entities", "documents"):
            table_path = self.status_service.table_path(table_name)
            if table_path is None:
                continue
            terms.extend(_read_term_columns(table_path))
        known_terms = tuple(
            dict.fromkeys(_usable_term(term) for term in terms if _usable_term(term))
        )
        self._known_terms_cache = (cache_key, known_terms)
        return known_terms


def _read_term_columns(path: Path) -> Iterable[str]:
    columns = _available_term_columns(path)
    if not columns:
        return []
    try:
        frame = pd.read_parquet(path, columns=columns)
    except Exception:
        return []
    terms: list[str] = []
    for column in columns:
        for value in frame[column].dropna().tolist():
            text = str(value).strip()
            if len(text) >= 3:
                terms.append(text)
    return terms


def _available_term_columns(path: Path) -> list[str]:
    candidates = ("title", "name", "human_readable_id", "id")
    try:
        import pyarrow.parquet as parquet

        available = set(parquet.read_schema(path).names)
    except Exception:
        try:
            frame = pd.read_parquet(path, columns=None)
        except Exception:
            return []
        available = set(frame.columns)
    return [name for name in candidates if name in available]


def _usable_term(term: str) -> str:
    normalized = term.casefold().strip()
    if len(normalized) < 3 or normalized in GENERIC_GRAPH_TERMS:
        return ""
    return term.strip()


def _term_in_question(term: str, normalized_question: str) -> bool:
    normalized_term = term.casefold().strip()
    if not normalized_term:
        return False
    if " " in normalized_term or "-" in normalized_term:
        return normalized_term in normalized_question
    return (
        re.search(rf"\b{re.escape(normalized_term)}\b", normalized_question) is not None
    )
