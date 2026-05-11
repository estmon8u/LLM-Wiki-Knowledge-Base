from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from src.services.graphrag_query_service import GRAPH_QUERY_METHODS
from src.services.graphrag_status_service import (
    GRAPH_OUTPUT_TABLES,
    GraphRAGStatusService,
)


GRAPH_ASK_METHODS = ("auto", *GRAPH_QUERY_METHODS)
GLOBAL_KEYWORDS = (
    "main theme",
    "main themes",
    "overall",
    "across",
    "patterns",
    "landscape",
    "whole corpus",
    "corpus",
    "dataset",
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


class QueryRouterError(ValueError):
    pass


@dataclass(frozen=True)
class QueryRoute:
    method: str
    planner: str = "heuristic"
    reason: str = "fallback"


class QueryRouterService:
    def __init__(self, status_service: GraphRAGStatusService | None = None) -> None:
        self.status_service = status_service

    def route(self, question: str, *, method: str = "auto") -> QueryRoute:
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
        if any(keyword in text for keyword in GLOBAL_KEYWORDS):
            return QueryRoute(method="global", reason="global corpus keyword")
        if any(keyword in text for keyword in DRIFT_KEYWORDS):
            return QueryRoute(method="drift", reason="comparison keyword")
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
        output_dir = self.status_service.output_dir
        for table_name in ("entities", "documents"):
            table_path = _find_table_path(output_dir, *GRAPH_OUTPUT_TABLES[table_name])
            if table_path is None:
                continue
            yield from _read_term_columns(table_path)


def _find_table_path(output_dir: Path, *tokens: str) -> Path | None:
    if not output_dir.exists():
        return None
    lowered = tuple(token.casefold() for token in tokens)
    for path in sorted(output_dir.rglob("*.parquet")):
        stem = path.stem.casefold()
        if any(stem == token or token in stem for token in lowered):
            return path
    return None


def _read_term_columns(path: Path) -> Iterable[str]:
    try:
        frame = pd.read_parquet(path, columns=None)
    except Exception:
        return []
    columns = [
        name for name in ("title", "name", "human_readable_id", "id") if name in frame
    ]
    terms: list[str] = []
    for column in columns:
        for value in frame[column].dropna().tolist():
            text = str(value).strip()
            if len(text) >= 3:
                terms.append(text)
    return terms


def _term_in_question(term: str, normalized_question: str) -> bool:
    normalized_term = term.casefold().strip()
    if not normalized_term:
        return False
    if " " in normalized_term or "-" in normalized_term:
        return normalized_term in normalized_question
    return (
        re.search(rf"\b{re.escape(normalized_term)}\b", normalized_question) is not None
    )
