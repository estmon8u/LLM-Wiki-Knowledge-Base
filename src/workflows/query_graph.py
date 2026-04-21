"""LangGraph StateGraph for self-consistency query workflow."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from src.services.query_service import QueryService

from src.models.wiki_models import SearchResult
from src.providers.base import TextProvider
from src.schemas.claims import CandidateAnswer, EvidenceBundle, MergedAnswer
from src.schemas.runs import RunRecord
from src.services.query_service import QueryAnswer
from src.storage.run_store import RunStore


# ── Prompt version must match the service's constant ──────────────
_QUERY_PROMPT_VERSION = "query-self-consistency-v1"


class QueryGraphState(TypedDict, total=False):
    """Mutable state flowing through the query graph."""

    # Inputs (set by caller before invocation)
    question: str
    matches: list[SearchResult]
    sample_count: int
    provider: TextProvider
    run_store: RunStore | None
    service: Any  # QueryService – avoids circular import at runtime

    # Intermediate
    evidence_bundle: EvidenceBundle
    candidates: list[CandidateAnswer]
    merged: MergedAnswer
    started_at: float
    wall_time_ms: int
    model_id: str
    unresolved_disagreement: bool
    run_id: str | None

    # Output
    query_answer: QueryAnswer


# ── Node functions ────────────────────────────────────────────────


def build_evidence(state: QueryGraphState) -> dict:
    svc = state["service"]
    bundle = svc._build_evidence_bundle(state["question"], state["matches"])
    return {"evidence_bundle": bundle, "started_at": time.perf_counter()}


def sample_candidates(state: QueryGraphState) -> dict:
    svc = state["service"]
    candidates = asyncio.run(
        svc._sample_candidates(
            state["question"],
            state["matches"],
            state["evidence_bundle"],
            state["sample_count"],
            state["provider"],
        )
    )

    from src.providers import ProviderExecutionError

    sample_errors = [c.error for c in candidates if c.error]
    if sample_errors:
        raise ProviderExecutionError(
            "Self-consistency query failed: " + "; ".join(sample_errors)
        )
    successful = [c for c in candidates if not c.error]
    if not successful:
        raise ProviderExecutionError(
            "Self-consistency query failed: all provider samples failed."
        )
    return {"candidates": successful}


def merge_candidates(state: QueryGraphState) -> dict:
    svc = state["service"]
    merged = svc._merge_candidates(state["candidates"], state["evidence_bundle"])
    wall_time_ms = int((time.perf_counter() - state["started_at"]) * 1000)
    model_id = next(
        (c.model_name for c in state["candidates"] if c.model_name),
        getattr(state["provider"], "name", ""),
    )
    unresolved = bool(
        merged.dropped_claims or any(c.error for c in state["candidates"])
    )
    return {
        "merged": merged,
        "wall_time_ms": wall_time_ms,
        "model_id": model_id,
        "unresolved_disagreement": unresolved,
    }


def persist_run(state: QueryGraphState) -> dict:
    run_store = state.get("run_store")
    if run_store is None:
        return {"run_id": None}
    record = RunRecord(
        command="query",
        model_id=state["model_id"],
        prompt_version=_QUERY_PROMPT_VERSION,
        evidence_bundle=state["evidence_bundle"],
        context_hash=state["evidence_bundle"].context_hash,
        candidates=state["candidates"],
        merged_answer=state["merged"],
        final_text=state["merged"].text,
        token_cost=0,
        wall_time_ms=state["wall_time_ms"],
        unresolved_disagreement=state["unresolved_disagreement"],
    )
    run_id = run_store.save_run(record)
    return {"run_id": run_id}


def render_answer(state: QueryGraphState) -> dict:
    merged = state["merged"]
    model_id = state["model_id"]
    sample_count = state["sample_count"]
    return {
        "query_answer": QueryAnswer(
            answer=merged.text,
            citations=state["matches"],
            mode=f"self-consistency:{model_id}:{sample_count}",
            run_id=state.get("run_id"),
        )
    }


# ── Graph construction ────────────────────────────────────────────


def _build_query_graph() -> StateGraph:
    graph = StateGraph(QueryGraphState)
    graph.add_node("build_evidence", build_evidence)
    graph.add_node("sample_candidates", sample_candidates)
    graph.add_node("merge_candidates", merge_candidates)
    graph.add_node("persist_run", persist_run)
    graph.add_node("render_answer", render_answer)

    graph.set_entry_point("build_evidence")
    graph.add_edge("build_evidence", "sample_candidates")
    graph.add_edge("sample_candidates", "merge_candidates")
    graph.add_edge("merge_candidates", "persist_run")
    graph.add_edge("persist_run", "render_answer")
    graph.add_edge("render_answer", END)
    return graph


def run_query_graph(
    service: "QueryService",
    question: str,
    matches: list[SearchResult],
    sample_count: int,
    provider: TextProvider,
) -> QueryAnswer:
    """Execute the self-consistency query as a LangGraph StateGraph."""
    graph = _build_query_graph()
    compiled = graph.compile()

    result = compiled.invoke(
        {
            "question": question,
            "matches": matches,
            "sample_count": sample_count,
            "provider": provider,
            "run_store": service.run_store,
            "service": service,
        },
    )
    return result["query_answer"]
