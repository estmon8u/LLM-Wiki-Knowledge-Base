"""LangGraph StateGraph for adversarial review workflow."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from src.services.review_service import ReviewService

from src.models.wiki_models import ReviewReport
from src.schemas.claims import EvidenceBundle
from src.schemas.review import ReviewFinding
from src.services.review_service import (
    PageSnapshot,
    PairReviewResult,
    ReviewPair,
)


class ReviewGraphState(TypedDict, total=False):
    """Mutable state flowing through the review graph."""

    # Inputs
    service: Any  # ReviewService – avoids circular import at runtime
    provider: Any  # TextProvider
    run_store: Any  # RunStore | None

    # Intermediate
    snapshots: list[PageSnapshot]
    pairs: list[ReviewPair]
    evidence_bundle: EvidenceBundle
    pair_results: list[PairReviewResult]
    findings: list[ReviewFinding]
    started_at: float
    wall_time_ms: int
    model_name: str
    unresolved: bool
    run_id: str | None

    # Output
    result_findings: list[ReviewFinding]
    result_mode: str
    result_run_id: str | None


# ── Node functions ────────────────────────────────────────────────


def load_snapshots(state: ReviewGraphState) -> dict:
    svc = state["service"]
    snapshots = svc._load_source_page_snapshots()
    return {"snapshots": snapshots, "started_at": time.perf_counter()}


def build_pairs(state: ReviewGraphState) -> dict:
    svc = state["service"]
    pairs = svc._build_candidate_pairs(state["snapshots"])
    return {"pairs": pairs}


def build_evidence(state: ReviewGraphState) -> dict:
    svc = state["service"]
    bundle = svc._build_review_evidence_bundle(state["pairs"], state["snapshots"])
    return {"evidence_bundle": bundle}


def run_pair_reviews(state: ReviewGraphState) -> dict:
    svc = state["service"]
    pairs = state["pairs"]
    if not pairs:
        return {
            "pair_results": [],
            "wall_time_ms": 0,
        }

    from src.providers import ProviderExecutionError

    try:
        results = asyncio.run(svc._run_adversarial_pairs(pairs))
    except Exception as exc:
        raise ProviderExecutionError(f"Adversarial review failed: {exc}") from exc

    wall_time_ms = int((time.perf_counter() - state["started_at"]) * 1000)
    return {"pair_results": results, "wall_time_ms": wall_time_ms}


def flatten_findings(state: ReviewGraphState) -> dict:
    results = state["pair_results"]
    findings = [f for r in results for f in r.findings]
    errors = [r.error for r in results if r.error]

    if results and len(errors) == len(results):
        from src.providers import ProviderExecutionError

        raise ProviderExecutionError("Adversarial review failed: " + "; ".join(errors))

    model_name = ""
    for r in results:
        if r.model_name:
            model_name = r.model_name
            break
    if not model_name:
        model_name = getattr(state["provider"], "name", "")

    from src.schemas.review import Verdict

    unresolved = any(f.verdict == Verdict.NEEDS_REVIEW for f in findings) or bool(
        errors
    )

    return {
        "findings": findings,
        "model_name": model_name,
        "unresolved": unresolved,
    }


def persist_run(state: ReviewGraphState) -> dict:
    svc = state["service"]
    run_id = svc._persist_review_run(
        evidence_bundle=state["evidence_bundle"],
        findings=state["findings"],
        model_id=state["model_name"],
        wall_time_ms=state.get("wall_time_ms", 0),
        unresolved_disagreement=state["unresolved"],
    )
    return {"run_id": run_id}


def render_report(state: ReviewGraphState) -> dict:
    return {
        "result_findings": state["findings"],
        "result_mode": f"adversarial:{state['model_name']}",
        "result_run_id": state.get("run_id"),
    }


# ── Graph construction ────────────────────────────────────────────


def _build_review_graph() -> StateGraph:
    graph = StateGraph(ReviewGraphState)
    graph.add_node("load_snapshots", load_snapshots)
    graph.add_node("build_pairs", build_pairs)
    graph.add_node("build_evidence", build_evidence)
    graph.add_node("run_pair_reviews", run_pair_reviews)
    graph.add_node("flatten_findings", flatten_findings)
    graph.add_node("persist_run", persist_run)
    graph.add_node("render_report", render_report)

    graph.set_entry_point("load_snapshots")
    graph.add_edge("load_snapshots", "build_pairs")
    graph.add_edge("build_pairs", "build_evidence")
    graph.add_edge("build_evidence", "run_pair_reviews")
    graph.add_edge("run_pair_reviews", "flatten_findings")
    graph.add_edge("flatten_findings", "persist_run")
    graph.add_edge("persist_run", "render_report")
    graph.add_edge("render_report", END)
    return graph


def run_review_graph(
    service: "ReviewService",
) -> tuple[list[ReviewFinding], str, str | None]:
    """Execute the adversarial review as a LangGraph StateGraph."""
    graph = _build_review_graph()
    compiled = graph.compile()

    result = compiled.invoke(
        {
            "service": service,
            "provider": service.provider,
            "run_store": service.run_store,
        },
    )
    return (
        result["result_findings"],
        result["result_mode"],
        result.get("result_run_id"),
    )
