"""Tests for the research agent tool and ResearchService."""

from __future__ import annotations

from typing import Any

import pytest

from graphwiki_kb.agents.models import (
    ResearchInput,
    SourceRecommendation,
    WebFinding,
    WebResearchResult,
)
from graphwiki_kb.agents.tools.research import run_research
from graphwiki_kb.services.research_service import (
    ResearchService,
    derive_kb_gaps,
    project_ask_kb_output,
    renumber_recommendations,
)
from graphwiki_kb.services.source_recommendation_store import (
    SourceRecommendationStore,
)
from graphwiki_kb.services.web_research_service import WebResearchError


class _StubController:
    def __init__(self, *, answer: str = "local answer", method: str = "global") -> None:
        from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryAnswer

        self.answer = GraphRAGQueryAnswer(
            question="q",
            answer=answer,
            raw_output=answer,
            method=method,
            created_at="2026-05-19T00:00:00+00:00",
            index_run_id="run-1",
            command=(),
            stdout="",
            stderr="",
            graph_input_hash="h",
            input_manifest_hash="m",
            claim_support="cited-graph-answer" if "[Data:" in answer else "unverified",
            source_trace={"input_path": "graph/input"},
            staleness_warnings=[],
        )

    def ask(self, question, **kwargs):
        return self.answer


class _StubWebService:
    def __init__(
        self,
        *,
        result: WebResearchResult | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def research(
        self,
        *,
        question,
        local_answer,
        kb_gaps,
        search_context_size,
        max_recommendations,
    ):
        self.calls.append(
            {
                "question": question,
                "kb_gaps": list(kb_gaps),
                "max_recommendations": max_recommendations,
                "search_context_size": search_context_size,
            }
        )
        if self.raises is not None:
            raise self.raises
        assert self.result is not None
        return self.result


def _sample_web_result() -> WebResearchResult:
    return WebResearchResult(
        findings=[
            WebFinding(
                title="Paper",
                url="https://arxiv.org/p",
                summary="s",
                relevance="high",
                supports_recommendation=True,
            )
        ],
        recommendations=[
            SourceRecommendation(
                id=99,
                title="Paper",
                url="https://arxiv.org/p",
                source_type="paper",
                retrieved_at="2026-05-19T00:00:00+00:00",
                why_add="addresses recent benchmark gap",
                novelty="high",
            )
        ],
        sources=["https://arxiv.org/p"],
        raw_text="{}",
    )


def test_research_service_renumbers_and_persists(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    service = ResearchService(
        test_project.paths,
        _StubController(),
        store,
        web_service=_StubWebService(result=_sample_web_result()),
    )
    result = service.research(ResearchInput(question="benchmarks", use_web=True))
    assert result.web_used is True
    assert len(result.recommendations) == 1
    assert result.recommendations[0].id == 1  # renumbered
    assert result.saved_report_path is not None
    record = store.latest()
    assert record is not None
    assert record.recommendations[0].id == 1


def test_research_service_skips_web_when_disabled(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    web_service = _StubWebService(result=_sample_web_result())
    service = ResearchService(
        test_project.paths,
        _StubController(),
        store,
        web_service=web_service,
    )
    result = service.research(ResearchInput(question="q", use_web=False))
    assert result.web_used is False
    assert result.web_findings == []
    assert result.recommendations == []
    assert web_service.calls == []


def test_research_service_handles_web_error_gracefully(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    web_service = _StubWebService(raises=WebResearchError("api down"))
    service = ResearchService(
        test_project.paths,
        _StubController(),
        store,
        web_service=web_service,
    )
    result = service.research(ResearchInput(question="q", use_web=True))
    assert result.web_used is False
    assert result.recommendations == []
    # Run is still persisted even when web fails
    assert store.latest() is not None


def test_run_research_tool_records_trace(runtime, test_project) -> None:
    service = ResearchService(
        test_project.paths,
        _StubController(),
        runtime.recommendation_store,
        web_service=_StubWebService(result=_sample_web_result()),
    )
    runtime.metadata["research_service"] = service

    result = run_research(runtime, ResearchInput(question="benchmarks"))

    assert len(result.recommendations) == 1
    trace = runtime.tool_results[-1]
    assert trace.tool_name == "research"
    assert trace.ok is True
    assert trace.data["recommendation_count"] == 1


def test_run_research_requires_service_in_metadata(runtime) -> None:
    runtime.metadata.pop("research_service", None)
    with pytest.raises(RuntimeError):
        run_research(runtime, ResearchInput(question="x"))


def test_derive_kb_gaps_picks_up_missing_answer() -> None:
    from graphwiki_kb.agents.models import AskKbOutput

    gaps = derive_kb_gaps(
        AskKbOutput(answer="", method="auto", claim_support="no-answer"),
        answer_obj=None,
    )
    assert any("no answer" in gap.lower() for gap in gaps)


def test_renumber_recommendations_starts_at_one() -> None:
    rec = SourceRecommendation(
        id=42,
        title="x",
        url="https://x",
        source_type="paper",
        retrieved_at="2026-05-19T00:00:00+00:00",
        why_add="r",
    )
    out = renumber_recommendations([rec, rec.model_copy(update={"id": 99})])
    assert [r.id for r in out] == [1, 2]


def test_project_ask_kb_output_normalizes_claim_support() -> None:
    from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryAnswer

    raw = GraphRAGQueryAnswer(
        question="q",
        answer="a",
        raw_output="a",
        method="local",
        created_at="2026-05-19T00:00:00+00:00",
        index_run_id=None,
        command=(),
        stdout="",
        stderr="",
        graph_input_hash="h",
        claim_support="unknown-value",
    )
    projection = project_ask_kb_output(raw)
    assert projection.claim_support == "unverified"
