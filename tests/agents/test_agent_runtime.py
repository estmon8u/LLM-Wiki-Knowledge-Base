"""Tests for agent runtime and additional tools."""

from __future__ import annotations

import json
from dataclasses import dataclass

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    FindKbInput,
    IngestRecommendationInput,
)
from graphwiki_kb.agents.runtime import run_agent_turn
from graphwiki_kb.agents.tools.find_kb import run_find_kb
from graphwiki_kb.agents.tools.ingest_recommendation import run_ingest_recommendation
from graphwiki_kb.agents.tools.lint import run_lint_kb
from graphwiki_kb.agents.tools.status import run_status_kb
from graphwiki_kb.services.research_service import ResearchService
from graphwiki_kb.services.source_recommendation_store import SourceRecommendationStore
from graphwiki_kb.services.web_research_service import WebResearchService


@dataclass
class _FakeInterruption:
    name: str
    arguments: str


@dataclass
class _FakeResult:
    final_output: str
    interruptions: list[_FakeInterruption]
    _state: _FakeState | None = None

    def to_state(self) -> _FakeState:
        if self._state is None:
            self._state = _FakeState()
        return self._state


class _FakeState:
    def __init__(self) -> None:
        self.approved: list[str] = []
        self.rejected: list[str] = []

    def approve(self, interruption: _FakeInterruption, **kwargs: object) -> None:
        self.approved.append(interruption.name)

    def reject(self, interruption: _FakeInterruption, **kwargs: object) -> None:
        self.rejected.append(interruption.name)


def test_run_agent_turn_handles_approval(test_project, monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_run_sync(agent, input_value, **kwargs: object) -> _FakeResult:
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResult(
                final_output="",
                interruptions=[
                    _FakeInterruption("update_kb", '{"graph_method":"auto"}')
                ],
            )
        return _FakeResult(final_output="Update done.", interruptions=[])

    monkeypatch.setattr("agents.run.Runner.run_sync", _fake_run_sync)

    runtime = AgentRuntimeContext(
        command_context=test_project.command_context,
        services=test_project.services,
    )
    output, pending = run_agent_turn(
        runtime,
        "update the kb",
        approval_callback=lambda item: item.name == "update_kb",
    )
    assert "Update done" in output
    assert pending[0]["status"] == "approved"


def test_readonly_tools_return_json(test_project) -> None:
    runtime = AgentRuntimeContext(
        command_context=test_project.command_context,
        services=test_project.services,
    )
    status_payload = json.loads(run_status_kb(runtime))
    assert "summary" in status_payload
    lint_payload = json.loads(run_lint_kb(runtime))
    assert "error_count" in lint_payload
    find_payload = json.loads(run_find_kb(runtime, FindKbInput(query="graph", limit=3)))
    assert find_payload["query"] == "graph"


def test_research_service_local_only(test_project, monkeypatch) -> None:

    class _Ask:
        def ask(self, question: str, **kwargs: object):
            from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryAnswer

            return GraphRAGQueryAnswer(
                question=question,
                answer="Local only",
                raw_output="",
                method="local",
                created_at="2026-05-19T00:00:00+00:00",
                index_run_id=None,
                command=(),
                stdout="",
                stderr="",
                graph_input_hash="",
                claim_support="graph-index-answer",
            )

    web = WebResearchService(client=object())
    monkeypatch.setattr(
        web,
        "research",
        lambda **kwargs: web.research_from_text(
            question=kwargs["question"],
            local_answer=kwargs["local_answer"],
            kb_gaps=kwargs["kb_gaps"],
            summary_text="Web summary",
            source_urls=["https://example.com/doc"],
        ),
    )
    service = ResearchService(
        test_project.paths,
        test_project.config,
        ask_controller=_Ask(),  # type: ignore[arg-type]
        web_research=web,
    )
    output = service.run(question="test topic", use_web=True)
    assert output.local_answer.answer == "Local only"
    assert output.recommendations
    assert output.recommendations[0].id == 1


def test_ingest_recommendation_tool(test_project, monkeypatch) -> None:
    from graphwiki_kb.agents.models import SourceRecommendation
    from graphwiki_kb.services.ingest_service import IngestResult
    from graphwiki_kb.services.web_source_acquisition_service import StagedSource

    store = SourceRecommendationStore(test_project.paths)
    from graphwiki_kb.agents.models import ResearchRunRecord

    store.save_run(
        ResearchRunRecord(
            run_id="run_ingest",
            question="q",
            created_at="2026-05-19T00:00:00+00:00",
            local_answer={},
            recommendations=[
                SourceRecommendation(
                    id=1,
                    title="Doc",
                    url="https://example.com/x",
                    source_type="article",
                    retrieved_at="2026-05-19T00:00:00+00:00",
                    why_add="r",
                    knowledge_gap="g",
                    novelty="low",
                    confidence="medium",
                    ingestable=True,
                )
            ],
        )
    )

    class _Acquisition:
        def stage(self, recommendation, *, run_id: str) -> StagedSource:
            path = test_project.root / "raw" / "web_staging" / "rec.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# doc\n", encoding="utf-8")
            return StagedSource(
                1, "Doc", "https://example.com/x", path, "raw/web_staging/rec.md"
            )

    monkeypatch.setattr(
        "graphwiki_kb.agents.tools.ingest_recommendation.WebSourceAcquisitionService",
        lambda paths: _Acquisition(),
    )
    test_project.services.ingest.ingest_path = lambda path: IngestResult(  # type: ignore[method-assign]
        created=True,
        source=None,
        message="ok",
    )
    runtime = AgentRuntimeContext(
        command_context=test_project.command_context,
        services=test_project.services,
    )
    text = run_ingest_recommendation(
        runtime,
        IngestRecommendationInput(recommendation_ids=[1]),
    )
    assert "ingested" in text.lower()
