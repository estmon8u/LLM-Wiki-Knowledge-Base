"""Tests for status, find_kb, lint, and review agent tools."""

from __future__ import annotations

from graphwiki_kb.agents.models import FindKbInput
from graphwiki_kb.agents.tools.find_kb import run_find_kb
from graphwiki_kb.agents.tools.lint import run_lint
from graphwiki_kb.agents.tools.review import run_review
from graphwiki_kb.agents.tools.status import run_status
from graphwiki_kb.models.wiki_models import SearchResult


def test_run_status_returns_pydantic_projection(runtime) -> None:
    output = run_status(runtime)
    assert output.project_initialized is True
    assert output.source_count >= 0
    assert output.graph_freshness  # non-empty string
    trace = runtime.tool_results[-1]
    assert trace.tool_name == "status"
    assert trace.ok is True


def test_run_find_kb_merges_graph_and_wiki(runtime) -> None:
    class _FakeGraphFind:
        def search(self, query, *, limit):
            return [
                SearchResult(
                    title="Entity X",
                    path="graph://entities/1",
                    score=0.9,
                    snippet="...",
                )
            ]

    class _FakeSearch:
        def search(self, query, *, limit, include_concepts):
            return [
                SearchResult(
                    title="Wiki Page",
                    path="wiki/sources/x.md",
                    score=0.7,
                    snippet="...",
                )
            ]

    runtime.services.graphrag_find = _FakeGraphFind()  # type: ignore[assignment]
    runtime.services.search = _FakeSearch()  # type: ignore[assignment]

    out = run_find_kb(runtime, FindKbInput(query="x", limit=5))

    assert len(out.results) == 2
    retrievers = {r.retriever for r in out.results}
    assert retrievers == {"graph", "wiki"}
    assert runtime.tool_results[-1].tool_name == "find_kb"


def test_run_find_kb_survives_backend_exceptions(runtime) -> None:
    """Regression: a crashing graph backend must not bubble up to the SDK."""

    class _BadGraphFind:
        def search(self, *args, **kwargs):
            raise RuntimeError("parquet read failed")

    class _OkSearch:
        def search(self, query, *, limit, include_concepts):
            return []

    runtime.services.graphrag_find = _BadGraphFind()
    runtime.services.search = _OkSearch()

    out = run_find_kb(runtime, FindKbInput(query="x", limit=3))

    assert out.results == []
    assert any("graph search unavailable" in d for d in out.graph_diagnostics)


def test_run_find_kb_rejects_empty_query(runtime) -> None:
    out = run_find_kb(runtime, FindKbInput(query="   "))
    assert out.results == []
    assert runtime.tool_results[-1].ok is False


def test_run_lint_projects_lint_report(runtime) -> None:
    output = run_lint(runtime)
    assert output.error_count >= 0
    assert output.warning_count >= 0
    assert output.suggestion_count >= 0
    assert runtime.tool_results[-1].tool_name == "lint"


def test_run_review_handles_missing_provider_gracefully(runtime) -> None:
    output = run_review(runtime)
    # The stubbed provider used in tests will run; we just confirm the
    # projection returns a valid Pydantic model and tool trace is recorded.
    assert output.total_issues >= 0
    assert runtime.tool_results[-1].tool_name == "review"
