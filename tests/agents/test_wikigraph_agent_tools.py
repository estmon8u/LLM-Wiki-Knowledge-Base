"""Tests for the WikiGraphRAG paths in agent ask_kb/find_kb/update/status."""

from __future__ import annotations

import textwrap
from typing import Any

import pytest

from graphwiki_kb.agents.models import (
    AskKbInput,
    AskKbOutput,
    FindKbInput,
    UpdateInput,
)
from graphwiki_kb.agents.tools.ask_kb import run_ask_kb
from graphwiki_kb.agents.tools.find_kb import run_find_kb
from graphwiki_kb.agents.tools.status import run_status
from graphwiki_kb.agents.tools.update import _run_inprocess, _run_subprocess
from graphwiki_kb.wikigraph.models import (
    WikiGraphAnswer,
    WikiGraphFindResult,
    WikiGraphRetrievedContext,
)

REALM_PAGE = textwrap.dedent(
    """\
    ---
    title: REALM
    type: source
    source_id: realm
    aliases:
      - Retrieval-Augmented Language Model
    summary: REALM pretrains a retriever with masked language modeling.
    ---

    # REALM

    ## Summary

    REALM jointly trains a retriever and a masked language model.

    ## Methods

    REALM backpropagates through retrieval. See [[RAG]].
    """
)
RAG_PAGE = textwrap.dedent(
    """\
    ---
    title: RAG
    type: source
    source_id: rag
    aliases:
      - Retrieval-Augmented Generation
    summary: RAG augments a seq2seq generator with retrieved passages.
    ---

    # RAG

    ## Summary

    RAG combines a frozen retriever and a seq2seq generator.

    ## Methods

    RAG decouples retrieval and generation. See [[REALM]].
    """
)


@pytest.fixture
def seeded_runtime(runtime):
    """Seed two source pages and build the WikiGraphRAG index."""
    (runtime.command_context.project_root / "wiki/sources/realm.md").write_text(
        REALM_PAGE, encoding="utf-8"
    )
    (runtime.command_context.project_root / "wiki/sources/rag.md").write_text(
        RAG_PAGE, encoding="utf-8"
    )
    runtime.services.wikigraph_index.build()
    return runtime


# --------------------------------------------------------------------------- #
# ask_kb engine routing                                                       #
# --------------------------------------------------------------------------- #


def test_ask_kb_wikigraph_provider_free(seeded_runtime) -> None:
    # Ensure the wikigraph query service is using no provider so we exercise
    # the deterministic provider-free synthesis path.
    seeded_runtime.services.wikigraph_query.provider = None

    result = run_ask_kb(
        seeded_runtime,
        AskKbInput(
            question="How does REALM differ from RAG?",
            engine="wikigraph",
            method="local",
        ),
    )
    assert isinstance(result, AskKbOutput)
    assert result.method == "local"
    assert result.planner == "wikigraph"
    assert result.source_trace.get("engine") == "wikigraph"
    # Provider-free synthesis still produces a non-empty answer with citations.
    assert "REALM" in result.answer or "RAG" in result.answer
    trace = seeded_runtime.tool_results[-1]
    assert trace.tool_name == "ask_kb"
    assert trace.ok is True
    assert trace.data["engine"] == "wikigraph"


def test_ask_kb_wikigraph_rejects_drift_method(seeded_runtime) -> None:
    result = run_ask_kb(
        seeded_runtime,
        AskKbInput(question="test", engine="wikigraph", method="drift"),
    )
    assert result.claim_support == "no-answer"
    assert result.answer == ""
    assert any("wikigraph" in w for w in result.staleness_warnings)
    assert seeded_runtime.tool_results[-1].ok is False


def test_ask_kb_graphrag_rejects_drift_lite_method(runtime) -> None:
    result = run_ask_kb(
        runtime, AskKbInput(question="test", engine="graphrag", method="drift-lite")
    )
    assert result.claim_support == "no-answer"
    assert result.answer == ""
    assert any("graphrag" in w for w in result.staleness_warnings)
    assert runtime.tool_results[-1].ok is False


def test_ask_kb_wikigraph_handles_missing_index(runtime) -> None:
    # No build has been run, so wikigraph_query.find() will raise.
    result = run_ask_kb(
        runtime,
        AskKbInput(question="anything", engine="wikigraph", method="local"),
    )
    assert result.answer == ""
    assert result.claim_support == "no-answer"
    assert runtime.tool_results[-1].ok is False


def test_ask_kb_wikigraph_handles_unexpected_exception(seeded_runtime) -> None:
    class _ExplodingService:
        def ask(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    seeded_runtime.services.wikigraph_query = _ExplodingService()  # type: ignore[assignment]
    result = run_ask_kb(
        seeded_runtime,
        AskKbInput(question="x", engine="wikigraph", method="auto"),
    )
    assert result.claim_support == "no-answer"
    assert any("boom" in w for w in result.staleness_warnings)
    assert seeded_runtime.tool_results[-1].ok is False


# --------------------------------------------------------------------------- #
# ask_kb projection edge cases                                                #
# --------------------------------------------------------------------------- #


def test_ask_kb_wikigraph_projection_insufficient_evidence(seeded_runtime) -> None:
    class _FakeService:
        def ask(self, question, *, method, save, **_):
            return WikiGraphAnswer(
                method=method,
                question=question,
                answer="No evidence is available.",
                contexts=[],
                citations=[],
                trace=[],
                warnings=["no_context"],
                insufficient_evidence=True,
                provider_status={"mode": "provider-free"},
            )

    seeded_runtime.services.wikigraph_query = _FakeService()  # type: ignore[assignment]
    result = run_ask_kb(
        seeded_runtime,
        AskKbInput(question="missing", engine="wikigraph", method="auto"),
    )
    # No citations and insufficient_evidence -> stale-index in projection.
    assert result.claim_support == "stale-index"


def test_ask_kb_wikigraph_projection_unverified_when_no_citations(
    seeded_runtime,
) -> None:
    class _FakeService:
        def ask(self, question, *, method, save, **_):
            return WikiGraphAnswer(
                method=method,
                question=question,
                answer="Answer text without citations.",
                contexts=[],
                citations=[],
                trace=[],
                warnings=[],
                insufficient_evidence=False,
                provider_status={"mode": "provider"},
            )

    seeded_runtime.services.wikigraph_query = _FakeService()  # type: ignore[assignment]
    result = run_ask_kb(
        seeded_runtime,
        AskKbInput(question="x", engine="wikigraph", method="auto"),
    )
    assert result.claim_support == "unverified"


# --------------------------------------------------------------------------- #
# find_kb engine wiring                                                       #
# --------------------------------------------------------------------------- #


def test_find_kb_auto_engine_includes_wikigraph(seeded_runtime) -> None:
    result = run_find_kb(
        seeded_runtime, FindKbInput(query="REALM", limit=5, engine="auto")
    )
    retrievers = {r.retriever for r in result.results}
    assert "wikigraph" in retrievers
    assert seeded_runtime.tool_results[-1].ok is True


def test_find_kb_engine_wikigraph_only(seeded_runtime) -> None:
    result = run_find_kb(
        seeded_runtime, FindKbInput(query="REALM", limit=5, engine="wikigraph")
    )
    retrievers = {r.retriever for r in result.results}
    assert retrievers == {"wikigraph"}


def test_find_kb_engine_wikigraph_errors_without_index(runtime) -> None:
    result = run_find_kb(
        runtime, FindKbInput(query="anything", limit=5, engine="wikigraph")
    )
    assert result.results == []
    assert runtime.tool_results[-1].ok is False


def test_find_kb_auto_falls_back_when_wikigraph_missing(runtime) -> None:
    (runtime.command_context.project_root / "wiki/sources/realm.md").write_text(
        REALM_PAGE, encoding="utf-8"
    )
    # No wikigraph build -- auto mode should warn but still return wiki hits.
    result = run_find_kb(runtime, FindKbInput(query="REALM", limit=5, engine="auto"))
    assert any("WikiGraphRAG" in diag for diag in result.graph_diagnostics)
    assert runtime.tool_results[-1].ok is True


def test_find_kb_engine_wikigraph_runtime_error_diagnostic(seeded_runtime) -> None:
    class _ExplodingService:
        def find(self, *args: Any, **kwargs: Any) -> WikiGraphFindResult:
            raise RuntimeError("unexpected")

    seeded_runtime.services.wikigraph_query = _ExplodingService()  # type: ignore[assignment]
    result = run_find_kb(
        seeded_runtime, FindKbInput(query="REALM", limit=5, engine="auto")
    )
    assert any("WikiGraphRAG" in diag for diag in result.graph_diagnostics)


def test_find_kb_empty_query_short_circuits(runtime) -> None:
    result = run_find_kb(runtime, FindKbInput(query="   ", limit=5))
    assert result.results == []
    assert runtime.tool_results[-1].ok is False


def test_find_kb_merge_uses_wikigraph_contexts_directly(runtime) -> None:
    class _FakeService:
        def find(self, query, *, method):
            ctx = WikiGraphRetrievedContext(
                node_id="chunk::x",
                node_kind="chunk",
                title="X",
                path="wiki/sources/x.md",
                text="long evidence text",
                score=0.9,
            )
            return WikiGraphFindResult(query=query, method="local", contexts=[ctx])

    runtime.services.wikigraph_query = _FakeService()  # type: ignore[assignment]
    result = run_find_kb(
        runtime, FindKbInput(query="anything", limit=5, engine="wikigraph")
    )
    assert len(result.results) == 1
    assert result.results[0].retriever == "wikigraph"
    assert result.results[0].title == "X"


# --------------------------------------------------------------------------- #
# update_kb wikigraph wiring                                                  #
# --------------------------------------------------------------------------- #


def test_update_kb_in_process_runs_wikigraph(seeded_runtime, monkeypatch) -> None:
    # The compile preflight requires a provider; the test_project stubs one,
    # but UpdateService.preflight() also checks config["provider"]["name"].
    seeded_runtime.command_context.config.setdefault("provider", {})["name"] = "stub"

    class _StubProvider:
        def ensure_available(self) -> None:
            return None

    seeded_runtime.services.compile.provider = _StubProvider()  # type: ignore[assignment]

    output = _run_inprocess(
        seeded_runtime,
        UpdateInput(no_graph=True, wikigraph=True),
    )
    wg = output.details.get("wikigraph", {})
    assert wg.get("ran") is True
    assert wg.get("node_count", 0) > 0
    assert "wikigraph(" in output.summary


def test_update_kb_subprocess_adds_no_wikigraph_flag(runtime, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_run(command, **kwargs):
        captured["command"] = command

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr("subprocess.run", _fake_run)
    out = _run_subprocess(
        runtime,
        UpdateInput(
            no_graph=True,
            wikigraph=False,
            wikigraph_include_graphrag_export_pages=False,
        ),
    )
    assert out.ok is True
    assert "--no-wikigraph" in captured["command"]
    assert "--wikigraph-include-graphrag-export-pages" not in captured["command"]


def test_update_kb_subprocess_passes_ablation_flag(runtime, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_run(command, **kwargs):
        captured["command"] = command

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr("subprocess.run", _fake_run)
    _run_subprocess(
        runtime,
        UpdateInput(
            no_graph=True,
            wikigraph=True,
            wikigraph_include_graphrag_export_pages=True,
        ),
    )
    assert "--wikigraph-include-graphrag-export-pages" in captured["command"]
    assert "--no-wikigraph" not in captured["command"]


# --------------------------------------------------------------------------- #
# status surfacing                                                            #
# --------------------------------------------------------------------------- #


def test_status_includes_wikigraph_block(seeded_runtime) -> None:
    output = run_status(seeded_runtime)
    assert output.wikigraph is not None
    assert output.wikigraph.initialized is True
    assert output.wikigraph.node_count > 0
    last = seeded_runtime.tool_results[-1]
    assert last.data["wikigraph_initialized"] is True


def test_status_wikigraph_block_when_not_initialized(runtime) -> None:
    output = run_status(runtime)
    assert output.wikigraph is not None
    assert output.wikigraph.initialized is False
    assert "kb update" in output.wikigraph.message
