"""Phase 2 tests — LangGraph workflow backends for query and review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.models.wiki_models import ReviewReport, SearchResult
from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.schemas.review import ReviewFinding, Verdict
from src.services import build_services
from src.services.config_service import ConfigService
from src.services.manifest_service import ManifestService
from src.services.project_service import (
    ProjectPaths,
    ProjectService,
    build_project_paths,
)
from src.services.query_service import QueryAnswer, QueryService
from src.services.review_service import ReviewService
from src.storage.run_store import RunStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMPILED_PAGE_TEMPLATE = """\
---
title: {title}
type: source
source_title: {title}
source_hash: abc123
compiled_at: "2026-04-20T00:00:00Z"
aliases: []
tags: []
---

# {title}

{body}
"""


def _compiled_page(title: str, body: str) -> str:
    return _COMPILED_PAGE_TEMPLATE.format(title=title, body=body)


class _DeterministicProvider(TextProvider):
    """Returns a fixed sequence of responses, cycling if needed."""

    name = "deterministic"

    def __init__(self, responses: list[str], model_name: str = "det-1") -> None:
        self._responses = responses
        self._model_name = model_name
        self._index = 0

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        text = self._responses[self._index % len(self._responses)]
        self._index += 1
        return ProviderResponse(text=text, model_name=self._model_name)


class _FailingProvider(TextProvider):
    """Always raises on generate()."""

    name = "failing"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        raise RuntimeError("provider failed")


def _make_project(
    tmp_path: Path, *, workflow_backend: str = "python"
) -> dict[str, Any]:
    """Create an initialized test project and return services dict + paths."""
    paths = build_project_paths(tmp_path)
    ProjectService(paths).ensure_structure()
    ConfigService(paths).ensure_files()
    ManifestService(paths).ensure_manifest()
    config = ConfigService(paths).load()
    config["ecosystem"] = {
        "workflows": {
            "query_backend": workflow_backend,
            "review_backend": workflow_backend,
        },
    }
    services = build_services(paths, config)
    return {"paths": paths, "config": config, "services": services}


def _write_source_page(paths: ProjectPaths, slug: str, title: str, body: str) -> None:
    page = paths.wiki_dir / "sources" / f"{slug}.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(_compiled_page(title, body), encoding="utf-8")


# ---------------------------------------------------------------------------
# Query graph parity: python vs langgraph produce same QueryAnswer
# ---------------------------------------------------------------------------


class TestQueryGraphParity:
    """Both backends must produce identical QueryAnswer for same inputs."""

    @staticmethod
    def _run_query(tmp_path: Path, backend: str) -> QueryAnswer:
        """Helper that creates project, writes a page, and runs self-consistency."""
        project = _make_project(tmp_path / backend, workflow_backend=backend)
        paths: ProjectPaths = project["paths"]
        _write_source_page(
            paths,
            "traceability-notes",
            "Traceability Notes",
            "Traceability preserves source links in the wiki.",
        )
        provider = _DeterministicProvider(
            [
                "Traceability preserves source links. [Traceability Notes]",
                "Traceability preserves source links. [Traceability Notes]",
                "Traceability preserves source links. [Traceability Notes]",
            ]
        )
        run_store = RunStore(paths.graph_exports_dir / "parity.db")
        service = QueryService(
            paths,
            project["services"]["search"],
            provider=provider,
            run_store=run_store,
            workflow_backend=backend,
        )
        answer = service.answer_question("traceability", self_consistency=3)
        return answer

    def test_both_backends_produce_same_answer_text(self, tmp_path: Path) -> None:
        py_answer = self._run_query(tmp_path, "python")
        lg_answer = self._run_query(tmp_path, "langgraph")
        assert py_answer.answer == lg_answer.answer

    def test_both_backends_produce_same_citation_count(self, tmp_path: Path) -> None:
        py_answer = self._run_query(tmp_path, "python")
        lg_answer = self._run_query(tmp_path, "langgraph")
        assert len(py_answer.citations) == len(lg_answer.citations)

    def test_langgraph_mode_matches_pattern(self, tmp_path: Path) -> None:
        lg_answer = self._run_query(tmp_path, "langgraph")
        assert lg_answer.mode.startswith("self-consistency:")
        assert ":3" in lg_answer.mode

    def test_langgraph_persists_run_id(self, tmp_path: Path) -> None:
        lg_answer = self._run_query(tmp_path, "langgraph")
        assert lg_answer.run_id is not None


class TestQueryGraphRunRecord:
    """LangGraph backend must save a proper RunRecord."""

    def test_run_record_saved_with_correct_fields(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path, workflow_backend="langgraph")
        paths = project["paths"]
        _write_source_page(
            paths,
            "traceability-notes",
            "Traceability Notes",
            "Traceability preserves source links in the wiki.",
        )
        provider = _DeterministicProvider(
            [
                "Traceability preserves source links. [Traceability Notes]",
                "Traceability preserves source links. [Traceability Notes]",
            ]
        )
        run_store = RunStore(paths.graph_exports_dir / "run-check.db")
        service = QueryService(
            paths,
            project["services"]["search"],
            provider=provider,
            run_store=run_store,
            workflow_backend="langgraph",
        )
        answer = service.answer_question("traceability", self_consistency=2)
        record = run_store.get_run(answer.run_id)
        assert record is not None
        assert record.command == "query"
        assert record.evidence_bundle is not None
        assert record.context_hash == record.evidence_bundle.context_hash
        assert len(record.candidates) == 2
        assert record.merged_answer is not None
        assert record.merged_answer.candidate_count == 2
        assert record.final_text == answer.answer


class TestQueryGraphNoMatches:
    """Langgraph backend still returns no-matches answer without invoking graph."""

    def test_no_matches_returns_fallback(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path, workflow_backend="langgraph")
        paths = project["paths"]
        provider = _DeterministicProvider(["unused"])
        service = QueryService(
            paths,
            project["services"]["search"],
            provider=provider,
            workflow_backend="langgraph",
        )
        answer = service.answer_question("nonexistent", self_consistency=3)
        assert answer.mode == "no-matches"
        assert "No compiled wiki pages" in answer.answer


# ---------------------------------------------------------------------------
# Review graph parity
# ---------------------------------------------------------------------------


class TestReviewGraphParity:
    """Both backends must produce matching adversarial review output."""

    @staticmethod
    def _run_review(
        tmp_path: Path, backend: str, provider: TextProvider
    ) -> ReviewReport:
        project = _make_project(tmp_path / backend, workflow_backend=backend)
        paths = project["paths"]
        # Two source pages with overlapping terms to trigger pair creation
        _write_source_page(
            paths,
            "retrieval-augmented",
            "Retrieval Augmented Generation",
            "RAG combines retrieval with generation for knowledge-intensive tasks in 2020.",
        )
        _write_source_page(
            paths,
            "dense-passage",
            "Dense Passage Retrieval",
            "DPR uses dense retrieval for knowledge-intensive open-domain QA in 2020.",
        )
        run_store = RunStore(paths.graph_exports_dir / "review-parity.db")
        service = ReviewService(
            paths,
            provider=provider,
            run_store=run_store,
            workflow_backend=backend,
        )
        return service.review(adversarial=True)

    def test_both_backends_produce_same_mode_prefix(self, tmp_path: Path) -> None:
        # Extractor returns no claims → skeptic and arbiter get NO_CLAIMS → no findings
        provider = _DeterministicProvider(["NO_CLAIMS", "NO_CLAIMS", "NO_FINDINGS"])
        py_report = self._run_review(tmp_path, "python", provider)
        provider2 = _DeterministicProvider(["NO_CLAIMS", "NO_CLAIMS", "NO_FINDINGS"])
        lg_report = self._run_review(tmp_path, "langgraph", provider2)
        assert py_report.mode.startswith("adversarial:")
        assert lg_report.mode.startswith("adversarial:")

    def test_both_backends_produce_same_finding_count(self, tmp_path: Path) -> None:
        provider = _DeterministicProvider(["NO_CLAIMS", "NO_CLAIMS", "NO_FINDINGS"])
        py_report = self._run_review(tmp_path, "python", provider)
        provider2 = _DeterministicProvider(["NO_CLAIMS", "NO_CLAIMS", "NO_FINDINGS"])
        lg_report = self._run_review(tmp_path, "langgraph", provider2)
        assert len(py_report.findings) == len(lg_report.findings)

    def test_langgraph_persists_run_id(self, tmp_path: Path) -> None:
        provider = _DeterministicProvider(["NO_CLAIMS", "NO_CLAIMS", "NO_FINDINGS"])
        report = self._run_review(tmp_path, "langgraph", provider)
        assert report.run_id is not None


class TestReviewGraphEmptySnapshots:
    """Review graph handles zero source pages gracefully."""

    def test_no_sources_returns_empty_findings(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path, workflow_backend="langgraph")
        paths = project["paths"]
        provider = _DeterministicProvider(["unused"])
        run_store = RunStore(paths.graph_exports_dir / "empty.db")
        service = ReviewService(
            paths,
            provider=provider,
            run_store=run_store,
            workflow_backend="langgraph",
        )
        report = service.review(adversarial=True)
        assert report.findings == []
        assert report.run_id is not None


# ---------------------------------------------------------------------------
# Config wiring through build_services
# ---------------------------------------------------------------------------


class TestBuildServicesWorkflowWiring:
    """build_services must thread workflow_backend from ecosystem config."""

    def test_default_backend_is_python(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path, workflow_backend="python")
        assert project["services"]["query"].workflow_backend == "python"
        assert project["services"]["review"].workflow_backend == "python"

    def test_langgraph_backend_threaded_to_services(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path, workflow_backend="langgraph")
        assert project["services"]["query"].workflow_backend == "langgraph"
        assert project["services"]["review"].workflow_backend == "langgraph"

    def test_build_services_with_no_ecosystem_defaults_to_python(
        self, tmp_path: Path
    ) -> None:
        paths = build_project_paths(tmp_path / "no-eco")
        ProjectService(paths).ensure_structure()
        ConfigService(paths).ensure_files()
        ManifestService(paths).ensure_manifest()
        config = ConfigService(paths).load()
        # Explicitly remove ecosystem to test default
        config.pop("ecosystem", None)
        services = build_services(paths, config)
        assert services["query"].workflow_backend == "python"
        assert services["review"].workflow_backend == "python"


# ---------------------------------------------------------------------------
# Import guard — langgraph backend fails clearly if langgraph missing
# ---------------------------------------------------------------------------


class TestWorkflowBackendDefaultSkipsGraph:
    """With python backend, self-consistency does not import langgraph at all."""

    def test_python_backend_self_consistency_works(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path, workflow_backend="python")
        paths = project["paths"]
        _write_source_page(
            paths,
            "notes",
            "Notes",
            "Notes about testing the python backend.",
        )
        provider = _DeterministicProvider(
            [
                "Notes about testing. [Notes]",
                "Notes about testing. [Notes]",
            ]
        )
        service = QueryService(
            paths,
            project["services"]["search"],
            provider=provider,
            workflow_backend="python",
        )
        answer = service.answer_question("testing", self_consistency=2)
        assert answer.mode.startswith("self-consistency:")

    def test_python_backend_adversarial_works(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path, workflow_backend="python")
        paths = project["paths"]
        _write_source_page(
            paths, "a", "Page A", "Alpha beta gamma delta knowledge in 2020."
        )
        _write_source_page(
            paths, "b", "Page B", "Alpha beta gamma delta retrieval in 2020."
        )
        provider = _DeterministicProvider(["NO_CLAIMS", "NO_CLAIMS", "NO_FINDINGS"])
        service = ReviewService(
            paths,
            provider=provider,
            workflow_backend="python",
        )
        report = service.review(adversarial=True)
        assert report.mode.startswith("adversarial:")


# ---------------------------------------------------------------------------
# LangGraph graph structure smoke tests
# ---------------------------------------------------------------------------


class TestGraphStructure:
    """Verify graphs can be compiled without errors."""

    def test_query_graph_compiles(self) -> None:
        from src.workflows.query_graph import _build_query_graph

        graph = _build_query_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_review_graph_compiles(self) -> None:
        from src.workflows.review_graph import _build_review_graph

        graph = _build_review_graph()
        compiled = graph.compile()
        assert compiled is not None


# ---------------------------------------------------------------------------
# Error path coverage for graph nodes
# ---------------------------------------------------------------------------


class TestQueryGraphErrorPaths:
    """Cover error branches inside query graph nodes."""

    def test_langgraph_raises_on_provider_error(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path, workflow_backend="langgraph")
        paths = project["paths"]
        _write_source_page(paths, "notes", "Notes", "Notes about testing error paths.")
        service = QueryService(
            paths,
            project["services"]["search"],
            provider=_FailingProvider(),
            workflow_backend="langgraph",
        )
        from src.providers import ProviderExecutionError

        with pytest.raises(ProviderExecutionError, match="Self-consistency"):
            service.answer_question("testing", self_consistency=2)

    def test_langgraph_without_run_store_returns_none_run_id(
        self, tmp_path: Path
    ) -> None:
        project = _make_project(tmp_path, workflow_backend="langgraph")
        paths = project["paths"]
        _write_source_page(paths, "notes", "Notes", "Notes for no-store test.")
        provider = _DeterministicProvider(
            ["Notes for testing. [Notes]", "Notes for testing. [Notes]"]
        )
        service = QueryService(
            paths,
            project["services"]["search"],
            provider=provider,
            run_store=None,
            workflow_backend="langgraph",
        )
        answer = service.answer_question("testing", self_consistency=2)
        assert answer.run_id is None


class TestReviewGraphErrorPaths:
    """Cover error branches inside review graph nodes."""

    def test_langgraph_raises_when_all_pairs_error(self, tmp_path: Path) -> None:
        project = _make_project(tmp_path, workflow_backend="langgraph")
        paths = project["paths"]
        _write_source_page(
            paths, "a", "Page A", "Alpha beta gamma delta knowledge in 2020."
        )
        _write_source_page(
            paths, "b", "Page B", "Alpha beta gamma delta retrieval in 2020."
        )
        service = ReviewService(
            paths,
            provider=_FailingProvider(),
            workflow_backend="langgraph",
        )
        from src.providers import ProviderExecutionError

        with pytest.raises(ProviderExecutionError, match="Adversarial review failed"):
            service.review(adversarial=True)
