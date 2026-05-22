"""Opt-in end-to-end test: ingest a real arXiv PDF and drive the agent tools.

This test exercises the full workflow that a user would run on a fresh
machine -- ``kb init``, ``kb add``, ``kb update --no-graph`` (which now
builds the WikiGraphRAG index automatically), and then ``status``,
``find_kb``, and ``ask_kb`` through the agent tool layer with both
``engine='graphrag'`` and ``engine='wikigraph'``.

It is **opt-in** for two reasons:

* It downloads a PDF over the network.
* It depends on the optional ``wikigraph`` extra (NetworkX, BM25S) and
  ``docling``/``markitdown`` for PDF conversion. Pure CI installs should
  skip it.

Enable with::

    RUN_AGENT_PDF_E2E=1 poetry run pytest tests/agents/test_agent_pdf_e2e.py \
        -q --no-cov
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import AskKbInput, FindKbInput
from graphwiki_kb.agents.tools.ask_kb import run_ask_kb
from graphwiki_kb.agents.tools.find_kb import run_find_kb
from graphwiki_kb.agents.tools.status import run_status
from graphwiki_kb.cli import main
from graphwiki_kb.models.command_models import CommandContext
from graphwiki_kb.providers.base import (
    ProviderRequest,
    ProviderResponse,
    TextProvider,
)
from graphwiki_kb.services import build_services
from graphwiki_kb.services.config_service import ConfigService
from graphwiki_kb.services.project_service import build_project_paths
from graphwiki_kb.services.source_recommendation_store import (
    SourceRecommendationStore,
)
from graphwiki_kb.wikigraph.deps import require_networkx

pytest.importorskip("networkx")

pytestmark = [
    pytest.mark.agent_pdf_e2e,
    pytest.mark.skipif(
        os.environ.get("RUN_AGENT_PDF_E2E") != "1",
        reason=(
            "set RUN_AGENT_PDF_E2E=1 to run the web PDF agent E2E "
            "(network + docling, ~30-90s)"
        ),
    ),
]

RAG_PAPER_PDF_URL = "https://arxiv.org/pdf/2005.11401v4.pdf"
RAG_PAPER_FILENAME = "rag-paper.pdf"


class _StubProvider(TextProvider):
    """Provider stub that returns a fixed payload for both providers used."""

    name = "stub"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        if request.response_schema_name == "kb_query_answer":
            return ProviderResponse(
                text=(
                    '{"answer_markdown": "Stub answer", '
                    '"claims": [], "citations": [], '
                    '"insufficient_evidence": true}'
                ),
                model_name="stub-1",
            )
        return ProviderResponse(
            text="Stub summary for the RAG paper.", model_name="stub-1"
        )


def _download_pdf(target: Path) -> None:
    try:
        urllib.request.urlretrieve(RAG_PAPER_PDF_URL, target)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        OSError,
    ) as exc:
        pytest.skip(f"PDF download failed: {exc}")


def _set_provider(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config.setdefault("provider", {})["name"] = "openai"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_agent_pdf_e2e(tmp_path: Path) -> None:
    """Download a real PDF, build the index, and drive the agent tools."""
    require_networkx()
    pdf_path = tmp_path / RAG_PAPER_FILENAME
    _download_pdf(pdf_path)

    project_root = tmp_path / "kb"
    runner = CliRunner()
    with patch("graphwiki_kb.services.build_provider", return_value=_StubProvider()):
        assert (
            runner.invoke(
                main,
                ["--project-root", str(project_root), "init"],
            ).exit_code
            == 0
        )
        _set_provider(project_root / "kb.config.yaml")
        add_result = runner.invoke(
            main,
            ["--project-root", str(project_root), "add", str(pdf_path)],
        )
        assert add_result.exit_code == 0, add_result.output
        update_result = runner.invoke(
            main,
            ["--project-root", str(project_root), "update", "--no-graph"],
        )
        assert update_result.exit_code == 0, update_result.output
        assert "WikiGraphRAG Summary" in update_result.output

    # Build a CommandContext + AgentRuntime that mirrors what `kb agent` uses.
    paths = build_project_paths(project_root)
    config_service = ConfigService(paths)
    config = config_service.load()
    schema_text = config_service.load_schema()
    services = build_services(paths, config)
    services["compile"].provider = _StubProvider()  # type: ignore[attr-defined]
    services["review"].provider = _StubProvider()  # type: ignore[attr-defined]
    command_context = CommandContext(
        project_root=paths.root,
        cwd=paths.root,
        config=config,
        schema_text=schema_text,
        services=services,
        verbose=False,
    )
    recommendation_store = SourceRecommendationStore(paths)
    recommendation_store.ensure_directory()
    runtime = AgentRuntimeContext(
        command_context=command_context,
        services=services,
        recommendation_store=recommendation_store,
        auto_approve=False,
        show_plan=False,
    )

    status = run_status(runtime)
    assert status.wikigraph is not None
    assert status.wikigraph.initialized is True
    assert status.wikigraph.node_count > 0
    assert status.wikigraph.source_count >= 1

    # Force the wikigraph_query service to use the deterministic provider-free
    # synthesis path so the test does not depend on a real LLM call.
    services["wikigraph_query"].provider = None  # type: ignore[attr-defined]

    find = run_find_kb(
        runtime, FindKbInput(query="retrieval generation", engine="wikigraph")
    )
    assert find.results
    assert all(r.retriever == "wikigraph" for r in find.results)

    ask = run_ask_kb(
        runtime,
        AskKbInput(
            question="What does RAG combine?",
            engine="wikigraph",
            method="local",
        ),
    )
    assert ask.method == "local"
    assert ask.planner == "wikigraph"
    assert ask.answer
