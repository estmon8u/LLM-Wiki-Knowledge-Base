"""Opt-in E2E: ingest a real web PDF and exercise kb agent tools.

Downloads the Lewis et al. RAG paper (arXiv:2005.11401), runs ``kb init``,
``kb add``, and ``kb update`` (wiki + WikiGraphRAG, no GraphRAG index), then
calls ``status``, ``find_kb``, and ``ask_kb`` through the agent tool layer.

Enable with::

    RUN_AGENT_PDF_E2E=1 poetry run pytest tests/agents/test_agent_pdf_e2e.py -q --no-cov

Requires network access, optional extras (``networkx``, ``docling`` / ``markitdown``),
and roughly 30-90 seconds depending on PDF conversion hardware.
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
from graphwiki_kb.services import build_services
from graphwiki_kb.services.config_service import ConfigService
from graphwiki_kb.services.project_service import build_project_paths
from graphwiki_kb.services.source_recommendation_store import (
    SourceRecommendationStore,
)
from graphwiki_kb.wikigraph.deps import require_networkx
from tests.conftest import _StubProvider

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

# Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
RAG_PAPER_PDF_URL = "https://arxiv.org/pdf/2005.11401.pdf"
RAG_PAPER_FILENAME = "rag-paper.pdf"


def _download_pdf(dest: Path) -> None:
    """Download the fixture PDF from arXiv."""
    try:
        urllib.request.urlretrieve(RAG_PAPER_PDF_URL, dest)
    except urllib.error.URLError as exc:
        pytest.skip(f"could not download PDF from {RAG_PAPER_PDF_URL}: {exc}")


def _set_openai_provider_config() -> None:
    config_path = Path("kb.config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["provider"] = {"name": "openai"}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _agent_runtime(project_root: Path) -> AgentRuntimeContext:
    """Build an agent runtime from an on-disk project after CLI steps."""
    paths = build_project_paths(project_root)
    config_service = ConfigService(paths)
    config = config_service.load()
    services = build_services(paths, config)
    services["compile"].provider = _StubProvider()
    command_context = CommandContext(
        project_root=project_root,
        cwd=project_root,
        config=config,
        schema_text=config_service.load_schema(),
        services=services,
        verbose=False,
    )
    store = SourceRecommendationStore(paths)
    store.ensure_directory()
    return AgentRuntimeContext(
        command_context=command_context,
        services=services,
        recommendation_store=store,
    )


def test_agent_tools_after_web_pdf_ingest_and_update() -> None:
    """Ingest arXiv RAG PDF via CLI, update KB, then query through agent tools."""
    require_networkx()
    runner = CliRunner()
    # Force docling/markitdown fallback so the test does not depend on Mistral OCR.
    pdf_env = {**os.environ, "MISTRAL_API_KEY": ""}

    with runner.isolated_filesystem():
        _download_pdf(Path(RAG_PAPER_FILENAME))

        init_result = runner.invoke(main, ["init"])
        assert init_result.exit_code == 0, init_result.output

        add_result = runner.invoke(main, ["add", RAG_PAPER_FILENAME], env=pdf_env)
        assert add_result.exit_code == 0, add_result.output
        assert "Ingested" in add_result.output

        normalized = Path("raw/normalized")
        assert normalized.exists()
        md_files = list(normalized.glob("*.md"))
        assert md_files, "expected normalized markdown from PDF conversion"
        normalized_text = md_files[0].read_text(encoding="utf-8")
        assert "retrieval" in normalized_text.casefold()

        _set_openai_provider_config()
        with patch(
            "graphwiki_kb.services.build_provider",
            return_value=_StubProvider(),
        ):
            update_result = runner.invoke(main, ["update", "--no-graph"], env=pdf_env)
        assert update_result.exit_code == 0, update_result.output
        assert "WikiGraphRAG" in update_result.output
        assert Path("graph/wikigraph/index.json").exists()

        runtime = _agent_runtime(Path())

        status = run_status(runtime)
        assert status.project_initialized is True
        assert status.source_count >= 1
        assert status.compiled_source_count >= 1
        assert status.wikigraph_built is True
        assert status.wikigraph_node_count > 0

        find_out = run_find_kb(
            runtime,
            FindKbInput(
                query="retrieval augmented generation",
                engine="wikigraph",
                method="local",
                limit=5,
            ),
        )
        assert find_out.results
        assert any(r.retriever == "wikigraph" for r in find_out.results)
        titles = " ".join(r.title.casefold() for r in find_out.results)
        assert "retrieval" in titles or "rag" in titles

        ask_out = run_ask_kb(
            runtime,
            AskKbInput(
                question="What is retrieval-augmented generation?",
                engine="wikigraph",
                method="local",
            ),
        )
        assert ask_out.answer.strip()
        assert ask_out.method
        assert ask_out.claim_support in {
            "cited-graph-answer",
            "graph-index-answer",
            "unverified",
        }
        answer_lower = ask_out.answer.casefold()
        assert "retrieval" in answer_lower or "rag" in answer_lower

        wiki_find = run_find_kb(
            runtime,
            FindKbInput(
                query="knowledge-intensive",
                engine="graph",
                limit=5,
            ),
        )
        assert wiki_find.results
        assert any(r.retriever == "wiki" for r in wiki_find.results)
