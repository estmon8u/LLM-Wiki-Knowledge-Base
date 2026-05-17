"""Tests for test graph ask controller service.

This module belongs to `tests.test_graph_ask_controller_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import json
import subprocess

import pandas as pd
import pytest

from graphwiki_kb.services.graph_ask_controller_service import (
    GraphAskControllerError,
    GraphAskControllerService,
)
from graphwiki_kb.services.graphrag_command_service import (
    GraphRAGCommandResult,
    GraphRAGCommandService,
)
from graphwiki_kb.services.graphrag_defaults import env_file_has_key
from graphwiki_kb.services.graphrag_freshness_service import (
    file_digest,
    graph_input_source_hashes,
    graph_runtime_digest,
)
from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryService
from graphwiki_kb.services.graphrag_status_service import GraphRAGStatusService
from graphwiki_kb.services.query_router_service import QueryRouterService


def _write_ready_graph(test_project) -> None:
    """Handles write ready graph.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "src-1", "text": "RAG text"}]),
    )
    output_dir = test_project.paths.graph_dir / "graphrag" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"id": "doc-1", "title": "RAG Document", "text": "RAG text"}]
    ).to_parquet(output_dir / "documents.parquet")
    pd.DataFrame([{"id": "tu-1", "text": "RAG text"}]).to_parquet(
        output_dir / "text_units.parquet"
    )
    pd.DataFrame([{"id": "entity-1", "title": "RAG"}]).to_parquet(
        output_dir / "entities.parquet"
    )
    pd.DataFrame([{"id": "rel-1", "source": "RAG", "target": "REALM"}]).to_parquet(
        output_dir / "relationships.parquet"
    )
    pd.DataFrame([{"id": "community-0", "community": 0, "title": "RAG"}]).to_parquet(
        output_dir / "communities.parquet"
    )
    pd.DataFrame(
        [{"id": "report-0", "community": 0, "title": "RAG", "summary": "RAG summary."}]
    ).to_parquet(output_dir / "community_reports.parquet")
    test_project.write_file(
        "graph/graphrag/output/lancedb/vector-store.marker",
        "ready",
    )
    workspace_dir = test_project.paths.graph_dir / "graphrag"
    input_path = workspace_dir / "input" / "sources.json"
    GraphRAGStatusService(test_project.paths).record_index_run(
        method="fast",
        dry_run=False,
        result=GraphRAGCommandResult(
            command=("python", "-m", "graphrag", "index"),
            cwd=test_project.paths.root,
            returncode=0,
            stdout="indexed",
            stderr="",
        ),
        input_digest=file_digest(input_path),
        config_digest=graph_runtime_digest(workspace_dir),
        input_source_count=1,
        source_hashes=graph_input_source_hashes(input_path),
        output_state="complete",
    )


def _build_controller(test_project, runner) -> GraphAskControllerService:
    """Handles build controller.

    Args:
        test_project: Test project value used by the operation.
        runner: Runner value used by the operation.

    Returns:
        GraphAskControllerService produced by the operation.
    """
    status_service = GraphRAGStatusService(test_project.paths)
    command_service = GraphRAGCommandService(test_project.paths, runner=runner)
    query_service = GraphRAGQueryService(
        test_project.paths,
        command_service,
        status_service,
        test_project.services["search"],
        refresh_index=test_project.services["compile"].refresh_index,
    )
    router = QueryRouterService(status_service)
    return GraphAskControllerService(
        test_project.paths,
        test_project.config,
        status_service,
        router,
        query_service,
    )


def test_controller_accepts_graph_env_file_credentials(
    test_project, monkeypatch
) -> None:
    """Verifies that controller accepts graph env file credentials.

    Args:
        test_project: Test project value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _write_ready_graph(test_project)
    test_project.write_file("graph/graphrag/.env", "OPENAI_API_KEY=local-key\n")

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
        return subprocess.CompletedProcess(
            command, 0, stdout="Graph answer.\n", stderr=""
        )

    controller = _build_controller(test_project, runner)

    answer = controller.ask("How does REALM differ from RAG?")

    assert answer.retriever == "graph"
    assert answer.method == "drift"
    assert answer.planner == "heuristic"
    assert answer.claim_support == "graph-index-answer"


def test_controller_saves_answer_when_requested(test_project, monkeypatch) -> None:
    """Verifies that controller saves answer when requested.

    Args:
        test_project: Test project value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_ready_graph(test_project)

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
        return subprocess.CompletedProcess(
            command, 0, stdout="Graph answer.\n", stderr=""
        )

    controller = _build_controller(test_project, runner)

    answer = controller.ask("What is RAG?", save=True, save_as="rag-note")

    assert answer.saved_path == "wiki/analysis/graphrag-rag-note.md"
    assert (test_project.root / answer.saved_path).exists()


def test_controller_reports_missing_graph_credentials(
    test_project, monkeypatch
) -> None:
    """Verifies that controller reports missing graph credentials.

    Args:
        test_project: Test project value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _write_ready_graph(test_project)
    controller = _build_controller(
        test_project,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    with pytest.raises(GraphAskControllerError, match="OPENAI_API_KEY"):
        controller.ask("What is RAG?")


def test_controller_stops_before_query_when_vector_store_missing(
    test_project, monkeypatch
) -> None:
    """Regression: kb ask should not call GraphRAG when the index is incomplete."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_ready_graph(test_project)
    marker = (
        test_project.paths.graph_dir
        / "graphrag"
        / "output"
        / "lancedb"
        / "vector-store.marker"
    )
    marker.unlink()

    def fail_runner(command, **kwargs):
        """Fail if the controller reaches the query subprocess."""
        raise AssertionError("GraphRAG query should not run")

    controller = _build_controller(test_project, fail_runner)

    with pytest.raises(GraphAskControllerError, match="vector store"):
        controller.ask("What is RAG?")


def test_controller_requires_separate_embedding_credentials(
    test_project, monkeypatch
) -> None:
    """Verifies that controller requires separate embedding credentials.

    Args:
        test_project: Test project value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    test_project.config["graph"] = {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "embedding_provider": "gemini",
        "embedding_model": "gemini-embedding-001",
    }
    _write_ready_graph(test_project)
    controller = _build_controller(
        test_project,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    with pytest.raises(GraphAskControllerError, match="GEMINI_API_KEY"):
        controller.ask("What is RAG?")


def test_controller_reports_graph_readiness_before_credentials(
    test_project,
    monkeypatch,
) -> None:
    """Verifies that controller reports graph readiness before credentials.

    Args:
        test_project: Test project value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    controller = _build_controller(
        test_project,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    with pytest.raises(GraphAskControllerError, match="workspace is not initialized"):
        controller.ask("What is RAG?")


def test_controller_wraps_invalid_graph_config(test_project) -> None:
    """Verifies that controller wraps invalid graph config.

    Args:
        test_project: Test project value used by the operation.
    """
    controller = _build_controller(
        test_project,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )
    controller.config["graph"] = {
        "provider": "",
        "model": "gpt-4.1-mini",
        "embedding_model": "text-embedding-3-small",
        "api_key_env": "OPENAI_API_KEY",
    }

    with pytest.raises(GraphAskControllerError, match="provider"):
        controller._resolve_graph_config()


def test_env_file_has_key_ignores_invalid_lines_and_io_errors(tmp_path) -> None:
    """Verifies that env file has key ignores invalid lines and io errors.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n# comment\nMISSING_EQUALS\nGRAPHRAG_API_KEY=\n", encoding="utf-8"
    )
    directory_path = tmp_path / "directory.env"
    directory_path.mkdir()

    assert not env_file_has_key(env_file, "GRAPHRAG_API_KEY")
    assert not env_file_has_key(directory_path, "GRAPHRAG_API_KEY")


def test_controller_claim_support_reports_stale_index(
    test_project, monkeypatch
) -> None:
    """When manifest is newer than graph input, claim_support should be stale-index."""
    _write_ready_graph(test_project)
    test_project.write_file("graph/graphrag/.env", "OPENAI_API_KEY=local-key\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    import os
    import time

    now = time.time()
    input_path = test_project.paths.graph_dir / "graphrag" / "input" / "sources.json"
    os.utime(input_path, (now - 120, now - 120))
    manifest_path = test_project.paths.raw_manifest_file
    os.utime(manifest_path, (now, now))

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
        return subprocess.CompletedProcess(command, 0, stdout="Answer.\n", stderr="")

    controller = _build_controller(test_project, runner)
    answer = controller.ask("What is RAG?")

    assert answer.claim_support == "stale-index"
    assert answer.staleness_warnings == [
        "Manifest is newer than graph input. Run `kb update`."
    ]


def test_controller_claim_support_reports_no_answer(test_project, monkeypatch) -> None:
    """When GraphRAG returns empty output, claim_support should be no-answer."""
    _write_ready_graph(test_project)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    controller = _build_controller(test_project, runner)
    answer = controller.ask("What is RAG?")

    assert answer.claim_support == "no-answer"
