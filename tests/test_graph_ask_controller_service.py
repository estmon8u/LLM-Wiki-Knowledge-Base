from __future__ import annotations

import json
import subprocess

import pytest

from src.services.graph_ask_controller_service import (
    GraphAskControllerError,
    GraphAskControllerService,
)
from src.services.graphrag_defaults import env_file_has_key
from src.services.graphrag_command_service import (
    GraphRAGCommandResult,
    GraphRAGCommandService,
)
from src.services.graphrag_query_service import GraphRAGQueryError, GraphRAGQueryService
from src.services.graphrag_status_service import GraphRAGStatusService
from src.services.query_router_service import QueryRouterService


def _write_ready_graph(test_project) -> None:
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "src-1", "text": "RAG text"}]),
    )
    test_project.write_file("graph/graphrag/output/entities.parquet", "")
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
    )


def _build_controller(test_project, runner) -> GraphAskControllerService:
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
        manifest_service=test_project.services["manifest"],
    )


def test_controller_accepts_graph_env_file_credentials(
    test_project, monkeypatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _write_ready_graph(test_project)
    test_project.write_file("graph/graphrag/.env", "OPENAI_API_KEY=local-key\n")

    def runner(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            command, 0, stdout="Graph answer.\n", stderr=""
        )

    controller = _build_controller(test_project, runner)

    answer = controller.ask("How does REALM differ from RAG?")

    assert answer.retriever == "graph"
    assert answer.method == "drift"
    assert answer.planner == "heuristic"
    assert answer.claim_support == "graph-grounded"


def test_controller_saves_answer_when_requested(test_project, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _write_ready_graph(test_project)

    def runner(command, *, cwd, capture_output, text):
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
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _write_ready_graph(test_project)
    controller = _build_controller(
        test_project,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    with pytest.raises(GraphAskControllerError, match="OPENAI_API_KEY"):
        controller.ask("What is RAG?")


def test_controller_requires_separate_embedding_credentials(
    test_project, monkeypatch
) -> None:
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
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    controller = _build_controller(
        test_project,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    with pytest.raises(GraphRAGQueryError, match="workspace is not initialized"):
        controller.ask("What is RAG?")


def test_controller_wraps_invalid_graph_config(test_project) -> None:
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

    import os, time

    now = time.time()
    input_path = test_project.paths.graph_dir / "graphrag" / "input" / "sources.json"
    os.utime(input_path, (now - 120, now - 120))
    manifest_path = test_project.paths.raw_manifest_file
    os.utime(manifest_path, (now, now))

    def runner(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(command, 0, stdout="Answer.\n", stderr="")

    controller = _build_controller(test_project, runner)
    answer = controller.ask("What is RAG?")

    assert answer.claim_support == "stale-index"


def test_controller_claim_support_reports_no_answer(test_project, monkeypatch) -> None:
    """When GraphRAG returns empty output, claim_support should be no-answer."""
    _write_ready_graph(test_project)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def runner(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    controller = _build_controller(test_project, runner)
    answer = controller.ask("What is RAG?")

    assert answer.claim_support == "no-answer"
