from __future__ import annotations

import hashlib
import json
import subprocess

import pytest
import yaml

from src.services.graphrag_command_service import (
    GraphRAGCommandResult,
    GraphRAGCommandService,
)
from src.services.graphrag_query_service import (
    GraphRAGQueryError,
    GraphRAGQueryService,
)
from src.services.graphrag_status_service import GraphRAGStatusService


def _write_ready_graph(test_project, *, index_success: bool = True) -> None:
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
            returncode=0 if index_success else 2,
            stdout="indexed" if index_success else "",
            stderr="" if index_success else "failed",
        ),
    )


def _build_query_service(test_project, runner):
    command_service = GraphRAGCommandService(test_project.paths, runner=runner)
    status_service = GraphRAGStatusService(test_project.paths)
    return GraphRAGQueryService(
        test_project.paths,
        command_service,
        status_service,
        test_project.services["search"],
        refresh_index=test_project.services["compile"].refresh_index,
    )


def test_graph_query_runs_explicit_method_and_options(test_project) -> None:
    _write_ready_graph(test_project)
    calls = []

    def runner(command, *, cwd, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="GraphRAG answer\n",
            stderr="",
        )

    service = _build_query_service(test_project, runner)

    answer = service.ask(
        "How does REALM differ from RAG?",
        method="local",
        community_level=1,
        dynamic_community_selection=False,
        response_type="Single Sentence",
        verbose=True,
    )

    command = calls[0]
    assert command[1:4] == ("-m", "graphrag", "query")
    assert "--method" in command
    assert "local" in command
    assert "--community-level" in command
    assert "1" in command
    assert "--no-dynamic-selection" in command
    assert "--response-type" in command
    assert "Single Sentence" in command
    assert "--verbose" in command
    assert command[-1] == "How does REALM differ from RAG?"
    assert answer.retriever == "graphrag"
    assert answer.method == "local"
    assert answer.answer == "GraphRAG answer"
    assert answer.raw_output == "GraphRAG answer"
    assert answer.index_run_id is not None
    assert (
        answer.input_manifest_hash
        == hashlib.sha256(
            test_project.paths.graph_dir.joinpath(
                "graphrag",
                "input",
                "sources.json",
            ).read_bytes()
        ).hexdigest()
    )


def test_graph_query_raw_output_prefers_stdout_over_progress_stderr(
    test_project,
) -> None:
    _write_ready_graph(test_project)

    def runner(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="GraphRAG answer\n",
            stderr="Warning: noisy dependency output\nProgress: 100%\n",
        )

    service = _build_query_service(test_project, runner)

    answer = service.ask("What is RAG?", method="basic")
    saved_path = service.save_answer(answer)

    assert answer.answer == "GraphRAG answer"
    assert answer.raw_output == "GraphRAG answer"
    assert "Progress" in answer.stderr
    saved_text = (test_project.root / saved_path).read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(saved_text.split("---", 2)[1])
    assert frontmatter["citations"] == [f"GraphRAG index {answer.index_run_id}"]
    assert frontmatter["citation_count"] == 1
    assert "GraphRAG answer" in saved_text
    assert "Warning: noisy dependency output" not in saved_text
    assert "Progress: 100%" not in saved_text


def test_graph_query_save_writes_analysis_page_and_refreshes_index(
    test_project,
) -> None:
    _write_ready_graph(test_project)

    def runner(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "REALM augments pretraining, while RAG augments generation. "
                "[Data: Sources (1)]\n"
            ),
            stderr="",
        )

    service = _build_query_service(test_project, runner)
    answer = service.ask(
        "How does REALM differ from RAG?",
        method="drift",
        community_level=2,
        dynamic_community_selection=True,
        response_type="Multiple Paragraphs",
    )
    answer.planner = "heuristic"
    answer.route_reason = "comparison question"
    answer.claim_support = "graph-grounded"

    saved_path = service.save_answer(answer)

    assert saved_path == "wiki/analysis/graphrag-how-does-realm-differ-from-rag.md"
    page = test_project.root / saved_path
    text = page.read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(text.split("---", 2)[1])
    assert frontmatter["type"] == "analysis"
    assert frontmatter["retriever"] == "graphrag"
    assert frontmatter["method"] == "drift"
    assert frontmatter["question"] == "How does REALM differ from RAG?"
    assert frontmatter["saved_at"] == answer.created_at
    assert frontmatter["citations"] == ["[Data: Sources (1)]"]
    assert frontmatter["citation_count"] == 1
    assert frontmatter["claim_count"] == 0
    assert frontmatter["claims"] == []
    assert frontmatter["insufficient_evidence"] is False
    assert frontmatter["index_run_id"] == answer.index_run_id
    assert frontmatter["input_manifest_hash"] == answer.input_manifest_hash
    assert frontmatter["planner"] == "heuristic"
    assert frontmatter["claim_support"] == "graph-grounded"
    assert "## Retrieval Mode" in text
    assert "- Planner: heuristic" in text
    assert "- Route reason: comparison question" in text
    assert "- Claim support: graph-grounded" in text
    assert "- Community level: 2" in text
    assert "- Dynamic community selection: True" in text
    assert "- Response type: Multiple Paragraphs" in text
    assert "## Source Trace" in text
    assert "## Raw GraphRAG Output" in text
    assert "graph ask --save" in test_project.paths.wiki_log_file.read_text(
        encoding="utf-8"
    )
    assert "graphrag-how-does-realm-differ-from-rag" in (
        test_project.paths.wiki_index_markdown.read_text(encoding="utf-8")
    )


def test_graph_query_requires_index_output(test_project) -> None:
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "src-1", "text": "RAG text"}]),
    )

    service = _build_query_service(
        test_project,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    with pytest.raises(GraphRAGQueryError, match="GraphRAG index output not found"):
        service.ask("What is RAG?", method="basic")


def test_graph_query_rejects_failed_last_index_run(test_project) -> None:
    _write_ready_graph(test_project, index_success=False)

    service = _build_query_service(
        test_project,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    with pytest.raises(GraphRAGQueryError, match="last GraphRAG index run failed"):
        service.ask("What is RAG?", method="basic")


def test_graph_query_surfaces_command_failure(test_project) -> None:
    _write_ready_graph(test_project)

    def runner(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            command,
            2,
            stdout="",
            stderr="query failed\n",
        )

    service = _build_query_service(test_project, runner)

    with pytest.raises(GraphRAGQueryError, match="query failed"):
        service.ask("What is RAG?", method="basic")
