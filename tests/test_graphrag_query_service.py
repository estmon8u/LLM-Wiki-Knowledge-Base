"""Tests for test graphrag query service.

This module belongs to `tests.test_graphrag_query_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import pandas as pd
import pytest
import yaml

from graphwiki_kb.services.graphrag_command_service import (
    GraphRAGCommandResult,
    GraphRAGCommandService,
)
from graphwiki_kb.services.graphrag_freshness_service import (
    file_digest,
    graph_input_source_hashes,
    graph_runtime_digest,
)
from graphwiki_kb.services.graphrag_query_service import (
    GraphRAGQueryError,
    GraphRAGQueryService,
)
from graphwiki_kb.services.graphrag_status_service import GraphRAGStatusService


def _write_ready_graph(test_project, *, index_success: bool = True) -> None:
    """Handles write ready graph.

    Args:
        test_project: Test project value used by the operation.
        index_success: Index success value used by the operation.
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
            returncode=0 if index_success else 2,
            stdout="indexed" if index_success else "",
            stderr="" if index_success else "failed",
        ),
        input_digest=file_digest(input_path),
        config_digest=graph_runtime_digest(workspace_dir),
        input_source_count=1,
        source_hashes=graph_input_source_hashes(input_path),
        output_state="complete",
    )


def _build_query_service(test_project, runner):
    """Handles build query service.

    Args:
        test_project: Test project value used by the operation.
        runner: Runner value used by the operation.
    """
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
    """Verifies that graph query runs explicit method and options.

    Args:
        test_project: Test project value used by the operation.
    """
    _write_ready_graph(test_project)
    calls = []

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
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
    assert command[:2] == ("graphrag.api", "local_search")
    assert "--method" in command
    assert "local" in command
    assert "--data" in command
    assert str(test_project.paths.graph_dir / "graphrag" / "output") in command
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
    assert answer.raw_output == "GraphRAG answer\n"
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
    """Verifies that graph query raw output prefers stdout over progress stderr.

    Args:
        test_project: Test project value used by the operation.
    """
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
            command,
            0,
            stdout="GraphRAG answer\n",
            stderr="Warning: noisy dependency output\nProgress: 100%\n",
        )

    service = _build_query_service(test_project, runner)

    answer = service.ask("What is RAG?", method="basic")
    saved_path = service.save_answer(answer)

    assert answer.answer == "GraphRAG answer"
    assert answer.raw_output == "GraphRAG answer\n"
    assert "Progress" in answer.stderr
    saved_text = (test_project.root / saved_path).read_text(encoding="utf-8")
    frontmatter = yaml.safe_load(saved_text.split("---", 2)[1])
    assert frontmatter["citations"] == []
    assert frontmatter["citation_count"] == 0
    assert "GraphRAG answer" in saved_text
    assert "## Raw GraphRAG Stdout" in saved_text
    assert "## Raw GraphRAG Stderr" in saved_text
    assert "Warning: noisy dependency output" in saved_text
    assert "Progress: 100%" in saved_text


def test_graph_query_does_not_treat_stderr_as_answer(test_project) -> None:
    """Regression: successful stderr-only GraphRAG noise is not answer text."""
    _write_ready_graph(test_project)

    def runner(command, *, cwd, capture_output, text):
        """Runner."""
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="",
            stderr="Warning: progress-only output\n",
        )

    service = _build_query_service(test_project, runner)

    answer = service.ask("What is RAG?", method="basic")

    assert answer.answer == ""
    assert answer.raw_output == ""
    assert "progress-only" in answer.stderr


def test_graph_query_filters_noisy_stdout_before_saved_answer(test_project) -> None:
    """Regression: stdout progress/log lines are not saved as answer prose."""
    _write_ready_graph(test_project)

    def runner(command, *, cwd, capture_output, text):
        """Runner."""
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "INFO: loading GraphRAG output\n"
                "Running workflow: query\n"
                "Search Response:\n"
                "GraphRAG answer after logs.\n"
            ),
            stderr="",
        )

    service = _build_query_service(test_project, runner)

    answer = service.ask("What is RAG?", method="basic")
    saved_path = service.save_answer(answer)
    saved_text = (test_project.root / saved_path).read_text(encoding="utf-8")

    assert answer.answer == "GraphRAG answer after logs."
    assert "INFO: loading GraphRAG output" in answer.raw_output
    assert "Running workflow: query" in answer.raw_output
    answer_section = saved_text.split("## Retrieval Mode", 1)[0]
    assert "INFO: loading" not in answer_section
    assert "Running workflow" not in answer_section
    assert "INFO: loading" in saved_text
    assert "Running workflow" in saved_text


def test_graph_query_save_writes_analysis_page_and_refreshes_index(
    test_project,
) -> None:
    """Verifies that graph query save writes analysis page and refreshes index.

    Args:
        test_project: Test project value used by the operation.
    """
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
    answer.claim_support = "cited-graph-answer"

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
    assert frontmatter["claim_support"] == "cited-graph-answer"
    assert "## Retrieval Mode" in text
    assert "- Planner: heuristic" in text
    assert "- Route reason: comparison question" in text
    assert "- Support level: cited-graph-answer" in text
    assert "- Community level: 2" in text
    assert "- Dynamic community selection: True" in text
    assert "- Response type: Multiple Paragraphs" in text
    assert "## Source Trace" in text
    assert "## Raw GraphRAG Stdout" in text
    assert "## Raw GraphRAG Stderr" in text
    assert "graph ask --save" in test_project.paths.wiki_log_file.read_text(
        encoding="utf-8"
    )
    assert "graphrag-how-does-realm-differ-from-rag" in (
        test_project.paths.wiki_index_markdown.read_text(encoding="utf-8")
    )


def test_graph_query_requires_index_output(test_project) -> None:
    """Verifies that graph query requires index output.

    Args:
        test_project: Test project value used by the operation.
    """
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


def test_graph_query_requires_vector_store(test_project) -> None:
    """Regression: table-only output is not enough for GraphRAG query readiness."""
    _write_ready_graph(test_project)
    marker = (
        test_project.paths.graph_dir
        / "graphrag"
        / "output"
        / "lancedb"
        / "vector-store.marker"
    )
    marker.unlink()

    service = _build_query_service(
        test_project,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    with pytest.raises(GraphRAGQueryError, match="vector store"):
        service.ask("What is RAG?", method="basic")


def test_graph_query_allows_global_queries_without_vector_store(test_project) -> None:
    """Regression: global queries only require community tables and fresh metadata."""
    _write_ready_graph(test_project)
    marker = (
        test_project.paths.graph_dir
        / "graphrag"
        / "output"
        / "lancedb"
        / "vector-store.marker"
    )
    marker.unlink()

    def runner(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Global answer\n",
            stderr="",
        )

    service = _build_query_service(test_project, runner)

    answer = service.ask("What are the graph-wide themes?", method="global")

    assert answer.answer == "Global answer"


def test_graph_query_rejects_stale_graph_index_metadata(test_project) -> None:
    """Regression: complete output is not enough when input/config digests drift."""
    _write_ready_graph(test_project)
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "src-1", "text": "Changed RAG text"}]),
    )

    service = _build_query_service(
        test_project,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    with pytest.raises(GraphRAGQueryError, match="GraphRAG index is stale"):
        service.ask("What is RAG?", method="basic")


def test_graph_query_save_rejects_blank_answers(test_project) -> None:
    """Regression: blank GraphRAG output must not create misleading analysis pages."""
    _write_ready_graph(test_project)
    service = _build_query_service(
        test_project,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    with pytest.raises(GraphRAGQueryError, match="empty GraphRAG answer"):
        service.save_answer(
            service.ask("What is RAG?", method="basic"),
        )


def test_graph_query_save_does_not_overwrite_existing_analysis_page(
    test_project,
) -> None:
    """Regression: repeated saved GraphRAG answers receive unique slugs."""
    _write_ready_graph(test_project)

    def runner(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="GraphRAG answer\n",
            stderr="",
        )

    service = _build_query_service(test_project, runner)

    first = service.save_answer(service.ask("What is RAG?", method="basic"))
    second = service.save_answer(service.ask("What is RAG?", method="basic"))

    assert first == "wiki/analysis/graphrag-what-is-rag.md"
    assert second == "wiki/analysis/graphrag-what-is-rag-2.md"
    assert (test_project.root / first).read_text(encoding="utf-8")
    assert (test_project.root / second).read_text(encoding="utf-8")


def test_graph_query_rejects_failed_last_index_run(test_project) -> None:
    """Verifies that graph query rejects failed last index run.

    Args:
        test_project: Test project value used by the operation.
    """
    _write_ready_graph(test_project, index_success=False)

    service = _build_query_service(
        test_project,
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )

    with pytest.raises(GraphRAGQueryError, match="last GraphRAG index run failed"):
        service.ask("What is RAG?", method="basic")


def test_graph_query_surfaces_command_failure(test_project) -> None:
    """Verifies that graph query surfaces command failure.

    Args:
        test_project: Test project value used by the operation.
    """
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
            command,
            2,
            stdout="",
            stderr="query failed\n",
        )

    service = _build_query_service(test_project, runner)

    with pytest.raises(GraphRAGQueryError, match="query failed"):
        service.ask("What is RAG?", method="basic")
