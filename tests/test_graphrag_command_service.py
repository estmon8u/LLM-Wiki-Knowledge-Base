"""Tests for GraphRAG command service API adapter behavior."""

from __future__ import annotations

import subprocess
import sys

import pytest

from graphwiki_kb.services import graphrag_command_service as command_module
from graphwiki_kb.services.graphrag_command_service import (
    GraphRAGApiBackend,
    GraphRAGCommandError,
    GraphRAGCommandResult,
    GraphRAGCommandService,
    _extract_progress_label,
    _index_command,
    _init_command,
    _query_command,
    _split_index_method,
)


class _FakeApiBackend:
    def __init__(self, *, returncode: int = 0, stderr: str = "") -> None:
        self.calls: list[tuple[str, dict]] = []
        self.returncode = returncode
        self.stderr = stderr

    def init_workspace(self, **kwargs):
        self.calls.append(("init", kwargs))
        return GraphRAGCommandResult(
            command=("graphrag.api", "initialize_project_at"),
            cwd=kwargs["workspace_dir"],
            returncode=self.returncode,
            stdout="initialized\n",
            stderr=self.stderr,
        )

    def index(self, **kwargs):
        self.calls.append(("index", kwargs))
        return GraphRAGCommandResult(
            command=("graphrag.api", "build_index", "--method", kwargs["method"]),
            cwd=kwargs["workspace_dir"],
            returncode=self.returncode,
            stdout="indexed\n" if self.returncode == 0 else "",
            stderr=self.stderr,
        )

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        return GraphRAGCommandResult(
            command=("graphrag.api", f"{kwargs['method']}_search"),
            cwd=kwargs["workspace_dir"],
            returncode=self.returncode,
            stdout="answer\n",
            stderr=self.stderr,
        )


def test_init_uses_python_api_backend(test_project) -> None:
    """Verifies workspace init is delegated to the Python API backend."""
    backend = _FakeApiBackend()
    service = GraphRAGCommandService(test_project.paths, api_backend=backend)

    result = service.init_workspace(
        model="chat-model", embedding="embed-model", force=False
    )

    assert result.stdout == "initialized\n"
    assert backend.calls == [
        (
            "init",
            {
                "workspace_dir": test_project.paths.graph_dir / "graphrag",
                "model": "chat-model",
                "embedding": "embed-model",
                "force": False,
            },
        )
    ]


def test_index_uses_python_api_backend(test_project) -> None:
    """Verifies indexing is delegated to GraphRAG's Python API backend."""
    backend = _FakeApiBackend()
    service = GraphRAGCommandService(test_project.paths, api_backend=backend)
    labels: list[str] = []

    result = service.index(
        method="fast-update",
        dry_run=True,
        cache=False,
        skip_validation=True,
        verbose=True,
        status_callback=labels.append,
    )

    assert result.returncode == 0
    assert result.stdout == "indexed\n"
    name, kwargs = backend.calls[0]
    assert name == "index"
    assert kwargs["method"] == "fast-update"
    assert kwargs["dry_run"] is True
    assert kwargs["cache"] is False
    assert kwargs["skip_validation"] is True
    assert kwargs["verbose"] is True
    assert kwargs["status_callback"].__self__ is labels


def test_query_uses_python_api_backend_with_data_dir(test_project) -> None:
    """Verifies queries carry active data directory and options into the API backend."""
    backend = _FakeApiBackend()
    service = GraphRAGCommandService(test_project.paths, api_backend=backend)
    data_dir = test_project.paths.graph_dir / "graphrag" / "output"

    result = service.query(
        "What changed?",
        method="local",
        data_dir=data_dir,
        community_level=2,
        dynamic_community_selection=True,
        response_type="Multiple paragraphs",
        verbose=True,
    )

    assert result.stdout == "answer\n"
    name, kwargs = backend.calls[0]
    assert name == "query"
    assert kwargs["question"] == "What changed?"
    assert kwargs["method"] == "local"
    assert kwargs["data_dir"] == data_dir
    assert kwargs["community_level"] == 2
    assert kwargs["dynamic_community_selection"] is True
    assert kwargs["response_type"] == "Multiple paragraphs"
    assert kwargs["verbose"] is True


def test_api_backend_failure_raises_with_stderr_detail(test_project) -> None:
    """Verifies non-zero GraphRAG API results still raise command errors."""
    service = GraphRAGCommandService(
        test_project.paths,
        api_backend=_FakeApiBackend(
            returncode=2, stderr="warning\nfatal index error\n"
        ),
    )

    with pytest.raises(GraphRAGCommandError, match="fatal index error") as exc_info:
        service.index(
            method="fast",
            dry_run=False,
            cache=True,
            skip_validation=False,
        )

    assert exc_info.value.result is not None
    assert exc_info.value.result.returncode == 2


def test_api_backend_failure_without_output_uses_generic_message(test_project) -> None:
    """Verifies failed API results without output keep the existing generic message."""
    service = GraphRAGCommandService(
        test_project.paths,
        api_backend=_FakeApiBackend(returncode=2, stderr=""),
    )

    with pytest.raises(GraphRAGCommandError, match="^GraphRAG command failed$"):
        service.index(
            method="fast",
            dry_run=False,
            cache=True,
            skip_validation=False,
        )


def test_api_backend_index_reports_progress_from_entrypoint_output(
    monkeypatch, test_project
) -> None:
    """Verifies Python entrypoint output updates live graph index status."""

    def fake_run_index_entrypoint(**_kwargs):
        print("Running workflow: extract_graph")
        print("Warning: noisy dependency output")
        print("Progress: 50%", file=sys.stderr)
        print("Running step: summarize", file=sys.stderr)

    monkeypatch.setattr(
        command_module,
        "_run_index_entrypoint",
        fake_run_index_entrypoint,
    )
    backend = GraphRAGApiBackend(test_project.paths)
    labels: list[str] = []

    result = backend.index(
        workspace_dir=test_project.paths.graph_dir / "graphrag",
        method="fast",
        dry_run=False,
        cache=True,
        skip_validation=False,
        verbose=False,
        status_callback=labels.append,
    )

    assert result.returncode == 0
    assert "Running workflow: extract_graph" in result.stdout
    assert "Progress: 50%" in result.stderr
    assert labels == [
        "starting graph index",
        "Running workflow: extract_graph",
        "Progress: 50%",
        "Running step: summarize",
    ]


def test_default_runner_sets_utf8_encoding(monkeypatch) -> None:
    """Verifies that default runner sets utf8 encoding."""
    calls = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "graphwiki_kb.services.graphrag_command_service.subprocess.run",
        fake_run,
    )

    result = GraphRAGCommandService._default_runner(("python", "-m", "graphrag"))

    assert result.returncode == 0
    assert calls["kwargs"]["encoding"] == "utf-8"


def test_command_tuple_helpers_include_api_and_options(test_project) -> None:
    """Verifies diagnostic command tuples preserve the meaningful options."""
    workspace_dir = test_project.paths.graph_dir / "graphrag"

    init_command = _init_command(workspace_dir, "chat", "embed", force=True)
    index_command = _index_command(
        workspace_dir,
        method="fast",
        dry_run=True,
        cache=False,
        skip_validation=True,
        verbose=True,
    )
    query_command = _query_command(
        workspace_dir,
        "What changed?",
        method="local",
        data_dir=workspace_dir / "output",
        community_level=2,
        dynamic_community_selection=True,
        response_type="Multiple paragraphs",
        streaming=False,
        verbose=True,
    )

    assert init_command[:2] == ("graphrag.api", "initialize_project_at")
    assert "--force" in init_command
    assert index_command[:2] == ("graphrag.api", "build_index")
    assert "--dry-run" in index_command
    assert "--no-cache" in index_command
    assert "--skip-validation" in index_command
    assert query_command[:2] == ("graphrag.api", "local_search")
    assert "--data" in query_command
    assert "--no-streaming" in query_command
    assert query_command[-1] == "What changed?"


def test_split_index_method_maps_update_suffix() -> None:
    """Verifies GraphRAG update methods map to build_index update mode."""
    assert _split_index_method("fast") == ("fast", False)
    assert _split_index_method("fast-update") == ("fast", True)
    assert _split_index_method("standard-update") == ("standard", True)


def test_system_exit_code_handles_non_numeric_payloads() -> None:
    """Verifies SystemExit payloads are normalized before recording failures."""
    assert command_module._system_exit_code(SystemExit(None)) == 0
    assert command_module._system_exit_code(SystemExit("")) == 0
    assert command_module._system_exit_code(SystemExit(True)) == 1
    assert command_module._system_exit_code(SystemExit({"error": "boom"})) == 1


def test_extract_progress_label_suppresses_noisy_warning() -> None:
    """Verifies that extract progress label suppresses noisy warning."""
    assert _extract_progress_label("Warning: noisy dependency output") == ""
