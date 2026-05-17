"""Tests for GraphRAG command service API adapter behavior."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from graphwiki_kb.services import graphrag_command_service as command_module
from graphwiki_kb.services import graphrag_runtime as runtime_module
from graphwiki_kb.services.graphrag_command_service import (
    GraphRAGApiBackend,
    GraphRAGCommandError,
    GraphRAGCommandResult,
    GraphRAGCommandService,
    SubprocessGraphRAGApiBackend,
    _call_graphrag_entrypoint,
    _extract_progress_label,
    _index_command,
    _init_command,
    _query_command,
    _query_return_to_text,
    _split_index_method,
)
from graphwiki_kb.services.graphrag_runtime import (
    GraphRAGCompatibilityError,
    _require_parameters,
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


def test_subprocess_backend_builds_documented_cli_commands(test_project) -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    backend = SubprocessGraphRAGApiBackend(test_project.paths, runner)

    result = backend.index(
        workspace_dir=test_project.paths.graph_dir / "graphrag",
        method="fast",
        dry_run=True,
        cache=True,
        skip_validation=False,
        verbose=False,
        status_callback=None,
    )

    assert result.command[:2] == ("graphrag", "index")
    assert "--dry-run" in result.command
    assert calls[0][1]["cwd"] == test_project.paths.root
    assert calls[0][1]["encoding"] == "utf-8"
    assert calls[0][1]["errors"] == "replace"
    assert calls[0][1]["env"]["PYTHONIOENCODING"] == "utf-8"


def test_subprocess_backend_builds_init_index_and_query_options(
    test_project,
) -> None:
    calls: list[tuple[str, ...]] = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    backend = SubprocessGraphRAGApiBackend(test_project.paths, runner)
    workspace_dir = test_project.paths.graph_dir / "graphrag"
    data_dir = workspace_dir / "output"
    labels: list[str] = []

    init_result = backend.init_workspace(
        workspace_dir=workspace_dir,
        model="chat",
        embedding="embed",
        force=True,
    )
    index_result = backend.index(
        workspace_dir=workspace_dir,
        method="standard-update",
        dry_run=False,
        cache=False,
        skip_validation=True,
        verbose=True,
        status_callback=labels.append,
    )
    query_result = backend.query(
        workspace_dir=workspace_dir,
        question="What changed?",
        method="global",
        data_dir=data_dir,
        community_level=1,
        dynamic_community_selection=False,
        response_type="Multiple paragraphs",
        streaming=True,
        verbose=True,
    )

    assert init_result.command[:2] == ("graphrag", "init")
    assert "--force" in init_result.command
    assert index_result.command[:2] == ("graphrag", "index")
    assert "--no-cache" in index_result.command
    assert "--skip-validation" in index_result.command
    assert "--verbose" in index_result.command
    assert labels == ["starting graph index", "ok"]
    assert query_result.command[:2] == ("graphrag", "query")
    assert "--data" in query_result.command
    assert str(data_dir) in query_result.command
    assert "--community-level" in query_result.command
    assert "--no-dynamic-selection" in query_result.command
    assert "--response-type" in query_result.command
    assert "--streaming" in query_result.command
    assert calls[-1][-1] == "What changed?"


def test_command_service_falls_back_to_cli_on_entrypoint_contract_error(
    test_project,
) -> None:
    class BrokenBackend:
        def index(self, **kwargs):
            raise GraphRAGCommandError(
                "graphrag.cli.index.index_cli has unsupported required parameter(s): x."
            )

    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="cli indexed", stderr="")

    service = GraphRAGCommandService(
        test_project.paths,
        api_backend=BrokenBackend(),
        fallback_backend=SubprocessGraphRAGApiBackend(test_project.paths, runner),
    )

    result = service.index(
        method="fast",
        dry_run=False,
        cache=True,
        skip_validation=False,
        verbose=False,
    )

    assert result.stdout == "cli indexed"
    assert calls[0][:2] == ("graphrag", "index")


def test_command_service_falls_back_to_cli_on_contract_error_stderr(
    test_project,
) -> None:
    backend = _FakeApiBackend(
        stderr="graphrag.cli.query missing expected parameter: query"
    )
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="cli answer", stderr="")

    service = GraphRAGCommandService(
        test_project.paths,
        api_backend=backend,
        fallback_backend=SubprocessGraphRAGApiBackend(test_project.paths, runner),
    )

    result = service.query("What changed?", method="basic")

    assert result.stdout == "cli answer"
    assert backend.calls[0][0] == "query"
    assert calls[0][:2] == ("graphrag", "query")


def test_command_service_falls_back_to_cli_on_entrypoint_encoding_error(
    test_project,
) -> None:
    backend = _FakeApiBackend(
        stderr="'charmap' codec can't encode character at position 1166: "
        "character maps to <undefined>"
    )
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="cli answer", stderr="")

    service = GraphRAGCommandService(
        test_project.paths,
        api_backend=backend,
        fallback_backend=SubprocessGraphRAGApiBackend(test_project.paths, runner),
    )

    result = service.query("What changed?", method="global")

    assert result.stdout == "cli answer"
    assert backend.calls[0][0] == "query"
    assert calls[0][:2] == ("graphrag", "query")


def test_command_service_can_use_subprocess_backend_directly(test_project) -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        return subprocess.CompletedProcess(command, 0, stdout="cli ok", stderr="")

    service = GraphRAGCommandService(
        test_project.paths,
        runner=runner,
        use_subprocess_backend=True,
    )

    result = service.init_workspace(model="chat", embedding="embed", force=False)

    assert result.stdout == "cli ok"
    assert calls[0][0][:2] == ("graphrag", "init")
    assert calls[0][1] == test_project.paths.root


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


def test_api_backend_validates_runtime_lazily(monkeypatch, test_project) -> None:
    """Regression: constructing services must not fail before GraphRAG is used."""
    calls = []

    def fail_runtime_validation() -> None:
        calls.append("validated")
        raise GraphRAGCompatibilityError("GraphRAG runtime is incompatible")

    monkeypatch.setattr(
        command_module,
        "validate_graphrag_runtime",
        fail_runtime_validation,
    )

    backend = GraphRAGApiBackend(test_project.paths)

    assert calls == []
    with pytest.raises(GraphRAGCommandError, match="runtime is incompatible"):
        backend.query(
            workspace_dir=test_project.paths.graph_dir / "graphrag",
            question="What is RAG?",
            method="basic",
            data_dir=test_project.paths.graph_dir / "graphrag" / "output",
            community_level=None,
            dynamic_community_selection=None,
            response_type=None,
            streaming=False,
            verbose=False,
        )
    assert calls == ["validated"]


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


def test_graphrag_entrypoint_adapter_supplies_current_index_defaults(
    test_project,
) -> None:
    """Regression: GraphRAG 3.0.x index signatures receive additive defaults."""
    calls = {}

    def fake_index_cli(
        root_dir,
        method,
        verbose,
        memprofile,
        cache,
        config_filepath,
        dry_run,
        skip_validation,
        output_dir,
    ):
        calls.update(locals())

    _call_graphrag_entrypoint(
        fake_index_cli,
        "fake.index_cli",
        root_dir=test_project.paths.graph_dir / "graphrag",
        method="fast",
        verbose=True,
        cache=False,
        dry_run=True,
        skip_validation=True,
    )

    assert calls["memprofile"] is False
    assert calls["config_filepath"] is None
    assert calls["output_dir"] is None
    assert calls["cache"] is False


def test_graphrag_entrypoint_adapter_filters_unsupported_basic_response_type(
    test_project,
) -> None:
    """Regression: GraphRAG basic search no longer accepts response_type."""
    calls = {}

    def fake_basic_search(
        config_filepath, data_dir, root_dir, streaming, query, verbose
    ):
        calls.update(locals())

    _call_graphrag_entrypoint(
        fake_basic_search,
        "fake.run_basic_search",
        config_filepath=None,
        data_dir=test_project.paths.graph_dir / "graphrag" / "output",
        root_dir=test_project.paths.graph_dir / "graphrag",
        response_type="Multiple Paragraphs",
        streaming=False,
        query="What is RAG?",
        verbose=False,
    )

    assert "response_type" not in calls
    assert calls["config_filepath"] is None
    assert calls["query"] == "What is RAG?"


def test_graphrag_entrypoint_adapter_rejects_unknown_required_parameter() -> None:
    """Regression: new upstream required parameters fail before a raw TypeError."""

    def fake_query(root_dir, query, new_required_parameter):
        return None

    with pytest.raises(GraphRAGCompatibilityError, match="new_required_parameter"):
        _call_graphrag_entrypoint(
            fake_query,
            "fake.query",
            root_dir="workspace",
            query="What is RAG?",
        )


def test_runtime_validation_rejects_unknown_required_graphrag_parameter() -> None:
    """Regression: runtime validation rejects new required upstream parameters."""

    def fake_index(root_dir, method, cache, skip_validation, unexpected):
        return None

    with pytest.raises(GraphRAGCompatibilityError, match="unexpected"):
        _require_parameters(
            fake_index,
            "fake.index",
            {"root_dir", "method", "cache", "skip_validation"},
        )


def test_runtime_validation_rejects_missing_expected_parameter() -> None:
    """Regression: runtime validation still catches removed GraphRAG parameters."""

    def fake_index(root_dir, cache, skip_validation):
        return None

    with pytest.raises(GraphRAGCompatibilityError, match="method"):
        _require_parameters(
            fake_index,
            "fake.index",
            {"root_dir", "method", "cache", "skip_validation"},
        )


def test_runtime_validation_rejects_unsupported_versions(monkeypatch) -> None:
    """Regression: GraphRAG version checks fail early before imports."""
    cases = [
        ("uninstalled", "not installed"),
        ("3.0.8", "too old"),
        ("3.1.0", "outside the supported range"),
    ]
    for version, message in cases:
        monkeypatch.setattr(
            runtime_module, "installed_graphrag_version", lambda v=version: v
        )
        with pytest.raises(GraphRAGCompatibilityError, match=message):
            runtime_module.validate_graphrag_runtime()


def test_runtime_validation_rejects_python_313(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module.sys, "version_info", (3, 13, 0))

    with pytest.raises(GraphRAGCompatibilityError, match="Python 3.10-3.12"):
        runtime_module.validate_graphrag_runtime()


def test_runtime_validation_rejects_unavailable_entrypoints(monkeypatch) -> None:
    """Regression: missing GraphRAG CLI entrypoints fail with a crisp message."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "graphrag.cli.index":
            raise ImportError("no index entrypoint")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(runtime_module, "installed_graphrag_version", lambda: "3.0.9")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(GraphRAGCompatibilityError, match="entrypoints are unavailable"):
        runtime_module.validate_graphrag_runtime()


def test_installed_graphrag_version_reports_uninstalled(monkeypatch) -> None:
    """Verifies missing GraphRAG package metadata maps to a stable label."""

    def fake_version(_name):
        raise runtime_module.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(runtime_module.importlib.metadata, "version", fake_version)

    assert runtime_module.installed_graphrag_version() == "uninstalled"


def test_parse_version_rejects_unknown_format() -> None:
    """Regression: unparseable GraphRAG versions produce a compatibility error."""
    with pytest.raises(GraphRAGCompatibilityError, match="Unable to parse"):
        runtime_module._parse_version("main")


def test_api_backend_query_uses_returned_answer_when_stdout_empty(
    monkeypatch, test_project
) -> None:
    """Regression: non-streaming GraphRAG responses can return instead of print."""

    def fake_run_query_entrypoint(**_kwargs):
        return ("Returned GraphRAG answer", {"context": []})

    monkeypatch.setattr(command_module, "validate_graphrag_runtime", lambda: None)
    monkeypatch.setattr(
        command_module,
        "_run_query_entrypoint",
        fake_run_query_entrypoint,
    )
    backend = GraphRAGApiBackend(test_project.paths)

    result = backend.query(
        workspace_dir=test_project.paths.graph_dir / "graphrag",
        question="What is RAG?",
        method="basic",
        data_dir=test_project.paths.graph_dir / "graphrag" / "output",
        community_level=None,
        dynamic_community_selection=None,
        response_type=None,
        streaming=False,
        verbose=False,
    )

    assert result.returncode == 0
    assert result.stdout == "Returned GraphRAG answer\n"


def test_run_query_entrypoint_preserves_zero_community_level(
    monkeypatch,
    test_project,
) -> None:
    """Regression: community_level=0 is explicit and must not become the default."""
    calls = {}
    graph_module = types.ModuleType("graphrag")
    cli_module = types.ModuleType("graphrag.cli")
    query_module = types.ModuleType("graphrag.cli.query")

    def fake_search(**kwargs):
        calls.update(kwargs)
        return "answer"

    query_module.run_basic_search = fake_search
    query_module.run_drift_search = fake_search
    query_module.run_global_search = fake_search
    query_module.run_local_search = fake_search
    cli_module.query = query_module
    graph_module.cli = cli_module
    monkeypatch.setitem(sys.modules, "graphrag", graph_module)
    monkeypatch.setitem(sys.modules, "graphrag.cli", cli_module)
    monkeypatch.setitem(sys.modules, "graphrag.cli.query", query_module)

    result = command_module._run_query_entrypoint(
        workspace_dir=test_project.paths.graph_dir / "graphrag",
        data_dir=test_project.paths.graph_dir / "graphrag" / "output",
        method="global",
        community_level=0,
        dynamic_community_selection=None,
        response_type=None,
        streaming=False,
        question="What is RAG?",
        verbose=False,
    )

    assert result == "answer"
    assert calls["community_level"] == 0
    assert "dynamic_community_selection" not in calls


def test_run_query_entrypoint_passes_dynamic_selection_when_explicit(
    monkeypatch,
    test_project,
) -> None:
    """Regression: None is omitted, but explicit true/false are preserved."""
    calls = []
    graph_module = types.ModuleType("graphrag")
    cli_module = types.ModuleType("graphrag.cli")
    query_module = types.ModuleType("graphrag.cli.query")

    def fake_search(**kwargs):
        calls.append(kwargs)
        return "answer"

    query_module.run_basic_search = fake_search
    query_module.run_drift_search = fake_search
    query_module.run_global_search = fake_search
    query_module.run_local_search = fake_search
    cli_module.query = query_module
    graph_module.cli = cli_module
    monkeypatch.setitem(sys.modules, "graphrag", graph_module)
    monkeypatch.setitem(sys.modules, "graphrag.cli", cli_module)
    monkeypatch.setitem(sys.modules, "graphrag.cli.query", query_module)

    for value in (True, False):
        command_module._run_query_entrypoint(
            workspace_dir=test_project.paths.graph_dir / "graphrag",
            data_dir=test_project.paths.graph_dir / "graphrag" / "output",
            method="global",
            community_level=None,
            dynamic_community_selection=value,
            response_type=None,
            streaming=None,
            question="What is RAG?",
            verbose=False,
        )

    assert calls[0]["dynamic_community_selection"] is True
    assert calls[1]["dynamic_community_selection"] is False


def test_query_return_to_text_uses_first_tuple_item() -> None:
    """Verifies GraphRAG query return payload conversion stays conservative."""
    assert _query_return_to_text(("answer", {"context": []})) == "answer"
    assert _query_return_to_text((b"answer",)) == "answer"
    assert _query_return_to_text(()) == ""


def test_query_return_to_text_handles_response_like_objects() -> None:
    class TextValue:
        text = " text answer "

    class FallbackValue:
        def __str__(self) -> str:
            return " fallback answer "

    assert _query_return_to_text(TextValue()) == "text answer"
    assert _query_return_to_text(FallbackValue()) == "fallback answer"
    assert _query_return_to_text(None) == ""


def test_progress_capture_flushes_pending_status_line() -> None:
    labels: list[str] = []
    capture = command_module._ProgressCapture(labels.append)

    assert capture.write("Running workflow: extract_graph\nProgress: 50") > 0
    capture.flush()

    assert labels == ["Running workflow: extract_graph", "Progress: 50"]


def test_command_output_helper_edges() -> None:
    assert command_module._append_error("", "detail") == "detail"
    assert command_module._append_error("stderr", "") == "stderr"
    assert command_module._append_error("stderr\n", "detail") == "stderr\ndetail"
    assert command_module._is_entrypoint_contract_error(
        "GraphRAG CLI entrypoints are unavailable"
    )
    assert not command_module._is_entrypoint_contract_error("regular runtime error")
    assert command_module._sanitize_stderr(("graphrag", "index"), "err") == "err"
    assert (
        command_module._sanitize_stderr(
            ("graphrag", "index", "--dry-run"),
            "--- Logging error ---\n"
            "Dry run complete, exiting...\n"
            "logger.info\n"
            "Arguments: (True,)",
        )
        == ""
    )
    long_line = "x" * (command_module.MAX_PROGRESS_LABEL_LENGTH + 10)
    assert _extract_progress_label(long_line).endswith("...")
