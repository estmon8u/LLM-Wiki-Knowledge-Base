from __future__ import annotations

import io
import subprocess

import pytest

from src.services.graphrag_command_service import (
    GraphRAGCommandError,
    GraphRAGCommandService,
    _extract_progress_label,
)


def test_index_builds_python_module_command(test_project) -> None:
    calls = []

    def runner(command, *, cwd, capture_output, text):
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "capture_output": capture_output,
                "text": text,
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    service = GraphRAGCommandService(test_project.paths, runner=runner)

    result = service.index(
        method="fast",
        dry_run=True,
        cache=False,
        skip_validation=True,
        verbose=True,
    )

    assert result.returncode == 0
    assert result.stdout == "ok\n"
    command = calls[0]["command"]
    assert command[1:4] == ("-m", "graphrag", "index")
    assert "--root" in command
    assert str(test_project.paths.graph_dir / "graphrag") in command
    assert "--method" in command
    assert "fast" in command
    assert "--dry-run" in command
    assert "--no-cache" in command
    assert "--skip-validation" in command
    assert "--verbose" in command
    assert calls[0]["cwd"] == test_project.paths.root
    assert calls[0]["capture_output"] is True
    assert calls[0]["text"] is True


def test_default_runner_sets_utf8_encoding(monkeypatch) -> None:
    calls = {}

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )

    result = GraphRAGCommandService._default_runner(("python", "-m", "graphrag"))

    assert result.returncode == 0
    assert calls["kwargs"]["encoding"] == "utf-8"


def test_query_builds_full_option_command(test_project) -> None:
    calls = []

    def runner(command, *, cwd, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="answer\n", stderr="")

    service = GraphRAGCommandService(test_project.paths, runner=runner)

    result = service.query(
        "What changed?",
        method="local",
        community_level=2,
        dynamic_community_selection=True,
        response_type="Multiple paragraphs",
        verbose=True,
    )

    command = calls[0]
    assert result.stdout == "answer\n"
    assert command[1:4] == ("-m", "graphrag", "query")
    assert "--method" in command
    assert "local" in command
    assert "--community-level" in command
    assert "2" in command
    assert "--dynamic-community-selection" in command
    assert "--response-type" in command
    assert "Multiple paragraphs" in command
    assert "--verbose" in command
    assert command[-1] == "What changed?"


def test_streaming_index_sets_unbuffered_env_and_reports_progress(
    monkeypatch, test_project
) -> None:
    call = {}

    class FakePopen:
        def __init__(
            self,
            command,
            *,
            cwd,
            stdout,
            stderr,
            text,
            encoding=None,
            errors=None,
            bufsize=None,
            env=None,
        ):
            call.update(
                {
                    "command": command,
                    "cwd": cwd,
                    "stdout": stdout,
                    "stderr": stderr,
                    "text": text,
                    "encoding": encoding,
                    "errors": errors,
                    "bufsize": bufsize,
                    "env": env,
                }
            )
            self.returncode = 0
            self.stdout = io.StringIO("50% complete\nindexed\n")
            self.stderr = io.StringIO(
                "graphrag.index Running workflow: extract_graph\n"
            )

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.Popen",
        FakePopen,
    )
    labels: list[str] = []
    service = GraphRAGCommandService(test_project.paths)

    result = service.index(
        method="fast",
        dry_run=False,
        cache=True,
        skip_validation=False,
        status_callback=labels.append,
    )

    assert result.returncode == 0
    assert result.stdout == "50% complete\nindexed\n"
    assert "Running workflow: extract_graph\n" in result.stderr
    assert call["command"][1:4] == ("-m", "graphrag", "index")
    assert call["cwd"] == test_project.paths.root
    assert call["text"] is True
    assert call["encoding"] == "utf-8"
    assert call["errors"] == "replace"
    assert call["bufsize"] == 1
    assert call["env"]["PYTHONUNBUFFERED"] == "1"
    assert labels[0] == "starting graph index"
    assert "50% complete" in labels
    assert "Running workflow: extract_graph" in labels


def test_streaming_index_missing_python_raises_clear_error(
    monkeypatch, test_project
) -> None:
    def raise_missing_python(*_args, **_kwargs):
        raise FileNotFoundError("python")

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.Popen",
        raise_missing_python,
    )
    service = GraphRAGCommandService(test_project.paths)

    with pytest.raises(GraphRAGCommandError, match="Python executable was not found"):
        service.index(
            method="fast",
            dry_run=False,
            cache=True,
            skip_validation=False,
            status_callback=lambda _message: None,
        )


def test_streaming_index_failure_raises_with_stderr_detail(
    monkeypatch, test_project
) -> None:
    class FailedPopen:
        def __init__(self, *_args, **_kwargs):
            self.returncode = 2
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("warning\nfatal index error\n")

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.Popen",
        FailedPopen,
    )
    labels: list[str] = []
    service = GraphRAGCommandService(test_project.paths)

    with pytest.raises(GraphRAGCommandError, match="fatal index error") as exc_info:
        service.index(
            method="fast",
            dry_run=False,
            cache=True,
            skip_validation=False,
            status_callback=labels.append,
        )

    assert labels[0] == "starting graph index"
    assert exc_info.value.result is not None
    assert exc_info.value.result.returncode == 2
    assert "fatal index error" in exc_info.value.result.stderr


def test_extract_progress_label_suppresses_noisy_warning() -> None:
    assert _extract_progress_label("Warning: noisy dependency output") == ""


def test_init_builds_command_without_force_when_disabled(test_project) -> None:
    calls = []

    def runner(command, *, cwd, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    service = GraphRAGCommandService(test_project.paths, runner=runner)

    service.init_workspace(model="chat-model", embedding="embed-model", force=False)

    command = calls[0]
    assert command[1:4] == ("-m", "graphrag", "init")
    assert "chat-model" in command
    assert "embed-model" in command
    assert "--force" not in command


def test_failed_command_raises_with_stderr_detail(test_project) -> None:
    def runner(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            command,
            2,
            stdout="",
            stderr="warning\nfatal index error\n",
        )

    service = GraphRAGCommandService(test_project.paths, runner=runner)

    with pytest.raises(GraphRAGCommandError, match="fatal index error") as exc_info:
        service.index(
            method="fast",
            dry_run=False,
            cache=True,
            skip_validation=False,
        )

    assert exc_info.value.result is not None
    assert exc_info.value.result.returncode == 2


def test_successful_dry_run_filters_known_graphrag_logging_error(test_project) -> None:
    logging_error = (
        "--- Logging error ---\n"
        "Traceback (most recent call last):\n"
        '  File "graphrag/cli/index.py", line 119, in _run_index\n'
        '    logger.info("Dry run complete, exiting...", True)\n'
        "Message: 'Dry run complete, exiting...'\n"
        "Arguments: (True,)\n"
    )

    def runner(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr=logging_error)

    service = GraphRAGCommandService(test_project.paths, runner=runner)

    result = service.index(
        method="fast",
        dry_run=True,
        cache=True,
        skip_validation=False,
    )

    assert result.stderr == ""


def test_non_dry_run_keeps_stderr(test_project) -> None:
    def runner(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="warning\n")

    service = GraphRAGCommandService(test_project.paths, runner=runner)

    result = service.index(
        method="fast",
        dry_run=False,
        cache=True,
        skip_validation=False,
    )

    assert result.stderr == "warning\n"


def test_failed_command_without_output_uses_generic_message(test_project) -> None:
    def runner(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="")

    service = GraphRAGCommandService(test_project.paths, runner=runner)

    with pytest.raises(GraphRAGCommandError, match="^GraphRAG command failed$"):
        service.index(
            method="fast",
            dry_run=False,
            cache=True,
            skip_validation=False,
        )


def test_missing_python_executable_raises_clear_error(test_project) -> None:
    def runner(command, *, cwd, capture_output, text):
        raise FileNotFoundError("python")

    service = GraphRAGCommandService(test_project.paths, runner=runner)

    with pytest.raises(GraphRAGCommandError, match="Python executable was not found"):
        service.index(
            method="fast",
            dry_run=False,
            cache=True,
            skip_validation=False,
        )
