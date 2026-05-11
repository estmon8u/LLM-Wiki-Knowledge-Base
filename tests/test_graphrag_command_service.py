from __future__ import annotations

import subprocess

import pytest

from src.services.graphrag_command_service import (
    GraphRAGCommandError,
    GraphRAGCommandService,
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
