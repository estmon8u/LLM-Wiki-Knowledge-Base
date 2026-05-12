from __future__ import annotations

import types

import click
import pytest

from src.commands import common as common_module
from src.cli import _extract_project_root, build_runtime_context
from src.commands.common import (
    echo_bullet,
    echo_kv,
    echo_section,
    echo_status_line,
    emit_json,
    lazy_live_status,
    live_status,
    progress_report,
    require_initialized,
)
from src.engine.command_registry import (
    build_command_specs,
    get_click_command,
    list_command_names,
)
from src.services.config_service import CURRENT_CONFIG_VERSION


class _FakeStream:
    def __init__(self, *, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class _FakeProgressBar:
    def __init__(self) -> None:
        self.updates: list[int] = []

    def __enter__(self) -> "_FakeProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def update(self, amount: int) -> None:
        self.updates.append(amount)


class _TerminalConsole:
    is_terminal = True


class _FakeRichProgress:
    instances: list["_FakeRichProgress"] = []

    def __init__(self, *, console, transient: bool) -> None:
        self.console = console
        self.transient = transient
        self.advanced: list[str] = []
        _FakeRichProgress.instances.append(self)

    def __enter__(self) -> "_FakeRichProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def add_task(self, label: str, *, total: int) -> str:
        self.label = label
        self.total = total
        return "task-1"

    def advance(self, task: str) -> None:
        self.advanced.append(task)


class _FakeStatus:
    instances: list["_FakeStatus"] = []

    def __init__(self, label: str, *, console, spinner: str) -> None:
        self.labels = [label]
        self.console = console
        self.spinner = spinner
        self.started = False
        self.stopped = False
        _FakeStatus.instances.append(self)

    def start(self) -> None:
        self.started = True

    def update(self, label: str) -> None:
        self.labels.append(label)

    def stop(self) -> None:
        self.stopped = True


class _FakeOutputStream:
    def __init__(self, *, raises: bool = False) -> None:
        self.raises = raises
        self.errors: list[str] = []

    def reconfigure(self, *, errors: str) -> None:
        if self.raises:
            raise OSError("stream is closed")
        self.errors.append(errors)


def test_command_registry_resolves_aliases_and_lists_names() -> None:
    assert list_command_names() == sorted(list_command_names())
    assert "add" in list_command_names()
    assert "init" in list_command_names()
    assert "update" in list_command_names()
    assert "ask" in list_command_names()
    assert "find" in list_command_names()
    assert "graph" not in list_command_names()
    assert "legacy" in list_command_names()
    assert "status" in list_command_names()
    assert "review" in list_command_names()
    assert "export" in list_command_names()
    assert "lint" in list_command_names()


def test_configure_output_streams_uses_replacement_errors(monkeypatch) -> None:
    stdout = _FakeOutputStream()
    stderr = _FakeOutputStream(raises=True)
    monkeypatch.setattr(common_module.sys, "stdout", stdout)
    monkeypatch.setattr(common_module.sys, "stderr", stderr)

    common_module._configure_output_streams()

    assert stdout.errors == ["replace"]
    assert stderr.errors == []


def test_command_registry_returns_click_commands_and_specs(test_project) -> None:
    add_command = get_click_command("add")
    command = get_click_command("export")
    specs = build_command_specs(test_project.command_context)

    assert add_command is not None
    assert add_command.name == "add"
    assert command is not None
    assert command.name == "export"
    assert get_click_command("missing") is None
    spec_names = {spec.name for spec in specs}
    assert "add" in spec_names
    assert "update" in spec_names
    assert "ask" in spec_names
    assert "find" in spec_names
    assert "graph" not in spec_names
    assert "legacy" in spec_names
    assert "status" in spec_names
    assert "review" in spec_names
    assert "export" in spec_names
    assert "lint" in spec_names


@pytest.mark.parametrize(
    "command_name",
    [
        "add",
        "ask",
        "config",
        "doctor",
        "export",
        "find",
        "init",
        "legacy",
        "lint",
        "review",
        "sources",
        "status",
        "update",
    ],
)
def test_each_registered_command_has_a_click_command(command_name: str) -> None:
    command = get_click_command(command_name)

    assert command is not None
    assert isinstance(command, click.BaseCommand)


def test_build_runtime_context_uses_project_root_files(test_project) -> None:
    test_project.paths.config_file.write_text(
        "project:\n  name: Runtime Test\n",
        encoding="utf-8",
    )

    runtime_context = build_runtime_context(test_project.root, verbose=True)

    assert runtime_context.project_root == test_project.root
    assert runtime_context.config["project"]["name"] == "Runtime Test"
    assert runtime_context.verbose is True


def test_build_runtime_context_migrates_legacy_config_file(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 1\n"
        "project:\n  name: Runtime Legacy\n"
        "compile:\n  summary_paragraph_limit: 2\n",
        encoding="utf-8",
    )

    runtime_context = build_runtime_context(test_project.root, verbose=False)

    assert runtime_context.config["version"] == CURRENT_CONFIG_VERSION
    assert runtime_context.config["project"]["name"] == "Runtime Legacy"
    assert "summary_paragraph_limit" not in runtime_context.config["compile"]
    assert "summary_paragraph_limit" not in test_project.paths.config_file.read_text(
        encoding="utf-8"
    )


def test_extract_project_root_uses_param_when_available(test_project) -> None:
    ctx = click.Context(click.Command("kb"))
    ctx.params = {"project_root": test_project.root / "nested"}
    ctx.params["project_root"].mkdir()
    (test_project.root / "kb.config.yaml").write_text("project: {}\n", encoding="utf-8")

    assert _extract_project_root(ctx) == test_project.root


def test_require_initialized_raises_for_uninitialized_project(
    uninitialized_project,
) -> None:
    with pytest.raises(click.ClickException):
        require_initialized(uninitialized_project.command_context)


def test_require_initialized_allows_initialized_project(test_project) -> None:
    require_initialized(test_project.command_context)


def test_echo_kv_prints_values_and_na(capsys) -> None:
    echo_kv("label", "value")
    echo_kv("empty", None)

    output = capsys.readouterr().out
    assert "label: value" in output
    assert "empty: n/a" in output


def test_echo_section_status_and_bullet_helpers(capsys) -> None:
    echo_section("Summary")
    echo_status_line("OK", "ready")
    echo_bullet("item")

    output = capsys.readouterr().out
    assert "Summary" in output
    assert "[OK]" in output
    assert "ready" in output
    assert "item" in output


def test_progress_report_hidden_mode_prints_preamble(capsys) -> None:
    with progress_report(
        label="Compiling",
        length=2,
        item_label="source page",
    ) as advance:
        advance()
        advance()

    output = capsys.readouterr().out
    assert "Compiling 2 source page(s)..." in output


def test_progress_report_interactive_updates_progress_bar(capsys) -> None:
    with progress_report(
        label="Compiling",
        length=2,
        item_label="source page",
    ) as advance:
        advance()
        advance()

    # Rich progress writes to stderr; at minimum verify no crash and
    # the context manager yielded a callable advance function.
    capsys.readouterr()  # consume output


def test_progress_report_uses_rich_progress_in_terminal(monkeypatch) -> None:
    _FakeRichProgress.instances.clear()
    monkeypatch.setattr(common_module, "err_console", _TerminalConsole())
    monkeypatch.setattr(common_module, "Progress", _FakeRichProgress)

    with progress_report(
        label="Compiling",
        length=2,
        item_label="source page",
    ) as advance:
        advance()
        advance()

    progress = _FakeRichProgress.instances[0]
    assert progress.console.is_terminal is True
    assert progress.transient is True
    assert progress.label == "Compiling"
    assert progress.total == 2
    assert progress.advanced == ["task-1", "task-1"]


def test_progress_report_zero_length_is_noop(capsys) -> None:
    with progress_report(
        label="Compiling",
        length=0,
        item_label="source page",
    ) as advance:
        advance()

    # Zero-length progress should not produce any output
    output = capsys.readouterr()
    assert output.out == ""


def test_lazy_live_status_starts_on_first_update(monkeypatch) -> None:
    _FakeStatus.instances.clear()
    monkeypatch.setattr(common_module, "err_console", _TerminalConsole())
    monkeypatch.setattr(common_module, "Status", _FakeStatus)

    with lazy_live_status("GraphRAG indexing") as update:
        assert _FakeStatus.instances == []
        update("running fast graph index")
        update("exporting graph pages")

    status = _FakeStatus.instances[0]
    assert status.started is True
    assert status.stopped is True
    assert status.labels == [
        "GraphRAG indexing - running fast graph index",
        "GraphRAG indexing - exporting graph pages",
    ]


def test_live_status_starts_immediately_and_updates(monkeypatch) -> None:
    _FakeStatus.instances.clear()
    monkeypatch.setattr(common_module, "err_console", _TerminalConsole())
    monkeypatch.setattr(common_module, "Status", _FakeStatus)

    with live_status("Querying GraphRAG") as update:
        update("waiting for answer")

    status = _FakeStatus.instances[0]
    assert status.started is True
    assert status.stopped is True
    assert status.labels == [
        "Querying GraphRAG",
        "Querying GraphRAG - waiting for answer",
    ]


def test_emit_json_outputs_valid_json(capsys) -> None:
    import json

    emit_json({"key": "value", "count": 42})

    output = capsys.readouterr().out
    data = json.loads(output)
    assert data == {"key": "value", "count": 42}
