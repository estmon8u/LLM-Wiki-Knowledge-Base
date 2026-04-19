from __future__ import annotations

import types

import click
import pytest

from src.cli import _extract_project_root, build_runtime_context
from src.commands.common import (
    echo_bullet,
    echo_kv,
    echo_section,
    echo_status_line,
    progress_report,
    require_initialized,
)
from src.engine.command_registry import (
    build_command_specs,
    get_click_command,
    list_command_names,
)
from src.engine.tool_registry import (
    _lint_wiki,
    _read_manifest,
    _review_wiki,
    _search_wiki,
    _unsupported,
    build_tool_specs,
)
from src.services.config_service import CURRENT_CONFIG_VERSION
from src.models.tool_models import ToolContext


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


def test_command_registry_resolves_aliases_and_lists_names() -> None:
    assert list_command_names() == sorted(list_command_names())
    assert "add" in list_command_names()
    assert "init" in list_command_names()
    assert "check" in list_command_names()
    assert "show" in list_command_names()
    assert "export" in list_command_names()
    assert "query" in list_command_names()


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
    assert "compile" in spec_names
    assert "check lint" in spec_names
    assert "show status" in spec_names
    assert "export vault" in spec_names
    assert "query search" in spec_names
    assert "query ask" in spec_names


@pytest.mark.parametrize(
    "command_name",
    [
        "add",
        "init",
        "ingest",
        "compile",
        "query",
        "check",
        "show",
        "export",
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


def test_echo_kv_prints_values_and_na(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(click, "echo", captured.append)

    echo_kv("label", "value")
    echo_kv("empty", None)

    assert captured == ["label: value", "empty: n/a"]


def test_echo_section_status_and_bullet_helpers(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(click, "echo", captured.append)

    echo_section("Summary")
    echo_status_line("OK", "ready")
    echo_bullet("item")

    assert captured == ["Summary", "=======", "[OK] ready", "- item"]


def test_progress_report_hidden_mode_prints_preamble(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(click, "echo", captured.append)
    monkeypatch.setattr(click, "get_text_stream", lambda _name: _FakeStream(tty=False))

    with progress_report(
        label="Compiling",
        length=2,
        item_label="source page",
    ) as advance:
        advance()
        advance()

    assert captured == ["Compiling 2 source page(s)..."]


def test_progress_report_interactive_updates_progress_bar(monkeypatch) -> None:
    progress_bar = _FakeProgressBar()
    monkeypatch.setattr(click, "get_text_stream", lambda _name: _FakeStream(tty=True))
    monkeypatch.setattr(click, "progressbar", lambda *args, **kwargs: progress_bar)

    with progress_report(
        label="Compiling",
        length=2,
        item_label="source page",
    ) as advance:
        advance()
        advance()

    assert progress_bar.updates == [1, 1]


def test_progress_report_zero_length_is_noop(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(click, "echo", captured.append)

    with progress_report(
        label="Compiling",
        length=0,
        item_label="source page",
    ) as advance:
        advance()

    assert captured == []


def test_build_tool_specs_contains_expected_contracts() -> None:
    specs = build_tool_specs()

    assert [spec.name for spec in specs] == [
        "SearchWiki",
        "ReadManifest",
        "IngestSource",
        "LintWiki",
        "ReviewWiki",
        "ExportVault",
    ]
    assert specs[0].is_concurrency_safe is True
    assert specs[2].is_concurrency_safe is False


def test_search_wiki_tool_requires_query_and_returns_results(test_project) -> None:
    test_project.write_file("wiki/sources/result.md", "traceability traceability")
    tool_context = ToolContext(
        project_root=str(test_project.root),
        config=test_project.config,
        session_state={},
        services=test_project.services,
        messages=[],
        cancel_requested=lambda: False,
    )

    missing = _search_wiki({}, tool_context)
    found = _search_wiki({"query": "traceability", "limit": 1}, tool_context)

    assert missing.ok is False
    assert missing.content == "Missing search query."
    assert found.ok is True
    assert found.data["results"][0]["path"] == "wiki/sources/result.md"


def test_read_manifest_and_lint_tools_return_structured_payloads(test_project) -> None:
    source_path = test_project.write_file("notes/manifest.md", "# Manifest\n\nBody\n")
    test_project.services["ingest"].ingest_path(source_path)
    tool_context = ToolContext(
        project_root=str(test_project.root),
        config=test_project.config,
        session_state={},
        services=test_project.services,
        messages=[],
        cancel_requested=lambda: False,
    )

    manifest_result = _read_manifest({}, tool_context)
    lint_result = _lint_wiki({}, tool_context)

    assert manifest_result.ok is True
    assert manifest_result.data["sources"][0]["slug"] == "manifest"
    assert lint_result.ok is True
    assert any(
        issue["code"] == "stale-source-page" for issue in lint_result.data["issues"]
    )


def test_review_wiki_tool_returns_structured_payload(test_project) -> None:
    tool_context = ToolContext(
        project_root=str(test_project.root),
        config=test_project.config,
        session_state={},
        services=test_project.services,
        messages=[],
        cancel_requested=lambda: False,
    )

    result = _review_wiki({}, tool_context)

    assert result.ok is True
    assert result.data["mode"] == "no-sources"
    assert isinstance(result.data["issues"], list)


def test_unsupported_tool_reports_stubbed_execution(test_project) -> None:
    tool_context = ToolContext(
        project_root=str(test_project.root),
        config=test_project.config,
        session_state={},
        services=test_project.services,
        messages=[],
        cancel_requested=lambda: False,
    )

    result = _unsupported({}, tool_context)

    assert result.ok is False
    assert "not wired into the CLI yet" in result.content
