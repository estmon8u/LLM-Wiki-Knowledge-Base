from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import DummyOutput

from src.models.wiki_models import LintIssue, LintReport
from src.services.tui_service import TuiService


def test_tui_service_render_supports_stacked_and_split_layouts(
    uninitialized_project,
) -> None:
    service = TuiService(uninitialized_project.command_context)

    stacked = service.render(width=92, height=26)
    split = service.render(width=140, height=30)

    assert "KB Terminal Workspace" in stacked
    assert "Project" in stacked
    assert "Session" in stacked
    assert "Active Pane: Session" in split
    assert "plain text => query" in split
    assert "initialized: false" in split


def test_tui_service_scripted_workflow_supports_plain_question_queries(
    test_project,
) -> None:
    test_project.write_file(
        "sample.md",
        "# Sample Research Note\n\n"
        "Markdown-first knowledge bases preserve source traceability.\n\n"
        "They can be linted for broken links and missing citations.\n",
    )
    service = TuiService(test_project.command_context)

    summary = service.run_scripted(
        ["ingest sample.md", "compile", "How does traceability help?", "quit"]
    )

    assert summary.had_errors is False
    assert "Ingested Sample Research Note" in summary.transcript
    assert "Compiled 1 source page(s)" in summary.transcript
    assert "Citations:" in summary.transcript
    assert "Session ended." in summary.transcript


def test_tui_service_scripted_mode_reports_errors(uninitialized_project) -> None:
    service = TuiService(uninitialized_project.command_context)

    summary = service.run_scripted(["compile"])

    assert summary.had_errors is True
    assert "Error: Project not initialized. Run :init first." in summary.transcript


def test_tui_service_scripted_mode_stops_after_quit(uninitialized_project) -> None:
    service = TuiService(uninitialized_project.command_context)

    summary = service.run_scripted(["quit", "status"])

    assert "Session ended." in summary.transcript
    assert "YOU: status" not in summary.transcript


def test_tui_service_clear_resets_transcript(uninitialized_project) -> None:
    service = TuiService(uninitialized_project.command_context)

    summary = service.run_scripted(["status", "clear", ":help"])

    assert summary.had_errors is False
    assert "Transcript cleared. Use Tab to move between panes." in summary.transcript
    assert ":pane <session|status|search|citations|history|help>" in summary.transcript
    assert "initialized: false" not in summary.transcript


def test_tui_service_handles_blank_and_unknown_explicit_commands(
    uninitialized_project,
) -> None:
    service = TuiService(uninitialized_project.command_context)

    assert service.handle_input("   ") is True
    assert service.handle_input(":bogus") is True

    assert service.had_errors is True
    assert "Unknown TUI command: bogus" in service.serialize_transcript()


def test_tui_service_init_can_be_repeated_without_new_scaffold(
    uninitialized_project,
) -> None:
    service = TuiService(uninitialized_project.command_context)

    summary = service.run_scripted(["init", "init"])

    assert summary.had_errors is False
    assert "project already had the required scaffold" in summary.transcript


def test_tui_service_supports_pane_switching_and_history_view(test_project) -> None:
    service = TuiService(test_project.command_context)

    summary = service.run_scripted([":pane history", ":pane help"])

    assert summary.had_errors is False
    assert service.active_pane == "help"
    assert "Focused history pane." in summary.transcript
    history_lines = service._history_pane_lines()
    assert any(":pane history" in line for line in history_lines)


def test_tui_service_validates_ingest_search_and_query_arguments(test_project) -> None:
    service = TuiService(test_project.command_context)

    assert service.handle_input("ingest") is True
    assert service.handle_input("search") is True
    assert service.handle_input("query") is True

    transcript = service.serialize_transcript()
    assert "Provide a path to ingest." in transcript
    assert "Provide at least one search term." in transcript
    assert "Provide a question to answer." in transcript


def test_tui_service_supports_search_lint_and_export_commands(test_project) -> None:
    test_project.write_file(
        "sample.md",
        "# Sample Research Note\n\n"
        "Markdown-first knowledge bases preserve source traceability.\n\n"
        "They can be linted for broken links and missing citations.\n",
    )
    service = TuiService(test_project.command_context)

    setup_summary = service.run_scripted(["ingest sample.md", "compile --force"])
    assert setup_summary.had_errors is False

    search_summary = service.run_scripted(
        ["search traceability", "search zzznomatchtoken"]
    )
    assert "score=" in search_summary.transcript
    assert "No wiki pages matched that query." in search_summary.transcript
    assert service.active_pane == "search"

    lint_summary = service.run_scripted(["lint"])
    assert "No lint issues found." in lint_summary.transcript

    export_summary = service.run_scripted(["export-vault"])
    assert "Exported " in export_summary.transcript
    assert "vault/obsidian/sources/sample-research-note.md" in export_summary.transcript


def test_tui_service_private_helpers_cover_dispatch_and_formatting_paths(
    test_project,
    tmp_path: Path,
) -> None:
    service = TuiService(test_project.command_context)
    absolute_source = tmp_path / "quoted.md"
    absolute_source.write_text("# Quoted\n\nBody\n", encoding="utf-8")

    resolved = service._resolve_user_path(f'"{absolute_source}"')

    assert resolved == absolute_source
    assert service._run_pane("status") == "Focused status pane."
    assert service.active_pane == "status"

    report = LintReport(
        issues=[
            LintIssue(
                severity="error",
                code="broken-link",
                path="wiki/sources/bad.md",
                message="Broken link detected.",
            ),
            LintIssue(
                severity="warning",
                code="orphan-page",
                path="wiki/sources/orphan.md",
                message="No inbound links.",
            ),
            LintIssue(
                severity="suggestion",
                code="cross-link",
                path="wiki/sources/idea.md",
                message="Consider adding backlinks.",
            ),
        ]
    )

    formatted = service._format_lint_report(report)
    partial_formatted = service._format_lint_report(
        LintReport(issues=[report.issues[0]])
    )

    assert "ERRORS (1):" in formatted
    assert "WARNINGS (1):" in formatted
    assert "SUGGESTIONS (1):" in formatted
    assert "WARNINGS" not in partial_formatted

    test_project.write_file("wiki/sources/bad.md", "# Bad\n\n[[Missing Target]]\n")
    lint_output = service._run_lint()
    assert "broken-link" in lint_output

    service.messages = []
    service.active_pane = "session"
    assert service._pane_lines() == ["No session output yet."]

    with pytest.raises(ValueError, match="Unknown TUI command"):
        service._dispatch("bogus", "")


def test_tui_service_builds_interactive_application_and_keybindings(
    uninitialized_project,
) -> None:
    service = TuiService(uninitialized_project.command_context)
    with create_pipe_input() as pipe_input:
        application = service._build_interactive_application(
            input_stream=pipe_input,
            output_stream=DummyOutput(),
        )

    assert application is service._application
    assert service._input_field is not None
    assert service._sidebar_area is not None
    assert service._main_area is not None
    assert "KB Terminal Workspace" in service._header_text()
    assert "plain text => query" in service._footer_text()
    assert "KB Terminal Workspace ready" in service._pane_text()

    bindings = service._build_key_bindings(KeyBindings())

    class FakeLayout:
        def __init__(self) -> None:
            self.focus_calls = 0

        def focus(self, _: object) -> None:
            self.focus_calls += 1

    class FakeApp:
        def __init__(self) -> None:
            self.layout = FakeLayout()
            self.exited = False

        def exit(self) -> None:
            self.exited = True

    class FakeEvent:
        def __init__(self) -> None:
            self.app = FakeApp()

    def run_binding(key_name: str) -> FakeEvent:
        event = FakeEvent()
        for binding in bindings.bindings:
            values = [getattr(key, "value", str(key)) for key in binding.keys]
            if values == [key_name]:
                binding.handler(event)
                return event
        raise AssertionError(f"Binding not found for {key_name}")

    run_binding("c-i")
    assert service.active_pane == "status"

    run_binding("s-tab")
    assert service.active_pane == "session"

    run_binding("f1")
    assert service.active_pane == "help"

    run_binding("f2")
    assert service.active_pane == "status"

    run_binding("f3")
    assert service.active_pane == "search"

    run_binding("f4")
    assert service.active_pane == "citations"

    run_binding("f5")
    assert service.active_pane == "session"

    run_binding("f6")
    assert service.active_pane == "history"

    refresh_event = run_binding("c-r")
    assert "Status snapshot refreshed." in service.serialize_transcript()
    assert refresh_event.app.layout.focus_calls > 0

    clear_event = run_binding("c-l")
    assert (
        "Transcript cleared. Use Tab to move between panes."
        in service.serialize_transcript()
    )
    assert clear_event.app.layout.focus_calls > 0

    quit_event = run_binding("c-q")
    assert quit_event.app.exited is True


def test_tui_service_accept_buffer_exits_when_command_requests_shutdown(
    uninitialized_project,
) -> None:
    service = TuiService(uninitialized_project.command_context)

    class FakeBuffer:
        def __init__(self, text: str) -> None:
            self.text = text
            self.reset_called = False

        def reset(self) -> None:
            self.reset_called = True

    class FakeLayout:
        def __init__(self) -> None:
            self.focus_calls = 0

        def focus(self, _: object) -> None:
            self.focus_calls += 1

    class FakeApplication:
        def __init__(self) -> None:
            self.layout = FakeLayout()
            self.exited = False
            self.invalidated = False

        def exit(self) -> None:
            self.exited = True

        def invalidate(self) -> None:
            self.invalidated = True

    service._input_field = object()
    service._application = FakeApplication()

    buffer = FakeBuffer("quit")
    keep_running = service._accept_buffer(buffer)

    assert keep_running is False
    assert buffer.reset_called is True
    assert service._application.exited is True
