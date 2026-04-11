from __future__ import annotations

from pathlib import Path

import pytest

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
    assert "Any plain sentence is treated as a query request." in split
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
    assert "Transcript cleared. Type :help for commands." in summary.transcript
    assert "Use :command syntax for explicit actions" in summary.transcript
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
    assert service._session_panel_lines(20) == ["No session output yet."]

    with pytest.raises(ValueError, match="Unknown TUI command"):
        service._dispatch("bogus", "")
