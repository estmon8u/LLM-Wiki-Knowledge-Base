from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import textwrap
from typing import Optional, Sequence

import click

from src.models.command_models import CommandContext
from src.models.wiki_models import LintReport, SearchResult, StatusSnapshot


WELCOME_MESSAGE = (
    "KB Terminal Workspace ready. Type :help for commands, or enter a plain "
    "question to run query directly."
)


@dataclass
class SessionMessage:
    speaker: str
    text: str


@dataclass
class TuiRunSummary:
    transcript: str
    had_errors: bool


class TuiService:
    def __init__(self, command_context: CommandContext) -> None:
        self.command_context = command_context
        self.messages: list[SessionMessage] = []
        self.had_errors = False
        self._reset_session(WELCOME_MESSAGE)

    def run_interactive(self) -> None:  # pragma: no cover - requires a real TTY
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise RuntimeError(
                "kb tui requires an interactive terminal. Use --command to run "
                "scripted TUI actions in non-interactive environments."
            )

        keep_running = True
        while keep_running:
            click.clear()
            click.echo(self.render())
            click.echo("")
            try:
                click.echo("kb> ", nl=False)
                raw_input = input()
            except EOFError:
                raw_input = "quit"
            except KeyboardInterrupt:
                self._append(
                    "system",
                    "Interrupted. Type :quit when you want to leave the terminal "
                    "workspace cleanly.",
                )
                continue

            keep_running = self.handle_input(raw_input)

        click.clear()
        click.echo(self.render())

    def run_scripted(self, commands: Sequence[str]) -> TuiRunSummary:
        keep_running = True
        for command in commands:
            if not keep_running:
                break
            keep_running = self.handle_input(command)

        if keep_running:
            self._append("system", "Scripted session complete.")

        return TuiRunSummary(
            transcript=self.serialize_transcript(),
            had_errors=self.had_errors,
        )

    def render(
        self, *, width: Optional[int] = None, height: Optional[int] = None
    ) -> str:
        terminal_size = shutil.get_terminal_size((120, 36))
        content_width = max(width or terminal_size.columns, 80)
        content_height = max(height or terminal_size.lines, 24)

        header_lines = [
            "KB Terminal Workspace",
            f"Project: {self.command_context.project_root}",
            "Persistent terminal workflow over the current knowledge-base services.",
        ]

        project_panel = self._build_box(
            "Project",
            self._project_panel_lines(),
            min(42, max(34, content_width // 3)),
        )
        session_width = max(40, content_width - len(project_panel[0]) - 2)
        session_panel = self._build_box(
            "Session",
            self._session_panel_lines(content_height),
            session_width,
        )

        body_lines: list[str]
        if content_width >= 118:
            body_lines = self._combine_columns(project_panel, session_panel)
        else:
            stacked_project = self._build_box(
                "Project", self._project_panel_lines(), content_width
            )
            stacked_session = self._build_box(
                "Session", self._session_panel_lines(content_height), content_width
            )
            body_lines = [*stacked_project, "", *stacked_session]

        footer_lines = [
            "Commands: :help, :status, :ingest <path>, :compile [--force], :search <terms>,",
            "          :query <question>, :lint, :export-vault, :clear, :quit",
            "Any plain sentence is treated as a query request.",
        ]

        return "\n".join([*header_lines, "", *body_lines, "", *footer_lines])

    def serialize_transcript(self) -> str:
        lines = [
            "KB Terminal Workspace",
            f"Project: {self.command_context.project_root}",
            "",
        ]
        for message in self.messages:
            label = message.speaker.upper()
            text_lines = message.text.splitlines() or [""]
            for index, line in enumerate(text_lines):
                prefix = f"{label}: " if index == 0 else " " * (len(label) + 2)
                lines.append(f"{prefix}{line}")
        return "\n".join(lines)

    def handle_input(self, raw_input: str) -> bool:
        command_text = raw_input.strip()
        if not command_text:
            return True

        self._append("you", command_text)

        try:
            command_name, argument_text = self._parse_command(command_text)
            response_text, keep_running = self._dispatch(command_name, argument_text)
        except Exception as error:
            self.had_errors = True
            self._append("kb", f"Error: {error}")
            return True

        if response_text:
            self._append("kb", response_text)
        if not keep_running:
            self._append("system", "Session ended.")
        return keep_running

    def _dispatch(self, command_name: str, argument_text: str) -> tuple[str, bool]:
        if command_name == "help":
            return self._help_text(), True
        if command_name == "quit":
            return "Leaving KB terminal workspace.", False
        if command_name == "clear":
            self._reset_session("Transcript cleared. Type :help for commands.")
            return "", True
        if command_name == "init":
            return self._run_init(), True
        if command_name == "status":
            return self._format_status_output(self._snapshot_status()), True
        if command_name == "ingest":
            self._ensure_initialized()
            return self._run_ingest(argument_text), True
        if command_name == "compile":
            self._ensure_initialized()
            return self._run_compile(argument_text), True
        if command_name == "search":
            self._ensure_initialized()
            return self._run_search(argument_text), True
        if command_name == "query":
            self._ensure_initialized()
            return self._run_query(argument_text), True
        if command_name == "lint":
            self._ensure_initialized()
            return self._run_lint(), True
        if command_name == "export-vault":
            self._ensure_initialized()
            return self._run_export(), True
        raise ValueError(f"Unknown TUI command: {command_name}")

    def _parse_command(self, command_text: str) -> tuple[str, str]:
        stripped = command_text.strip()
        explicit_command = stripped.startswith(":") or stripped.startswith("/")
        normalized = stripped[1:].strip() if explicit_command else stripped
        command_name, _, argument_text = normalized.partition(" ")
        canonical_name = self._canonical_command(command_name.lower())

        if explicit_command:
            if canonical_name is None:
                raise ValueError(
                    f"Unknown TUI command: {command_name}. Type :help to list commands."
                )
            return canonical_name, argument_text.strip()

        if canonical_name is not None:
            return canonical_name, argument_text.strip()

        return "query", stripped

    def _canonical_command(self, command_name: str) -> Optional[str]:
        aliases = {
            "?": "help",
            "help": "help",
            "clear": "clear",
            "cls": "clear",
            "init": "init",
            "status": "status",
            "ingest": "ingest",
            "compile": "compile",
            "search": "search",
            "query": "query",
            "lint": "lint",
            "export": "export-vault",
            "export-vault": "export-vault",
            "export_vault": "export-vault",
            "quit": "quit",
            "exit": "quit",
        }
        return aliases.get(command_name)

    def _run_init(self) -> str:
        project_service = self.command_context.services["project"]
        config_service = self.command_context.services["config"]
        manifest_service = self.command_context.services["manifest"]

        created_items = project_service.ensure_structure()
        created_items.extend(config_service.ensure_files())
        if manifest_service.ensure_manifest():
            created_items.append("raw/_manifest.json")

        lines = [f"Initialized project at {self.command_context.project_root}"]
        if created_items:
            lines.extend(f"- created {item}" for item in created_items)
        else:
            lines.append("- project already had the required scaffold")
        return "\n".join(lines)

    def _run_ingest(self, argument_text: str) -> str:
        if not argument_text:
            raise ValueError("Provide a path to ingest.")

        ingest_service = self.command_context.services["ingest"]
        source_path = self._resolve_user_path(argument_text)
        result = ingest_service.ingest_path(source_path)

        lines = [result.message]
        if result.source is not None:
            lines.append(f"- slug: {result.source.slug}")
            lines.append(f"- raw path: {result.source.raw_path}")
        return "\n".join(lines)

    def _run_compile(self, argument_text: str) -> str:
        compile_service = self.command_context.services["compile"]
        force = "--force" in argument_text.split()
        result = compile_service.compile(force=force)

        lines = [
            f"Compiled {result.compiled_count} source page(s)",
            f"Skipped {result.skipped_count} source page(s)",
        ]
        lines.extend(f"- updated {path}" for path in result.compiled_paths)
        return "\n".join(lines)

    def _run_search(self, argument_text: str) -> str:
        query_text = argument_text.strip()
        if not query_text:
            raise ValueError("Provide at least one search term.")

        search_service = self.command_context.services["search"]
        results = search_service.search(query_text, limit=5)
        if not results:
            return "No wiki pages matched that query."

        lines: list[str] = []
        for result in results:
            lines.extend(self._format_search_result(result))
        return "\n".join(lines)

    def _run_query(self, argument_text: str) -> str:
        question = argument_text.strip()
        if not question:
            raise ValueError("Provide a question to answer.")

        query_service = self.command_context.services["query"]
        answer = query_service.answer_question(question, limit=3)
        lines = [answer.answer]
        if answer.citations:
            lines.append("")
            lines.append("Citations:")
            lines.extend(
                f"- {citation.title} [{citation.path}]" for citation in answer.citations
            )
        return "\n".join(lines)

    def _run_lint(self) -> str:
        lint_service = self.command_context.services["lint"]
        report = lint_service.lint()
        if not report.issues:
            return "No lint issues found."

        return self._format_lint_report(report)

    def _run_export(self) -> str:
        export_service = self.command_context.services["export"]
        result = export_service.export_vault()

        lines = [f"Exported {len(result.exported_paths)} markdown file(s) to the vault"]
        lines.extend(f"- {path}" for path in result.exported_paths)
        return "\n".join(lines)

    def _help_text(self) -> str:
        return "\n".join(
            [
                "Use :command syntax for explicit actions, or enter a plain sentence to run query.",
                ":init",
                ":status",
                ":ingest <path>",
                ":compile [--force]",
                ":search <terms>",
                ":query <question>",
                ":lint",
                ":export-vault",
                ":clear",
                ":quit",
            ]
        )

    def _project_panel_lines(self) -> list[str]:
        snapshot = self._snapshot_status()
        return [
            f"project_root: {self.command_context.project_root}",
            f"cwd: {self.command_context.cwd}",
            "",
            *self._status_lines(snapshot),
            "",
            "Quick Commands",
            ":help",
            ":init",
            ":status",
            ":ingest <path>",
            ":compile [--force]",
            ":search <terms>",
            ":query <question>",
            ":lint",
            ":export-vault",
            ":clear",
            ":quit",
        ]

    def _session_panel_lines(self, content_height: int) -> list[str]:
        available_lines = max(content_height - 14, 10)
        lines: list[str] = []
        for message in self.messages:
            lines.extend(self._message_lines(message))
            lines.append("")
        if lines:
            lines.pop()
        if not lines:
            lines = ["No session output yet."]
        return lines[-available_lines:]

    def _message_lines(self, message: SessionMessage) -> list[str]:
        label = message.speaker.upper()
        raw_lines = message.text.splitlines() or [""]
        formatted: list[str] = []
        for index, raw_line in enumerate(raw_lines):
            prefix = f"{label}: " if index == 0 else " " * (len(label) + 2)
            formatted.append(f"{prefix}{raw_line}")
        return formatted

    def _snapshot_status(self) -> StatusSnapshot:
        project_service = self.command_context.services["project"]
        status_service = self.command_context.services["status"]
        return status_service.snapshot(initialized=project_service.is_initialized())

    def _format_status_output(self, snapshot: StatusSnapshot) -> str:
        return "\n".join(self._status_lines(snapshot))

    def _status_lines(self, snapshot: StatusSnapshot) -> list[str]:
        return [
            f"initialized: {str(snapshot.initialized).lower()}",
            f"source_count: {snapshot.source_count}",
            f"compiled_source_count: {snapshot.compiled_source_count}",
            f"concept_page_count: {snapshot.concept_page_count}",
            f"last_compile_at: {snapshot.last_compile_at or 'n/a'}",
        ]

    def _format_lint_report(self, report: LintReport) -> str:
        lines: list[str] = []
        for severity in ("error", "warning", "suggestion"):
            scoped = [issue for issue in report.issues if issue.severity == severity]
            if not scoped:
                continue
            lines.append(f"{severity.upper()}S ({len(scoped)}):")
            lines.extend(
                f"- {issue.code} [{issue.path}] {issue.message}" for issue in scoped
            )
        return "\n".join(lines)

    def _format_search_result(self, result: SearchResult) -> list[str]:
        return [
            f"- {result.title} [{result.path}] score={result.score}",
            f"  {result.snippet}",
        ]

    def _resolve_user_path(self, raw_path: str) -> Path:
        normalized = raw_path.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1]:
            if normalized[0] in {"'", '"'}:
                normalized = normalized[1:-1]

        source_path = Path(normalized)
        if source_path.is_absolute():
            return source_path
        return (self.command_context.cwd / source_path).resolve()

    def _ensure_initialized(self) -> None:
        project_service = self.command_context.services["project"]
        if not project_service.is_initialized():
            raise ValueError("Project not initialized. Run :init first.")

    def _append(self, speaker: str, text: str) -> None:
        self.messages.append(SessionMessage(speaker=speaker, text=text))
        self.messages = self.messages[-60:]

    def _reset_session(self, system_message: str) -> None:
        self.messages = [SessionMessage(speaker="system", text=system_message)]

    def _build_box(self, title: str, lines: list[str], width: int) -> list[str]:
        box_width = max(width, 24)
        inner_width = box_width - 4
        rendered_lines = [
            f"+{'-' * (box_width - 2)}+",
            f"| {title.ljust(inner_width)} |",
        ]
        rendered_lines.append(f"+{'-' * (box_width - 2)}+")

        for line in lines or [""]:
            wrapped = textwrap.wrap(line, width=inner_width) or [""]
            for segment in wrapped:
                rendered_lines.append(f"| {segment.ljust(inner_width)} |")

        rendered_lines.append(f"+{'-' * (box_width - 2)}+")
        return rendered_lines

    def _combine_columns(
        self, left_lines: list[str], right_lines: list[str], gap: int = 2
    ) -> list[str]:
        left_width = max(len(line) for line in left_lines)
        total_rows = max(len(left_lines), len(right_lines))
        padded_left = left_lines + [" " * left_width] * (total_rows - len(left_lines))
        padded_right = right_lines + [""] * (total_rows - len(right_lines))
        return [
            f"{left.ljust(left_width)}{' ' * gap}{right}"
            for left, right in zip(padded_left, padded_right)
        ]
