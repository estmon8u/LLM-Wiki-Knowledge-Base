from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import textwrap
from typing import Optional, Sequence

from src.models.command_models import CommandContext
from src.models.wiki_models import LintReport, SearchResult, StatusSnapshot


WELCOME_MESSAGE = (
    "KB Terminal Workspace ready. Use Tab to switch panes, Up/Down for command "
    "history, and :help for commands."
)

PANE_ORDER = ("session", "status", "search", "citations", "history", "help")

PANE_TITLES = {
    "session": "Session",
    "status": "Status",
    "search": "Search Results",
    "citations": "Citations",
    "history": "History",
    "help": "Help",
}


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
        self.command_history: list[str] = []
        self.had_errors = False
        self.active_pane = "session"
        self.last_status_snapshot = self._snapshot_status()
        self.last_search_query: Optional[str] = None
        self.last_search_results: list[SearchResult] = []
        self.last_question: Optional[str] = None
        self.last_answer_text: Optional[str] = None
        self.last_citations: list[SearchResult] = []
        self.last_lint_report: Optional[LintReport] = None
        self.last_export_paths: list[str] = []
        self._application: Optional[object] = None
        self._input_field: Optional[object] = None
        self._sidebar_area: Optional[object] = None
        self._main_area: Optional[object] = None
        self._main_frame: Optional[object] = None
        self._reset_session(WELCOME_MESSAGE)

    def run_interactive(self) -> None:  # pragma: no cover - requires a real TTY
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise RuntimeError(
                "kb tui requires an interactive terminal. Use --snapshot or "
                "--command to preview it in non-interactive environments."
            )

        application = self._build_interactive_application()
        self._refresh_interactive_buffers()
        application.run()

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
        content_width = max(width or terminal_size.columns, 90)
        content_height = max(height or terminal_size.lines, 28)

        header_lines = [
            "KB Terminal Workspace",
            f"Project: {self.command_context.project_root}",
            f"Active Pane: {self._pane_title()}",
        ]

        project_panel = self._build_box(
            "Project",
            self._sidebar_lines(),
            min(42, max(34, content_width // 3)),
        )
        pane_width = max(46, content_width - len(project_panel[0]) - 2)
        pane_panel = self._build_box(
            self._pane_title(),
            self._pane_lines(max_lines=max(content_height - 12, 12)),
            pane_width,
        )

        if content_width >= 118:
            body_lines = self._combine_columns(project_panel, pane_panel)
        else:
            stacked_project = self._build_box(
                "Project", self._sidebar_lines(), content_width
            )
            stacked_pane = self._build_box(
                self._pane_title(),
                self._pane_lines(max_lines=max(content_height - 18, 12)),
                content_width,
            )
            body_lines = [*stacked_project, "", *stacked_pane]

        return "\n".join([*header_lines, "", *body_lines, "", *self._footer_lines()])

    def serialize_transcript(self) -> str:
        lines = [
            "KB Terminal Workspace",
            f"Project: {self.command_context.project_root}",
            f"Active Pane: {self._pane_title()}",
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

        self.command_history.append(command_text)
        self.command_history = self.command_history[-120:]
        self._append("you", command_text)

        try:
            command_name, argument_text = self._parse_command(command_text)
            response_text, keep_running = self._dispatch(command_name, argument_text)
        except Exception as error:
            self.had_errors = True
            self._append("kb", f"Error: {error}")
            self._refresh_interactive_buffers()
            return True

        if response_text:
            self._append("kb", response_text)
        if not keep_running:
            self._append("system", "Session ended.")

        self._refresh_interactive_buffers()
        return keep_running

    def _dispatch(self, command_name: str, argument_text: str) -> tuple[str, bool]:
        if command_name == "help":
            self._set_active_pane("help")
            return self._help_text(), True
        if command_name == "quit":
            return "Leaving KB terminal workspace.", False
        if command_name == "clear":
            self._reset_session("Transcript cleared. Use Tab to move between panes.")
            self._set_active_pane("session")
            return "", True
        if command_name == "pane":
            return self._run_pane(argument_text), True
        if command_name == "init":
            return self._run_init(), True
        if command_name == "status":
            self._set_active_pane("status")
            return self._format_status_output(self._refresh_status_snapshot()), True
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
            "pane": "pane",
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

    def _run_pane(self, argument_text: str) -> str:
        target = argument_text.strip().lower()
        if not target:
            raise ValueError(
                "Provide a pane name: session, status, search, citations, history, or help."
            )
        if target not in PANE_ORDER:
            raise ValueError(
                "Unknown pane. Choose session, status, search, citations, history, or help."
            )
        self._set_active_pane(target)
        return f"Focused {self._pane_title().lower()} pane."

    def _run_init(self) -> str:
        project_service = self.command_context.services["project"]
        config_service = self.command_context.services["config"]
        manifest_service = self.command_context.services["manifest"]

        created_items = project_service.ensure_structure()
        created_items.extend(config_service.ensure_files())
        if manifest_service.ensure_manifest():
            created_items.append("raw/_manifest.json")

        self._refresh_status_snapshot()
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

        self._refresh_status_snapshot()
        lines = [result.message]
        if result.source is not None:
            lines.append(f"- slug: {result.source.slug}")
            lines.append(f"- raw path: {result.source.raw_path}")
        return "\n".join(lines)

    def _run_compile(self, argument_text: str) -> str:
        compile_service = self.command_context.services["compile"]
        force = "--force" in argument_text.split()
        result = compile_service.compile(force=force)

        self._refresh_status_snapshot()
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
        self.last_search_query = query_text
        self.last_search_results = results
        self._set_active_pane("search")
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
        self.last_question = question
        self.last_answer_text = answer.answer
        self.last_citations = answer.citations
        self._set_active_pane("citations")

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
        self.last_lint_report = report
        if not report.issues:
            return "No lint issues found."
        return self._format_lint_report(report)

    def _run_export(self) -> str:
        export_service = self.command_context.services["export"]
        result = export_service.export_vault()
        self.last_export_paths = result.exported_paths

        lines = [f"Exported {len(result.exported_paths)} markdown file(s) to the vault"]
        lines.extend(f"- {path}" for path in result.exported_paths)
        return "\n".join(lines)

    def _help_text(self) -> str:
        return "\n".join(self._help_lines())

    def _help_lines(self) -> list[str]:
        return [
            "Commands",
            ":init",
            ":status",
            ":ingest <path>",
            ":compile [--force]",
            ":search <terms>",
            ":query <question>",
            ":lint",
            ":export-vault",
            ":pane <session|status|search|citations|history|help>",
            ":clear",
            ":quit",
            "",
            "Keys",
            "Tab / Shift+Tab: cycle panes",
            "F1: help pane",
            "F2: status pane",
            "F3: search pane",
            "F4: citations pane",
            "F5: session pane",
            "F6: history pane",
            "Ctrl-R: refresh status",
            "Ctrl-L: clear transcript",
            "Ctrl-Q: quit",
            "Up / Down: command history in the prompt",
            "",
            "Any plain sentence is treated as a query request.",
        ]

    def _sidebar_lines(self) -> list[str]:
        snapshot = self.last_status_snapshot
        lines = [
            f"cwd: {self.command_context.cwd}",
            "",
            "Panes",
        ]
        for pane_name in PANE_ORDER:
            marker = ">" if pane_name == self.active_pane else " "
            lines.append(f"{marker} {PANE_TITLES[pane_name]}")

        lines.extend(
            [
                "",
                "Status",
                *self._status_lines(snapshot),
                "",
                "Shortcuts",
                "Tab cycle panes",
                "Up/Down prompt history",
                "Ctrl-R refresh status",
                "Ctrl-L clear transcript",
                "Ctrl-Q quit",
            ]
        )
        return lines

    def _sidebar_text(self) -> str:
        return "\n".join(self._sidebar_lines())

    def _pane_lines(self, max_lines: Optional[int] = None) -> list[str]:
        lines = self._current_pane_lines()
        if max_lines is None or len(lines) <= max_lines:
            return lines
        if self.active_pane in {"session", "history"}:
            return lines[-max_lines:]
        return lines[:max_lines]

    def _current_pane_lines(self) -> list[str]:
        if self.active_pane == "session":
            lines = self._session_lines()
            return lines or ["No session output yet."]
        if self.active_pane == "status":
            return self._status_pane_lines()
        if self.active_pane == "search":
            return self._search_pane_lines()
        if self.active_pane == "citations":
            return self._citations_pane_lines()
        if self.active_pane == "history":
            return self._history_pane_lines()
        return self._help_lines()

    def _session_lines(self) -> list[str]:
        lines: list[str] = []
        for message in self.messages:
            lines.extend(self._message_lines(message))
            lines.append("")
        if lines:
            lines.pop()
        return lines

    def _status_pane_lines(self) -> list[str]:
        snapshot = self.last_status_snapshot
        lines = [
            "Focused status view.",
            "",
            *self._status_lines(snapshot),
        ]
        if self.last_export_paths:
            lines.extend(
                [
                    "",
                    "Recent exports",
                    *[f"- {path}" for path in self.last_export_paths[-5:]],
                ]
            )
        return lines

    def _search_pane_lines(self) -> list[str]:
        lines = ["Focused search results view.", ""]
        if self.last_search_query is None:
            lines.append("No search has been run yet.")
            lines.append("Run :search <terms> to populate this pane.")
            return lines

        lines.append(f"Last search: {self.last_search_query}")
        lines.append("")
        if not self.last_search_results:
            lines.append("No wiki pages matched that query.")
            return lines

        for result in self.last_search_results:
            lines.extend(self._format_search_result(result))
            lines.append("")
        if lines[-1] == "":
            lines.pop()
        return lines

    def _citations_pane_lines(self) -> list[str]:
        lines = ["Focused citation view.", ""]
        if self.last_question is None or self.last_answer_text is None:
            lines.append("No question has been answered yet.")
            lines.append("Type a plain sentence or use :query <question>.")
            return lines

        lines.append(f"Question: {self.last_question}")
        lines.append("")
        lines.append("Answer")
        lines.extend(self._wrapped_lines(self.last_answer_text, width=84))
        lines.append("")

        if not self.last_citations:
            lines.append("No citations returned for the last answer.")
            return lines

        lines.append("Citations")
        for citation in self.last_citations:
            lines.append(f"- {citation.title} [{citation.path}]")
        return lines

    def _history_pane_lines(self) -> list[str]:
        lines = [
            "Focused command history view.",
            "Use Up / Down in the prompt to replay recent commands.",
            "",
        ]
        if not self.command_history:
            lines.append("No commands entered yet.")
            return lines

        start_index = max(1, len(self.command_history) - 19)
        for index, entry in enumerate(self.command_history[-20:], start=start_index):
            lines.append(f"{index}. {entry}")
        return lines

    def _footer_lines(self) -> list[str]:
        return [
            "Keys: Tab panes | F1 help | F2 status | F3 search | F4 citations | F5 session | F6 history",
            "Prompt: Up/Down history | Ctrl-R refresh | Ctrl-L clear | Ctrl-Q quit | plain text => query",
        ]

    def _pane_title(self) -> str:
        return PANE_TITLES[self.active_pane]

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

    def _refresh_status_snapshot(self) -> StatusSnapshot:
        self.last_status_snapshot = self._snapshot_status()
        return self.last_status_snapshot

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
        self.messages = self.messages[-80:]

    def _reset_session(self, system_message: str) -> None:
        self.messages = [SessionMessage(speaker="system", text=system_message)]

    def _set_active_pane(self, pane_name: str) -> None:
        if pane_name not in PANE_ORDER:
            raise ValueError(f"Unknown pane: {pane_name}")
        self.active_pane = pane_name

    def _wrapped_lines(self, text: str, *, width: int) -> list[str]:
        wrapped_lines: list[str] = []
        for raw_line in text.splitlines() or [""]:
            segments = textwrap.wrap(raw_line, width=width) or [""]
            wrapped_lines.extend(segments)
        return wrapped_lines

    def _build_box(self, title: str, lines: list[str], width: int) -> list[str]:
        box_width = max(width, 24)
        inner_width = box_width - 4
        rendered_lines = [
            f"+{'-' * (box_width - 2)}+",
            f"| {title.ljust(inner_width)} |",
            f"+{'-' * (box_width - 2)}+",
        ]

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

    def _build_interactive_application(
        self,
        *,
        input_stream: Optional[object] = None,
        output_stream: Optional[object] = None,
    ) -> object:
        try:
            from prompt_toolkit import Application
            from prompt_toolkit.history import InMemoryHistory
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.styles import Style
            from prompt_toolkit.widgets import Frame, TextArea
        except ImportError as error:  # pragma: no cover - exercised via packaging
            raise RuntimeError(
                "Interactive kb tui requires prompt-toolkit. Run poetry install and try again."
            ) from error

        self._sidebar_area = TextArea(
            text=self._sidebar_text(),
            read_only=True,
            focusable=False,
            scrollbar=True,
            wrap_lines=False,
            width=38,
        )
        self._main_area = TextArea(
            text=self._pane_text(),
            read_only=True,
            focusable=False,
            scrollbar=True,
            wrap_lines=True,
        )
        self._input_field = TextArea(
            prompt="kb> ",
            multiline=False,
            wrap_lines=False,
            history=InMemoryHistory(),
            accept_handler=self._accept_buffer,
            height=1,
        )
        self._main_frame = Frame(self._main_area, title=self._pane_title())

        header = Window(
            height=1,
            content=FormattedTextControl(text=lambda: self._header_text()),
            style="class:header",
            always_hide_cursor=True,
        )
        footer = Window(
            height=2,
            content=FormattedTextControl(text=lambda: self._footer_text()),
            style="class:footer",
            always_hide_cursor=True,
        )

        root_container = HSplit(
            [
                header,
                VSplit(
                    [Frame(self._sidebar_area, title="Project"), self._main_frame],
                    padding=1,
                ),
                footer,
                Frame(self._input_field, title="Command"),
            ]
        )

        application = Application(
            layout=Layout(root_container, focused_element=self._input_field),
            key_bindings=self._build_key_bindings(KeyBindings()),
            full_screen=True,
            mouse_support=False,
            refresh_interval=0.2,
            input=input_stream,
            output=output_stream,
            style=Style.from_dict(
                {
                    "header": "reverse bold",
                    "footer": "reverse",
                    "frame.label": "bold",
                }
            ),
        )
        self._application = application
        return application

    def _build_key_bindings(self, key_bindings: object) -> object:
        @key_bindings.add("tab")
        def _next_pane(event: object) -> None:
            self._cycle_pane(1)
            self._focus_input(event)

        @key_bindings.add("s-tab")
        def _previous_pane(event: object) -> None:
            self._cycle_pane(-1)
            self._focus_input(event)

        @key_bindings.add("f1")
        def _help(event: object) -> None:
            self._set_active_pane("help")
            self._refresh_interactive_buffers()
            self._focus_input(event)

        @key_bindings.add("f2")
        def _status(event: object) -> None:
            self._refresh_status_snapshot()
            self._set_active_pane("status")
            self._refresh_interactive_buffers()
            self._focus_input(event)

        @key_bindings.add("f3")
        def _search(event: object) -> None:
            self._set_active_pane("search")
            self._refresh_interactive_buffers()
            self._focus_input(event)

        @key_bindings.add("f4")
        def _citations(event: object) -> None:
            self._set_active_pane("citations")
            self._refresh_interactive_buffers()
            self._focus_input(event)

        @key_bindings.add("f5")
        def _session(event: object) -> None:
            self._set_active_pane("session")
            self._refresh_interactive_buffers()
            self._focus_input(event)

        @key_bindings.add("f6")
        def _history(event: object) -> None:
            self._set_active_pane("history")
            self._refresh_interactive_buffers()
            self._focus_input(event)

        @key_bindings.add("c-r")
        def _refresh(event: object) -> None:
            self._refresh_status_snapshot()
            self._append("system", "Status snapshot refreshed.")
            self._refresh_interactive_buffers()
            self._focus_input(event)

        @key_bindings.add("c-l")
        def _clear(event: object) -> None:
            self._reset_session("Transcript cleared. Use Tab to move between panes.")
            self._set_active_pane("session")
            self._refresh_interactive_buffers()
            self._focus_input(event)

        @key_bindings.add("c-q")
        def _quit(event: object) -> None:
            event.app.exit()

        return key_bindings

    def _accept_buffer(self, buffer: object) -> bool:
        raw_input = buffer.text
        keep_running = self.handle_input(raw_input)
        buffer.reset()
        if self._application is not None:
            if not keep_running:
                self._application.exit()
            else:
                self._application.layout.focus(self._input_field)
        return False

    def _refresh_interactive_buffers(self) -> None:
        if self._sidebar_area is not None:
            self._sidebar_area.text = self._sidebar_text()
        if self._main_area is not None:
            self._main_area.text = self._pane_text()
        if self._main_frame is not None:
            self._main_frame.title = self._pane_title()
        if self._application is not None:
            self._application.invalidate()

    def _cycle_pane(self, step: int) -> None:
        current_index = PANE_ORDER.index(self.active_pane)
        next_index = (current_index + step) % len(PANE_ORDER)
        self._set_active_pane(PANE_ORDER[next_index])
        self._refresh_interactive_buffers()

    def _focus_input(self, event: object) -> None:
        if self._input_field is not None:
            event.app.layout.focus(self._input_field)

    def _header_text(self) -> str:
        return (
            f" KB Terminal Workspace | {self._pane_title()} | "
            f"{self.command_context.project_root} "
        )

    def _footer_text(self) -> str:
        return "\n".join(self._footer_lines())

    def _pane_text(self) -> str:
        return "\n".join(self._pane_lines())
