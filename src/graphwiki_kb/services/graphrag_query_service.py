"""Graphrag query service service behavior for the knowledge-base workflow.

This module belongs to `graphwiki_kb.services.graphrag_query_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from graphwiki_kb.services.file_lock import file_lock
from graphwiki_kb.services.graphrag_command_service import (
    GraphRAGCommandError,
    GraphRAGCommandResult,
    GraphRAGCommandService,
)
from graphwiki_kb.services.graphrag_status_service import (
    GraphRAGStatus,
    GraphRAGStatusService,
    graph_not_ready_message,
)
from graphwiki_kb.services.graphrag_sync_service import file_digest
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
    unique_markdown_heading,
    utc_now_iso,
)
from graphwiki_kb.services.search_service import SearchService

GRAPH_QUERY_METHODS = ("local", "global", "drift", "basic")
GRAPH_DATA_REFERENCE_PATTERN = re.compile(r"\[Data:\s*[^\]]+\]")
_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_ANSWER_MARKERS = (
    "answer:",
    "response:",
    "search response:",
    "final answer:",
)
_DIAGNOSTIC_PREFIXES = (
    "debug:",
    "info:",
    "warning:",
    "error:",
    "running workflow",
    "running verb",
    "running step",
    "progress:",
    "loading ",
    "loaded ",
    "writing ",
    "wrote ",
    "initializing ",
    "querying ",
)


class GraphRAGQueryError(RuntimeError):
    """Error raised for graph ragquery failures.

    Attributes:
        See annotated class attributes for stored values.
    """

    pass


@dataclass
class GraphRAGQueryAnswer:
    """Represents graph ragquery answer behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    question: str
    answer: str
    raw_output: str
    method: str
    created_at: str
    index_run_id: str | None
    input_manifest_hash: str
    command: tuple[str, ...]
    stdout: str
    stderr: str
    community_level: int | None = None
    dynamic_community_selection: bool | None = None
    response_type: str | None = None
    saved_path: str | None = None
    retriever: str = "graphrag"
    planner: str | None = None
    route_reason: str | None = None
    claim_support: str | None = None
    source_trace: dict[str, str | None] = field(default_factory=dict)
    staleness_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serializes this value to a dictionary.

        Returns:
            dict[str, object] produced by the operation.
        """
        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload


class GraphRAGQueryService:
    """Coordinates graph ragquery operations.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        command_service: GraphRAGCommandService,
        status_service: GraphRAGStatusService,
        search_service: SearchService,
        *,
        refresh_index: Callable[[], None] | None = None,
    ) -> None:
        self.paths = paths
        self.command_service = command_service
        self.status_service = status_service
        self.search_service = search_service
        self._refresh_index = refresh_index

    def ask(
        self,
        question: str,
        *,
        method: str,
        community_level: int | None = None,
        dynamic_community_selection: bool | None = None,
        response_type: str | None = None,
        streaming: bool | None = None,
        verbose: bool = False,
    ) -> GraphRAGQueryAnswer:
        """Ask.

        Args:
            question: User question to answer from available evidence.
            method: Method value used by the operation.
            community_level: Community level value used by the operation.
            dynamic_community_selection: Dynamic community selection value used by the operation.
            response_type: Response type value used by the operation.
            streaming: GraphRAG streaming flag forwarded to the query CLI.
            verbose: Whether to emit verbose command output.

        Returns:
            GraphRAGQueryAnswer produced by the operation.
        """
        status = self.status_service.status()
        self._require_query_ready(status)
        input_manifest_hash = self._input_manifest_hash(status.input_path)
        try:
            result = self.command_service.query(
                question,
                method=method,
                data_dir=status.active_output_dir,
                community_level=community_level,
                dynamic_community_selection=dynamic_community_selection,
                response_type=response_type,
                streaming=streaming,
                verbose=verbose,
            )
        except GraphRAGCommandError as exc:
            raise GraphRAGQueryError(str(exc)) from exc

        answer = _answer_from_result(result)
        return GraphRAGQueryAnswer(
            question=question,
            answer=answer,
            raw_output=answer,
            method=method,
            created_at=utc_now_iso(),
            index_run_id=status.last_index_run_id,
            input_manifest_hash=input_manifest_hash,
            command=result.command,
            stdout=result.stdout,
            stderr=result.stderr,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            source_trace={
                "input_path": _relative_path(status.input_path, self.paths.root),
                "output_dir": _relative_path(
                    status.active_output_dir or status.output_dir,
                    self.paths.root,
                ),
                "index_run_id": status.last_index_run_id,
                "input_manifest_hash": input_manifest_hash,
            },
        )

    def save_answer(
        self,
        answer: GraphRAGQueryAnswer,
        *,
        slug: str | None = None,
    ) -> str:
        """Saves answer.

        Args:
            answer: Answer value used by the operation.
            slug: Slug value used by the operation.

        Returns:
            str produced by the operation.
        """
        safe_slug = slugify(slug) if slug else slugify(answer.question)
        if not safe_slug or safe_slug == "untitled":
            safe_slug = "analysis"
        safe_slug = f"graphrag-{safe_slug}"
        dest = self.paths.wiki_analysis_dir / f"{safe_slug}.md"
        page_text = self._render_saved_page(answer)
        atomic_write_text(dest, page_text)
        self.search_service.refresh_file(dest)
        self._append_log(answer.question, dest)
        if self._refresh_index is not None:
            self._refresh_index()
        answer.saved_path = dest.relative_to(self.paths.root).as_posix()
        return answer.saved_path

    def _render_saved_page(self, answer: GraphRAGQueryAnswer) -> str:
        summary = answer.answer.replace("\n", " ").strip()[:280].rstrip()
        insufficient_evidence = answer.claim_support == "no-answer" or not bool(
            answer.answer.strip()
        )
        citations = _graph_data_references(answer.answer)
        frontmatter = {
            "title": answer.question,
            "summary": summary,
            "type": "analysis",
            "retriever": answer.retriever,
            "method": answer.method,
            "question": answer.question,
            "created_at": answer.created_at,
            "saved_at": answer.created_at,
            "citations": citations,
            "insufficient_evidence": insufficient_evidence,
            "claim_count": 0,
            "citation_count": len(citations),
            "claims": [],
            "index_run_id": answer.index_run_id,
            "input_manifest_hash": answer.input_manifest_hash,
        }
        if answer.planner:
            frontmatter["planner"] = answer.planner
        if answer.claim_support:
            frontmatter["claim_support"] = answer.claim_support
        yaml_block = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        retrieval_lines = [
            f"- Retriever: {answer.retriever}",
            f"- GraphRAG method: {answer.method}",
            f"- Index run: {answer.index_run_id or 'unknown'}",
            f"- Input manifest hash: {answer.input_manifest_hash}",
        ]
        if answer.planner:
            retrieval_lines.append(f"- Planner: {answer.planner}")
        if answer.route_reason:
            retrieval_lines.append(f"- Route reason: {answer.route_reason}")
        if answer.claim_support:
            retrieval_lines.append(f"- Support level: {answer.claim_support}")
        if answer.community_level is not None:
            retrieval_lines.append(f"- Community level: {answer.community_level}")
        if answer.dynamic_community_selection is not None:
            retrieval_lines.append(
                "- Dynamic community selection: "
                f"{answer.dynamic_community_selection}"
            )
        if answer.response_type:
            retrieval_lines.append(f"- Response type: {answer.response_type}")
        trace_lines = [
            f"- GraphRAG input: {answer.source_trace.get('input_path')}",
            f"- GraphRAG output: {answer.source_trace.get('output_dir')}",
            "- Direct citation parsing: not available from this raw CLI wrapper yet",
        ]
        raw_output = answer.raw_output or "No raw GraphRAG output captured."
        return (
            f"---\n{yaml_block}\n---\n\n"
            f"# {answer.question}\n\n"
            "## Answer\n\n"
            f"{answer.answer or 'No answer text returned.'}\n\n"
            "## Retrieval Mode\n\n"
            f"{chr(10).join(retrieval_lines)}\n\n"
            "## Source Trace\n\n"
            f"{chr(10).join(trace_lines)}\n\n"
            "## Raw GraphRAG Output\n\n"
            "```text\n"
            f"{raw_output}\n"
            "```\n"
        )

    def _append_log(self, question: str, dest: Path) -> None:
        timestamp = utc_now_iso()
        with file_lock(self.paths.wiki_log_file):
            current = "# Activity Log\n"
            if self.paths.wiki_log_file.exists():
                current = self.paths.wiki_log_file.read_text(encoding="utf-8")
            if not current.endswith("\n"):
                current += "\n"
            rel = dest.relative_to(self.paths.root).as_posix()
            heading = unique_markdown_heading(
                current,
                f"## [{timestamp}] graph ask --save | {json.dumps(question)} -> {rel}",
            )
            current += f"\n{heading}\n"
            atomic_write_text(self.paths.wiki_log_file, current)

    def _require_query_ready(self, status: GraphRAGStatus) -> None:
        if not status.workspace_initialized:
            raise GraphRAGQueryError(graph_not_ready_message(status))
        if not status.input_exists:
            raise GraphRAGQueryError(graph_not_ready_message(status))
        if status.input_document_count == 0:
            raise GraphRAGQueryError(graph_not_ready_message(status))
        if not status.output_present:
            raise GraphRAGQueryError(graph_not_ready_message(status))
        if not status.vector_store_exists or not status.vector_store_readable:
            raise GraphRAGQueryError(graph_not_ready_message(status))
        if not status.output_complete:
            raise GraphRAGQueryError(graph_not_ready_message(status))
        if status.last_index_success is False:
            raise GraphRAGQueryError(graph_not_ready_message(status))

    @staticmethod
    def _input_manifest_hash(input_path: Path) -> str:
        return file_digest(input_path)


def _answer_from_result(result: GraphRAGCommandResult) -> str:
    cleaned = _ANSI_PATTERN.sub("", result.stdout).strip()
    if not cleaned:
        return ""
    lines = [line.rstrip() for line in cleaned.splitlines()]
    answer_start = _answer_start_index(lines)
    if answer_start is not None:
        answer_lines = lines[answer_start:]
        if answer_lines:
            answer_lines[0] = _strip_answer_marker(answer_lines[0])
        return "\n".join(line for line in answer_lines).strip()

    answer_lines = [
        line
        for line in lines
        if line.strip() and not _looks_like_stdout_diagnostic(line)
    ]
    return "\n".join(answer_lines).strip()


def _answer_start_index(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        lowered = line.strip().lower()
        for marker in _ANSWER_MARKERS:
            if lowered == marker[:-1]:
                return index + 1 if index + 1 < len(lines) else index
            if lowered.startswith(marker):
                return index
    return None


def _strip_answer_marker(line: str) -> str:
    stripped = line.strip()
    lowered = stripped.lower()
    for marker in _ANSWER_MARKERS:
        if lowered.startswith(marker):
            return stripped[len(marker) :].strip()
    return line


def _looks_like_stdout_diagnostic(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if lowered.startswith(_DIAGNOSTIC_PREFIXES):
        return True
    if re.match(r"^\d{1,3}%(\s|$)", stripped):
        return True
    if re.match(r"^\[[#=\-.\s]+\]\s*\d{1,3}%", stripped):
        return True
    return False


def _graph_data_references(text: str) -> list[str]:
    return list(
        dict.fromkeys(
            match.group(0) for match in GRAPH_DATA_REFERENCE_PATTERN.finditer(text)
        )
    )


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
