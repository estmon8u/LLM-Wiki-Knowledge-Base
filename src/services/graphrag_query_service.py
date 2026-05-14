"""Graphrag query service service behavior for the knowledge-base workflow.

This module belongs to `src.services.graphrag_query_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""


from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Callable

import yaml

from src.services.graphrag_command_service import (
    GraphRAGCommandError,
    GraphRAGCommandResult,
    GraphRAGCommandService,
)
from src.services.graphrag_status_service import GraphRAGStatus, GraphRAGStatusService
from src.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
    unique_markdown_heading,
    utc_now_iso,
)
from src.services.search_service import SearchService


GRAPH_QUERY_METHODS = ("local", "global", "drift", "basic")
GRAPH_DATA_REFERENCE_PATTERN = re.compile(r"\[Data:\s*[^\]]+\]")


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
        verbose: bool = False,
    ) -> GraphRAGQueryAnswer:
        """Ask.

        Args:
            question: User question to answer from available evidence.
            method: Method value used by the operation.
            community_level: Community level value used by the operation.
            dynamic_community_selection: Dynamic community selection value used by the operation.
            response_type: Response type value used by the operation.
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
                community_level=community_level,
                dynamic_community_selection=dynamic_community_selection,
                response_type=response_type,
                verbose=verbose,
            )
        except GraphRAGCommandError as exc:
            raise GraphRAGQueryError(str(exc)) from exc

        answer = _answer_from_result(result)
        return GraphRAGQueryAnswer(
            question=question,
            answer=answer,
            raw_output=_raw_output(result),
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
                "output_dir": _relative_path(status.output_dir, self.paths.root),
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
        if not citations and not insufficient_evidence:
            citations = _source_trace_citations(answer)
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
            retrieval_lines.append(f"- Claim support: {answer.claim_support}")
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
            raise GraphRAGQueryError(
                "GraphRAG workspace is not initialized. Run `kb init` first."
            )
        if not status.input_exists:
            raise GraphRAGQueryError("GraphRAG input not found. Run `kb update` first.")
        if status.input_document_count == 0:
            raise GraphRAGQueryError(
                "GraphRAG input has no documents. Add and compile sources, then run "
                "`kb update`."
            )
        if not status.output_present:
            raise GraphRAGQueryError(
                "GraphRAG index output not found. Run `kb update`."
            )
        if status.last_index_success is False:
            raise GraphRAGQueryError(
                "The last GraphRAG index run failed. Re-run `kb update` before asking."
            )

    @staticmethod
    def _input_manifest_hash(input_path: Path) -> str:
        digest = hashlib.sha256()
        with input_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def _answer_from_result(result: GraphRAGCommandResult) -> str:
    return result.stdout.strip() or result.stderr.strip()


def _raw_output(result: GraphRAGCommandResult) -> str:
    return result.stdout.strip() or result.stderr.strip()


def _graph_data_references(text: str) -> list[str]:
    return list(
        dict.fromkeys(
            match.group(0) for match in GRAPH_DATA_REFERENCE_PATTERN.finditer(text)
        )
    )


def _source_trace_citations(answer: GraphRAGQueryAnswer) -> list[str]:
    if answer.index_run_id:
        return [f"GraphRAG index {answer.index_run_id}"]
    input_path = answer.source_trace.get("input_path")
    if input_path:
        return [f"GraphRAG input {input_path}"]
    return []


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
