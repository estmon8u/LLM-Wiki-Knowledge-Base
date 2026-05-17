"""GraphRAG query execution and saved-answer artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from graphwiki_kb.services.file_lock import file_lock
from graphwiki_kb.services.graphrag_command_service import (
    GraphRAGCommandError,
    GraphRAGCommandResult,
    GraphRAGCommandService,
)
from graphwiki_kb.services.graphrag_freshness_service import (
    file_digest,
    graph_input_manifest_hash,
)
from graphwiki_kb.services.graphrag_status_service import (
    GraphRAGStatus,
    GraphRAGStatusService,
    graph_not_ready_message,
    graph_ready_for_query,
)
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
GRAPH_DATA_REFERENCE_DETAIL_PATTERN = re.compile(r"\[Data:\s*(?P<body>[^\]]+)\]")
GRAPH_DATA_REFERENCE_PART_PATTERN = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z_\s-]*?)\s*\((?P<ids>[^)]*)\)"
)
GRAPH_DATA_REFERENCE_KINDS = {
    "source": "source",
    "sources": "source",
    "document": "document",
    "documents": "document",
    "text_unit": "text_unit",
    "text_units": "text_unit",
    "entity": "entity",
    "entities": "entity",
    "relationship": "relationship",
    "relationships": "relationship",
    "community": "community",
    "communities": "community",
    "community_report": "community_report",
    "community_reports": "community_report",
}
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


@dataclass
class GraphRAGQueryAnswer:
    """Answer text, diagnostics, and provenance from a GraphRAG query."""

    question: str
    answer: str
    raw_output: str
    method: str
    created_at: str
    index_run_id: str | None
    command: tuple[str, ...]
    stdout: str
    stderr: str
    graph_input_hash: str
    input_manifest_hash: str | None = None
    community_level: int | None = None
    dynamic_community_selection: bool | None = None
    response_type: str | None = None
    saved_path: str | None = None
    retriever: str = "graphrag"
    planner: str | None = None
    route_reason: str | None = None
    route_confidence: str | None = None
    route_matched_terms: list[str] = field(default_factory=list)
    claim_support: str | None = None
    source_trace: dict[str, str | None] = field(default_factory=dict)
    graph_data_references: list[dict[str, object]] = field(default_factory=list)
    staleness_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation of the answer."""
        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload


class GraphRAGQueryService:
    """Executes GraphRAG queries and persists optional analysis pages."""

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
        """Run a GraphRAG query after checking method-specific readiness."""
        status = self.status_service.status()
        self._require_query_ready(status, method=method)
        graph_input_hash = file_digest(status.input_path)
        input_manifest_hash = graph_input_manifest_hash(status.input_path)
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
            raw_output=result.stdout,
            method=method,
            created_at=utc_now_iso(),
            index_run_id=status.last_index_run_id,
            command=result.command,
            stdout=result.stdout,
            stderr=result.stderr,
            graph_input_hash=graph_input_hash,
            input_manifest_hash=input_manifest_hash,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            graph_data_references=_graph_data_reference_details(answer),
            source_trace={
                "input_path": _relative_path(status.input_path, self.paths.root),
                "output_dir": _relative_path(
                    status.active_output_dir or status.output_dir,
                    self.paths.root,
                ),
                "index_run_id": status.last_index_run_id,
                "graph_input_hash": graph_input_hash,
                "input_manifest_hash": input_manifest_hash,
                "graph_run_id": status.last_index_run_id,
            },
        )

    def save_answer(
        self,
        answer: GraphRAGQueryAnswer,
        *,
        slug: str | None = None,
    ) -> str:
        """Persist a non-empty GraphRAG answer as a unique analysis page."""
        if not answer.answer.strip():
            raise GraphRAGQueryError(
                "Refusing to save an empty GraphRAG answer. Re-run `kb ask` after "
                "refreshing the graph index or inspect the terminal diagnostics."
            )
        safe_slug = slugify(slug) if slug else slugify(answer.question)
        if not safe_slug or safe_slug == "untitled":
            safe_slug = "analysis"
        safe_slug = f"graphrag-{safe_slug}"
        dest = _unique_analysis_path(self.paths.wiki_analysis_dir / f"{safe_slug}.md")
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
            "graph_run_id": answer.index_run_id,
            "graph_input_hash": answer.graph_input_hash,
            "input_manifest_hash": answer.input_manifest_hash,
            "graph_data_references": answer.graph_data_references,
        }
        if answer.planner:
            frontmatter["planner"] = answer.planner
        if answer.route_reason:
            frontmatter["route_reason"] = answer.route_reason
        if answer.route_confidence:
            frontmatter["route_confidence"] = answer.route_confidence
        if answer.route_matched_terms:
            frontmatter["route_matched_terms"] = answer.route_matched_terms
        if answer.claim_support:
            frontmatter["claim_support"] = answer.claim_support
        yaml_block = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        retrieval_lines = [
            f"- Retriever: {answer.retriever}",
            f"- GraphRAG method: {answer.method}",
            f"- Index run: {answer.index_run_id or 'unknown'}",
            f"- Graph input hash: {answer.graph_input_hash}",
            f"- Input manifest hash: {answer.input_manifest_hash or 'unknown'}",
        ]
        if answer.planner:
            retrieval_lines.append(f"- Planner: {answer.planner}")
        if answer.route_reason:
            retrieval_lines.append(f"- Route reason: {answer.route_reason}")
        if answer.route_confidence:
            retrieval_lines.append(f"- Route confidence: {answer.route_confidence}")
        if answer.route_matched_terms:
            retrieval_lines.append(
                "- Route matched terms: " + ", ".join(answer.route_matched_terms)
            )
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
            f"- Graph input hash: {answer.source_trace.get('graph_input_hash')}",
            f"- Input manifest hash: {answer.source_trace.get('input_manifest_hash')}",
            "- Parsed GraphRAG data references: "
            f"{len(answer.graph_data_references)}",
        ]
        if answer.graph_data_references:
            for item in answer.graph_data_references:
                ids = item.get("ids", ())
                id_list = [str(value) for value in ids] if isinstance(ids, list) else []
                trace_lines.append(
                    f"- `{item.get('kind', 'unknown')}` ids: {', '.join(id_list)}"
                )
        raw_stdout = answer.raw_output or "No raw GraphRAG stdout captured."
        raw_stderr = answer.stderr or "No raw GraphRAG stderr captured."
        return (
            f"---\n{yaml_block}\n---\n\n"
            f"# {answer.question}\n\n"
            "## Answer\n\n"
            f"{answer.answer or 'No answer text returned.'}\n\n"
            "## Retrieval Mode\n\n"
            f"{chr(10).join(retrieval_lines)}\n\n"
            "## Source Trace\n\n"
            f"{chr(10).join(trace_lines)}\n\n"
            "## Raw GraphRAG Stdout\n\n"
            "```text\n"
            f"{raw_stdout}"
            f"{'' if raw_stdout.endswith(chr(10)) else chr(10)}"
            "```\n\n"
            "## Raw GraphRAG Stderr\n\n"
            "```text\n"
            f"{raw_stderr}"
            f"{'' if raw_stderr.endswith(chr(10)) else chr(10)}"
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

    def _require_query_ready(self, status: GraphRAGStatus, *, method: str) -> None:
        if not graph_ready_for_query(status, method=method):
            raise GraphRAGQueryError(graph_not_ready_message(status, method=method))


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


def _graph_data_reference_details(text: str) -> list[dict[str, object]]:
    references: list[dict[str, object]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for match in GRAPH_DATA_REFERENCE_DETAIL_PATTERN.finditer(text):
        body = match.group("body")
        for part in GRAPH_DATA_REFERENCE_PART_PATTERN.finditer(body):
            kind = _normalize_reference_kind(part.group("label"))
            ids = tuple(_split_reference_ids(part.group("ids")))
            if not kind or not ids:
                continue
            key = (kind, ids)
            if key in seen:
                continue
            seen.add(key)
            references.append(
                {
                    "kind": kind,
                    "ids": list(ids),
                    "raw": part.group(0),
                }
            )
    return references


def _normalize_reference_kind(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return GRAPH_DATA_REFERENCE_KINDS.get(normalized, "")


def _split_reference_ids(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,;]", value) if item.strip()]


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _unique_analysis_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}-{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
