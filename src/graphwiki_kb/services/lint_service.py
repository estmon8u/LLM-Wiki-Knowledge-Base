"""Lint service service behavior for the knowledge-base workflow.

This module belongs to `graphwiki_kb.services.lint_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.models.wiki_models import LintIssue, LintReport
from graphwiki_kb.services.compile_service import (
    SOURCE_PAGE_CONTRACT_VERSION,
    SOURCE_PAGE_CONTRACT_VERSION_KEY,
)
from graphwiki_kb.services.graphrag_freshness_service import graph_input_manifest_hash
from graphwiki_kb.services.graphrag_status_service import GraphRAGStatusService
from graphwiki_kb.services.manifest_service import ManifestService
from graphwiki_kb.services.markdown_document import (
    headings as markdown_headings,
)
from graphwiki_kb.services.markdown_document import (
    markdown_links,
    without_fenced_code_blocks,
)
from graphwiki_kb.services.project_service import ProjectPaths, slugify

WIKI_LINK_PATTERN = re.compile(
    r"\[\[(?P<target>[^\]\|#\n]+)(?:#(?P<fragment>[^\]\|\n]+))?(?:\|[^\]\n]+)?\]\]"
)
MARKDOWN_LINK_PATTERN = re.compile(
    r"(?<!!)(?<!\[)\[(?P<text>[^\]]*)\]\((?P<target>[^)]*)\)"
)
ATX_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
EXTERNAL_LINK_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?")

_FIELD_TYPE_SPEC: Dict[str, str] = {
    "title": "string",
    "summary": "string",
    "source_id": "string",
    "raw_path": "string",
    "source_hash": "string",
    "origin": "string",
    "normalized_path": "string",
    "compiled_at": "date",
    "ingested_at": "date",
    "tags": "list",
    "type": "string",
    "question": "string",
    "saved_at": "date",
    "citations": "list",
    "insufficient_evidence": "bool",
    "claim_count": "int",
    "citation_count": "int",
    "claims": "list",
    "provider_citations": "list",
    "generated_at": "date",
    "generator": "string",
    "source_pages": "list",
    "topic_terms": "list",
    "key_points": "list",
    "open_questions": "list",
    "title_suggestion": "string",
}


@dataclass
class _PageState:
    """Represents page state behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    file_path: Path
    relative_path: str
    frontmatter: Optional[Dict[str, Any]]
    frontmatter_error: str | None
    content: str
    analysis_text: str
    headings: list[tuple[int, str]]
    anchors: set[str]
    page_title: str


class LintService:
    """Coordinates lint operations.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        config: dict[str, Any],
        manifest_service: ManifestService,
        graphrag_status_service: GraphRAGStatusService | None = None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.manifest_service = manifest_service
        self.graphrag_status_service = graphrag_status_service

    def lint(self) -> LintReport:
        """Lint.

        Returns:
            LintReport produced by the operation.
        """
        issues: list[LintIssue] = []
        source_records = self.manifest_service.list_sources()
        source_records_by_slug = {source.slug: source for source in source_records}
        markdown_files = sorted(self.paths.wiki_dir.rglob("*.md"))
        page_states = [
            _build_page_state(file_path, self.paths) for file_path in markdown_files
        ]
        page_states_by_slug = {state.file_path.stem: state for state in page_states}
        page_states_by_path = {
            state.file_path.resolve(): state for state in page_states
        }
        incoming_links: dict[str, int] = {
            state.file_path.stem: 0 for state in page_states
        }
        page_titles: dict[str, list[_PageState]] = {}

        for state in page_states:
            if state.file_path.parent in {
                self.paths.wiki_sources_dir,
                self.paths.wiki_concepts_dir,
                self.paths.wiki_analysis_dir,
            }:
                is_analysis_page = (
                    state.file_path.parent == self.paths.wiki_analysis_dir
                )
                if state.frontmatter_error is not None:
                    issues.append(
                        LintIssue(
                            severity="warning",
                            code="invalid-frontmatter-yaml",
                            path=state.relative_path,
                            message=(
                                "YAML frontmatter could not be parsed; "
                                f"{state.frontmatter_error}"
                            ),
                        )
                    )
                elif state.frontmatter is None:
                    issues.append(
                        LintIssue(
                            severity="error",
                            code="missing-frontmatter",
                            path=state.relative_path,
                            message="Compiled wiki pages must include YAML frontmatter.",
                        )
                    )
                else:
                    page_type = state.frontmatter.get("type")
                    is_analysis_page = is_analysis_page or (
                        state.file_path.parent == self.paths.wiki_concepts_dir
                        and page_type == "analysis"
                    )
                    is_concept_page = (
                        state.file_path.parent == self.paths.wiki_concepts_dir
                        and page_type == "concept"
                    )
                    is_source_page = (
                        state.file_path.parent == self.paths.wiki_sources_dir
                    )
                    if is_analysis_page:
                        required_fields = [
                            "title",
                            "summary",
                            "type",
                            "question",
                            "saved_at",
                            "citations",
                            "insufficient_evidence",
                            "claim_count",
                            "citation_count",
                        ]
                    elif is_concept_page:
                        required_fields = [
                            "title",
                            "summary",
                            "type",
                            "generated_at",
                            "source_pages",
                        ]
                    else:
                        required_fields = self.config["lint"][
                            "required_frontmatter_fields"
                        ]
                    for field_name in required_fields:
                        if field_name not in state.frontmatter:
                            issues.append(
                                LintIssue(
                                    severity="error",
                                    code="missing-field",
                                    path=state.relative_path,
                                    message=f"Missing required frontmatter field: {field_name}",
                                )
                            )
                    if is_source_page and "type" not in state.frontmatter:
                        issues.append(
                            self._source_type_issue(
                                state,
                                source_records_by_slug.get(state.file_path.stem),
                            )
                        )
                    if not str(state.frontmatter.get("summary", "")).strip():
                        issues.append(
                            LintIssue(
                                severity="warning",
                                code="empty-summary",
                                path=state.relative_path,
                                message="Summary field is empty.",
                            )
                        )
                    issues.extend(self._lint_frontmatter_types(state))
                    issues.extend(self._lint_analysis_page(state))

                if not is_analysis_page:
                    normalized_title = state.page_title.casefold().strip()
                    page_titles.setdefault(normalized_title, []).append(state)

            issues.extend(self._lint_heading_structure(state))
            issues.extend(self._lint_empty_page(state))
            issues.extend(
                self._lint_links(
                    state,
                    page_states_by_slug,
                    page_states_by_path,
                    incoming_links,
                )
            )

        for duplicates in page_titles.values():
            if len(duplicates) <= 1:
                continue
            duplicate_title = duplicates[0].page_title
            for state in duplicates:
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="duplicate-title",
                        path=state.relative_path,
                        message=(
                            "Page title duplicates another compiled page title: "
                            f'"{duplicate_title}".'
                        ),
                    )
                )

        for state in page_states:
            if state.file_path.parent not in {
                self.paths.wiki_sources_dir,
                self.paths.wiki_concepts_dir,
            }:
                continue
            slug = state.file_path.stem
            if incoming_links.get(slug, 0) == 0:
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="orphan-page",
                        path=state.relative_path,
                        message="Page has no inbound wiki links.",
                    )
                )

        for source in source_records:
            issues.extend(self._lint_manifest_state(source))

        issues.extend(self._lint_graph_staleness())

        return LintReport(issues=issues)

    def _source_type_issue(
        self,
        page_state: _PageState,
        source_record: Optional[RawSourceRecord],
    ) -> LintIssue:
        contract_version = _source_page_contract_version(source_record)
        if source_record is None or contract_version >= SOURCE_PAGE_CONTRACT_VERSION:
            return LintIssue(
                severity="error",
                code="missing-type",
                path=page_state.relative_path,
                message="Source page missing required frontmatter field: type.",
            )
        return LintIssue(
            severity="warning",
            code="missing-type",
            path=page_state.relative_path,
            message=(
                "Source page missing 'type' field; " "run kb update --force to refresh."
            ),
        )

    def _lint_frontmatter_types(self, page_state: _PageState) -> list[LintIssue]:
        issues: list[LintIssue] = []
        if page_state.frontmatter is None:
            return issues
        for field_name, expected in _FIELD_TYPE_SPEC.items():
            if field_name not in page_state.frontmatter:
                continue
            value = page_state.frontmatter[field_name]
            if expected == "string":
                if not isinstance(value, str):
                    issues.append(
                        LintIssue(
                            severity="warning",
                            code="invalid-field-type",
                            path=page_state.relative_path,
                            message=(
                                f"Frontmatter field '{field_name}' should be a "
                                f"string but got {type(value).__name__}."
                            ),
                        )
                    )
            elif expected == "date":
                raw = str(value).strip()
                if not ISO_DATE_PATTERN.match(raw):
                    issues.append(
                        LintIssue(
                            severity="warning",
                            code="invalid-date-format",
                            path=page_state.relative_path,
                            message=(
                                f"Frontmatter field '{field_name}' does not look "
                                f"like an ISO date: '{raw}'."
                            ),
                        )
                    )
            elif expected == "list":
                if not isinstance(value, list):
                    issues.append(
                        LintIssue(
                            severity="warning",
                            code="invalid-field-type",
                            path=page_state.relative_path,
                            message=(
                                f"Frontmatter field '{field_name}' should be a "
                                f"list but got {type(value).__name__}."
                            ),
                        )
                    )
            elif expected == "bool":
                if not isinstance(value, bool):
                    issues.append(
                        LintIssue(
                            severity="warning",
                            code="invalid-field-type",
                            path=page_state.relative_path,
                            message=(
                                f"Frontmatter field '{field_name}' should be a "
                                f"bool but got {type(value).__name__}."
                            ),
                        )
                    )
            elif expected == "int":
                if not isinstance(value, int) or isinstance(value, bool):
                    issues.append(
                        LintIssue(
                            severity="warning",
                            code="invalid-field-type",
                            path=page_state.relative_path,
                            message=(
                                f"Frontmatter field '{field_name}' should be an "
                                f"int but got {type(value).__name__}."
                            ),
                        )
                    )
        return issues

    def _lint_analysis_page(self, page_state: _PageState) -> list[LintIssue]:
        if page_state.frontmatter is None:
            return []
        if page_state.frontmatter.get("type") != "analysis":
            return []

        citations = page_state.frontmatter.get("citations")
        citation_count = page_state.frontmatter.get("citation_count")
        insufficient = page_state.frontmatter.get("insufficient_evidence") is True
        if not isinstance(citations, list):
            citations = []
        if not isinstance(citation_count, int) or isinstance(citation_count, bool):
            citation_count = len(citations)

        if not insufficient and citation_count == 0:
            return [
                LintIssue(
                    severity="warning",
                    code="analysis-without-citations",
                    path=page_state.relative_path,
                    message=(
                        "Saved analysis page has no citations and is not marked "
                        "as insufficient evidence."
                    ),
                )
            ]
        return []

    def _lint_empty_page(self, page_state: _PageState) -> list[LintIssue]:
        if page_state.file_path.parent not in {
            self.paths.wiki_sources_dir,
            self.paths.wiki_concepts_dir,
            self.paths.wiki_analysis_dir,
        }:
            return []
        for line in page_state.content.splitlines():
            stripped = line.strip()
            if not stripped or ATX_HEADING_PATTERN.match(stripped):
                continue
            return []
        return [
            LintIssue(
                severity="warning",
                code="empty-page",
                path=page_state.relative_path,
                message="Page has no body content beyond headings.",
            )
        ]

    def _lint_heading_structure(self, page_state: _PageState) -> list[LintIssue]:
        issues: list[LintIssue] = []
        if not page_state.headings:
            return issues

        previous_level: int | None = None
        h1_count = 0
        heading_counts: Counter[str] = Counter()
        heading_examples: dict[str, str] = {}

        for level, title in page_state.headings:
            if previous_level is not None and level > previous_level + 1:
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="heading-level-skip",
                        path=page_state.relative_path,
                        message=(
                            f"Heading level jumps from H{previous_level} to H{level} "
                            f'at "{title}".'
                        ),
                    )
                )
            previous_level = level

            if level == 1:
                h1_count += 1

            normalized = title.casefold().strip()
            if normalized:
                heading_counts[normalized] += 1
                heading_examples.setdefault(normalized, title)

        if h1_count > 1:
            issues.append(
                LintIssue(
                    severity="warning",
                    code="multiple-h1",
                    path=page_state.relative_path,
                    message="Document contains multiple H1 headings.",
                )
            )

        for normalized, count in heading_counts.items():
            if count <= 1:
                continue
            issues.append(
                LintIssue(
                    severity="warning",
                    code="duplicate-heading",
                    path=page_state.relative_path,
                    message=f'Document repeats heading "{heading_examples[normalized]}".',
                )
            )

        return issues

    def _lint_links(
        self,
        page_state: _PageState,
        page_states_by_slug: dict[str, _PageState],
        page_states_by_path: dict[Path, _PageState],
        incoming_links: dict[str, int],
    ) -> list[LintIssue]:
        issues: list[LintIssue] = []

        for match in WIKI_LINK_PATTERN.finditer(page_state.analysis_text):
            raw_target = match.group("target").strip()
            raw_fragment = (match.group("fragment") or "").strip()
            target_slug = slugify(raw_target)
            target_state = page_states_by_slug.get(target_slug)
            if target_state is None:
                issues.append(
                    LintIssue(
                        severity="error",
                        code="broken-link",
                        path=page_state.relative_path,
                        message=f"Wiki link target not found: [[{raw_target}]]",
                    )
                )
                continue

            incoming_links[target_state.file_path.stem] += 1

            if raw_fragment and not _fragment_exists(target_state, raw_fragment):
                issues.append(
                    LintIssue(
                        severity="error",
                        code="broken-fragment",
                        path=page_state.relative_path,
                        message=(
                            "Wiki link fragment not found: "
                            f"[[{raw_target}#{raw_fragment}]]"
                        ),
                    )
                )

        markdown_link_pairs: list[tuple[str, str]] = [
            (link.target.strip(), link.text)
            for link in markdown_links(page_state.analysis_text)
        ]
        seen_markdown_links = set(markdown_link_pairs)
        for match in MARKDOWN_LINK_PATTERN.finditer(page_state.analysis_text):
            pair = (match.group("target").strip(), match.group("text"))
            if pair not in seen_markdown_links:
                markdown_link_pairs.append(pair)
                seen_markdown_links.add(pair)

        for raw_target, link_text in markdown_link_pairs:
            if not raw_target:
                issues.append(
                    LintIssue(
                        severity="error",
                        code="empty-markdown-link",
                        path=page_state.relative_path,
                        message=f"Markdown link target is empty: [{link_text}]().",
                    )
                )
                continue

            destination, fragment = _split_markdown_target(raw_target)
            if destination and EXTERNAL_LINK_PATTERN.match(destination):
                continue

            if not destination:
                if fragment and not _fragment_exists(page_state, fragment):
                    issues.append(
                        LintIssue(
                            severity="error",
                            code="broken-fragment",
                            path=page_state.relative_path,
                            message=(
                                "Markdown link fragment not found: "
                                f"[{link_text}]({raw_target})"
                            ),
                        )
                    )
                continue

            target_path = _resolve_markdown_target(page_state.file_path, destination)
            if not target_path.exists() or not target_path.is_file():
                issues.append(
                    LintIssue(
                        severity="error",
                        code="broken-markdown-link",
                        path=page_state.relative_path,
                        message=(
                            "Markdown link target not found: "
                            f"[{link_text}]({raw_target})"
                        ),
                    )
                )
                continue

            target_state = page_states_by_path.get(target_path.resolve())
            if target_state is not None:
                incoming_links[target_state.file_path.stem] += 1
                if fragment and not _fragment_exists(target_state, fragment):
                    issues.append(
                        LintIssue(
                            severity="error",
                            code="broken-fragment",
                            path=page_state.relative_path,
                            message=(
                                "Markdown link fragment not found: "
                                f"[{link_text}]({raw_target})"
                            ),
                        )
                    )

        return issues

    def _lint_manifest_state(self, source: RawSourceRecord) -> list[LintIssue]:
        issues: list[LintIssue] = []
        article_path = self.paths.wiki_sources_dir / f"{source.slug}.md"
        raw_path = _resolve_project_path(self.paths, source.raw_path)
        if not raw_path.exists():
            issues.append(
                LintIssue(
                    severity="error",
                    code="missing-raw-source",
                    path=source.raw_path,
                    message=(
                        "Manifest raw source file is missing. Re-ingest the source "
                        "or remove the stale manifest entry."
                    ),
                )
            )
        if source.normalized_path:
            normalized_path = _resolve_project_path(self.paths, source.normalized_path)
            if not normalized_path.exists():
                issues.append(
                    LintIssue(
                        severity="error",
                        code="missing-normalized-source",
                        path=source.normalized_path,
                        message=(
                            "Manifest normalized artifact is missing. Run "
                            "`kb update --force` or re-ingest the source."
                        ),
                    )
                )
        if source.compiled_from_hash != source.content_hash:
            issues.append(
                LintIssue(
                    severity="warning",
                    code="stale-source-page",
                    path=f"wiki/sources/{source.slug}.md",
                    message="Source page is stale and should be recompiled.",
                )
            )
        if (
            source.compiled_from_hash == source.content_hash
            and not article_path.exists()
        ):
            issues.append(
                LintIssue(
                    severity="error",
                    code="missing-compiled-page",
                    path=f"wiki/sources/{source.slug}.md",
                    message="Manifest says the source was compiled, but the source page is missing.",
                )
            )
        return issues

    def _lint_graph_staleness(self) -> list[LintIssue]:
        if self.graphrag_status_service is None:
            return []
        status = self.graphrag_status_service.status()
        issues: list[LintIssue] = []
        if status.output_present and not status.output_complete:
            issues.append(
                LintIssue(
                    severity="error",
                    code="graph-output-incomplete",
                    path=status.output_dir.relative_to(self.paths.root).as_posix(),
                    message=(
                        "GraphRAG output is missing required table(s): "
                        f"{', '.join(status.missing_tables)}. Run `kb update`."
                    ),
                )
            )
        if status.input_exists and self.paths.raw_manifest_file.exists():
            current_manifest_hash = _file_sha256(self.paths.raw_manifest_file)
            synced_manifest_hash = _input_manifest_hash(status.input_path)
            if (
                current_manifest_hash
                and synced_manifest_hash
                and current_manifest_hash != synced_manifest_hash
            ):
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="graph-input-stale",
                        path=status.input_path.relative_to(self.paths.root).as_posix(),
                        message="Manifest changed since last sync. Run `kb update`.",
                    )
                )
        last_run = self.graphrag_status_service.last_successful_index_run()
        if status.input_exists and last_run:
            current_input_hash = _file_sha256(status.input_path)
            indexed_input_hash = last_run.get("input_hash") or last_run.get(
                "input_digest"
            )
            if (
                current_input_hash
                and indexed_input_hash
                and current_input_hash != indexed_input_hash
            ):
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="graph-index-stale",
                        path=status.input_path.relative_to(self.paths.root).as_posix(),
                        message="Graph input changed since last index. Run `kb update`.",
                    )
                )
        if (
            status.output_present
            and status.output_complete
            and status.graph_freshness_state in {"stale", "missing-metadata"}
        ):
            reason = (
                status.graph_stale_reasons[0]
                if status.graph_stale_reasons
                else "Graph index metadata no longer matches current inputs."
            )
            issues.append(
                LintIssue(
                    severity="warning",
                    code="graph-index-stale",
                    path=(status.active_output_dir or status.output_dir)
                    .relative_to(self.paths.root)
                    .as_posix(),
                    message=f"{reason} Run `kb update --graph-only`.",
                )
            )
        graph_index_path = self.paths.wiki_dir / "graph" / "index.md"
        if status.last_index_run_id and graph_index_path.exists():
            try:
                text = graph_index_path.read_text(encoding="utf-8")
            except OSError:
                text = ""
            frontmatter, _ = _split_frontmatter(text)
            exported_run_id = (
                str(frontmatter.get("index_run_id", "")).strip() if frontmatter else ""
            )
            if exported_run_id and exported_run_id != status.last_index_run_id:
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="graph-export-stale",
                        path=graph_index_path.relative_to(self.paths.root).as_posix(),
                        message="Index newer than wiki export. Run `kb update`.",
                    )
                )
        return issues


def _split_frontmatter(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    frontmatter, content, _ = _split_frontmatter_with_error(text)
    return frontmatter, content


def _split_frontmatter_with_error(
    text: str,
) -> Tuple[Optional[Dict[str, Any]], str, str | None]:
    if not text.startswith("---\n"):
        return None, text, None
    marker = text.find("\n---\n", 4)
    if marker == -1:
        return None, text, None
    payload = text[4:marker]
    content = text[marker + 5 :]
    try:
        parsed = yaml.safe_load(payload) or {}
    except yaml.YAMLError as exc:
        return None, content, str(exc).splitlines()[0]
    return parsed if isinstance(parsed, dict) else {}, content, None


def _build_page_state(file_path: Path, paths: ProjectPaths) -> _PageState:
    text = file_path.read_text(encoding="utf-8")
    frontmatter, content, frontmatter_error = _split_frontmatter_with_error(text)
    analysis_text = _strip_fenced_code_blocks(_strip_excerpt_section(content))
    headings = _extract_headings(content)
    anchors = {slugify(title) for _, title in headings if slugify(title)}
    page_title = _page_title(file_path, frontmatter, headings)
    return _PageState(
        file_path=file_path,
        relative_path=file_path.relative_to(paths.root).as_posix(),
        frontmatter=frontmatter,
        frontmatter_error=frontmatter_error,
        content=content,
        analysis_text=analysis_text,
        headings=headings,
        anchors=anchors,
        page_title=page_title,
    )


def _extract_headings(content: str) -> list[tuple[int, str]]:
    return [
        (heading.level, heading.title)
        for heading in markdown_headings(_strip_fenced_code_blocks(content))
    ]


def _normalize_heading_title(title: str) -> str:
    normalized = re.sub(r"\s+#+\s*$", "", title.strip())
    return re.sub(r"\s+", " ", normalized).strip()


def _page_title(
    file_path: Path,
    frontmatter: Optional[Dict[str, Any]],
    headings: list[tuple[int, str]],
) -> str:
    if frontmatter is not None:
        title = str(frontmatter.get("title", "")).strip()
        if title:
            return title
    if headings:
        return headings[0][1]
    return file_path.stem


def _fragment_exists(page_state: _PageState, fragment: str) -> bool:
    normalized = slugify(fragment.strip())
    if not normalized:
        return False
    return normalized in page_state.anchors


def _split_markdown_target(raw_target: str) -> tuple[str, str]:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if " " in target and not target.startswith("#"):
        target = target.split(" ", 1)[0]
    destination, _, fragment = target.partition("#")
    return destination, fragment


def _resolve_markdown_target(current_file: Path, destination: str) -> Path:
    path = Path(destination)
    if path.is_absolute():
        return path
    return (current_file.parent / path).resolve()


def _resolve_project_path(paths: ProjectPaths, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return paths.root / path


def _strip_excerpt_section(text: str) -> str:
    """Remove the Key Excerpt section to avoid linting links from source material."""
    match = re.search(r"^## Key Excerpt\s*$", text, re.MULTILINE)
    if match is None:
        return text
    return text[: match.start()]


def _strip_fenced_code_blocks(text: str) -> str:
    return without_fenced_code_blocks(text)


def _source_page_contract_version(source: Optional[RawSourceRecord]) -> int:
    if source is None or not isinstance(source.metadata, dict):
        return 0
    version = source.metadata.get(SOURCE_PAGE_CONTRACT_VERSION_KEY, 0)
    try:
        parsed = int(version)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _input_manifest_hash(path: Path) -> str | None:
    return graph_input_manifest_hash(path)
