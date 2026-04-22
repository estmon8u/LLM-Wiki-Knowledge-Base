from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

import yaml

from src.models.source_models import RawSourceRecord
from src.models.wiki_models import LintIssue, LintReport
from src.services.compile_service import (
    SOURCE_PAGE_CONTRACT_VERSION,
    SOURCE_PAGE_CONTRACT_VERSION_KEY,
)
from src.services.manifest_service import ManifestService
from src.services.project_service import ProjectPaths, slugify


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
    "generated_at": "date",
    "generator": "string",
    "source_pages": "list",
    "topic_terms": "list",
}


@dataclass
class _PageState:
    file_path: Path
    relative_path: str
    frontmatter: Optional[Dict[str, Any]]
    content: str
    analysis_text: str
    headings: list[tuple[int, str]]
    anchors: set[str]
    page_title: str


class LintService:
    def __init__(
        self,
        paths: ProjectPaths,
        config: dict[str, Any],
        manifest_service: ManifestService,
    ) -> None:
        self.paths = paths
        self.config = config
        self.manifest_service = manifest_service

    def lint(self) -> LintReport:
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
            }:
                if state.frontmatter is None:
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
                    is_concept_page = (
                        state.file_path.parent == self.paths.wiki_concepts_dir
                        and page_type in {"analysis", "concept"}
                    )
                    is_source_page = (
                        state.file_path.parent == self.paths.wiki_sources_dir
                    )
                    if is_concept_page and page_type == "concept":
                        required_fields = [
                            "title",
                            "summary",
                            "type",
                            "generated_at",
                            "source_pages",
                        ]
                    elif is_concept_page:
                        required_fields = ["title"]
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
        return issues

    def _lint_empty_page(self, page_state: _PageState) -> list[LintIssue]:
        if page_state.file_path.parent not in {
            self.paths.wiki_sources_dir,
            self.paths.wiki_concepts_dir,
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

        for match in MARKDOWN_LINK_PATTERN.finditer(page_state.analysis_text):
            raw_target = match.group("target").strip()
            link_text = match.group("text")
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


def _split_frontmatter(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    if not text.startswith("---\n"):
        return None, text
    marker = text.find("\n---\n", 4)
    if marker == -1:
        return None, text
    payload = text[4:marker]
    content = text[marker + 5 :]
    return yaml.safe_load(payload) or {}, content


def _build_page_state(file_path: Path, paths: ProjectPaths) -> _PageState:
    text = file_path.read_text(encoding="utf-8")
    frontmatter, content = _split_frontmatter(text)
    analysis_text = _strip_fenced_code_blocks(_strip_excerpt_section(content))
    headings = _extract_headings(content)
    anchors = {slugify(title) for _, title in headings if slugify(title)}
    page_title = _page_title(file_path, frontmatter, headings)
    return _PageState(
        file_path=file_path,
        relative_path=file_path.relative_to(paths.root).as_posix(),
        frontmatter=frontmatter,
        content=content,
        analysis_text=analysis_text,
        headings=headings,
        anchors=anchors,
        page_title=page_title,
    )


def _extract_headings(content: str) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    for line in _strip_fenced_code_blocks(content).splitlines():
        match = ATX_HEADING_PATTERN.match(line)
        if not match:
            continue
        title = _normalize_heading_title(match.group(2))
        if not title:
            continue
        headings.append((len(match.group(1)), title))
    return headings


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


def _strip_excerpt_section(text: str) -> str:
    """Remove the Key Excerpt section to avoid linting links from source material."""
    match = re.search(r"^## Key Excerpt\s*$", text, re.MULTILINE)
    if match is None:
        return text
    return text[: match.start()]


def _strip_fenced_code_blocks(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            continue

        if not in_fence:
            lines.append(line)

    return "\n".join(lines)


def _source_page_contract_version(source: Optional[RawSourceRecord]) -> int:
    if source is None or not isinstance(source.metadata, dict):
        return 0
    version = source.metadata.get(SOURCE_PAGE_CONTRACT_VERSION_KEY, 0)
    try:
        parsed = int(version)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)
