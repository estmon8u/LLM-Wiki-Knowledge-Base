from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

import yaml

from src.models.source_models import RawSourceRecord
from src.models.wiki_models import LintIssue, LintReport
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
                    for field_name in self.config["lint"][
                        "required_frontmatter_fields"
                    ]:
                        if field_name not in state.frontmatter:
                            issues.append(
                                LintIssue(
                                    severity="error",
                                    code="missing-field",
                                    path=state.relative_path,
                                    message=f"Missing required frontmatter field: {field_name}",
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

                normalized_title = state.page_title.casefold().strip()
                page_titles.setdefault(normalized_title, []).append(state)

            issues.extend(self._lint_heading_structure(state))
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

        for source in self.manifest_service.list_sources():
            issues.extend(self._lint_manifest_state(source))

        return LintReport(issues=issues)

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
    analysis_text = _strip_fenced_code_blocks(content)
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
