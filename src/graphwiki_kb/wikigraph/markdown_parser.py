"""Parse wiki markdown pages into structured page records."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from graphwiki_kb.services.markdown_document import headings as markdown_headings
from graphwiki_kb.services.project_service import slugify

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]]+)?(?:\|[^\]]+)?\]\]")
_ALIAS_SPLIT_RE = re.compile(r"[,;]")


@dataclass(frozen=True)
class ParsedWikiPage:
    """One wiki page parsed for graph indexing."""

    path: str
    page_kind: str
    title: str
    source_id: str | None
    aliases: tuple[str, ...]
    tags: tuple[str, ...]
    summary: str
    body: str
    sections: tuple[tuple[str, str], ...]
    wikilinks: tuple[str, ...]
    frontmatter: dict[str, Any]


@dataclass
class ParsedChunk:
    """A retrievable section-level chunk."""

    chunk_id: str
    page_path: str
    page_kind: str
    title: str
    heading: str
    text: str
    source_id: str | None
    aliases: tuple[str, ...] = field(default_factory=tuple)


def parse_wiki_page(file_path: Path, project_root: Path) -> ParsedWikiPage | None:
    """Parse a wiki markdown file into a page record."""
    text = file_path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    relative = file_path.relative_to(project_root).as_posix()
    page_kind = _infer_page_kind(relative, frontmatter)
    page_headings = markdown_headings(body)
    title = _page_title(file_path, frontmatter, page_headings)
    source_id = _optional_str(frontmatter.get("source_id"))
    aliases = _collect_aliases(frontmatter, title)
    tags = tuple(
        str(item).strip()
        for item in frontmatter.get("tags", []) or []
        if str(item).strip()
    )
    summary = str(frontmatter.get("summary", "")).strip()
    sections = _split_sections(body)
    wikilinks = tuple(sorted(set(_WIKILINK_RE.findall(body))))
    return ParsedWikiPage(
        path=relative,
        page_kind=page_kind,
        title=title,
        source_id=source_id,
        aliases=aliases,
        tags=tags,
        summary=summary,
        body=body,
        sections=sections,
        wikilinks=wikilinks,
        frontmatter=frontmatter,
    )


def chunks_from_page(page: ParsedWikiPage) -> list[ParsedChunk]:
    """Split a parsed page into section-level chunks."""
    chunks: list[ParsedChunk] = []
    if page.summary:
        chunks.append(
            ParsedChunk(
                chunk_id=f"{page.path}#summary",
                page_path=page.path,
                page_kind=page.page_kind,
                title=page.title,
                heading="Summary",
                text=page.summary,
                source_id=page.source_id,
                aliases=page.aliases,
            )
        )
    for heading, section_text in page.sections:
        normalized = section_text.strip()
        if not normalized:
            continue
        chunk_id = f"{page.path}#{slugify(heading) or 'section'}"
        chunks.append(
            ParsedChunk(
                chunk_id=chunk_id,
                page_path=page.path,
                page_kind=page.page_kind,
                title=page.title,
                heading=heading,
                text=normalized,
                source_id=page.source_id,
                aliases=page.aliases,
            )
        )
    if not chunks and page.body.strip():
        chunks.append(
            ParsedChunk(
                chunk_id=f"{page.path}#body",
                page_path=page.path,
                page_kind=page.page_kind,
                title=page.title,
                heading=page.title,
                text=page.body.strip()[:4000],
                source_id=page.source_id,
                aliases=page.aliases,
            )
        )
    return chunks


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    marker = text.find("\n---\n", 4)
    if marker == -1:
        return {}, text
    payload = text[4:marker]
    content = text[marker + 5 :]
    try:
        parsed = yaml.safe_load(payload) or {}
    except yaml.YAMLError:
        return {}, content
    return parsed if isinstance(parsed, dict) else {}, content


def _infer_page_kind(relative_path: str, frontmatter: dict[str, Any]) -> str:
    page_type = str(frontmatter.get("type", "")).strip().lower()
    if page_type in {"source", "concept", "analysis"}:
        return f"{page_type}_page"
    if relative_path.startswith("wiki/sources/"):
        return "source_page"
    if relative_path.startswith("wiki/concepts/"):
        return "concept_page"
    if relative_path.startswith("wiki/analysis/"):
        return "analysis_page"
    return "source_page"


def _page_title(
    file_path: Path,
    frontmatter: dict[str, Any],
    page_headings: list[Any],
) -> str:
    title = str(frontmatter.get("title", "")).strip()
    if title:
        return title
    if page_headings:
        return page_headings[0].title
    return file_path.stem.replace("-", " ")


def _collect_aliases(frontmatter: dict[str, Any], title: str) -> tuple[str, ...]:
    aliases: set[str] = {title}
    raw_aliases = frontmatter.get("aliases", [])
    if isinstance(raw_aliases, str):
        raw_aliases = _ALIAS_SPLIT_RE.split(raw_aliases)
    if isinstance(raw_aliases, list):
        for item in raw_aliases:
            cleaned = str(item).strip()
            if cleaned:
                aliases.add(cleaned)
    return tuple(sorted(aliases))


def _split_sections(body: str) -> tuple[tuple[str, str], ...]:
    sections: list[tuple[str, str]] = []
    current_heading = "Overview"
    current_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[3:].strip()
            current_lines = []
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return tuple(sections)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None
