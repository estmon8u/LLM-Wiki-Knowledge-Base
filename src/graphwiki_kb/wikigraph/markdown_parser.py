"""Wiki-aware Markdown parsing helpers for the WikiGraphRAG pipeline.

This module wraps :mod:`graphwiki_kb.services.markdown_document` with helpers
tuned to the wiki page conventions: wikilinks (``[[Target]]`` and
``[[Target|Label]]``), section-level chunks bounded by character budgets,
and frontmatter-aware page identity (title/aliases/source ids).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphwiki_kb.services.markdown_document import (
    markdown_links,
    parse_document,
    plain_text,
)
from graphwiki_kb.services.markdown_document import (
    sections as markdown_sections,
)

WIKILINK_PATTERN = re.compile(r"\[\[([^\[\]\n]+?)\]\]")

_DEFAULT_CHUNK_CHAR_LIMIT = 1200
_NON_EVIDENCE_SECTION_TITLES: frozenset[str] = frozenset(
    {
        "citations",
        "related concept pages",
        "source details",
        "source pages",
    }
)


@dataclass(frozen=True)
class WikiPageChunk:
    """A section-level chunk extracted from a wiki page."""

    section: str
    body: str
    chunk_index: int


@dataclass(frozen=True)
class WikiLink:
    """A parsed wikilink occurrence in a page body."""

    target: str
    label: str


@dataclass
class WikiPage:
    """A parsed wiki markdown page."""

    relative_path: str
    page_type: str
    title: str
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    body: str = ""
    frontmatter: dict[str, Any] = field(default_factory=dict)
    chunks: list[WikiPageChunk] = field(default_factory=list)
    wikilinks: list[WikiLink] = field(default_factory=list)
    markdown_links: list[tuple[str, str]] = field(default_factory=list)


def page_type_from_path(relative_path: str) -> str:
    """Return the semantic page type from the wiki path prefix."""
    if relative_path.startswith("wiki/sources/"):
        return "source"
    if relative_path.startswith("wiki/concepts/"):
        return "concept"
    if relative_path.startswith("wiki/analysis/"):
        return "analysis"
    if relative_path.startswith("wiki/graph/"):
        return "graph"
    if relative_path.startswith("wiki/wikigraph/"):
        return "wikigraph_generated"
    return ""


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for entry in value:
            text = str(entry).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def _title_from_frontmatter(frontmatter: dict[str, Any], fallback: str) -> str:
    candidate = frontmatter.get("title")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return fallback.replace("-", " ").strip().title() or fallback


def _source_ids_from_frontmatter(frontmatter: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("source_id", "source_ids", "sources"):
        candidates.extend(_coerce_string_list(frontmatter.get(key)))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _chunk_sections(
    body: str,
    *,
    title: str,
    char_limit: int,
) -> list[WikiPageChunk]:
    """Build section-bounded chunks within ``char_limit`` characters each."""
    chunks: list[WikiPageChunk] = []
    chunk_index = 0
    for section in markdown_sections(body, default_title=title):
        if section.title.strip().casefold() in _NON_EVIDENCE_SECTION_TITLES:
            continue
        current_parts: list[str] = []
        current_length = 0
        for paragraph in section.paragraphs:
            normalized = " ".join(paragraph.split()).strip()
            if not normalized:
                continue
            addition = len(normalized) + 2
            if current_parts and current_length + addition > char_limit:
                chunks.append(
                    WikiPageChunk(
                        section=section.title or title,
                        body="\n\n".join(current_parts),
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
                current_parts = []
                current_length = 0
            current_parts.append(normalized)
            current_length += addition
        if current_parts:
            chunks.append(
                WikiPageChunk(
                    section=section.title or title,
                    body="\n\n".join(current_parts),
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
    if not chunks:
        plain = plain_text(body).strip()
        if plain:
            chunks.append(
                WikiPageChunk(
                    section=title,
                    body=plain[:char_limit],
                    chunk_index=0,
                )
            )
    return chunks


def _extract_wikilinks(text: str) -> list[WikiLink]:
    """Return [[Target]] and [[Target|Label]] occurrences."""
    found: list[WikiLink] = []
    for match in WIKILINK_PATTERN.finditer(text):
        inner = match.group(1).strip()
        if not inner:
            continue
        if "|" in inner:
            target, _, label = inner.partition("|")
            target = target.strip()
            label = label.strip() or target
        else:
            target = inner
            label = inner
        if target:
            found.append(WikiLink(target=target, label=label))
    return found


def parse_wiki_page(
    file_path: Path,
    relative_path: str,
    *,
    chunk_char_limit: int = _DEFAULT_CHUNK_CHAR_LIMIT,
) -> WikiPage | None:
    """Parse a wiki markdown file into a :class:`WikiPage`.

    Returns ``None`` for unreadable files. Skips files that look like
    maintenance pages (``wiki/index.md``, ``wiki/log.md``).
    """
    if relative_path in {"wiki/index.md", "wiki/log.md"}:
        return None
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError:
        return None

    document = parse_document(text)
    frontmatter = document.frontmatter if document.valid_frontmatter else {}

    page_type = ""
    if isinstance(frontmatter.get("type"), str):
        page_type = str(frontmatter["type"]).strip()
    if not page_type:
        page_type = page_type_from_path(relative_path)

    title = _title_from_frontmatter(frontmatter, file_path.stem)
    aliases = _coerce_string_list(frontmatter.get("aliases"))
    tags = _coerce_string_list(frontmatter.get("tags"))
    source_ids = _source_ids_from_frontmatter(frontmatter)

    body = document.body
    chunks = _chunk_sections(body, title=title, char_limit=chunk_char_limit)
    wikilinks = _extract_wikilinks(body)
    md_links = [(link.text, link.target) for link in markdown_links(body)]

    return WikiPage(
        relative_path=relative_path,
        page_type=page_type,
        title=title,
        aliases=aliases,
        tags=tags,
        source_ids=source_ids,
        body=body,
        frontmatter=dict(frontmatter.items()),
        chunks=chunks,
        wikilinks=wikilinks,
        markdown_links=md_links,
    )
