"""Shared Markdown/frontmatter parsing helpers.

This module keeps Markdown-aware behavior in one place so services do not need
to maintain parallel regex/state-machine implementations for frontmatter,
plain-text extraction, headings, paragraphs, sections, links, or fenced code.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import frontmatter
from markdown_it import MarkdownIt
from markdown_it.token import Token


_MD_PARSER = MarkdownIt()


@dataclass(frozen=True)
class MarkdownDocument:
    frontmatter: dict[str, Any]
    body: str
    has_frontmatter: bool
    valid_frontmatter: bool


@dataclass(frozen=True)
class MarkdownHeading:
    level: int
    title: str


@dataclass(frozen=True)
class MarkdownSection:
    title: str
    paragraphs: list[str]


@dataclass(frozen=True)
class MarkdownLink:
    text: str
    target: str


def normalize_newlines(contents: str) -> str:
    return contents.replace("\r\n", "\n").replace("\r", "\n")


def parse_document(contents: str) -> MarkdownDocument:
    """Parse YAML frontmatter and return the body.

    Invalid or unterminated frontmatter is treated as plain Markdown body. This
    keeps read-only commands resilient while still allowing lint/config layers
    to decide whether malformed frontmatter should be reported.
    """

    normalized = normalize_newlines(contents)
    has_frontmatter = normalized.startswith("---\n")
    if not has_frontmatter:
        return MarkdownDocument({}, normalized, False, True)
    marker = normalized.find("\n---\n", 4)
    if marker == -1:
        return MarkdownDocument({}, normalized, True, False)

    try:
        post = frontmatter.loads(normalized)
    except Exception:
        return MarkdownDocument({}, normalized, True, False)

    metadata = dict(post.metadata) if isinstance(post.metadata, dict) else {}
    return MarkdownDocument(metadata, normalized[marker + 5 :], True, True)


def parse_frontmatter(contents: str) -> dict[str, Any]:
    document = parse_document(contents)
    return document.frontmatter if document.valid_frontmatter else {}


def strip_frontmatter(contents: str) -> str:
    document = parse_document(contents)
    return document.body if document.valid_frontmatter else normalize_newlines(contents)


def inline_text(token: Token) -> str:
    if not token.children:
        return token.content or ""

    parts: list[str] = []
    for child in token.children:
        if child.type == "image":
            continue
        if child.type in {"text", "code_inline"}:
            parts.append(child.content)
            continue
        if child.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
            continue
        if child.children:
            parts.append(inline_text(child))
    return "".join(parts)


def is_link_only_inline(token: Token) -> bool:
    if token.type != "inline" or not token.children:
        return False

    seen_link = False
    outside_text: list[str] = []
    inside_link = 0

    for child in token.children:
        if child.type == "link_open":
            inside_link += 1
            seen_link = True
            continue
        if child.type == "link_close":
            inside_link = max(0, inside_link - 1)
            continue
        if child.type == "image":
            seen_link = True
            continue
        if inside_link == 0 and child.type in {"text", "code_inline"}:
            outside_text.append(child.content)

    return seen_link and not "".join(outside_text).strip()


def is_content_paragraph(paragraph: str) -> bool:
    tokens = _MD_PARSER.parse(paragraph)
    inline_tokens = [token for token in tokens if token.type == "inline"]
    if inline_tokens and all(is_link_only_inline(token) for token in inline_tokens):
        return False

    link_count = len(markdown_links(paragraph))
    if link_count:
        outside_links = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", paragraph)
        outside_links = re.sub(r"\[[^\]]*\]\([^)]+\)", " ", outside_links)
        outside_words = re.findall(r"[A-Za-z]{2,}", outside_links)
        if link_count >= 2 and len(outside_words) <= 3:
            return False

    text = plain_text(paragraph).strip()
    if len(text) < 8:
        return False

    words = [word for word in text.split() if any(ch.isalpha() for ch in word)]
    if len(words) < 2:
        return False

    return True


def paragraphs(
    contents: str,
    *,
    content_only: bool = True,
    trim_leading_boilerplate: bool = True,
) -> list[str]:
    body = strip_frontmatter(contents)
    tokens = _MD_PARSER.parse(body)
    extracted: list[str] = []

    for index, token in enumerate(tokens):
        if token.type != "paragraph_open":
            continue
        if index + 1 >= len(tokens) or tokens[index + 1].type != "inline":
            continue

        paragraph = inline_text(tokens[index + 1]).strip()
        if not paragraph:
            continue
        if content_only and not is_content_paragraph(paragraph):
            continue
        extracted.append(paragraph)

    if trim_leading_boilerplate and extracted:
        while extracted and _looks_like_boilerplate(extracted[0]):
            extracted.pop(0)

    return extracted


def plain_text(contents: str) -> str:
    body = strip_frontmatter(contents)
    tokens = _MD_PARSER.parse(body)
    lines: list[str] = []

    for token in tokens:
        if token.type in {
            "heading_open",
            "paragraph_open",
            "bullet_list_open",
            "ordered_list_open",
        }:
            continue
        if token.type in {
            "heading_close",
            "paragraph_close",
            "bullet_list_close",
            "ordered_list_close",
        }:
            continue
        if token.type in {"fence", "code_block", "html_block"}:
            continue
        if token.type == "inline":
            text = inline_text(token).strip()
            if text:
                lines.append(text)
            continue
        if token.content and token.type not in {"hr"}:
            text = token.content.strip()
            if text:
                lines.append(text)

    return "\n".join(lines)


def headings(contents: str) -> list[MarkdownHeading]:
    body = strip_frontmatter(contents)
    tokens = _MD_PARSER.parse(body)
    extracted: list[MarkdownHeading] = []

    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        if index + 1 >= len(tokens) or tokens[index + 1].type != "inline":
            continue
        level_text = token.tag[1:] if token.tag.startswith("h") else "0"
        try:
            level = int(level_text)
        except ValueError:
            level = 0
        title = inline_text(tokens[index + 1]).strip()
        if title:
            extracted.append(MarkdownHeading(level, title))

    return extracted


def sections(contents: str, *, default_title: str = "content") -> list[MarkdownSection]:
    body = strip_frontmatter(contents)
    tokens = _MD_PARSER.parse(body)
    current_title = default_title
    current_paragraphs: list[str] = []
    extracted: list[MarkdownSection] = []

    def flush() -> None:
        nonlocal current_paragraphs
        if current_paragraphs:
            extracted.append(MarkdownSection(current_title, current_paragraphs))
            current_paragraphs = []

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            token.type == "heading_open"
            and index + 1 < len(tokens)
            and tokens[index + 1].type == "inline"
        ):
            flush()
            current_title = inline_text(tokens[index + 1]).strip() or default_title
            index += 3
            continue
        if (
            token.type == "paragraph_open"
            and index + 1 < len(tokens)
            and tokens[index + 1].type == "inline"
        ):
            paragraph = inline_text(tokens[index + 1]).strip()
            if paragraph:
                current_paragraphs.append(paragraph)
            index += 3
            continue
        index += 1

    flush()
    return extracted


def section_paragraphs(
    contents: str,
    heading_title: str,
    *,
    content_only: bool = True,
) -> list[str]:
    target = heading_title.strip().casefold()
    matching: list[str] = []
    for section in sections(contents, default_title="content"):
        if section.title.strip().casefold() != target:
            continue
        for paragraph in section.paragraphs:
            if not content_only or is_content_paragraph(paragraph):
                matching.append(paragraph)
    return matching


def without_fenced_code_blocks(contents: str) -> str:
    body = normalize_newlines(contents)
    tokens = _MD_PARSER.parse(body)
    lines = body.splitlines()
    excluded: set[int] = set()

    for token in tokens:
        if token.type not in {"fence", "code_block"} or not token.map:
            continue
        start, end = token.map
        excluded.update(range(start, end))

    return "\n".join(line for index, line in enumerate(lines) if index not in excluded)


def markdown_links(contents: str) -> list[MarkdownLink]:
    tokens = _MD_PARSER.parse(normalize_newlines(contents))
    links: list[MarkdownLink] = []

    for token in tokens:
        if token.type != "inline" or not token.children:
            continue

        link_text: list[str] = []
        target: str | None = None
        inside_link = False

        for child in token.children:
            if child.type == "link_open":
                target = child.attrGet("href") or ""
                link_text = []
                inside_link = True
                continue
            if child.type == "link_close":
                links.append(MarkdownLink("".join(link_text).strip(), target or ""))
                target = None
                link_text = []
                inside_link = False
                continue
            if inside_link and child.type == "text":
                link_text.append(child.content)
            elif inside_link and child.type == "code_inline":
                link_text.append(child.content)

    return links


def _looks_like_boilerplate(paragraph: str) -> bool:
    normalized = paragraph.strip().lower()
    if not normalized:
        return True
    if normalized.startswith(("view source", "source:", "back to ")):
        return True
    words = normalized.split()
    if normalized in {"table of contents", "contents"}:
        return True
    if "documentation search changelog" in normalized:
        return True
    if len(words) <= 6 and not re.search(r"[.!?]$", normalized):
        return True
    sentence_word_counts = [
        len(re.findall(r"[a-z0-9]+", sentence))
        for sentence in re.split(r"[.!?]+", normalized)
        if sentence.strip()
    ]
    if len(sentence_word_counts) >= 3 and all(
        count <= 3 for count in sentence_word_counts
    ):
        return True
    return False
