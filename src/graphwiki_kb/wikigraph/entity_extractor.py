"""Extract entity nodes from wiki pages and chunks."""

from __future__ import annotations

import re
from collections import defaultdict

from rapidfuzz import fuzz

from graphwiki_kb.services.project_service import slugify
from graphwiki_kb.wikigraph.markdown_parser import ParsedChunk, ParsedWikiPage
from graphwiki_kb.wikigraph.models import WikiGraphNode

_ENTITY_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "about",
    }
)
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,}\b")
_TITLE_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)+\b")


def build_entity_nodes(
    pages: list[ParsedWikiPage],
    chunks: list[ParsedChunk],
) -> list[WikiGraphNode]:
    """Create entity nodes from titles, aliases, tags, and capitalized phrases."""
    registry: dict[str, WikiGraphNode] = {}

    def register(label: str, *, origin: str, source_id: str | None) -> None:
        cleaned = label.strip()
        if len(cleaned) < 2 or cleaned.lower() in _ENTITY_STOPWORDS:
            return
        entity_id = f"entity:{slugify(cleaned)}"
        if entity_id in registry:
            metadata = registry[entity_id].metadata
            origins = set(metadata.get("origins", []))
            origins.add(origin)
            metadata["origins"] = sorted(origins)
            if source_id:
                source_ids = set(metadata.get("source_ids", []))
                source_ids.add(source_id)
                metadata["source_ids"] = sorted(source_ids)
            return
        registry[entity_id] = WikiGraphNode(
            id=entity_id,
            kind="entity",
            title=cleaned,
            path=None,
            text=cleaned,
            metadata={
                "origins": [origin],
                "source_ids": [source_id] if source_id else [],
            },
        )

    for page in pages:
        register(page.title, origin=page.path, source_id=page.source_id)
        for alias in page.aliases:
            register(alias, origin=page.path, source_id=page.source_id)
        for tag in page.tags:
            register(tag, origin=page.path, source_id=page.source_id)
        for phrase in _extract_phrases(page.summary):
            register(phrase, origin=page.path, source_id=page.source_id)
        for _, section_text in page.sections:
            for phrase in _extract_phrases(section_text):
                register(phrase, origin=page.path, source_id=page.source_id)

    for chunk in chunks:
        for alias in chunk.aliases:
            register(alias, origin=chunk.page_path, source_id=chunk.source_id)

    return list(registry.values())


def match_entities(
    question: str,
    entities: list[WikiGraphNode],
    *,
    limit: int = 8,
    score_cutoff: int = 70,
) -> list[tuple[WikiGraphNode, float]]:
    """Fuzzy-match question text to entity nodes."""
    tokens = _question_tokens(question)
    if not tokens:
        return []
    scored: list[tuple[WikiGraphNode, float]] = []
    for entity in entities:
        best = 0.0
        for token in tokens:
            best = max(
                best, float(fuzz.partial_ratio(token.lower(), entity.title.lower()))
            )
        if best >= score_cutoff:
            scored.append((entity, best / 100.0))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


def co_mentioned_entities(
    chunks: list[ParsedChunk],
    *,
    min_count: int = 2,
) -> dict[str, set[str]]:
    """Map entity labels that co-occur in the same chunk."""
    pairs: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        labels = sorted({alias for alias in chunk.aliases if alias})
        for index, left in enumerate(labels):
            for right in labels[index + 1 :]:
                pairs[left].add(right)
                pairs[right].add(left)
    return {
        key: values for key, values in pairs.items() if len(values) >= min_count - 1
    }


def _extract_phrases(text: str) -> set[str]:
    phrases: set[str] = set()
    for match in _ACRONYM_RE.findall(text):
        phrases.add(match)
    for match in _TITLE_CASE_RE.findall(text):
        phrases.add(match)
    return phrases


def _question_tokens(question: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{1,}", question)
    return [token for token in raw if token.lower() not in _ENTITY_STOPWORDS]
