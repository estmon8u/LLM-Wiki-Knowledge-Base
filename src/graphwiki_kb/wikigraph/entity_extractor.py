"""Lightweight entity and claim extraction for the WikiGraphRAG pipeline.

The pipeline is deliberately deterministic and provider-free:

* **Entities** come from page titles, aliases, and ``[[wikilinks]]`` plus
  capitalized noun-like tokens that recur frequently across a page. This is
  not as expressive as the Microsoft GraphRAG LLM extractor, but it gives a
  transparent, source-grounded entity catalog that is good enough to
  demonstrate local search and to keep the comparison honest.
* **Claims** come from short bullet items under headings such as
  ``Key Points``, ``Findings``, ``Claims``, or ``Summary`` -- sections we
  already encourage in wiki source-page schemas.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from graphwiki_kb.services.markdown_document import sections as markdown_sections
from graphwiki_kb.services.stopwords import STOPWORDS
from graphwiki_kb.wikigraph.markdown_parser import WikiPage

_CAPITALIZED_PHRASE = re.compile(
    r"\b([A-Z][A-Za-z0-9][A-Za-z0-9\-]*"
    r"(?:\s+(?:of|for|and|in|on|the|a)?\s*[A-Z][A-Za-z0-9][A-Za-z0-9\-]*){0,3})\b"
)
_ACRONYM = re.compile(r"\b([A-Z]{2,6})\b")
_BULLET_LINE = re.compile(r"^\s*[-*+]\s+(.*)$")
_MAX_ENTITY_LENGTH = 80
_MIN_ENTITY_LENGTH = 2
_CLAIM_SECTION_TITLES: frozenset[str] = frozenset(
    {
        "key points",
        "key findings",
        "findings",
        "claims",
        "summary",
        "main contributions",
        "contributions",
        "methods",
        "results",
    }
)


def _normalize_entity(text: str) -> str:
    cleaned = " ".join(text.split()).strip(" .,:;")
    return cleaned


def _is_acceptable_entity(text: str) -> bool:
    if not text:
        return False
    if len(text) < _MIN_ENTITY_LENGTH or len(text) > _MAX_ENTITY_LENGTH:
        return False
    if text.casefold() in STOPWORDS:
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    return True


@dataclass(frozen=True)
class ExtractedEntity:
    """A surface mention of an entity together with provenance."""

    name: str
    aliases: tuple[str, ...]
    page_path: str
    page_title: str
    occurrences: int
    source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractedClaim:
    """A short bullet-sized claim extracted from a structured section."""

    text: str
    page_path: str
    page_title: str
    section: str
    chunk_index: int | None = None
    source_ids: tuple[str, ...] = ()


@dataclass
class EntityCatalog:
    """The merged set of entity surface forms across the corpus."""

    entries: dict[str, ExtractedEntity] = field(default_factory=dict)
    by_alias: dict[str, str] = field(default_factory=dict)

    def add(self, entity: ExtractedEntity) -> None:
        """Merge ``entity`` into the catalog by case-folded canonical name."""
        key = entity.name.casefold()
        existing = self.entries.get(key)
        if existing is None:
            self.entries[key] = entity
            self.by_alias[key] = key
            for alias in entity.aliases:
                self.by_alias[alias.casefold()] = key
            return
        merged_aliases = tuple(
            dict.fromkeys([*existing.aliases, *entity.aliases, entity.name])
        )
        merged_sources = tuple(
            dict.fromkeys([*existing.source_ids, *entity.source_ids])
        )
        self.entries[key] = ExtractedEntity(
            name=existing.name,
            aliases=tuple(a for a in merged_aliases if a.casefold() != key),
            page_path=existing.page_path,
            page_title=existing.page_title,
            occurrences=existing.occurrences + entity.occurrences,
            source_ids=merged_sources,
        )
        for alias in merged_aliases:
            self.by_alias[alias.casefold()] = key

    def find(self, term: str) -> ExtractedEntity | None:
        """Look up an entity by case-folded name or alias."""
        key = self.by_alias.get(term.casefold())
        if key is None:
            return None
        return self.entries.get(key)

    def iter_entities(self) -> Iterable[ExtractedEntity]:
        """Iterate entities in lexicographic order for deterministic IO."""
        for key in sorted(self.entries):
            yield self.entries[key]


def _candidate_entity_phrases(text: str) -> list[str]:
    candidates: list[str] = []
    for match in _CAPITALIZED_PHRASE.finditer(text):
        phrase = _normalize_entity(match.group(1))
        if _is_acceptable_entity(phrase):
            candidates.append(phrase)
    for match in _ACRONYM.finditer(text):
        phrase = match.group(1)
        if _is_acceptable_entity(phrase):
            candidates.append(phrase)
    return candidates


def extract_page_entities(page: WikiPage) -> list[ExtractedEntity]:
    """Extract entities for a single wiki page."""
    sources = tuple(page.source_ids)
    seed_aliases = [page.title, *page.aliases]
    aliases_normalized = tuple(
        dict.fromkeys(alias for alias in seed_aliases if _is_acceptable_entity(alias))
    )

    candidate_counter: Counter[str] = Counter()
    for chunk in page.chunks:
        for phrase in _candidate_entity_phrases(chunk.body):
            candidate_counter[phrase] += 1
    for link in page.wikilinks:
        if _is_acceptable_entity(link.target):
            candidate_counter[link.target] += 1

    page_entity = ExtractedEntity(
        name=page.title,
        aliases=tuple(alias for alias in aliases_normalized if alias != page.title),
        page_path=page.relative_path,
        page_title=page.title,
        occurrences=max(1, candidate_counter.get(page.title, 0)),
        source_ids=sources,
    )

    extras: list[ExtractedEntity] = []
    for phrase, count in candidate_counter.items():
        if phrase.casefold() == page.title.casefold():
            continue
        if count < 2 and phrase not in {link.target for link in page.wikilinks}:
            continue
        extras.append(
            ExtractedEntity(
                name=phrase,
                aliases=(),
                page_path=page.relative_path,
                page_title=page.title,
                occurrences=count,
                source_ids=sources,
            )
        )
    return [page_entity, *extras]


def build_entity_catalog(pages: Iterable[WikiPage]) -> EntityCatalog:
    """Build an :class:`EntityCatalog` across the supplied pages."""
    catalog = EntityCatalog()
    for page in pages:
        for entity in extract_page_entities(page):
            catalog.add(entity)
    return catalog


def extract_page_claims(page: WikiPage) -> list[ExtractedClaim]:
    """Extract short bullet-sized claims from structured page sections."""
    claims: list[ExtractedClaim] = []
    sources = tuple(page.source_ids)
    chunk_lookup = {
        chunk.section.casefold(): chunk.chunk_index for chunk in page.chunks
    }
    for section in markdown_sections(page.body, default_title=page.title):
        if section.title.strip().casefold() not in _CLAIM_SECTION_TITLES:
            continue
        chunk_index = chunk_lookup.get(section.title.casefold())
        for paragraph in section.paragraphs:
            for line in paragraph.splitlines():
                bullet = _BULLET_LINE.match(line)
                text = bullet.group(1).strip() if bullet else paragraph.strip()
                if not text or len(text) < 12:
                    continue
                if not _looks_like_claim(text):
                    continue
                claims.append(
                    ExtractedClaim(
                        text=text,
                        page_path=page.relative_path,
                        page_title=page.title,
                        section=section.title,
                        chunk_index=chunk_index,
                        source_ids=sources,
                    )
                )
                if not bullet:
                    break
    return claims


def _looks_like_claim(text: str) -> bool:
    if len(text) > 400:
        return False
    return any(char.isalpha() for char in text)
