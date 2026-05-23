"""Per-chunk entity & relation extraction for the LightRAG-style backend.

Provider-backed extraction (LightRAG's published path) requires an LLM
that can return structured JSON. To keep tests and offline runs honest
(see project recommendation §24, Tier C "fallback diagnostic mode"),
this module also ships a **deterministic** extractor that mines
entities and relations from chunk text using the same regex / acronym
heuristics already used by the classic ``EntityCatalog`` builder.

The deterministic extractor is intentionally less capable than the
provider-backed path; the public report always records which extractor
ran so evaluation cannot accidentally pretend an offline run is a
strict LightRAG run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from graphwiki_kb.services.stopwords import STOPWORDS
from graphwiki_kb.wikigraph.light_models import (
    ExtractedEntity,
    ExtractedRelation,
    LightChunk,
    LightExtractionResult,
)

DEFAULT_ENTITY_TYPES: tuple[str, ...] = (
    "MODEL",
    "METHOD",
    "DATASET",
    "METRIC",
    "TASK",
    "PAPER",
    "TOOL",
    "ORGANIZATION",
    "PERSON",
    "CLAIM",
)

DEFAULT_RELATION_TYPES: tuple[str, ...] = (
    "USES",
    "EVALUATES_ON",
    "IMPROVES_OVER",
    "COMPARES_TO",
    "INTRODUCES",
    "DEPENDS_ON",
    "TRADEOFF_WITH",
    "CONTRADICTS",
    "SUPPORTS",
)

_CAPITALIZED_PHRASE = re.compile(
    r"\b([A-Z][A-Za-z0-9][A-Za-z0-9\-]*"
    r"(?:\s+(?:of|for|and|in|on|the|a)?\s*[A-Z][A-Za-z0-9][A-Za-z0-9\-]*){0,3})\b"
)
_ACRONYM = re.compile(r"\b([A-Z]{2,6})\b")
_MIN_ENT_LEN = 2
_MAX_ENT_LEN = 80


@dataclass(frozen=True)
class LightExtractorOptions:
    """Tunable knobs for the deterministic extractor."""

    entity_types: tuple[str, ...] = DEFAULT_ENTITY_TYPES
    relation_types: tuple[str, ...] = DEFAULT_RELATION_TYPES
    max_entities_per_chunk: int = 20
    max_relations_per_chunk: int = 30
    min_occurrences: int = 1


@runtime_checkable
class LightExtractor(Protocol):
    """Anything that can extract entities/relations from a single chunk."""

    name: str
    prompt_hash: str

    def extract(self, chunk: LightChunk) -> LightExtractionResult:
        """Return the extraction result for ``chunk``."""
        ...


def _is_acceptable_entity(text: str) -> bool:
    if not text:
        return False
    if len(text) < _MIN_ENT_LEN or len(text) > _MAX_ENT_LEN:
        return False
    if text.casefold() in STOPWORDS:
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    return True


def _short_quote(text: str, mention: str, *, span: int = 80) -> str:
    idx = text.find(mention)
    if idx < 0:
        return mention
    start = max(0, idx - span)
    end = min(len(text), idx + len(mention) + span)
    snippet = text[start:end].strip()
    return " ".join(snippet.split())


def _candidate_phrases(text: str) -> list[str]:
    candidates: list[str] = []
    for match in _CAPITALIZED_PHRASE.finditer(text):
        phrase = " ".join(match.group(1).split()).strip(" .,:;")
        if _is_acceptable_entity(phrase):
            candidates.append(phrase)
    for match in _ACRONYM.finditer(text):
        phrase = match.group(1)
        if _is_acceptable_entity(phrase):
            candidates.append(phrase)
    return candidates


def _co_occurrence_relations(
    chunk: LightChunk,
    entity_names: list[str],
    relation_types: tuple[str, ...],
) -> list[ExtractedRelation]:
    """Emit weak SUPPORTS relations between co-occurring entities.

    The deterministic extractor cannot reliably infer typed relations,
    so we emit a single ``SUPPORTS``-style edge per ordered pair of
    entities mentioned in the same chunk. This intentionally mirrors
    the classic backend's ``co_mentions`` edge, kept as a "weak
    fallback" per project recommendation §30.
    """
    if "SUPPORTS" in relation_types:
        rel_type = "SUPPORTS"
    else:
        rel_type = relation_types[0] if relation_types else "SUPPORTS"
    relations: list[ExtractedRelation] = []
    seen: set[tuple[str, str]] = set()
    for i, src in enumerate(entity_names):
        for tgt in entity_names[i + 1 :]:
            if src.casefold() == tgt.casefold():
                continue
            pair = (src.casefold(), tgt.casefold())
            if pair in seen:
                continue
            seen.add(pair)
            quote = _short_quote(chunk.text, src)
            relations.append(
                ExtractedRelation(
                    source=src,
                    target=tgt,
                    relation_type=rel_type,
                    description=(
                        f"{src} and {tgt} co-occur in source "
                        f"'{chunk.source_title or chunk.source_slug}'."
                    ),
                    keywords=[src.casefold(), tgt.casefold()],
                    chunk_ids=[chunk.id],
                    source_ids=[chunk.source_id],
                    evidence_quote=quote,
                    weight=0.5,
                    confidence=0.3,
                )
            )
    return relations


@dataclass
class DeterministicLightExtractor:
    """Provider-free chunk extractor — heuristic but reproducible.

    The extractor's ``prompt_hash`` is derived from its option set, so
    changing ``entity_types`` or ``relation_types`` correctly invalidates
    cached extraction results.
    """

    options: LightExtractorOptions = LightExtractorOptions()
    name: str = "deterministic"

    def __post_init__(self) -> None:
        signature = json.dumps(
            {
                "name": self.name,
                "entity_types": list(self.options.entity_types),
                "relation_types": list(self.options.relation_types),
                "max_entities_per_chunk": self.options.max_entities_per_chunk,
                "max_relations_per_chunk": self.options.max_relations_per_chunk,
            },
            sort_keys=True,
        )
        self.prompt_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]

    def extract(self, chunk: LightChunk) -> LightExtractionResult:
        """Return a deterministic extraction result for ``chunk``."""
        types = self.options.entity_types or DEFAULT_ENTITY_TYPES
        default_type = "CLAIM" if "CLAIM" in types else types[0]

        from collections import Counter

        counter: Counter[str] = Counter()
        for phrase in _candidate_phrases(chunk.text):
            counter[phrase] += 1

        # Add source title as an entity so the chunk always has at least
        # one provenance-rooted entity.
        seed: list[ExtractedEntity] = []
        if chunk.source_title:
            seed.append(
                ExtractedEntity(
                    name=chunk.source_title,
                    type="PAPER" if "PAPER" in types else default_type,
                    description=(
                        f"Source document '{chunk.source_title}' "
                        f"contributing chunk {chunk.chunk_index}."
                    ),
                    aliases=[chunk.source_slug],
                    chunk_ids=[chunk.id],
                    source_ids=[chunk.source_id],
                    evidence_quote=_short_quote(
                        chunk.text, chunk.source_title, span=60
                    ),
                    confidence=0.6,
                )
            )

        seen_keys: set[str] = {
            chunk.source_title.casefold() for _ in (1,) if chunk.source_title
        }

        ranked: list[ExtractedEntity] = list(seed)
        for phrase, count in counter.most_common():
            if count < self.options.min_occurrences:
                continue
            key = phrase.casefold()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            entity_type = "MODEL" if phrase.isupper() else default_type
            if entity_type not in types:
                entity_type = default_type
            ranked.append(
                ExtractedEntity(
                    name=phrase,
                    type=entity_type,
                    description=(
                        f"Mention extracted from chunk {chunk.chunk_index} "
                        f"of {chunk.source_slug}."
                    ),
                    aliases=[],
                    chunk_ids=[chunk.id],
                    source_ids=[chunk.source_id],
                    evidence_quote=_short_quote(chunk.text, phrase),
                    confidence=0.3,
                )
            )
            if len(ranked) >= self.options.max_entities_per_chunk:
                break

        relations = _co_occurrence_relations(
            chunk,
            [ent.name for ent in ranked],
            self.options.relation_types or DEFAULT_RELATION_TYPES,
        )[: self.options.max_relations_per_chunk]

        return LightExtractionResult(
            chunk_id=chunk.id,
            entities=ranked,
            relations=relations,
            warnings=[],
            extractor=self.name,
        )


@dataclass
class LightExtractionCache:
    """Filesystem cache for extraction results keyed by chunk+extractor."""

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, chunk: LightChunk, prompt_hash: str) -> str:
        material = "::".join([chunk.id, chunk.content_hash, prompt_hash]).encode(
            "utf-8"
        )
        return hashlib.sha256(material).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, chunk: LightChunk, prompt_hash: str) -> LightExtractionResult | None:
        """Return a cached :class:`LightExtractionResult` for ``chunk`` or None."""
        path = self._path(self._key(chunk, prompt_hash))
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return LightExtractionResult.model_validate(payload)
        except (OSError, ValueError):
            return None

    def put(
        self,
        chunk: LightChunk,
        prompt_hash: str,
        result: LightExtractionResult,
    ) -> None:
        """Persist ``result`` under the chunk/prompt key."""
        path = self._path(self._key(chunk, prompt_hash))
        path.write_text(
            json.dumps(result.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def extract_corpus(
    chunks: Iterable[LightChunk],
    extractor: LightExtractor,
    *,
    cache: LightExtractionCache | None = None,
) -> list[LightExtractionResult]:
    """Run ``extractor`` over ``chunks`` and return the results.

    When ``cache`` is supplied, results are cached by chunk content hash
    plus the extractor's ``prompt_hash`` (see project recommendation §23).
    """
    results: list[LightExtractionResult] = []
    for chunk in chunks:
        cached = cache.get(chunk, extractor.prompt_hash) if cache else None
        if cached is not None:
            results.append(cached)
            continue
        result = extractor.extract(chunk)
        if cache is not None:
            cache.put(chunk, extractor.prompt_hash, result)
        results.append(result)
    return results
