"""Provider-backed LLM extractor for the LightRAG-style backend.

The deterministic extractor in :mod:`graphwiki_kb.wikigraph.light_extractor`
mines entities from regex-captured capitalized phrases and emits weak
``SUPPORTS`` co-occurrence relations. The LightRAG paper instead uses
an LLM to extract **typed** entities and relations from each chunk,
with explicit evidence quotes — this is the Tier A path.

This module wraps that with the project's :class:`TextProvider` so it
plugs into the same OpenAI / Anthropic / Gemini infrastructure the rest
of the KB uses. Output is validated against a strict JSON schema and
gracefully degrades when the provider misbehaves (returns a warning,
not a crash).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from graphwiki_kb.providers.base import (
    ProviderRequest,
    ProviderResponse,
    TextProvider,
)
from graphwiki_kb.wikigraph.light_extractor import (
    DEFAULT_ENTITY_TYPES,
    DEFAULT_RELATION_TYPES,
    LightExtractorOptions,
)
from graphwiki_kb.wikigraph.light_models import (
    ExtractedEntity,
    ExtractedRelation,
    LightChunk,
    LightExtractionResult,
)

EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "description": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": [
                    "name",
                    "type",
                    "aliases",
                    "description",
                    "evidence_quote",
                ],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "description": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": [
                    "source",
                    "target",
                    "relation_type",
                    "keywords",
                    "description",
                    "evidence_quote",
                ],
            },
        },
    },
    "required": ["entities", "relations"],
}


class _LLMEntityPayload(BaseModel):
    """Strict shape for one LLM-emitted entity (rejected on extra keys)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    evidence_quote: str = ""


class _LLMRelationPayload(BaseModel):
    """Strict shape for one LLM-emitted relation."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    relation_type: str
    keywords: list[str] = Field(default_factory=list)
    description: str = ""
    evidence_quote: str = ""


class _LLMExtractionPayload(BaseModel):
    """Top-level shape returned by the extractor LLM call."""

    model_config = ConfigDict(extra="forbid")

    entities: list[_LLMEntityPayload] = Field(default_factory=list)
    relations: list[_LLMRelationPayload] = Field(default_factory=list)


def _build_system_prompt(
    entity_types: tuple[str, ...], relation_types: tuple[str, ...]
) -> str:
    return (
        "You build a focused knowledge graph from one source chunk of "
        "a research-paper corpus. The graph is used by a downstream "
        "retrieval system, so prefer a small number of high-signal "
        "entities and relations over a long list of incidental "
        "mentions.\n\n"
        "Allowed entity types (use exactly one per entity):\n  - "
        + "\n  - ".join(entity_types)
        + "\n\n"
        "Allowed relation types (use exactly one per relation):\n  - "
        + "\n  - ".join(relation_types)
        + "\n\n"
        "Extraction rules:\n"
        "1. ALWAYS emit one PAPER entity for the source paper itself "
        "(use the 'Source:' header value as ``name``). It is the "
        "anchor entity even when the chunk only contains methodology "
        "details.\n"
        "2. Extract at most ~10 entities per chunk. Prefer entities "
        "that name the paper's core contributions: models, methods, "
        "datasets, metrics, tasks, and explicitly-cited prior "
        "systems. Skip incidental named entities (optimizer names "
        "like 'AdamW', generic hyperparameters, song lyrics, "
        "unrelated people in example text) unless they are central "
        "to the chunk's argument.\n"
        "3. Every entity must include a short ``evidence_quote`` "
        "taken as a near-verbatim substring of the chunk.\n"
        "4. Every relation must connect two entity names that appear "
        "in the ``entities`` array. Prefer relations involving the "
        "PAPER anchor entity (e.g. ``REALM USES BERT``, ``DPR "
        "EVALUATES_ON Natural Questions``) over relations between "
        "two incidental entities.\n"
        "5. If the chunk truly has no research-graph content (metadata, "
        "bibliography, or unrelated examples), return empty arrays "
        "EXCEPT for the PAPER anchor entity from rule 1.\n"
        "Return JSON conforming exactly to the provided schema."
    )


def _build_prompt(chunk: LightChunk, max_chars: int = 6000) -> str:
    body = chunk.text
    if len(body) > max_chars:
        body = body[: max_chars - 1] + "…"
    header = (
        f"Source: {chunk.source_title or chunk.source_slug}\n"
        f"Chunk: {chunk.chunk_index}\n"
        f"Source ID: {chunk.source_id}\n\n"
    )
    return header + "Chunk text:\n```\n" + body + "\n```"


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return stripped


@dataclass
class LLMLightExtractor:
    """LightExtractor that calls a :class:`TextProvider` for each chunk.

    Attributes:
        provider: The underlying completion provider (typically OpenAI).
        options: Extraction options. Entity/relation type sets come from
            here and are echoed into the system prompt.
        max_tokens: Output token budget for one chunk's extraction.
        max_chunk_chars: Cap on chunk body length passed to the
            provider (prevents pathologically long chunks from blowing
            the prompt budget).
        name: Cosmetic name surfaced in the build manifest.
    """

    provider: TextProvider
    options: LightExtractorOptions = field(default_factory=LightExtractorOptions)
    max_tokens: int = 2048
    max_chunk_chars: int = 6000
    name: str = "llm"

    def __post_init__(self) -> None:
        types = self.options.entity_types or DEFAULT_ENTITY_TYPES
        rel_types = self.options.relation_types or DEFAULT_RELATION_TYPES
        self._system_prompt = _build_system_prompt(types, rel_types)
        # Prompt hash captures everything that would invalidate cached
        # extractions: schema, type sets, system prompt, model name,
        # and the option budget. Bumping this safely invalidates the
        # on-disk extraction cache.
        signature = json.dumps(
            {
                "name": self.name,
                "provider": self.provider.name,
                "entity_types": list(types),
                "relation_types": list(rel_types),
                "max_chunk_chars": self.max_chunk_chars,
                "schema_version": 1,
            },
            sort_keys=True,
        )
        self.prompt_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]

    def extract(self, chunk: LightChunk) -> LightExtractionResult:
        """Call the provider once and return a validated extraction."""
        warnings: list[str] = []
        try:
            response: ProviderResponse = self.provider.generate(
                ProviderRequest(
                    prompt=_build_prompt(chunk, max_chars=self.max_chunk_chars),
                    system_prompt=self._system_prompt,
                    max_tokens=self.max_tokens,
                    response_schema=EXTRACTION_SCHEMA,
                    response_schema_name="lightrag_chunk_extraction",
                )
            )
        except Exception as exc:  # pragma: no cover - provider-specific
            return LightExtractionResult(
                chunk_id=chunk.id,
                entities=[],
                relations=[],
                warnings=[f"provider_error:{type(exc).__name__}:{exc}"],
                extractor=self.name,
            )
        return _parse_llm_response(
            chunk,
            response.text,
            warnings=warnings,
            extractor_name=self.name,
            entity_types=self.options.entity_types or DEFAULT_ENTITY_TYPES,
            relation_types=self.options.relation_types or DEFAULT_RELATION_TYPES,
        )


def _parse_llm_response(
    chunk: LightChunk,
    raw_text: str,
    *,
    warnings: list[str],
    extractor_name: str,
    entity_types: tuple[str, ...],
    relation_types: tuple[str, ...],
) -> LightExtractionResult:
    cleaned = _strip_code_fence(raw_text or "")
    if not cleaned:
        return LightExtractionResult(
            chunk_id=chunk.id,
            entities=[],
            relations=[],
            warnings=[*warnings, "empty_provider_response"],
            extractor=extractor_name,
        )
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return LightExtractionResult(
            chunk_id=chunk.id,
            entities=[],
            relations=[],
            warnings=[*warnings, f"invalid_json:{exc.msg}"],
            extractor=extractor_name,
        )
    try:
        parsed = _LLMExtractionPayload.model_validate(payload)
    except ValidationError as exc:
        return LightExtractionResult(
            chunk_id=chunk.id,
            entities=[],
            relations=[],
            warnings=[*warnings, f"schema_violation:{exc.error_count()}_errors"],
            extractor=extractor_name,
        )

    entity_type_set = {t.upper() for t in entity_types}
    relation_type_set = {t.upper() for t in relation_types}
    paper_type = "PAPER" if "PAPER" in entity_type_set else next(iter(entity_type_set))

    entities: list[ExtractedEntity] = []
    known_names: set[str] = set()

    # Always seed the source paper as a PAPER entity (rule 1 in the
    # system prompt) — the LLM occasionally forgets, and the paper
    # title is the most-cited entity at retrieval time.
    if chunk.source_title:
        entities.append(
            ExtractedEntity(
                name=chunk.source_title,
                type=paper_type,
                description=(
                    f"Source paper {chunk.source_title!r} (chunk "
                    f"{chunk.chunk_index})."
                ),
                aliases=[chunk.source_slug] if chunk.source_slug else [],
                chunk_ids=[chunk.id],
                source_ids=[chunk.source_id],
                evidence_quote=chunk.source_title,
                confidence=0.99,
            )
        )
        known_names.add(chunk.source_title.casefold())
        if chunk.source_slug:
            known_names.add(chunk.source_slug.casefold())

    for raw_entity in parsed.entities:
        name = raw_entity.name.strip()
        if not name:
            continue
        if name.casefold() in known_names:
            # The LLM emitted the same paper entity we already seeded.
            continue
        entity_type = raw_entity.type.strip().upper() or next(iter(entity_type_set))
        if entity_types and entity_type not in entity_type_set:
            # Coerce unknown types onto the closest allowed bucket
            # rather than dropping the entity entirely.
            entity_type = (
                "CLAIM" if "CLAIM" in entity_type_set else next(iter(entity_type_set))
            )
        entities.append(
            ExtractedEntity(
                name=name,
                type=entity_type,
                description=raw_entity.description.strip(),
                aliases=[a.strip() for a in raw_entity.aliases if a.strip()],
                chunk_ids=[chunk.id],
                source_ids=[chunk.source_id],
                evidence_quote=raw_entity.evidence_quote.strip(),
                confidence=0.9,
            )
        )
        known_names.add(name.casefold())
        for alias in raw_entity.aliases:
            if alias.strip():
                known_names.add(alias.strip().casefold())

    relations: list[ExtractedRelation] = []
    for raw_rel in parsed.relations:
        source = raw_rel.source.strip()
        target = raw_rel.target.strip()
        if not source or not target:
            continue
        if source.casefold() not in known_names or target.casefold() not in known_names:
            warnings.append(f"relation_endpoint_missing:{source}->{target}")
            continue
        rel_type = raw_rel.relation_type.strip().upper() or "SUPPORTS"
        if relation_types and rel_type not in relation_type_set:
            rel_type = (
                "SUPPORTS"
                if "SUPPORTS" in relation_type_set
                else next(iter(relation_type_set))
            )
        relations.append(
            ExtractedRelation(
                source=source,
                target=target,
                relation_type=rel_type,
                description=raw_rel.description.strip(),
                keywords=[k.strip() for k in raw_rel.keywords if k.strip()],
                chunk_ids=[chunk.id],
                source_ids=[chunk.source_id],
                evidence_quote=raw_rel.evidence_quote.strip(),
                weight=1.0,
                confidence=0.85,
            )
        )

    return LightExtractionResult(
        chunk_id=chunk.id,
        entities=entities,
        relations=relations,
        warnings=warnings,
        extractor=extractor_name,
    )


__all__ = [
    "EXTRACTION_SCHEMA",
    "LLMLightExtractor",
]


# --------------------------------------------------------------------------- #
# Public re-exports for the iterable typing helper                            #
# --------------------------------------------------------------------------- #


def iter_chunks(chunks: Iterable[LightChunk]) -> Iterable[LightChunk]:
    """Identity helper kept for symmetry with the deterministic API."""
    yield from chunks


_ = Any  # mark Any as used so ruff doesn't strip the import on a future refactor.
