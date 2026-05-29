"""Chunk-level entity/relation extraction for the LightRAG backend.

Two tiers are supported:

* **Provider-backed (Tier A)** — a structured-output LLM call extracts typed
  entities and relations from a single chunk, with guardrails (relation
  endpoints must appear in the entity set; evidence quotes must be
  near-substrings of the chunk). Results are cached by
  ``(chunk_hash, prompt_hash, provider_identity)`` so re-runs are cheap and
  reproducible.
* **Deterministic fallback (Tier C)** — a provider-free heuristic extractor
  (capitalized phrases + acronyms + co-mentions) so ``kb`` stays usable
  locally. Runs are clearly labeled as a fallback.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from rapidfuzz import fuzz

from graphwiki_kb.providers.base import ProviderRequest, TextProvider
from graphwiki_kb.providers.structured import (
    parse_model_payload,
)
from graphwiki_kb.services.project_service import atomic_write_text
from graphwiki_kb.wikigraph.light_models import (
    ExtractedEntity,
    ExtractedRelation,
    LightChunk,
    LightExtractionResult,
)

_CAPITALIZED_PHRASE = re.compile(
    r"\b([A-Z][A-Za-z0-9][A-Za-z0-9\-]*"
    r"(?:\s+[A-Z][A-Za-z0-9][A-Za-z0-9\-]*){0,3})\b"
)
_ACRONYM = re.compile(r"\b([A-Z]{2,6})\b")
_MAX_FALLBACK_ENTITIES = 8
_MAX_FALLBACK_RELATIONS = 12
_EVIDENCE_FUZZ_THRESHOLD = 85


@dataclass(frozen=True)
class ExtractionConfig:
    """Entity/relation types and gleaning depth for extraction."""

    entity_types: tuple[str, ...]
    relation_types: tuple[str, ...]
    max_gleaning: int = 1


def extraction_prompt_hash(config: ExtractionConfig) -> str:
    """Stable hash of the extraction contract (prompt + types + gleaning)."""
    payload = json.dumps(
        {
            "prompt": _PROMPT_TEMPLATE,
            "entity_types": sorted(config.entity_types),
            "relation_types": sorted(config.relation_types),
            "max_gleaning": config.max_gleaning,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Provider prompt + schema                                                    #
# --------------------------------------------------------------------------- #

_PROMPT_TEMPLATE = (
    "You are extracting a technical knowledge graph from ONE source chunk.\n"
    "Only use information present in the chunk. Do not invent facts.\n\n"
    "Allowed entity types: {entity_types}.\n"
    "Allowed relation types: {relation_types}.\n\n"
    "Return JSON with 'entities' and 'relations'. Each entity has name, type, "
    "aliases, description, evidence_quote (a short quote from the chunk). Each "
    "relation has source, target (entity names from your entity list), "
    "relation_type, keywords, description, evidence_quote.\n\n"
    "Chunk:\n{chunk}"
)

_GLEAN_TEMPLATE = (
    "From the SAME chunk below, list ADDITIONAL entities and relations you "
    "missed previously. Already found entities: {known}. Use the same JSON "
    "schema and allowed types. If nothing else is present, return empty lists.\n\n"
    "Allowed entity types: {entity_types}.\n"
    "Allowed relation types: {relation_types}.\n\n"
    "Chunk:\n{chunk}"
)

_RESPONSE_SCHEMA = {
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
                    "aliases": {"type": "array", "items": {"type": "string"}},
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
                    "keywords": {"type": "array", "items": {"type": "string"}},
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


class _RawEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    type: str = ""
    aliases: list[str] = []
    description: str = ""
    evidence_quote: str = ""


class _RawRelation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    target: str
    relation_type: str = ""
    keywords: list[str] = []
    description: str = ""
    evidence_quote: str = ""


class _RawExtraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entities: list[_RawEntity] = []
    relations: list[_RawRelation] = []


# --------------------------------------------------------------------------- #
# Cache                                                                       #
# --------------------------------------------------------------------------- #


class ExtractionCache:
    """File-based cache for per-chunk extraction results."""

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir

    @staticmethod
    def key(content_hash: str, prompt_hash: str, provider_identity: str) -> str:
        """Return the cache key for a chunk under a given extraction contract."""
        raw = f"{content_hash}|{prompt_hash}|{provider_identity}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> LightExtractionResult | None:
        """Return a cached result or ``None`` on miss/corruption."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return LightExtractionResult.model_validate(payload["result"])
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return None

    def put(
        self,
        key: str,
        result: LightExtractionResult,
        *,
        provider_identity: str,
    ) -> None:
        """Persist a result for ``key``."""
        self._dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self._path(key),
            json.dumps(
                {"provider": provider_identity, "result": result.model_dump()},
                indent=2,
            ),
        )


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class ExtractionRun:
    """Aggregated extraction output across all processed chunks."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    tier: str = "fallback"


def run_extraction(
    chunks: list[LightChunk],
    *,
    config: ExtractionConfig,
    provider: TextProvider | None,
    provider_identity: str = "deterministic",
    cache: ExtractionCache | None = None,
    prompt_hash: str | None = None,
) -> ExtractionRun:
    """Extract entities/relations for every chunk (provider or fallback)."""
    use_provider = _provider_available(provider)
    run = ExtractionRun(tier="provider" if use_provider else "fallback")
    identity = provider_identity if use_provider else "deterministic"
    phash = prompt_hash or extraction_prompt_hash(config)

    for chunk in chunks:
        cache_key = ExtractionCache.key(chunk.content_hash, phash, identity)
        result: LightExtractionResult | None = None
        if cache is not None:
            result = cache.get(cache_key)
            if result is not None:
                run.cache_hits += 1
        if result is None:
            run.cache_misses += 1
            if use_provider and provider is not None:
                result = _extract_with_provider(chunk, provider, config)
            else:
                result = deterministic_extract_chunk(chunk, config)
            if cache is not None:
                cache.put(cache_key, result, provider_identity=identity)
        run.entities.extend(result.entities)
        run.relations.extend(result.relations)
        run.warnings.extend(result.warnings)
    return run


def _provider_available(provider: TextProvider | None) -> bool:
    if provider is None:
        return False
    ensure = getattr(provider, "ensure_available", None)
    if callable(ensure):
        try:
            ensure()
        except Exception:
            return False
    return True


def _extract_with_provider(
    chunk: LightChunk,
    provider: TextProvider,
    config: ExtractionConfig,
) -> LightExtractionResult:
    prompt = _PROMPT_TEMPLATE.format(
        entity_types=", ".join(config.entity_types),
        relation_types=", ".join(config.relation_types),
        chunk=chunk.text,
    )
    try:
        raw = _call_provider(provider, prompt)
    except Exception as exc:
        fallback = deterministic_extract_chunk(chunk, config)
        fallback.warnings.append(f"provider extraction failed for {chunk.id}: {exc}")
        return fallback

    entities, relations, warnings = _normalize_raw(raw, chunk)
    for _ in range(max(0, config.max_gleaning)):
        known = ", ".join(sorted({entity.name for entity in entities})) or "(none)"
        glean_prompt = _GLEAN_TEMPLATE.format(
            known=known,
            entity_types=", ".join(config.entity_types),
            relation_types=", ".join(config.relation_types),
            chunk=chunk.text,
        )
        try:
            extra = _call_provider(provider, glean_prompt)
        except Exception:
            break
        extra_entities, extra_relations, extra_warnings = _normalize_raw(extra, chunk)
        existing = {entity.name.casefold() for entity in entities}
        entities.extend(e for e in extra_entities if e.name.casefold() not in existing)
        relations.extend(extra_relations)
        warnings.extend(extra_warnings)

    return _validate_extraction(chunk, entities, relations, warnings)


def _call_provider(provider: TextProvider, prompt: str) -> _RawExtraction:
    response = provider.generate(
        ProviderRequest(
            prompt=prompt,
            system_prompt=(
                "Extract a faithful technical knowledge graph as JSON. "
                "Use only the chunk content."
            ),
            max_tokens=2048,
            response_schema=_RESPONSE_SCHEMA,
            response_schema_name="lightrag_extraction",
            reasoning_effort="low",
        )
    )
    return parse_model_payload(
        response.text, _RawExtraction, label="LightRAG extraction"
    )


def _normalize_raw(
    raw: _RawExtraction, chunk: LightChunk
) -> tuple[list[ExtractedEntity], list[ExtractedRelation], list[str]]:
    entities: list[ExtractedEntity] = []
    relations: list[ExtractedRelation] = []
    warnings: list[str] = []
    for raw_entity in raw.entities:
        name = raw_entity.name.strip()
        if not name:
            continue
        entities.append(
            ExtractedEntity(
                name=name,
                type=raw_entity.type.strip().upper() or "CONCEPT",
                description=raw_entity.description.strip(),
                aliases=[a.strip() for a in raw_entity.aliases if a.strip()],
                chunk_ids=[chunk.id],
                source_ids=[chunk.source_id],
                evidence_quote=raw_entity.evidence_quote.strip(),
            )
        )
    for raw_relation in raw.relations:
        source = raw_relation.source.strip()
        target = raw_relation.target.strip()
        if not source or not target:
            continue
        relations.append(
            ExtractedRelation(
                source=source,
                target=target,
                relation_type=raw_relation.relation_type.strip().upper()
                or "RELATED_TO",
                description=raw_relation.description.strip(),
                keywords=[k.strip() for k in raw_relation.keywords if k.strip()],
                chunk_ids=[chunk.id],
                source_ids=[chunk.source_id],
                evidence_quote=raw_relation.evidence_quote.strip(),
            )
        )
    return entities, relations, warnings


def _validate_extraction(
    chunk: LightChunk,
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
    warnings: list[str],
) -> LightExtractionResult:
    entity_names = {entity.name.casefold() for entity in entities}
    # Evidence-quote near-substring guardrail (warn, keep).
    for entity in entities:
        if entity.evidence_quote and not _near_substring(
            entity.evidence_quote, chunk.text
        ):
            warnings.append(
                f"entity {entity.name!r} evidence quote not found in {chunk.id}"
            )
    valid_relations: list[ExtractedRelation] = []
    for relation in relations:
        if (
            relation.source.casefold() not in entity_names
            or relation.target.casefold() not in entity_names
        ):
            warnings.append(
                f"relation {relation.source!r}->{relation.target!r} dropped: "
                f"endpoint not in extracted entities ({chunk.id})"
            )
            continue
        if relation.evidence_quote and not _near_substring(
            relation.evidence_quote, chunk.text
        ):
            warnings.append(
                f"relation {relation.source!r}->{relation.target!r} evidence "
                f"quote not found in {chunk.id}"
            )
        valid_relations.append(relation)
    return LightExtractionResult(
        entities=entities, relations=valid_relations, warnings=warnings
    )


def _near_substring(quote: str, text: str) -> bool:
    quote_norm = " ".join(quote.split()).casefold()
    text_norm = " ".join(text.split()).casefold()
    if not quote_norm:
        return True
    if quote_norm in text_norm:
        return True
    return fuzz.partial_ratio(quote_norm, text_norm) >= _EVIDENCE_FUZZ_THRESHOLD


# --------------------------------------------------------------------------- #
# Deterministic (provider-free) fallback                                      #
# --------------------------------------------------------------------------- #


def deterministic_extract_chunk(
    chunk: LightChunk, config: ExtractionConfig
) -> LightExtractionResult:
    """Heuristic, provider-free extraction (labeled as a fallback)."""
    counter: Counter[str] = Counter()
    for match in _CAPITALIZED_PHRASE.finditer(chunk.text):
        phrase = " ".join(match.group(1).split()).strip(" .,:;")
        if len(phrase) >= 2:
            counter[phrase] += 1
    for match in _ACRONYM.finditer(chunk.text):
        counter[match.group(1)] += 1

    ranked = [name for name, _ in counter.most_common(_MAX_FALLBACK_ENTITIES)]
    entities = [
        ExtractedEntity(
            name=name,
            type="CONCEPT",
            description="",
            chunk_ids=[chunk.id],
            source_ids=[chunk.source_id],
            confidence=0.3,
        )
        for name in ranked
    ]
    relations: list[ExtractedRelation] = []
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            if len(relations) >= _MAX_FALLBACK_RELATIONS:
                break
            relations.append(
                ExtractedRelation(
                    source=ranked[i],
                    target=ranked[j],
                    relation_type="RELATED_TO",
                    description="",
                    chunk_ids=[chunk.id],
                    source_ids=[chunk.source_id],
                    weight=1.0,
                    confidence=0.2,
                )
            )
    return LightExtractionResult(
        entities=entities,
        relations=relations,
        warnings=[],
    )
