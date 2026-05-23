"""Chunk-level entity/relation extraction for LightRAG-style indexing."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from graphwiki_kb.providers.base import ProviderRequest, TextProvider
from graphwiki_kb.wikigraph.light_models import (
    ExtractedEntity,
    ExtractedRelation,
    LightChunk,
)

_EXTRACTION_PROMPT_VERSION = "lightrag-extract-v1"


class LightExtractionResult(BaseModel):
    """Structured extraction output for one chunk."""

    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    warnings: list[str] = Field(default_factory=list)


def extraction_prompt_hash(
    *,
    entity_types: tuple[str, ...],
    relation_types: tuple[str, ...],
) -> str:
    """Stable hash for extraction prompt configuration."""
    payload = {
        "version": _EXTRACTION_PROMPT_VERSION,
        "entity_types": list(entity_types),
        "relation_types": list(relation_types),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LightExtractionConfig:
    """Runtime extraction settings."""

    entity_types: tuple[str, ...]
    relation_types: tuple[str, ...]
    entity_extract_max_gleaning: int = 1


def extract_entities_and_relations(
    chunk: LightChunk,
    *,
    provider: TextProvider | None,
    config: LightExtractionConfig,
    cache_dir: Path | None = None,
    provider_identity: str | None = None,
) -> LightExtractionResult:
    """Extract entities/relations from a chunk with optional cache."""
    prompt_hash = extraction_prompt_hash(
        entity_types=config.entity_types,
        relation_types=config.relation_types,
    )
    cache_key = _cache_key(
        chunk=chunk,
        prompt_hash=prompt_hash,
        provider_identity=provider_identity or "deterministic",
        config=config,
    )
    if cache_dir is not None:
        cached = _read_cache(cache_dir, cache_key)
        if cached is not None:
            return cached

    if provider is not None:
        result = _extract_with_provider(chunk, provider=provider, config=config)
    else:
        result = _extract_deterministic(chunk, config=config)

    result = _attach_provenance(chunk, result)
    result = _validate_evidence(chunk, result)

    if cache_dir is not None:
        _write_cache(cache_dir, cache_key, result, provider_identity=provider_identity)
    return result


def _extract_with_provider(
    chunk: LightChunk,
    *,
    provider: TextProvider,
    config: LightExtractionConfig,
) -> LightExtractionResult:
    entity_types = ", ".join(config.entity_types)
    relation_types = " | ".join(config.relation_types)
    prompt = (
        "You are extracting a technical knowledge graph from one source chunk.\n\n"
        f"Entity types:\n{entity_types}\n\n"
        f"Relation types:\n{relation_types}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "entities": [\n'
        '    {"name": "...", "type": "...", "aliases": ["..."], '
        '"description": "...", "evidence_quote": "..."}\n'
        "  ],\n"
        '  "relations": [\n'
        '    {"source": "...", "target": "...", "relation_type": "...", '
        '"keywords": ["..."], "description": "...", "evidence_quote": "..."}\n'
        "  ]\n"
        "}\n\n"
        f"Chunk:\n{chunk.text}\n"
    )
    response = provider.generate(
        ProviderRequest(
            prompt=prompt,
            system_prompt="Return strict JSON only.",
            max_tokens=2048,
            response_schema_name="light_extraction",
        )
    )
    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError:
        payload = _parse_json_loose(response.text)
    return _payload_to_result(payload, chunk=chunk, config=config)


def _extract_deterministic(
    chunk: LightChunk,
    *,
    config: LightExtractionConfig,
) -> LightExtractionResult:
    """Provider-free heuristic extraction for tests and fallback builds."""
    warnings: list[str] = []
    entities: list[ExtractedEntity] = []
    relations: list[ExtractedRelation] = []
    names = _candidate_entity_names(chunk.text)
    for name in names[:12]:
        entity_type = _guess_entity_type(name, config.entity_types)
        entities.append(
            ExtractedEntity(
                name=name,
                type=entity_type,
                description=f"{name} mentioned in source chunk.",
                aliases=_aliases_for(name),
                chunk_ids=[chunk.id],
                source_ids=[chunk.source_id],
            )
        )
    entity_names = {entity.name for entity in entities}
    if len(entities) >= 2:
        left, right = entities[0].name, entities[1].name
        relations.append(
            ExtractedRelation(
                source=left,
                target=right,
                relation_type=(
                    config.relation_types[0] if config.relation_types else "RELATED_TO"
                ),
                description=f"{left} is discussed alongside {right}.",
                keywords=["co-occurrence"],
                chunk_ids=[chunk.id],
                source_ids=[chunk.source_id],
            )
        )
    for relation in list(relations):
        if relation.source not in entity_names or relation.target not in entity_names:
            warnings.append(
                f"dropped relation with unknown endpoints: "
                f"{relation.source} -> {relation.target}"
            )
    return LightExtractionResult(
        entities=entities, relations=relations, warnings=warnings
    )


def _payload_to_result(
    payload: dict[str, Any],
    *,
    chunk: LightChunk,
    config: LightExtractionConfig,
) -> LightExtractionResult:
    entities: list[ExtractedEntity] = []
    for raw in payload.get("entities", []) or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        if not name:
            continue
        entity_type = str(raw.get("type", "CLAIM")).strip().upper()
        if entity_type not in config.entity_types:
            entity_type = config.entity_types[0] if config.entity_types else "CLAIM"
        entities.append(
            ExtractedEntity(
                name=name,
                type=entity_type,
                description=str(raw.get("description", "")).strip() or name,
                aliases=[
                    str(a).strip() for a in raw.get("aliases", []) if str(a).strip()
                ],
                chunk_ids=[chunk.id],
                source_ids=[chunk.source_id],
            )
        )
    entity_names = {entity.name for entity in entities}
    relations: list[ExtractedRelation] = []
    for raw in payload.get("relations", []) or []:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source", "")).strip()
        target = str(raw.get("target", "")).strip()
        if not source or not target:
            continue
        if source not in entity_names or target not in entity_names:
            continue
        relation_type = str(raw.get("relation_type", "SUPPORTS")).strip().upper()
        relations.append(
            ExtractedRelation(
                source=source,
                target=target,
                relation_type=relation_type,
                description=str(raw.get("description", "")).strip()
                or f"{source} {relation_type} {target}",
                keywords=[
                    str(k).strip() for k in raw.get("keywords", []) if str(k).strip()
                ],
                chunk_ids=[chunk.id],
                source_ids=[chunk.source_id],
            )
        )
    return LightExtractionResult(entities=entities, relations=relations)


def _attach_provenance(
    chunk: LightChunk, result: LightExtractionResult
) -> LightExtractionResult:
    for entity in result.entities:
        if chunk.id not in entity.chunk_ids:
            entity.chunk_ids.append(chunk.id)
        if chunk.source_id not in entity.source_ids:
            entity.source_ids.append(chunk.source_id)
    for relation in result.relations:
        if chunk.id not in relation.chunk_ids:
            relation.chunk_ids.append(chunk.id)
        if chunk.source_id not in relation.source_ids:
            relation.source_ids.append(chunk.source_id)
    return result


def _validate_evidence(
    chunk: LightChunk, result: LightExtractionResult
) -> LightExtractionResult:
    warnings = list(result.warnings)
    lowered = chunk.text.casefold()
    for entity in result.entities:
        if entity.name.casefold() not in lowered:
            warnings.append(f"entity name not found verbatim in chunk: {entity.name}")
    return result.model_copy(update={"warnings": warnings})


def _candidate_entity_names(text: str) -> list[str]:
    pattern = re.compile(
        r"\b(?:[A-Z][a-z]+(?:[-/][A-Z][a-z]+)*|[A-Z]{2,}(?:[-/][A-Z]{2,})*)\b"
    )
    seen: set[str] = set()
    names: list[str] = []
    for match in pattern.findall(text):
        if match in seen:
            continue
        seen.add(match)
        names.append(match)
    return names


def _guess_entity_type(name: str, entity_types: tuple[str, ...]) -> str:
    if name.isupper() and len(name) <= 6:
        return "MODEL" if "MODEL" in entity_types else entity_types[0]
    if "dataset" in name.casefold():
        return "DATASET" if "DATASET" in entity_types else entity_types[0]
    return entity_types[0] if entity_types else "CLAIM"


def _aliases_for(name: str) -> list[str]:
    if "-" in name:
        return [name.replace("-", " ")]
    return []


def _find_quote(text: str, needle: str) -> str:
    idx = text.find(needle)
    if idx < 0:
        return needle[:80]
    return text[max(0, idx - 20) : idx + len(needle) + 20].strip()


def _parse_json_loose(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    return {"entities": [], "relations": []}


def _cache_key(
    *,
    chunk: LightChunk,
    prompt_hash: str,
    provider_identity: str,
    config: LightExtractionConfig,
) -> str:
    payload = "|".join(
        [
            chunk.content_hash,
            prompt_hash,
            provider_identity,
            ",".join(config.entity_types),
            ",".join(config.relation_types),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(cache_dir: Path, key: str) -> LightExtractionResult | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    result_payload = payload.get("result", payload)
    return LightExtractionResult.model_validate(result_payload)


def _write_cache(
    cache_dir: Path,
    key: str,
    result: LightExtractionResult,
    *,
    provider_identity: str | None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    path.write_text(
        json.dumps(
            {
                "result": result.model_dump(),
                "provider": provider_identity,
                "prompt_hash": extraction_prompt_hash(
                    entity_types=(),
                    relation_types=(),
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
