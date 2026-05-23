"""Entity and relation deduplication for LightRAG-style profiles."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

from graphwiki_kb.services.project_service import slugify, utc_now_iso
from graphwiki_kb.wikigraph.light_extractor import LightExtractionResult
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    ExtractedEntity,
    ExtractedRelation,
    RelationProfile,
)

_INVERSE_RELATIONS: dict[str, str] = {
    "USES": "USED_BY",
    "USED_BY": "USES",
    "IMPROVES_OVER": "IS_IMPROVED_BY",
    "IS_IMPROVED_BY": "IMPROVES_OVER",
    "EVALUATES_ON": "IS_EVALUATION_DATASET_FOR",
    "IS_EVALUATION_DATASET_FOR": "EVALUATES_ON",
}


@dataclass(frozen=True)
class LightDedupeConfig:
    """Dedupe thresholds and behavior."""

    fuzzy_threshold: int = 88
    description_token_cap: int = 400


def dedupe_and_profile(
    extracted: list[tuple[str, LightExtractionResult]],
    *,
    existing_entities: list[EntityProfile] | None = None,
    existing_relations: list[RelationProfile] | None = None,
    config: LightDedupeConfig | None = None,
) -> tuple[list[EntityProfile], list[RelationProfile]]:
    """Merge extracted entities/relations into canonical profiles."""
    cfg = config or LightDedupeConfig()
    entity_index: dict[str, EntityProfile] = {
        profile.id: profile for profile in (existing_entities or [])
    }
    canonical_by_key: dict[str, str] = {
        _entity_key(profile.canonical_name): profile.id
        for profile in entity_index.values()
    }
    alias_to_id: dict[str, str] = {}
    for profile in entity_index.values():
        for alias in [profile.canonical_name, *profile.aliases]:
            alias_to_id[_entity_key(alias)] = profile.id

    timestamp = utc_now_iso()
    for _chunk_id, result in extracted:
        for entity in result.entities:
            profile_id = _upsert_entity(
                entity,
                entity_index=entity_index,
                canonical_by_key=canonical_by_key,
                alias_to_id=alias_to_id,
                config=cfg,
                timestamp=timestamp,
            )
            canonical_by_key[_entity_key(entity.name)] = profile_id

    relation_index: dict[str, RelationProfile] = {
        profile.id: profile for profile in (existing_relations or [])
    }
    for _chunk_id, result in extracted:
        for relation in result.relations:
            _upsert_relation(
                relation,
                entity_index=entity_index,
                alias_to_id=alias_to_id,
                relation_index=relation_index,
                timestamp=timestamp,
            )

    entities = list(entity_index.values())
    relations = list(relation_index.values())
    _attach_relation_ids(entities, relations)
    for profile in entities:
        profile.profile_text = _build_entity_profile_text(profile, relations)
        profile.embedding_text = _build_entity_embedding_text(profile)
    for profile in relations:
        profile.profile_text = _build_relation_profile_text(profile, entity_index)
        profile.embedding_text = _build_relation_embedding_text(profile, entity_index)
    return entities, relations


def _upsert_entity(
    entity: ExtractedEntity,
    *,
    entity_index: dict[str, EntityProfile],
    canonical_by_key: dict[str, str],
    alias_to_id: dict[str, str],
    config: LightDedupeConfig,
    timestamp: str,
) -> str:
    key = _entity_key(entity.name)
    profile_id = canonical_by_key.get(key) or alias_to_id.get(key)
    if profile_id is None:
        profile_id = _find_fuzzy_entity(
            entity.name, entity_index, config.fuzzy_threshold
        )
    if profile_id is None:
        profile_id = f"entity:{slugify(entity.name)}"
        entity_index[profile_id] = EntityProfile(
            id=profile_id,
            canonical_name=entity.name,
            type=entity.type,
            aliases=list(dict.fromkeys(entity.aliases)),
            description=entity.description,
            profile_text="",
            keywords=[],
            chunk_ids=list(entity.chunk_ids),
            source_ids=list(entity.source_ids),
            relation_ids=[],
            embedding_text="",
            updated_at=timestamp,
        )
        canonical_by_key[_entity_key(entity.name)] = profile_id
        for alias in entity.aliases:
            alias_to_id[_entity_key(alias)] = profile_id
        return profile_id

    profile = entity_index[profile_id]
    profile.chunk_ids = sorted(set(profile.chunk_ids).union(entity.chunk_ids))
    profile.source_ids = sorted(set(profile.source_ids).union(entity.source_ids))
    profile.aliases = sorted(
        set(profile.aliases)
        .union(entity.aliases)
        .union({_alias_if_distinct(profile.canonical_name, entity.name)})
    )
    profile.description = _merge_descriptions(
        profile.description, entity.description, cap=config.description_token_cap
    )
    profile.updated_at = timestamp
    alias_to_id[_entity_key(entity.name)] = profile_id
    return profile_id


def _upsert_relation(
    relation: ExtractedRelation,
    *,
    entity_index: dict[str, EntityProfile],
    alias_to_id: dict[str, str],
    relation_index: dict[str, RelationProfile],
    timestamp: str,
) -> None:
    source_id = alias_to_id.get(_entity_key(relation.source))
    target_id = alias_to_id.get(_entity_key(relation.target))
    if not source_id or not target_id:
        return
    normalized_type = _normalize_relation_type(relation.relation_type)
    relation_id = _relation_id(source_id, target_id, normalized_type)
    if relation_id not in relation_index:
        relation_index[relation_id] = RelationProfile(
            id=relation_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type=normalized_type,
            description=relation.description,
            profile_text="",
            keywords=list(relation.keywords),
            chunk_ids=list(relation.chunk_ids),
            source_ids=list(relation.source_ids),
            embedding_text="",
            weight=relation.weight,
            updated_at=timestamp,
        )
        return
    profile = relation_index[relation_id]
    profile.chunk_ids = sorted(set(profile.chunk_ids).union(relation.chunk_ids))
    profile.source_ids = sorted(set(profile.source_ids).union(relation.source_ids))
    profile.keywords = sorted(set(profile.keywords).union(relation.keywords))
    profile.description = _merge_descriptions(profile.description, relation.description)
    profile.weight = max(profile.weight, relation.weight)
    profile.updated_at = timestamp


def _relation_id(source_id: str, target_id: str, relation_type: str) -> str:
    digest = hashlib.sha256(
        f"{source_id}|{target_id}|{relation_type}".encode()
    ).hexdigest()[:16]
    return f"relation:{digest}"


def _normalize_relation_type(relation_type: str) -> str:
    normalized = relation_type.strip().upper().replace(" ", "_")
    inverse = _INVERSE_RELATIONS.get(normalized)
    if inverse and normalized > inverse:
        return inverse
    return normalized


def _entity_key(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _find_fuzzy_entity(
    name: str,
    entity_index: dict[str, EntityProfile],
    threshold: int,
) -> str | None:
    best_id: str | None = None
    best_score = 0
    for profile in entity_index.values():
        candidates = [profile.canonical_name, *profile.aliases]
        for candidate in candidates:
            score = fuzz.token_sort_ratio(name, candidate)
            if score >= threshold and score > best_score:
                if profile.type and profile.type:
                    best_score = score
                    best_id = profile.id
    return best_id


def _alias_if_distinct(left: str, right: str) -> str:
    return right if _entity_key(left) != _entity_key(right) else ""


def _merge_descriptions(left: str, right: str, *, cap: int = 400) -> str:
    parts = [part.strip() for part in (left, right) if part.strip()]
    merged = " ".join(dict.fromkeys(parts))
    words = merged.split()
    if len(words) > cap:
        return " ".join(words[:cap]) + " ..."
    return merged


def _attach_relation_ids(
    entities: list[EntityProfile], relations: list[RelationProfile]
) -> None:
    by_id = {entity.id: entity for entity in entities}
    for entity in by_id.values():
        entity.relation_ids = []
    for relation in relations:
        source = by_id.get(relation.source_entity_id)
        target = by_id.get(relation.target_entity_id)
        if source is not None:
            source.relation_ids.append(relation.id)
        if target is not None:
            target.relation_ids.append(relation.id)


def _build_entity_profile_text(
    profile: EntityProfile, relations: list[RelationProfile]
) -> str:
    lines = [
        f"Entity: {profile.canonical_name}",
        f"Type: {profile.type}",
    ]
    if profile.aliases:
        lines.append(f"Aliases: {', '.join(profile.aliases)}")
    lines.append(f"Description: {profile.description}")
    lines.append("Sources:")
    for source_id in profile.source_ids[:8]:
        lines.append(f"- source_id={source_id}")
    lines.append("Evidence snippets:")
    for chunk_id in profile.chunk_ids[:6]:
        lines.append(f"- chunk={chunk_id}")
    related = [
        rel
        for rel in relations
        if rel.source_entity_id == profile.id or rel.target_entity_id == profile.id
    ]
    if related:
        lines.append("Related relations:")
        for rel in related[:8]:
            lines.append(f"- {rel.relation_type} ({rel.id})")
    return "\n".join(lines)


def _build_entity_embedding_text(profile: EntityProfile) -> str:
    parts = [
        profile.canonical_name,
        *profile.aliases,
        profile.type,
        profile.description,
    ]
    return " ".join(part for part in parts if part).strip()


def _build_relation_profile_text(
    profile: RelationProfile, entity_index: dict[str, EntityProfile]
) -> str:
    source = entity_index.get(profile.source_entity_id)
    target = entity_index.get(profile.target_entity_id)
    source_name = source.canonical_name if source else profile.source_entity_id
    target_name = target.canonical_name if target else profile.target_entity_id
    lines = [
        f"Relation: {source_name} {profile.relation_type} {target_name}",
        f"Type: {profile.relation_type}",
        f"Keywords: {', '.join(profile.keywords)}",
        f"Description: {profile.description}",
        "Sources:",
    ]
    for source_id in profile.source_ids[:8]:
        lines.append(f"- source_id={source_id}")
    for chunk_id in profile.chunk_ids[:6]:
        lines.append(f"- chunk={chunk_id}")
    return "\n".join(lines)


def _build_relation_embedding_text(
    profile: RelationProfile, entity_index: dict[str, EntityProfile]
) -> str:
    source = entity_index.get(profile.source_entity_id)
    target = entity_index.get(profile.target_entity_id)
    source_name = source.canonical_name if source else profile.source_entity_id
    target_name = target.canonical_name if target else profile.target_entity_id
    parts = [
        source_name,
        profile.relation_type,
        target_name,
        *profile.keywords,
        profile.description,
    ]
    return " ".join(part for part in parts if part).strip()
