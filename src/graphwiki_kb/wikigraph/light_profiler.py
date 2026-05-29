"""Deterministic profiling of LightRAG entities and relations.

LightRAG profiles each entity/relation into a key-value text block: keys drive
retrieval, values summarize the supporting snippets and seed answer generation.
This module builds that text deterministically (no provider calls) so the index
is reproducible; a richer provider-backed summary can be layered on later
without changing the embedding keys.
"""

from __future__ import annotations

from collections import defaultdict

from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    RelationProfile,
)

_MAX_EVIDENCE_SNIPPETS = 3
_SNIPPET_CHARS = 200
_DESCRIPTION_EMBED_CHARS = 240


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) > _SNIPPET_CHARS:
        return flat[:_SNIPPET_CHARS].rstrip() + "..."
    return flat


def _evidence_snippets(
    chunk_ids: list[str], chunk_by_id: dict[str, LightChunk]
) -> list[str]:
    snippets: list[str] = []
    for chunk_id in chunk_ids:
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            continue
        snippets.append(f"({chunk.source_ref}) {_snippet(chunk.text)}")
        if len(snippets) >= _MAX_EVIDENCE_SNIPPETS:
            break
    return snippets


def build_entity_embedding_text(entity: EntityProfile) -> str:
    """Build a concise embedding key for an entity (name + aliases + keywords)."""
    parts = [entity.canonical_name, *entity.aliases, entity.type, *entity.keywords]
    if entity.description:
        parts.append(entity.description[:_DESCRIPTION_EMBED_CHARS])
    return " ".join(part for part in parts if part).strip()


def build_relation_embedding_text(
    relation: RelationProfile, source_name: str, target_name: str
) -> str:
    """Build a concise embedding key for a relation."""
    parts = [
        source_name,
        relation.relation_type.replace("_", " ").lower(),
        target_name,
        *relation.keywords,
    ]
    if relation.description:
        parts.append(relation.description[:_DESCRIPTION_EMBED_CHARS])
    return " ".join(part for part in parts if part).strip()


def profile_index(
    entities: list[EntityProfile],
    relations: list[RelationProfile],
    chunks: list[LightChunk],
    *,
    updated_at: str,
) -> None:
    """Fill ``profile_text``/``embedding_text``/``updated_at`` in place."""
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    name_by_id = {entity.id: entity.canonical_name for entity in entities}
    relations_by_entity: dict[str, list[RelationProfile]] = defaultdict(list)
    for relation in relations:
        relations_by_entity[relation.source_entity_id].append(relation)
        relations_by_entity[relation.target_entity_id].append(relation)

    for entity in entities:
        related = relations_by_entity.get(entity.id, [])
        entity.profile_text = _entity_profile_text(
            entity, related, name_by_id, chunk_by_id
        )
        entity.embedding_text = build_entity_embedding_text(entity)
        entity.updated_at = updated_at

    for relation in relations:
        source_name = name_by_id.get(
            relation.source_entity_id, relation.source_entity_id
        )
        target_name = name_by_id.get(
            relation.target_entity_id, relation.target_entity_id
        )
        relation.profile_text = _relation_profile_text(
            relation, source_name, target_name, chunk_by_id
        )
        relation.embedding_text = build_relation_embedding_text(
            relation, source_name, target_name
        )
        relation.updated_at = updated_at


def _entity_profile_text(
    entity: EntityProfile,
    related: list[RelationProfile],
    name_by_id: dict[str, str],
    chunk_by_id: dict[str, LightChunk],
) -> str:
    lines = [f"Entity: {entity.canonical_name}", f"Type: {entity.type}"]
    if entity.aliases:
        lines.append("Aliases: " + ", ".join(entity.aliases))
    if entity.description:
        lines.append(f"Description: {entity.description}")
    snippets = _evidence_snippets(entity.chunk_ids, chunk_by_id)
    if snippets:
        lines.append("Evidence snippets:")
        lines.extend(f"- {snippet}" for snippet in snippets)
    if related:
        lines.append("Related relations:")
        for relation in related[:_MAX_EVIDENCE_SNIPPETS]:
            src = name_by_id.get(relation.source_entity_id, relation.source_entity_id)
            tgt = name_by_id.get(relation.target_entity_id, relation.target_entity_id)
            lines.append(f"- {src} {relation.relation_type} {tgt}")
    return "\n".join(lines)


def _relation_profile_text(
    relation: RelationProfile,
    source_name: str,
    target_name: str,
    chunk_by_id: dict[str, LightChunk],
) -> str:
    lines = [
        f"Relation: {source_name} {relation.relation_type} {target_name}",
        f"Type: {relation.relation_type}",
    ]
    if relation.keywords:
        lines.append("Keywords: " + ", ".join(relation.keywords))
    if relation.description:
        lines.append(f"Description: {relation.description}")
    snippets = _evidence_snippets(relation.chunk_ids, chunk_by_id)
    if snippets:
        lines.append("Evidence snippets:")
        lines.extend(f"- {snippet}" for snippet in snippets)
    return "\n".join(lines)
