"""Canonical entity/relation deduplication for the LightRAG backend.

LightRAG deduplicates identical entities and relations extracted from different
chunks to shrink the graph and keep retrieval clean. This implementation goes
beyond exact matching: it canonicalizes via Unicode normalization, a lowercase
key, an alias table, acronym expansion, and RapidFuzz similarity (gated by type
compatibility). Relations are canonicalized by their (source, type, target)
identity with inverse-type normalization so ``DPR USED_BY RAG`` merges with
``RAG USES DPR``.

Provenance is preserved more strongly than the paper baseline: merges union
``source_ids``/``chunk_ids``/``keywords`` and append descriptions up to a cap so
no evidence chunk reference is ever dropped.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from rapidfuzz import fuzz
from slugify import slugify

from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    ExtractedEntity,
    ExtractedRelation,
    RelationProfile,
)

# Relation types whose inverse should be normalized to a canonical direction so
# duplicates merge regardless of which way the LLM phrased them.
_INVERSE_RELATION_TYPES: dict[str, str] = {
    "USED_BY": "USES",
    "IS_IMPROVED_BY": "IMPROVES_OVER",
    "IS_EVALUATION_DATASET_FOR": "EVALUATES_ON",
    "INTRODUCED_BY": "INTRODUCES",
    "DEPENDED_ON_BY": "DEPENDS_ON",
}


@dataclass(frozen=True)
class DedupeConfig:
    """Tuning for entity/relation canonicalization."""

    fuzzy_threshold: int = 88
    description_char_cap: int = 1500


def _canonical_key(name: str) -> str:
    """Return a normalized matching key (NFKC, lowercase, punctuation-stripped)."""
    normalized = unicodedata.normalize("NFKC", name)
    normalized = normalized.casefold().strip()
    kept = [ch if (ch.isalnum() or ch.isspace()) else " " for ch in normalized]
    return " ".join("".join(kept).split())


def _is_acronym(text: str) -> bool:
    stripped = text.strip()
    return 2 <= len(stripped) <= 6 and stripped.isupper() and stripped.isalpha()


def _initials(name: str) -> str:
    return "".join(word[0] for word in name.split() if word).upper()


@dataclass
class _EntityAccumulator:
    """Mutable accumulator merged into a final :class:`EntityProfile`."""

    entity_id: str
    canonical_name: str
    key: str
    type: str
    aliases: list[str] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    type_votes: dict[str, int] = field(default_factory=dict)
    name_lengths: dict[str, int] = field(default_factory=dict)


def _extend_unique(target: list[str], values: list[str]) -> None:
    seen = set(target)
    for value in values:
        if value and value not in seen:
            target.append(value)
            seen.add(value)


def _types_compatible(a: str, b: str) -> bool:
    return not a or not b or a == b


class EntityDeduper:
    """Accumulates extracted entities into canonical profiles."""

    def __init__(self, config: DedupeConfig | None = None) -> None:
        self._config = config or DedupeConfig()
        self._accumulators: list[_EntityAccumulator] = []
        self._key_to_acc: dict[str, _EntityAccumulator] = {}
        self._alias_to_acc: dict[str, _EntityAccumulator] = {}
        self._used_ids: set[str] = set()

    def add(self, entity: ExtractedEntity) -> _EntityAccumulator:
        """Merge ``entity`` into an existing accumulator or create a new one."""
        key = _canonical_key(entity.name)
        if not key:
            key = entity.name.casefold().strip() or "entity"
        etype = entity.type.strip().upper()
        alias_keys = [
            _canonical_key(alias) for alias in entity.aliases if _canonical_key(alias)
        ]
        acc = self._find_match(entity.name, key, etype, alias_keys)
        if acc is None:
            acc = self._new_accumulator(entity.name, key, etype)
        self._merge_into(acc, entity, key)
        return acc

    def _find_match(
        self, name: str, key: str, etype: str, alias_keys: list[str] | None = None
    ) -> _EntityAccumulator | None:
        # Direct hit on the canonical key or on any declared alias key.
        candidate_keys = [key, *(alias_keys or [])]
        for candidate in candidate_keys:
            direct = self._key_to_acc.get(candidate) or self._alias_to_acc.get(
                candidate
            )
            if direct is not None and _types_compatible(direct.type, etype):
                return direct
        # Acronym expansion (both directions).
        if _is_acronym(name):
            for acc in self._accumulators:
                if _initials(acc.canonical_name) == name.strip().upper():
                    if _types_compatible(acc.type, etype):
                        return acc
        else:
            initials = _initials(name)
            for acc in self._accumulators:
                if _is_acronym(acc.canonical_name) and acc.key == initials.casefold():
                    if _types_compatible(acc.type, etype):
                        return acc
        # Fuzzy match gated by type compatibility.
        best: _EntityAccumulator | None = None
        best_score = 0.0
        for acc in self._accumulators:
            if not _types_compatible(acc.type, etype):
                continue
            score = fuzz.token_sort_ratio(key, acc.key)
            if score >= self._config.fuzzy_threshold and score > best_score:
                best = acc
                best_score = score
        return best

    def _new_accumulator(self, name: str, key: str, etype: str) -> _EntityAccumulator:
        entity_id = self._allocate_id(name)
        acc = _EntityAccumulator(
            entity_id=entity_id,
            canonical_name=name,
            key=key,
            type=etype,
        )
        self._accumulators.append(acc)
        self._key_to_acc[key] = acc
        return acc

    def _allocate_id(self, name: str) -> str:
        base = slugify(name) or "entity"
        candidate = f"entity:{base}"
        suffix = 2
        while candidate in self._used_ids:
            candidate = f"entity:{base}-{suffix}"
            suffix += 1
        self._used_ids.add(candidate)
        return candidate

    def _merge_into(
        self, acc: _EntityAccumulator, entity: ExtractedEntity, key: str
    ) -> None:
        etype = entity.type.strip().upper()
        if etype:
            acc.type_votes[etype] = acc.type_votes.get(etype, 0) + 1
            acc.type = max(acc.type_votes, key=lambda t: (acc.type_votes[t], t))
        # Track the canonical surface form (prefer the longest non-acronym name).
        acc.name_lengths[entity.name] = len(entity.name)
        candidate = _preferred_name(acc.canonical_name, entity.name)
        if candidate != acc.canonical_name:
            acc.canonical_name = candidate
        # Aliases: every surface form other than the canonical name.
        _extend_unique(acc.aliases, [entity.name, *entity.aliases])
        if entity.description.strip():
            _extend_unique(acc.descriptions, [entity.description.strip()])
        _extend_unique(acc.chunk_ids, entity.chunk_ids)
        _extend_unique(acc.source_ids, entity.source_ids)
        self._key_to_acc[key] = acc
        for alias in [entity.name, *entity.aliases]:
            alias_key = _canonical_key(alias)
            if alias_key:
                self._alias_to_acc[alias_key] = acc

    def build(self) -> tuple[list[EntityProfile], dict[str, str]]:
        """Return canonical entity profiles plus an alias/name -> id map."""
        profiles: list[EntityProfile] = []
        name_to_id: dict[str, str] = {}
        for acc in self._accumulators:
            aliases = [a for a in acc.aliases if a != acc.canonical_name]
            description = _join_capped(
                acc.descriptions, self._config.description_char_cap
            )
            profiles.append(
                EntityProfile(
                    id=acc.entity_id,
                    canonical_name=acc.canonical_name,
                    type=acc.type or "CONCEPT",
                    aliases=aliases,
                    description=description,
                    keywords=list(acc.keywords),
                    chunk_ids=list(acc.chunk_ids),
                    source_ids=list(acc.source_ids),
                )
            )
            for surface in [acc.canonical_name, *acc.aliases]:
                surface_key = _canonical_key(surface)
                if surface_key:
                    name_to_id.setdefault(surface_key, acc.entity_id)
        return profiles, name_to_id


def _preferred_name(current: str, candidate: str) -> str:
    """Prefer a longer, non-acronym surface form as the canonical name."""
    if _is_acronym(current) and not _is_acronym(candidate):
        return candidate
    if not _is_acronym(current) and _is_acronym(candidate):
        return current
    if len(candidate) > len(current):
        return candidate
    return current


def _join_capped(parts: list[str], cap: int) -> str:
    joined = ""
    for part in parts:
        if not joined:
            joined = part
        elif len(joined) + len(part) + 1 <= cap:
            joined = f"{joined} {part}"
        else:
            break
    return joined


@dataclass
class _RelationAccumulator:
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    descriptions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    weight: float = 0.0


class RelationDeduper:
    """Canonicalizes and merges extracted relations by (src, type, tgt)."""

    def __init__(
        self,
        name_to_id: dict[str, str],
        *,
        config: DedupeConfig | None = None,
    ) -> None:
        self._name_to_id = name_to_id
        self._config = config or DedupeConfig()
        self._accumulators: dict[str, _RelationAccumulator] = {}

    def resolve_entity(self, name: str) -> str | None:
        """Resolve an entity surface form to its canonical id."""
        return self._name_to_id.get(_canonical_key(name))

    def add(self, relation: ExtractedRelation) -> _RelationAccumulator | None:
        """Merge a relation; returns ``None`` when endpoints are unresolved."""
        src_id = self.resolve_entity(relation.source)
        tgt_id = self.resolve_entity(relation.target)
        if src_id is None or tgt_id is None or src_id == tgt_id:
            return None
        rel_type = relation.relation_type.strip().upper() or "RELATED_TO"
        if rel_type in _INVERSE_RELATION_TYPES:
            src_id, tgt_id = tgt_id, src_id
            rel_type = _INVERSE_RELATION_TYPES[rel_type]
        rel_id = f"relation:{src_id.split(':')[-1]}:{rel_type.lower()}:{tgt_id.split(':')[-1]}"
        acc = self._accumulators.get(rel_id)
        if acc is None:
            acc = _RelationAccumulator(
                relation_id=rel_id,
                source_entity_id=src_id,
                target_entity_id=tgt_id,
                relation_type=rel_type,
            )
            self._accumulators[rel_id] = acc
        if relation.description.strip():
            _extend_unique(acc.descriptions, [relation.description.strip()])
        _extend_unique(acc.keywords, relation.keywords)
        _extend_unique(acc.chunk_ids, relation.chunk_ids)
        _extend_unique(acc.source_ids, relation.source_ids)
        acc.weight += float(relation.weight)
        return acc

    def build(self) -> list[RelationProfile]:
        """Return canonical relation profiles in deterministic id order."""
        profiles: list[RelationProfile] = []
        for rel_id in sorted(self._accumulators):
            acc = self._accumulators[rel_id]
            profiles.append(
                RelationProfile(
                    id=acc.relation_id,
                    source_entity_id=acc.source_entity_id,
                    target_entity_id=acc.target_entity_id,
                    relation_type=acc.relation_type,
                    description=_join_capped(
                        acc.descriptions, self._config.description_char_cap
                    ),
                    keywords=list(acc.keywords),
                    chunk_ids=list(acc.chunk_ids),
                    source_ids=list(acc.source_ids),
                    weight=acc.weight or 1.0,
                )
            )
        return profiles


def dedupe_entities_and_relations(
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
    *,
    config: DedupeConfig | None = None,
) -> tuple[list[EntityProfile], list[RelationProfile]]:
    """Deduplicate extracted entities/relations into canonical profiles.

    Cross-links entity ``relation_ids`` to the relations that touch them.
    """
    cfg = config or DedupeConfig()
    entity_deduper = EntityDeduper(cfg)
    for entity in sorted(entities, key=lambda e: (e.name.casefold(), e.type)):
        entity_deduper.add(entity)
    entity_profiles, name_to_id = entity_deduper.build()

    relation_deduper = RelationDeduper(name_to_id, config=cfg)
    for relation in relations:
        relation_deduper.add(relation)
    relation_profiles = relation_deduper.build()

    # Cross-link relation ids onto their endpoint entities.
    by_id = {profile.id: profile for profile in entity_profiles}
    for relation in relation_profiles:
        for endpoint in (relation.source_entity_id, relation.target_entity_id):
            profile = by_id.get(endpoint)
            if profile is not None and relation.id not in profile.relation_ids:
                profile.relation_ids.append(relation.id)
    return entity_profiles, relation_profiles
