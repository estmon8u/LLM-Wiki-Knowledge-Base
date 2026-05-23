"""Deduplication & merge logic for LightRAG-style entities and relations.

The deduper takes per-chunk :class:`ExtractedEntity` / :class:`ExtractedRelation`
and emits canonical :class:`EntityProfile` / :class:`RelationProfile`
records, unioning evidence (chunks, sources, descriptions) along the way.

Entity canonicalization layers (cheapest first):

1. Unicode-light normalization + casefold.
2. Acronym alias table — colon-prefixed titles such as
   ``REALM: Retrieval-Augmented Language Model Pre-Training`` lift
   ``REALM`` into the alias list.
3. Fuzzy (RapidFuzz) match above a configurable threshold, gated by
   type compatibility.

Relation canonicalization:

* Direction-preserving id derived from canonical endpoint ids and the
  normalized relation type.
* Inverse-type table folds ``USES`` / ``USED_BY`` and similar pairs
  into a single canonical direction.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from graphwiki_kb.services.project_service import slugify, utc_now_iso
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    ExtractedEntity,
    ExtractedRelation,
    RelationProfile,
)

_INVERSE_RELATION_TYPES: dict[str, tuple[str, bool]] = {
    "USES": ("USES", True),
    "USED_BY": ("USES", False),
    "IMPROVES_OVER": ("IMPROVES_OVER", True),
    "IS_IMPROVED_BY": ("IMPROVES_OVER", False),
    "EVALUATES_ON": ("EVALUATES_ON", True),
    "IS_EVALUATION_DATASET_FOR": ("EVALUATES_ON", False),
    "DEPENDS_ON": ("DEPENDS_ON", True),
    "IS_DEPENDED_ON_BY": ("DEPENDS_ON", False),
    "INTRODUCES": ("INTRODUCES", True),
    "IS_INTRODUCED_BY": ("INTRODUCES", False),
}


@dataclass(frozen=True)
class LightDeduperOptions:
    """Tunable knobs for :class:`LightDeduper`."""

    fuzzy_match_threshold: int = 88
    max_description_chars: int = 600
    type_compatible_only: bool = True


def _normalize_name(name: str) -> str:
    cleaned = " ".join(name.split()).strip(" .,:;")
    return cleaned


def _name_key(name: str) -> str:
    return _normalize_name(name).casefold()


def _entity_id(canonical_name: str, entity_type: str) -> str:
    slug = slugify(canonical_name) or "entity"
    digest = hashlib.sha1(f"{slug}:{entity_type.lower()}".encode()).hexdigest()[:8]
    return f"entity:{slug}:{digest}"


def _relation_id(source_id: str, target_id: str, relation_type: str) -> str:
    digest = hashlib.sha1(
        f"{source_id}|{relation_type.upper()}|{target_id}".encode()
    ).hexdigest()[:10]
    return f"relation:{digest}"


def _truncate_description(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - 1)].rstrip()
    return cut + "…"


def _canonical_relation_type(relation_type: str) -> tuple[str, bool]:
    """Return ``(canonical_type, keep_direction)`` for ``relation_type``."""
    normalized = relation_type.strip().upper()
    if normalized in _INVERSE_RELATION_TYPES:
        return _INVERSE_RELATION_TYPES[normalized]
    return normalized, True


def _lift_acronym_alias(name: str) -> list[str]:
    """Return implicit aliases derived from ``name`` (e.g. colon prefixes)."""
    aliases: list[str] = []
    title = name.strip()
    if ":" in title:
        prefix = title.split(":", 1)[0].strip()
        if prefix and prefix != title:
            aliases.append(prefix)
    return aliases


@dataclass
class _EntityBucket:
    canonical_name: str
    canonical_id: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    evidence_quotes: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    occurrences: int = 0


@dataclass
class LightDeduper:
    """Stateful canonicalizer for the LightRAG pipeline."""

    options: LightDeduperOptions = LightDeduperOptions()

    def __post_init__(self) -> None:
        self._buckets: OrderedDict[str, _EntityBucket] = OrderedDict()
        self._alias_to_canonical: dict[str, str] = {}

    @property
    def entity_count(self) -> int:
        """Return the number of canonical buckets currently tracked."""
        return len(self._buckets)

    def _resolve_canonical(
        self,
        entity: ExtractedEntity,
    ) -> _EntityBucket:
        names = [entity.name, *entity.aliases, *_lift_acronym_alias(entity.name)]
        for name in names:
            key = _name_key(name)
            if not key:
                continue
            target = self._alias_to_canonical.get(key)
            if target is None:
                continue
            existing = self._buckets[target]
            if (
                self.options.type_compatible_only
                and entity.type
                and existing.entity_type
                and entity.type != existing.entity_type
            ):
                continue
            return existing

        # Fuzzy match against existing canonical names of compatible type.
        best_key: str | None = None
        best_score: float = 0.0
        cmp_name = _normalize_name(entity.name)
        for key, bucket in self._buckets.items():
            if (
                self.options.type_compatible_only
                and entity.type
                and bucket.entity_type
                and entity.type != bucket.entity_type
            ):
                continue
            score = float(fuzz.token_set_ratio(cmp_name, bucket.canonical_name))
            if score > best_score:
                best_score = score
                best_key = key
        if best_key is not None and best_score >= self.options.fuzzy_match_threshold:
            return self._buckets[best_key]

        canonical_name = cmp_name or entity.name
        canonical_id = _entity_id(canonical_name, entity.type)
        bucket = _EntityBucket(
            canonical_name=canonical_name,
            canonical_id=canonical_id,
            entity_type=entity.type,
        )
        self._buckets[canonical_id] = bucket
        self._alias_to_canonical[_name_key(canonical_name)] = canonical_id
        return bucket

    def add_entity(self, entity: ExtractedEntity) -> str:
        """Merge ``entity`` into the working set and return its canonical id."""
        bucket = self._resolve_canonical(entity)
        for alias in [entity.name, *entity.aliases, *_lift_acronym_alias(entity.name)]:
            key = _name_key(alias)
            if not key:
                continue
            self._alias_to_canonical[key] = bucket.canonical_id
            if alias != bucket.canonical_name and alias not in bucket.aliases and alias:
                bucket.aliases.append(alias)
        if entity.description and entity.description not in bucket.descriptions:
            bucket.descriptions.append(entity.description)
        for cid in entity.chunk_ids:
            if cid not in bucket.chunk_ids:
                bucket.chunk_ids.append(cid)
        for sid in entity.source_ids:
            if sid not in bucket.source_ids:
                bucket.source_ids.append(sid)
        if (
            entity.evidence_quote
            and entity.evidence_quote not in bucket.evidence_quotes
        ):
            bucket.evidence_quotes.append(entity.evidence_quote)
        bucket.occurrences += 1
        return bucket.canonical_id

    def canonical_id_for(self, name: str) -> str | None:
        """Return the canonical entity id for ``name`` (or None)."""
        return self._alias_to_canonical.get(_name_key(name))

    def build_entity_profiles(self) -> list[EntityProfile]:
        """Materialize :class:`EntityProfile` for every canonical bucket."""
        timestamp = utc_now_iso()
        profiles: list[EntityProfile] = []
        for bucket in self._buckets.values():
            description = " ".join(bucket.descriptions).strip()
            if description:
                description = _truncate_description(
                    description, self.options.max_description_chars
                )
            keywords = sorted(
                {
                    _name_key(alias)
                    for alias in [bucket.canonical_name, *bucket.aliases]
                    if alias
                }
            )
            profile_text = _build_entity_profile_text(bucket, description)
            embedding_text = _build_entity_embedding_text(bucket)
            profiles.append(
                EntityProfile(
                    id=bucket.canonical_id,
                    canonical_name=bucket.canonical_name,
                    type=bucket.entity_type,
                    aliases=list(bucket.aliases),
                    description=description,
                    profile_text=profile_text,
                    keywords=list(keywords),
                    chunk_ids=list(bucket.chunk_ids),
                    source_ids=list(bucket.source_ids),
                    relation_ids=[],
                    embedding_text=embedding_text,
                    updated_at=timestamp,
                    metadata={"occurrences": bucket.occurrences},
                )
            )
        return profiles


def _build_entity_profile_text(bucket: _EntityBucket, description: str) -> str:
    lines: list[str] = [
        f"Entity: {bucket.canonical_name}",
        f"Type: {bucket.entity_type}",
    ]
    if bucket.aliases:
        lines.append("Aliases: " + ", ".join(bucket.aliases[:8]))
    if description:
        lines.append("Description: " + description)
    if bucket.source_ids:
        lines.append("Sources: " + ", ".join(bucket.source_ids[:8]))
    if bucket.evidence_quotes:
        lines.append("Evidence snippets:")
        for quote in bucket.evidence_quotes[:4]:
            lines.append(f"  - {quote}")
    return "\n".join(lines)


def _build_entity_embedding_text(bucket: _EntityBucket) -> str:
    parts = [bucket.canonical_name, bucket.entity_type, *bucket.aliases]
    if bucket.descriptions:
        parts.append(bucket.descriptions[0][:200])
    cleaned = [p for p in parts if p]
    return " ".join(cleaned)


@dataclass
class _RelationBucket:
    canonical_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    descriptions: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    evidence_quotes: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)


@dataclass
class LightRelationDeduper:
    """Deduplicates relations after entity canonicalization."""

    options: LightDeduperOptions = LightDeduperOptions()

    def __post_init__(self) -> None:
        self._buckets: dict[str, _RelationBucket] = {}
        self._endpoint_to_relations: dict[str, set[str]] = defaultdict(set)

    def add_relation(
        self,
        relation: ExtractedRelation,
        *,
        source_entity_id: str,
        target_entity_id: str,
    ) -> str:
        """Merge ``relation`` (with resolved endpoints) and return its id."""
        canonical_type, keep_direction = _canonical_relation_type(
            relation.relation_type
        )
        if not keep_direction:
            source_entity_id, target_entity_id = target_entity_id, source_entity_id
        rel_id = _relation_id(source_entity_id, target_entity_id, canonical_type)
        bucket = self._buckets.get(rel_id)
        if bucket is None:
            bucket = _RelationBucket(
                canonical_id=rel_id,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                relation_type=canonical_type,
            )
            self._buckets[rel_id] = bucket
        if relation.description and relation.description not in bucket.descriptions:
            bucket.descriptions.append(relation.description)
        for cid in relation.chunk_ids:
            if cid not in bucket.chunk_ids:
                bucket.chunk_ids.append(cid)
        for sid in relation.source_ids:
            if sid not in bucket.source_ids:
                bucket.source_ids.append(sid)
        for kw in relation.keywords:
            if kw not in bucket.keywords:
                bucket.keywords.append(kw)
        if (
            relation.evidence_quote
            and relation.evidence_quote not in bucket.evidence_quotes
        ):
            bucket.evidence_quotes.append(relation.evidence_quote)
        bucket.weights.append(relation.weight)
        self._endpoint_to_relations[source_entity_id].add(rel_id)
        self._endpoint_to_relations[target_entity_id].add(rel_id)
        return rel_id

    def build_relation_profiles(self) -> list[RelationProfile]:
        """Materialize :class:`RelationProfile` for every canonical bucket."""
        timestamp = utc_now_iso()
        profiles: list[RelationProfile] = []
        for bucket in self._buckets.values():
            description = " ".join(bucket.descriptions).strip()
            if description:
                description = _truncate_description(
                    description, self.options.max_description_chars
                )
            avg_weight = (
                sum(bucket.weights) / len(bucket.weights) if bucket.weights else 1.0
            )
            profile_text = _build_relation_profile_text(bucket, description)
            embedding_text = _build_relation_embedding_text(bucket)
            profiles.append(
                RelationProfile(
                    id=bucket.canonical_id,
                    source_entity_id=bucket.source_entity_id,
                    target_entity_id=bucket.target_entity_id,
                    relation_type=bucket.relation_type,
                    description=description,
                    profile_text=profile_text,
                    keywords=list(bucket.keywords),
                    chunk_ids=list(bucket.chunk_ids),
                    source_ids=list(bucket.source_ids),
                    embedding_text=embedding_text,
                    weight=float(avg_weight),
                    updated_at=timestamp,
                    metadata={"occurrences": len(bucket.weights)},
                )
            )
        return profiles

    def relations_for_entity(self, entity_id: str) -> set[str]:
        """Return the canonical relation ids that touch ``entity_id``."""
        return set(self._endpoint_to_relations.get(entity_id, set()))


def _build_relation_profile_text(bucket: _RelationBucket, description: str) -> str:
    lines: list[str] = [
        f"Relation: {bucket.source_entity_id} {bucket.relation_type} "
        f"{bucket.target_entity_id}",
        f"Type: {bucket.relation_type}",
    ]
    if bucket.keywords:
        lines.append("Keywords: " + ", ".join(bucket.keywords[:8]))
    if description:
        lines.append("Description: " + description)
    if bucket.source_ids:
        lines.append("Sources: " + ", ".join(bucket.source_ids[:8]))
    if bucket.evidence_quotes:
        lines.append("Evidence snippets:")
        for quote in bucket.evidence_quotes[:4]:
            lines.append(f"  - {quote}")
    return "\n".join(lines)


def _build_relation_embedding_text(bucket: _RelationBucket) -> str:
    parts = [
        bucket.relation_type,
        bucket.source_entity_id.replace("entity:", "").replace("-", " "),
        bucket.target_entity_id.replace("entity:", "").replace("-", " "),
        *bucket.keywords[:8],
    ]
    if bucket.descriptions:
        parts.append(bucket.descriptions[0][:200])
    return " ".join(p for p in parts if p)


def dedupe_and_profile(
    extracted: Iterable[tuple[object, object]],
    *,
    deduper_options: LightDeduperOptions | None = None,
) -> tuple[list[EntityProfile], list[RelationProfile]]:
    """Convenience wrapper that runs entity then relation dedupe in order.

    ``extracted`` is an iterable of ``(entities, relations)`` pairs — one
    pair per processed chunk. The function is provided for tests and
    notebooks; production code goes through :func:`build_lightgraph_index`
    in :mod:`graphwiki_kb.wikigraph.light_index_builder`.
    """
    options = deduper_options or LightDeduperOptions()
    ent_deduper = LightDeduper(options=options)
    rel_deduper = LightRelationDeduper(options=options)
    pending_relations: list[ExtractedRelation] = []
    for entities, relations in extracted:
        assert isinstance(entities, Iterable)
        assert isinstance(relations, Iterable)
        for ent in entities:
            assert isinstance(ent, ExtractedEntity)
            ent_deduper.add_entity(ent)
        for rel in relations:
            assert isinstance(rel, ExtractedRelation)
            pending_relations.append(rel)
    for rel in pending_relations:
        source_id = ent_deduper.canonical_id_for(rel.source)
        target_id = ent_deduper.canonical_id_for(rel.target)
        if not source_id or not target_id:
            continue
        rel_deduper.add_relation(
            rel,
            source_entity_id=source_id,
            target_entity_id=target_id,
        )

    entity_profiles = ent_deduper.build_entity_profiles()
    for profile in entity_profiles:
        profile.relation_ids[:] = sorted(rel_deduper.relations_for_entity(profile.id))
    return entity_profiles, rel_deduper.build_relation_profiles()


def normalize_relation_type(relation_type: str) -> str:
    """Public helper to canonicalize a relation-type string."""
    canonical, _ = _canonical_relation_type(relation_type)
    return canonical


def normalize_name_for_match(name: str) -> str:
    """Public helper exposing the casefolded name key used for matching."""
    cleaned = re.sub(r"\s+", " ", name).strip()
    return cleaned.casefold()
