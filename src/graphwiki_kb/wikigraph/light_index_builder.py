"""Build the LightRAG-style WikiGraphRAG index from normalized source text.

Pipeline (spec-aligned): chunk -> extract (cached) -> dedupe + profile ->
embed entity/relation profiles -> persist + manifest. Extraction is cached by
content hash so re-running over a mostly-unchanged corpus is cheap; for the MVP
the vector arrays are fully rewritten after each build ("incremental
extraction, full local vector rewrite"), which is honest and easy to validate.

Missing sources (present in the previous build manifest but gone from the
current manifest) are flagged for review -- never silently deleted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.providers.base import TextProvider
from graphwiki_kb.providers.embedding_base import (
    EmbeddingError,
    EmbeddingProvider,
)
from graphwiki_kb.services.config_service import (
    EmbeddingsRuntimeConfig,
    LightRagRuntimeConfig,
)
from graphwiki_kb.services.project_service import utc_now_iso
from graphwiki_kb.wikigraph.light_chunker import build_light_chunks
from graphwiki_kb.wikigraph.light_deduper import (
    DedupeConfig,
    dedupe_entities_and_relations,
)
from graphwiki_kb.wikigraph.light_extractor import (
    ExtractionCache,
    ExtractionConfig,
    extraction_prompt_hash,
    run_extraction,
)
from graphwiki_kb.wikigraph.light_graph_store import LightGraphStore
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    ExtractedEntity,
    ExtractedRelation,
    LightChunk,
    LightGraphBuildReport,
    LightGraphIndex,
    RelationProfile,
)
from graphwiki_kb.wikigraph.light_profiler import profile_index
from graphwiki_kb.wikigraph.light_tokenizer import Tokenizer, get_default_tokenizer
from graphwiki_kb.wikigraph.light_vector_store import LightVectorStore


@dataclass
class LightGraphUpdatePlan:
    """Source-level update plan derived from the previous build manifest."""

    new_source_ids: list[str] = field(default_factory=list)
    changed_source_ids: list[str] = field(default_factory=list)
    missing_source_ids: list[str] = field(default_factory=list)
    incremental: bool = False


def plan_lightgraph_update(
    sources: list[RawSourceRecord],
    previous_manifest: dict[str, Any] | None,
    *,
    force: bool,
    prompt_hash: str,
    embedding_identity: str,
) -> LightGraphUpdatePlan:
    """Compute new/changed/missing sources versus the previous build manifest."""
    current = {source.source_id: source.content_hash for source in sources}
    previous_hashes: dict[str, str] = {}
    contract_match = False
    if previous_manifest:
        previous_hashes = dict(previous_manifest.get("source_hashes", {}))
        contract_match = (
            previous_manifest.get("extraction_prompt_hash") == prompt_hash
            and previous_manifest.get("embedding_identity") == embedding_identity
        )
    new = [sid for sid in current if sid not in previous_hashes]
    changed = [
        sid
        for sid in current
        if sid in previous_hashes and previous_hashes[sid] != current[sid]
    ]
    missing = [sid for sid in previous_hashes if sid not in current]
    incremental = bool(previous_manifest) and not force and contract_match
    return LightGraphUpdatePlan(
        new_source_ids=sorted(new),
        changed_source_ids=sorted(changed),
        missing_source_ids=sorted(missing),
        incremental=incremental,
    )


def build_lightgraph_index(
    root: Path,
    sources: list[RawSourceRecord],
    *,
    store: LightGraphStore,
    lightrag_config: LightRagRuntimeConfig,
    embeddings_config: EmbeddingsRuntimeConfig,
    provider: TextProvider | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    provider_identity: str = "deterministic",
    tokenizer: Tokenizer | None = None,
    force: bool = False,
    previous_index: LightGraphIndex | None = None,
    previous_entity_vectors: LightVectorStore | None = None,
    previous_relation_vectors: LightVectorStore | None = None,
) -> LightGraphBuildReport:
    """Build and persist the LightRAG index; return a build report.

    When ``previous_index`` is supplied and the build is incremental
    (unchanged extraction/embedding contract, not forced), unchanged sources
    reuse their previous chunks (skipping re-chunking) and unchanged
    entity/relation profiles reuse their previous embedding vectors (skipping
    re-embedding) -- a source-level incremental update per LightRAG's design.
    """
    tok = tokenizer or get_default_tokenizer()
    built_at = utc_now_iso()

    extraction_config = ExtractionConfig(
        entity_types=tuple(lightrag_config.entity_types),
        relation_types=tuple(lightrag_config.relation_types),
        max_gleaning=lightrag_config.entity_extract_max_gleaning,
    )
    prompt_hash = extraction_prompt_hash(extraction_config)
    embedding_identity = (
        f"{embeddings_config.provider}:{embeddings_config.model}"
        if embedding_provider is not None
        else "bm25"
    )

    previous_manifest = store.load_build_manifest()
    plan = plan_lightgraph_update(
        sources,
        previous_manifest,
        force=force,
        prompt_hash=prompt_hash,
        embedding_identity=embedding_identity,
    )

    incremental = plan.incremental and previous_index is not None
    reprocessed_ids = set(plan.new_source_ids) | set(plan.changed_source_ids)
    if incremental:
        assert previous_index is not None
        reused_ids = {
            source.source_id
            for source in sources
            if source.source_id not in reprocessed_ids
        }
        reused_chunks = [
            chunk for chunk in previous_index.chunks if chunk.source_id in reused_ids
        ]
        sources_to_chunk = [
            source for source in sources if source.source_id in reprocessed_ids
        ]
        fresh_chunks = build_light_chunks(
            root,
            sources_to_chunk,
            tokenizer=tok,
            chunk_token_size=lightrag_config.chunk_token_size,
            overlap_tokens=lightrag_config.chunk_overlap_tokens,
        )
        chunks = [*reused_chunks, *fresh_chunks]
    else:
        reused_ids = set()
        reused_chunks = []
        reprocessed_ids = {source.source_id for source in sources}
        fresh_chunks = build_light_chunks(
            root,
            sources,
            tokenizer=tok,
            chunk_token_size=lightrag_config.chunk_token_size,
            overlap_tokens=lightrag_config.chunk_overlap_tokens,
        )
        chunks = fresh_chunks

    cache = ExtractionCache(store.paths.extraction_cache_dir)
    run = run_extraction(
        fresh_chunks,
        config=extraction_config,
        provider=provider,
        provider_identity=provider_identity,
        cache=cache,
        prompt_hash=prompt_hash,
    )
    if incremental and reused_chunks and previous_index is not None:
        replay_entities, replay_relations = _replay_extraction_from_previous_index(
            previous_index=previous_index,
            reused_chunks=reused_chunks,
        )
        run.entities = [*replay_entities, *run.entities]
        run.relations = [*replay_relations, *run.relations]

    entity_profiles, relation_profiles = dedupe_entities_and_relations(
        run.entities,
        run.relations,
        config=DedupeConfig(fuzzy_threshold=88),
    )
    profile_index(entity_profiles, relation_profiles, chunks, updated_at=built_at)

    reuse_entity_vectors = (
        _vectors_by_text(previous_index.entities, previous_entity_vectors)
        if incremental
        and previous_index is not None
        and previous_entity_vectors is not None
        else {}
    )
    reuse_relation_vectors = (
        _vectors_by_text(previous_index.relations, previous_relation_vectors)
        if incremental
        and previous_index is not None
        and previous_relation_vectors is not None
        else {}
    )
    entity_vectors, relation_vectors, embed_ok, embed_warnings, reused_vec_count = (
        _embed(
            entity_profiles,
            relation_profiles,
            embedding_provider=embedding_provider,
            embeddings_config=embeddings_config,
            reuse_entity_vectors=reuse_entity_vectors,
            reuse_relation_vectors=reuse_relation_vectors,
        )
    )

    embedding_mode = "embedded" if embed_ok else "bm25"
    embedding_tier = "strict" if embed_ok else "fallback"
    if embed_ok:
        embedding_tier_reason = (
            f"{embeddings_config.provider}:{embeddings_config.model} embeddings active"
        )
    elif embedding_provider is None:
        embedding_tier_reason = "BM25 fallback (no embedding provider configured)"
    else:
        embedding_tier_reason = (
            embed_warnings[-1] if embed_warnings else "BM25 fallback active"
        )
    tier = f"{run.tier}+{embedding_mode}"
    embedding_model = embeddings_config.model if embed_ok else ""
    embedding_dimension = embeddings_config.dimension if embed_ok else 0

    index = LightGraphIndex(
        built_at=built_at,
        chunks=chunks,
        entities=entity_profiles,
        relations=relation_profiles,
        source_hashes={source.source_id: source.content_hash for source in sources},
        extraction_prompt_hash=prompt_hash,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        provider_identity=provider_identity if run.tier == "provider" else None,
        embedding_identity=embedding_identity if embed_ok else "bm25",
        tier=tier,
        schema_version=1,
    )

    source_contributions = _build_source_contributions(
        chunks,
        entity_profiles,
        relation_profiles,
        reused_source_ids=reused_ids,
        reprocessed_source_ids=reprocessed_ids,
        missing_source_ids=plan.missing_source_ids,
    )
    build_manifest = {
        "built_at": built_at,
        "source_hashes": index.source_hashes,
        "chunking": {
            "chunk_token_size": lightrag_config.chunk_token_size,
            "overlap_tokens": lightrag_config.chunk_overlap_tokens,
        },
        "extraction_prompt_hash": prompt_hash,
        "provider_identity": index.provider_identity,
        "embedding_provider": embeddings_config.provider if embed_ok else "bm25",
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "embedding_tier": embedding_tier,
        "embedding_tier_reason": embedding_tier_reason,
        "embedding_identity": index.embedding_identity,
        "index_schema_version": 1,
        "tier": tier,
        "missing_sources": [
            {"source_id": sid, "status": "missing", "requires_review": True}
            for sid in plan.missing_source_ids
        ],
    }

    artifacts = store.save(
        index,
        entity_vectors=entity_vectors,
        relation_vectors=relation_vectors,
        source_contributions=source_contributions,
        build_manifest=build_manifest,
    )

    warnings = list(run.warnings) + embed_warnings
    if incremental:
        warnings.append(
            "incremental update: "
            f"{len(reused_ids)} source(s) reused, "
            f"{len(reprocessed_ids)} reprocessed, "
            f"{reused_vec_count} embedding vector(s) reused"
        )
    if plan.missing_source_ids:
        warnings.append(
            "Sources removed from the manifest are retained and flagged for "
            f"review (not deleted): {', '.join(plan.missing_source_ids)}"
        )

    return LightGraphBuildReport(
        built_at=built_at,
        tier=tier,
        chunk_count=index.chunk_count,
        entity_count=index.entity_count,
        relation_count=index.relation_count,
        source_count=len(sources),
        new_source_ids=plan.new_source_ids,
        changed_source_ids=plan.changed_source_ids,
        missing_source_ids=plan.missing_source_ids,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        embedding_tier=embedding_tier,
        embedding_tier_reason=embedding_tier_reason,
        incremental=plan.incremental,
        reused_source_count=len(reused_ids),
        reprocessed_source_count=len(reprocessed_ids),
        extraction_cache_hits=run.cache_hits,
        extraction_cache_misses=run.cache_misses,
        artifacts=artifacts,
        warnings=warnings,
    )


def _replay_extraction_from_previous_index(
    *,
    previous_index: LightGraphIndex,
    reused_chunks: list[LightChunk],
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    """Reconstruct extracted rows for reused chunks without calling extractors."""
    chunk_by_id = {chunk.id: chunk for chunk in reused_chunks}
    entities: list[ExtractedEntity] = []
    for profile in previous_index.entities:
        for chunk_id in profile.chunk_ids:
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            entities.append(
                ExtractedEntity(
                    name=profile.canonical_name,
                    type=profile.type,
                    description=profile.description,
                    aliases=list(profile.aliases),
                    chunk_ids=[chunk_id],
                    source_ids=[chunk.source_id],
                    evidence_quote="",
                    confidence=0.99,
                )
            )

    entity_id_to_name = {
        profile.id: profile.canonical_name for profile in previous_index.entities
    }
    relations: list[ExtractedRelation] = []
    for profile in previous_index.relations:
        source_name = entity_id_to_name.get(
            profile.source_entity_id, profile.source_entity_id
        )
        target_name = entity_id_to_name.get(
            profile.target_entity_id, profile.target_entity_id
        )
        for chunk_id in profile.chunk_ids:
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            relations.append(
                ExtractedRelation(
                    source=source_name,
                    target=target_name,
                    relation_type=profile.relation_type,
                    description=profile.description,
                    keywords=list(profile.keywords),
                    chunk_ids=[chunk_id],
                    source_ids=[chunk.source_id],
                    evidence_quote="",
                    weight=profile.weight,
                    confidence=0.99,
                )
            )
    return entities, relations


def _vectors_by_text(
    profiles: list,
    vector_store: LightVectorStore | None,
) -> dict[str, list[float]]:
    """Map ``embedding_text -> vector`` from a previous index + its vectors."""
    if vector_store is None:
        return {}
    vec_by_id = dict(zip(vector_store.ids, vector_store.vectors, strict=False))
    mapping: dict[str, list[float]] = {}
    for profile in profiles:
        text = getattr(profile, "embedding_text", "")
        vector = vec_by_id.get(profile.id)
        if text and vector is not None:
            mapping.setdefault(text, vector)
    return mapping


def _embed(
    entity_profiles: list[EntityProfile],
    relation_profiles: list[RelationProfile],
    *,
    embedding_provider: EmbeddingProvider | None,
    embeddings_config: EmbeddingsRuntimeConfig,
    reuse_entity_vectors: dict[str, list[float]] | None = None,
    reuse_relation_vectors: dict[str, list[float]] | None = None,
) -> tuple[LightVectorStore | None, LightVectorStore | None, bool, list[str], int]:
    """Embed entity/relation profiles; fall back to BM25 (no vectors) on error.

    Reuses vectors for profiles whose ``embedding_text`` is unchanged (passed in
    ``reuse_*`` maps), so an incremental build only embeds new/changed profiles.
    """
    if embedding_provider is None:
        return None, None, False, [], 0
    ensure = getattr(embedding_provider, "ensure_available", None)
    if callable(ensure):
        try:
            ensure()
        except EmbeddingError as exc:
            return None, None, False, [f"embeddings unavailable, using BM25: {exc}"], 0
    try:
        entity_vectors, entity_reused = _embed_profiles(
            embedding_provider,
            entity_profiles,
            embeddings_config,
            reuse_entity_vectors or {},
        )
        relation_vectors, relation_reused = _embed_profiles(
            embedding_provider,
            relation_profiles,
            embeddings_config,
            reuse_relation_vectors or {},
        )
    except EmbeddingError as exc:
        return None, None, False, [f"embedding call failed, using BM25: {exc}"], 0
    return entity_vectors, relation_vectors, True, [], entity_reused + relation_reused


def _embed_profiles(
    embedding_provider: EmbeddingProvider,
    profiles: list,
    embeddings_config: EmbeddingsRuntimeConfig,
    reuse: dict[str, list[float]],
) -> tuple[LightVectorStore, int]:
    if not profiles:
        return (
            LightVectorStore(
                model=embeddings_config.model,
                dimension=embeddings_config.dimension,
                ids=[],
                vectors=[],
            ),
            0,
        )
    ids = [profile.id for profile in profiles]
    vectors_by_index: dict[int, list[float]] = {}
    to_embed_indices: list[int] = []
    to_embed_texts: list[str] = []
    reused = 0
    for index, profile in enumerate(profiles):
        cached = reuse.get(profile.embedding_text)
        if cached is not None:
            vectors_by_index[index] = cached
            reused += 1
        else:
            to_embed_indices.append(index)
            to_embed_texts.append(profile.embedding_text)
    if to_embed_texts:
        embedded = embedding_provider.embed_texts(to_embed_texts)
        for position, index in enumerate(to_embed_indices):
            vectors_by_index[index] = embedded[position]
    ordered = [vectors_by_index[index] for index in range(len(profiles))]
    store = LightVectorStore.from_embeddings(
        ids,
        ordered,
        model=embeddings_config.model,
        dimension=embeddings_config.dimension,
    )
    return store, reused


def _build_source_contributions(
    chunks: list[LightChunk],
    entities: list[EntityProfile],
    relations: list[RelationProfile],
    *,
    reused_source_ids: set[str] | None = None,
    reprocessed_source_ids: set[str] | None = None,
    missing_source_ids: list[str] | None = None,
) -> dict[str, Any]:
    reused = reused_source_ids or set()
    reprocessed = reprocessed_source_ids or set()
    chunk_source = {chunk.id: chunk.source_id for chunk in chunks}

    def _status_for(source_id: str) -> str:
        if source_id in reprocessed:
            return "reprocessed"
        if source_id in reused:
            return "reused"
        return "fresh"

    contrib: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "status": "fresh",
            "requires_review": False,
            "chunk_ids": [],
            "entity_ids": [],
            "relation_ids": [],
            "entity_contributions": {},
            "relation_contributions": {},
        }
    )
    for chunk in chunks:
        contrib[chunk.source_id]["chunk_ids"].append(chunk.id)
    for entity in entities:
        for chunk_id in entity.chunk_ids:
            source_id = chunk_source.get(chunk_id)
            if source_id is None:
                continue
            bucket = contrib[source_id]
            if entity.id not in bucket["entity_ids"]:
                bucket["entity_ids"].append(entity.id)
            bucket["entity_contributions"].setdefault(entity.id, [])
            if chunk_id not in bucket["entity_contributions"][entity.id]:
                bucket["entity_contributions"][entity.id].append(chunk_id)
    for relation in relations:
        for chunk_id in relation.chunk_ids:
            source_id = chunk_source.get(chunk_id)
            if source_id is None:
                continue
            bucket = contrib[source_id]
            if relation.id not in bucket["relation_ids"]:
                bucket["relation_ids"].append(relation.id)
            bucket["relation_contributions"].setdefault(relation.id, [])
            if chunk_id not in bucket["relation_contributions"][relation.id]:
                bucket["relation_contributions"][relation.id].append(chunk_id)
    for source_id, bucket in contrib.items():
        bucket["status"] = _status_for(source_id)
    # Missing sources (in the previous index but gone now) are retained and
    # flagged for review -- never silently deleted (LightRAG §11).
    for source_id in missing_source_ids or []:
        contrib[source_id] = {
            "status": "missing",
            "requires_review": True,
            "chunk_ids": [],
            "entity_ids": [],
            "relation_ids": [],
            "entity_contributions": {},
            "relation_contributions": {},
        }
    return dict(contrib)
