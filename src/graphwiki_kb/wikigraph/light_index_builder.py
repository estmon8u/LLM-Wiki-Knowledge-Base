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
) -> LightGraphBuildReport:
    """Build and persist the LightRAG index; return a build report."""
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

    chunks = build_light_chunks(
        root,
        sources,
        tokenizer=tok,
        chunk_token_size=lightrag_config.chunk_token_size,
        overlap_tokens=lightrag_config.chunk_overlap_tokens,
    )

    cache = ExtractionCache(store.paths.extraction_cache_dir)
    run = run_extraction(
        chunks,
        config=extraction_config,
        provider=provider,
        provider_identity=provider_identity,
        cache=cache,
        prompt_hash=prompt_hash,
    )

    entity_profiles, relation_profiles = dedupe_entities_and_relations(
        run.entities,
        run.relations,
        config=DedupeConfig(fuzzy_threshold=88),
    )
    profile_index(entity_profiles, relation_profiles, chunks, updated_at=built_at)

    entity_vectors, relation_vectors, embed_ok, embed_warnings = _embed(
        entity_profiles,
        relation_profiles,
        embedding_provider=embedding_provider,
        embeddings_config=embeddings_config,
    )

    embedding_tier = "embedded" if embed_ok else "bm25"
    tier = f"{run.tier}+{embedding_tier}"
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
        chunks, entity_profiles, relation_profiles
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
        incremental=plan.incremental,
        extraction_cache_hits=run.cache_hits,
        extraction_cache_misses=run.cache_misses,
        artifacts=artifacts,
        warnings=warnings,
    )


def _embed(
    entity_profiles: list[EntityProfile],
    relation_profiles: list[RelationProfile],
    *,
    embedding_provider: EmbeddingProvider | None,
    embeddings_config: EmbeddingsRuntimeConfig,
) -> tuple[LightVectorStore | None, LightVectorStore | None, bool, list[str]]:
    """Embed entity/relation profiles; fall back to BM25 (no vectors) on error."""
    if embedding_provider is None:
        return None, None, False, []
    ensure = getattr(embedding_provider, "ensure_available", None)
    if callable(ensure):
        try:
            ensure()
        except EmbeddingError as exc:
            return None, None, False, [f"embeddings unavailable, using BM25: {exc}"]
    try:
        entity_vectors = _embed_profiles(
            embedding_provider,
            [profile.id for profile in entity_profiles],
            [profile.embedding_text for profile in entity_profiles],
            embeddings_config,
        )
        relation_vectors = _embed_profiles(
            embedding_provider,
            [profile.id for profile in relation_profiles],
            [profile.embedding_text for profile in relation_profiles],
            embeddings_config,
        )
    except EmbeddingError as exc:
        return None, None, False, [f"embedding call failed, using BM25: {exc}"]
    return entity_vectors, relation_vectors, True, []


def _embed_profiles(
    embedding_provider: EmbeddingProvider,
    ids: list[str],
    texts: list[str],
    embeddings_config: EmbeddingsRuntimeConfig,
) -> LightVectorStore:
    if not ids:
        return LightVectorStore(
            model=embeddings_config.model,
            dimension=embeddings_config.dimension,
            ids=[],
            vectors=[],
        )
    vectors = embedding_provider.embed_texts(texts)
    return LightVectorStore.from_embeddings(
        ids,
        vectors,
        model=embeddings_config.model,
        dimension=embeddings_config.dimension,
    )


def _build_source_contributions(
    chunks: list[LightChunk],
    entities: list[EntityProfile],
    relations: list[RelationProfile],
) -> dict[str, Any]:
    chunk_source = {chunk.id: chunk.source_id for chunk in chunks}
    contrib: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
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
    return dict(contrib)
