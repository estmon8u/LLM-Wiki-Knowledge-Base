"""Build and incrementally update the LightRAG-style WikiGraph index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.providers.base import TextProvider
from graphwiki_kb.providers.embedding_base import EmbeddingProvider
from graphwiki_kb.services.embedding_service import EmbeddingRuntimeConfig
from graphwiki_kb.services.project_service import ProjectPaths, utc_now_iso
from graphwiki_kb.wikigraph.light_chunker import build_light_chunks
from graphwiki_kb.wikigraph.light_deduper import dedupe_and_profile
from graphwiki_kb.wikigraph.light_extractor import (
    LightExtractionConfig,
    extract_entities_and_relations,
    extraction_prompt_hash,
)
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
)
from graphwiki_kb.wikigraph.light_models import LightGraphBuildReport, LightGraphIndex
from graphwiki_kb.wikigraph.light_tokenizer import WhitespaceTokenizer
from graphwiki_kb.wikigraph.light_vector_store import HybridRetriever, LightVectorStore


@dataclass(frozen=True)
class LightGraphConfig:
    """Resolved LightRAG build configuration."""

    chunk_token_size: int = 1200
    chunk_overlap_tokens: int = 100
    entity_extract_max_gleaning: int = 1
    entity_types: tuple[str, ...] = (
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
    relation_types: tuple[str, ...] = (
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
    embedding_runtime: EmbeddingRuntimeConfig | None = None
    local_fallback: str = "bm25"
    force: bool = False


@dataclass
class LightGraphUpdatePlan:
    """Source-level update plan."""

    new_source_ids: set[str]
    changed_source_ids: set[str]
    missing_source_ids: set[str]
    sources_to_process: set[str]
    can_incremental: bool


def plan_lightgraph_update(
    sources: list[RawSourceRecord],
    *,
    previous_manifest: dict[str, Any] | None,
    force: bool = False,
) -> LightGraphUpdatePlan:
    """Plan incremental source updates."""
    current = {source.source_id: source.content_hash for source in sources}
    current_ids = set(current)
    if previous_manifest is None or force:
        return LightGraphUpdatePlan(
            new_source_ids=current_ids,
            changed_source_ids=set(),
            missing_source_ids=set(),
            sources_to_process=current_ids,
            can_incremental=False,
        )
    previous_hashes = dict(previous_manifest.get("source_hashes", {}))
    previous_ids = set(previous_hashes)
    new_ids = current_ids - previous_ids
    missing_ids = previous_ids - current_ids
    changed_ids = {
        source_id
        for source_id in current_ids & previous_ids
        if current[source_id] != previous_hashes.get(source_id)
    }
    to_process = new_ids | changed_ids
    return LightGraphUpdatePlan(
        new_source_ids=new_ids,
        changed_source_ids=changed_ids,
        missing_source_ids=missing_ids,
        sources_to_process=to_process if to_process else current_ids,
        can_incremental=bool(previous_manifest) and not force,
    )


def build_lightgraph_index(
    paths: ProjectPaths,
    sources: list[RawSourceRecord],
    *,
    config: LightGraphConfig,
    provider: TextProvider | None,
    embedding_provider: EmbeddingProvider | None,
    previous_index: LightGraphIndex | None = None,
    changed_source_ids: set[str] | None = None,
) -> LightGraphBuildReport:
    """Build or incrementally refresh the LightGraph index."""
    store_paths = LightGraphStorePaths(paths.graph_dir / "wikigraph" / "lightrag")
    store = LightGraphStore(store_paths)
    previous_manifest = store.load_build_manifest()
    plan = plan_lightgraph_update(
        sources,
        previous_manifest=previous_manifest,
        force=config.force,
    )
    warnings: list[str] = []
    for source_id in sorted(plan.missing_source_ids):
        warnings.append(f"source_id={source_id} status=missing requires_review=true")

    remove_ids = changed_source_ids or plan.changed_source_ids
    if plan.can_incremental and previous_index is not None and remove_ids:
        working_entities = [
            entity
            for entity in previous_index.entities
            if not _entity_touched_by_sources(entity, remove_ids)
        ]
        working_relations = [
            relation
            for relation in previous_index.relations
            if not _relation_touched_by_sources(relation, remove_ids)
        ]
        working_chunks = [
            chunk
            for chunk in previous_index.chunks
            if chunk.source_id not in remove_ids
        ]
        selected_sources = [
            s for s in sources if s.source_id in plan.sources_to_process
        ]
    else:
        working_entities = []
        working_relations = []
        working_chunks = []
        selected_sources = list(sources)

    tokenizer = WhitespaceTokenizer()
    chunks = build_light_chunks(
        paths.root,
        selected_sources,
        tokenizer=tokenizer,
        chunk_token_size=config.chunk_token_size,
        overlap_tokens=config.chunk_overlap_tokens,
    )
    if plan.can_incremental and previous_index is not None:
        chunks = [*working_chunks, *chunks]

    extraction_config = LightExtractionConfig(
        entity_types=config.entity_types,
        relation_types=config.relation_types,
        entity_extract_max_gleaning=config.entity_extract_max_gleaning,
    )
    prompt_hash = extraction_prompt_hash(
        entity_types=config.entity_types,
        relation_types=config.relation_types,
    )
    provider_identity = provider.name if provider else "deterministic"
    extracted: list[tuple[str, Any]] = []
    for chunk in chunks:
        result = extract_entities_and_relations(
            chunk,
            provider=provider,
            config=extraction_config,
            cache_dir=store_paths.extraction_cache_dir,
            provider_identity=provider_identity,
        )
        extracted.append((chunk.id, result))

    entities, relations = dedupe_and_profile(
        extracted,
        existing_entities=working_entities,
        existing_relations=working_relations,
    )

    embedding_model = ""
    embedding_dimension = 0
    fallback_mode = ""
    if embedding_provider is not None:
        entity_vectors = embedding_provider.embed_texts(
            [entity.embedding_text for entity in entities]
        )
        relation_vectors = embedding_provider.embed_texts(
            [relation.embedding_text for relation in relations]
        )
        embedding_model = embedding_provider.model_name
        embedding_dimension = embedding_provider.dimension
        _save_vectors(
            store_paths.entity_vectors_dir,
            entities,
            entity_vectors,
            model=embedding_model,
            backend="embedding",
        )
        _save_vectors(
            store_paths.relation_vectors_dir,
            relations,
            relation_vectors,
            model=embedding_model,
            backend="embedding",
        )
    else:
        fallback_mode = f"{config.local_fallback} fallback"
        _save_bm25_vectors(
            store_paths.entity_vectors_dir, entities, backend=fallback_mode
        )
        _save_bm25_vectors(
            store_paths.relation_vectors_dir, relations, backend=fallback_mode
        )
        warnings.append(
            "LightGraph update: embedding provider unavailable; "
            f"using {config.local_fallback} lexical fallback."
        )

    built_at = utc_now_iso()
    source_hashes = {source.source_id: source.content_hash for source in sources}
    index = LightGraphIndex(
        built_at=built_at,
        chunks=chunks,
        entities=entities,
        relations=relations,
        source_hashes=source_hashes,
        extraction_prompt_hash=prompt_hash,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        provider_identity=provider_identity,
        chunk_count=len(chunks),
        entity_count=len(entities),
        relation_count=len(relations),
    )
    build_manifest = {
        "built_at": built_at,
        "source_hashes": source_hashes,
        "chunking": {
            "chunk_token_size": config.chunk_token_size,
            "overlap_tokens": config.chunk_overlap_tokens,
        },
        "extraction_prompt_hash": prompt_hash,
        "provider_identity": provider_identity,
        "embedding_provider": (
            config.embedding_runtime.provider if config.embedding_runtime else ""
        ),
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "index_schema_version": 1,
        "missing_sources": [
            {"source_id": source_id, "status": "missing", "requires_review": True}
            for source_id in sorted(plan.missing_source_ids)
        ],
    }
    contributions = _build_source_contributions(chunks, entities, relations)
    artifacts = store.save(
        index,
        build_manifest=build_manifest,
        source_contributions=contributions,
    )
    incremental = plan.can_incremental and bool(changed_source_ids)
    return LightGraphBuildReport(
        built_at=built_at,
        chunk_count=len(chunks),
        entity_count=len(entities),
        relation_count=len(relations),
        source_count=len(sources),
        incremental=incremental,
        vector_rewrite=True,
        fallback_mode=fallback_mode,
        artifacts=artifacts,
        warnings=warnings,
    )


def _entity_touched_by_sources(entity: Any, source_ids: set[str]) -> bool:
    return bool(set(entity.source_ids) & source_ids)


def _relation_touched_by_sources(relation: Any, source_ids: set[str]) -> bool:
    return bool(set(relation.source_ids) & source_ids)


def _save_vectors(
    root: Any,
    profiles: list[Any],
    vectors: list[list[float]],
    *,
    model: str,
    backend: str,
) -> None:
    store = LightVectorStore(root)
    store.save(
        ids=[profile.id for profile in profiles],
        vectors=vectors,
        model=model,
        backend=backend,
    )


def _save_bm25_vectors(root: Any, profiles: list[Any], *, backend: str) -> None:
    store = LightVectorStore(root)
    store.save(
        ids=[profile.id for profile in profiles],
        vectors=[],
        model="bm25",
        backend=backend,
    )
    HybridRetriever(
        ids=[profile.id for profile in profiles],
        texts=[profile.embedding_text for profile in profiles],
        vectors=None,
        backend_label=backend,
    )


def _build_source_contributions(
    chunks: list[Any],
    entities: list[Any],
    relations: list[Any],
) -> dict[str, Any]:
    contributions: dict[str, Any] = {}
    for chunk in chunks:
        entry = contributions.setdefault(
            chunk.source_id,
            {
                "chunk_ids": [],
                "entity_ids": [],
                "relation_ids": [],
                "entity_contributions": {},
                "relation_contributions": {},
            },
        )
        entry["chunk_ids"].append(chunk.id)
    for entity in entities:
        for source_id in entity.source_ids:
            entry = contributions.setdefault(
                source_id,
                {
                    "chunk_ids": [],
                    "entity_ids": [],
                    "relation_ids": [],
                    "entity_contributions": {},
                    "relation_contributions": {},
                },
            )
            entry["entity_ids"].append(entity.id)
            entry["entity_contributions"].setdefault(entity.id, []).extend(
                entity.chunk_ids
            )
    for relation in relations:
        for source_id in relation.source_ids:
            entry = contributions.setdefault(
                source_id,
                {
                    "chunk_ids": [],
                    "entity_ids": [],
                    "relation_ids": [],
                    "entity_contributions": {},
                    "relation_contributions": {},
                },
            )
            entry["relation_ids"].append(relation.id)
            entry["relation_contributions"].setdefault(relation.id, []).extend(
                relation.chunk_ids
            )
    return contributions
