"""Build a LightRAG-style :class:`LightGraphIndex` from normalized sources.

Pipeline:

1. Chunk normalized source files into ~1200-token windows.
2. Run a :class:`LightExtractor` (deterministic by default) over each
   chunk, with optional disk cache keyed on chunk hash + prompt hash.
3. Canonicalize entities and relations (alias merge, fuzzy match,
   inverse-type folding).
4. Profile each canonical entity/relation into ``embedding_text`` +
   ``profile_text``.
5. Embed entity and relation profiles using the provided
   :class:`EmbeddingProvider` (BM25 fallback by default).
6. Persist everything via :class:`LightGraphStore`.

Incremental updates are implemented at source granularity (see
project recommendation §11): callers can pass an existing index plus a
set of source ids that should be recomputed, and the builder rewires
only those source contributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.providers.embedding_base import EmbeddingProvider
from graphwiki_kb.services.embedding_service import (
    ResolvedEmbedding,
)
from graphwiki_kb.services.project_service import ProjectPaths, utc_now_iso
from graphwiki_kb.wikigraph.light_chunker import (
    LightChunkerOptions,
    build_light_chunks,
)
from graphwiki_kb.wikigraph.light_deduper import (
    LightDeduper,
    LightDeduperOptions,
    LightRelationDeduper,
)
from graphwiki_kb.wikigraph.light_embeddings import (
    BM25SparseEmbeddingProvider,
)
from graphwiki_kb.wikigraph.light_extractor import (
    DeterministicLightExtractor,
    LightExtractionCache,
    LightExtractor,
    LightExtractorOptions,
    extract_corpus,
)
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
    serialize_vectors,
)
from graphwiki_kb.wikigraph.light_models import (
    ExtractedEntity,
    ExtractedRelation,
    LightExtractionResult,
    LightGraphBuildManifest,
    LightGraphBuildReport,
    LightGraphIndex,
    SourceContribution,
)


@dataclass(frozen=True)
class LightGraphBuildOptions:
    """Tunable knobs for :func:`build_lightgraph_index`."""

    chunk_token_size: int = 1200
    overlap_tokens: int = 100
    min_chunk_tokens: int = 30
    entity_types: tuple[str, ...] = field(default_factory=tuple)
    relation_types: tuple[str, ...] = field(default_factory=tuple)
    fuzzy_match_threshold: int = 88
    max_description_chars: int = 600
    embed_chunks: bool = False
    extraction_min_occurrences: int = 1


def _wiki_source_path(source: RawSourceRecord, root: Path) -> str | None:
    candidate = root / "wiki" / "sources" / f"{source.slug}.md"
    if candidate.exists():
        return candidate.relative_to(root).as_posix()
    return None


def build_lightgraph_index(
    paths: ProjectPaths,
    sources: list[RawSourceRecord],
    *,
    options: LightGraphBuildOptions | None = None,
    extractor: LightExtractor | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_resolution: ResolvedEmbedding | None = None,
    previous_index: LightGraphIndex | None = None,
    changed_source_ids: set[str] | None = None,
    store: LightGraphStore | None = None,
    use_cache: bool = True,
) -> tuple[LightGraphIndex, LightGraphBuildReport]:
    """Build (or incrementally update) a :class:`LightGraphIndex`.

    Args:
        paths: Project paths used to resolve normalized artifacts.
        sources: Manifest sources to process.
        options: Optional :class:`LightGraphBuildOptions`.
        extractor: Optional :class:`LightExtractor`. Defaults to a
            deterministic offline extractor when omitted.
        embedding_provider: Optional :class:`EmbeddingProvider`.
            Defaults to BM25 fit on the corpus of entity/relation
            embedding texts.
        previous_index: Optional previously persisted index to use as
            the incremental starting point.
        changed_source_ids: Optional explicit set of source ids that
            should be re-extracted. When ``None``, ``previous_index``'s
            source hashes are compared with the current ones and the
            differences become the change set.
        store: Optional :class:`LightGraphStore`. When provided, the
            built index (and vectors) are persisted before returning.
        use_cache: Whether to use the on-disk extraction cache. Set to
            ``False`` to force a full re-extraction.
    """
    opts = options or LightGraphBuildOptions()

    if extractor is None:
        extractor = DeterministicLightExtractor(
            options=LightExtractorOptions(
                entity_types=(
                    tuple(opts.entity_types)
                    if opts.entity_types
                    else LightExtractorOptions().entity_types
                ),
                relation_types=(
                    tuple(opts.relation_types)
                    if opts.relation_types
                    else LightExtractorOptions().relation_types
                ),
                min_occurrences=opts.extraction_min_occurrences,
            )
        )

    store = store or LightGraphStore(
        LightGraphStorePaths(paths.graph_dir / "wikigraph" / "lightrag")
    )

    incremental = False
    reused_source_ids: set[str] = set()
    reprocessed_source_ids: set[str] = set()

    if previous_index is not None:
        prev_hashes = previous_index.manifest.source_hashes or {}
        current_hashes = {s.source_id: s.content_hash for s in sources}
        if changed_source_ids is None:
            changed_source_ids = {
                sid for sid, h in current_hashes.items() if prev_hashes.get(sid) != h
            }
        else:
            # Honor explicit caller-supplied change set, but make sure
            # genuinely-changed hashes are not silently skipped.
            changed_source_ids = set(changed_source_ids) | {
                sid for sid, h in current_hashes.items() if prev_hashes.get(sid) != h
            }
        incremental = True
        reused_source_ids = {
            s.source_id for s in sources if s.source_id not in changed_source_ids
        }
        reprocessed_source_ids = set(changed_source_ids)

    chunker_opts = LightChunkerOptions(
        chunk_token_size=opts.chunk_token_size,
        overlap_tokens=opts.overlap_tokens,
        min_tokens=opts.min_chunk_tokens,
    )

    # Source-level incremental chunking. Reuse previous chunks for
    # unchanged sources verbatim — they share content hashes with the
    # current normalized artifacts, so the extraction cache will hit on
    # them anyway, but skipping the re-chunk avoids re-reading the file.
    reused_chunks: list = []
    if previous_index is not None and reused_source_ids:
        reused_chunks = [
            chunk
            for chunk in previous_index.chunks
            if chunk.source_id in reused_source_ids
        ]
    sources_to_chunk = [
        s
        for s in sources
        if previous_index is None or s.source_id not in reused_source_ids
    ]
    if previous_index is None:
        reprocessed_source_ids = {s.source_id for s in sources_to_chunk}

    fresh_chunks = build_light_chunks(
        root=paths.root,
        sources=sources_to_chunk,
        options=chunker_opts,
        compiled_page_lookup=lambda s: _wiki_source_path(s, paths.root),
    )
    chunks = [*reused_chunks, *fresh_chunks]

    cache: LightExtractionCache | None = None
    if use_cache:
        cache = LightExtractionCache(store.paths.extraction_cache_dir)

    # True source-level incremental extraction:
    #   * fresh_chunks → run the extractor (with cache when enabled).
    #   * reused_chunks → reconstruct ExtractedEntity / ExtractedRelation
    #     records from the previous index instead of calling the
    #     extractor at all. This keeps incremental updates fast even
    #     when the on-disk extraction cache is disabled (e.g. in tests).
    fresh_results = extract_corpus(fresh_chunks, extractor, cache=cache)
    reused_results: list[LightExtractionResult] = []
    if previous_index is not None and reused_chunks:
        reused_results = _replay_extraction_from_previous_index(
            previous_index=previous_index,
            reused_chunks=reused_chunks,
        )
    extraction_results = [*reused_results, *fresh_results]
    extracted_entity_count = sum(len(r.entities) for r in extraction_results)
    extracted_relation_count = sum(len(r.relations) for r in extraction_results)

    deduper = LightDeduper(
        options=LightDeduperOptions(
            fuzzy_match_threshold=opts.fuzzy_match_threshold,
            max_description_chars=opts.max_description_chars,
        )
    )
    relation_deduper = LightRelationDeduper(
        options=LightDeduperOptions(
            fuzzy_match_threshold=opts.fuzzy_match_threshold,
            max_description_chars=opts.max_description_chars,
        )
    )

    # Entity dedupe pass.
    for result in extraction_results:
        for entity in result.entities:
            deduper.add_entity(entity)

    # Relation dedupe pass (after entities so endpoint ids resolve).
    for result in extraction_results:
        for relation in result.relations:
            source_id = deduper.canonical_id_for(relation.source)
            target_id = deduper.canonical_id_for(relation.target)
            if not source_id or not target_id:
                continue
            relation_deduper.add_relation(
                relation,
                source_entity_id=source_id,
                target_entity_id=target_id,
            )

    entity_profiles = deduper.build_entity_profiles()
    for profile in entity_profiles:
        profile.relation_ids[:] = sorted(
            relation_deduper.relations_for_entity(profile.id)
        )
    relation_profiles = relation_deduper.build_relation_profiles()

    # Per-source contribution accounting.
    contributions: dict[str, SourceContribution] = {}
    for source in sources:
        contributions[source.source_id] = SourceContribution(
            source_id=source.source_id,
            source_hash=source.content_hash,
            chunk_ids=[],
            entity_ids=[],
            relation_ids=[],
            status="fresh",
        )
    for chunk in chunks:
        contribution = contributions.get(chunk.source_id)
        if contribution is not None and chunk.id not in contribution.chunk_ids:
            contribution.chunk_ids.append(chunk.id)
    for entity in entity_profiles:
        for sid in entity.source_ids:
            contribution = contributions.get(sid)
            if contribution is not None and entity.id not in contribution.entity_ids:
                contribution.entity_ids.append(entity.id)
    for relation in relation_profiles:
        for sid in relation.source_ids:
            contribution = contributions.get(sid)
            if (
                contribution is not None
                and relation.id not in contribution.relation_ids
            ):
                contribution.relation_ids.append(relation.id)

    # Flag missing sources (in previous index but not in current set) for
    # review rather than silently deleting their evidence. See §11.
    missing_count = 0
    if previous_index is not None:
        prev_sources = {c.source_id for c in previous_index.contributions}
        current_source_ids = {s.source_id for s in sources}
        for missing_id in prev_sources - current_source_ids:
            missing_count += 1
            contributions[missing_id] = SourceContribution(
                source_id=missing_id,
                source_hash=previous_index.manifest.source_hashes.get(missing_id, ""),
                chunk_ids=[],
                entity_ids=[],
                relation_ids=[],
                status="missing",
                requires_review=True,
            )

    # Embedding step. The default tier is BM25 fallback (Tier C), but
    # callers can supply either a pre-built ``embedding_provider`` or a
    # full ``embedding_resolution`` (preferred) to opt into Tier A
    # provider-backed embeddings.
    embedding_tier = "fallback"
    embedding_tier_reason = "BM25 fallback (no embedding resolution supplied)"
    if embedding_resolution is not None:
        embedding_tier = embedding_resolution.tier
        embedding_tier_reason = embedding_resolution.reason
        if embedding_resolution.provider is not None and embedding_provider is None:
            embedding_provider = embedding_resolution.provider
    if embedding_provider is None:
        embedding_provider = _fit_default_embedder(
            [p.embedding_text for p in entity_profiles]
            + [p.embedding_text for p in relation_profiles]
        )
        if embedding_resolution is None:
            embedding_tier_reason = (
                "BM25 fallback (no embedding_provider or resolution supplied)"
            )

    try:
        entity_vectors = embedding_provider.embed_texts(
            [p.embedding_text or p.canonical_name for p in entity_profiles]
        )
        relation_vectors = embedding_provider.embed_texts(
            [p.embedding_text or p.relation_type for p in relation_profiles]
        )
        chunk_vectors: list[list[float]] | None = None
        if opts.embed_chunks:
            chunk_vectors = embedding_provider.embed_texts([c.text for c in chunks])
    except RuntimeError as exc:
        # Strict provider failed at embed-time (e.g. revoked API key).
        # Degrade to BM25 with a clear diagnostic rather than crashing.
        embedding_provider = _fit_default_embedder(
            [p.embedding_text for p in entity_profiles]
            + [p.embedding_text for p in relation_profiles]
        )
        embedding_tier = "fallback"
        embedding_tier_reason = (
            f"provider call failed ({exc}); degraded to BM25 fallback"
        )
        entity_vectors = embedding_provider.embed_texts(
            [p.embedding_text or p.canonical_name for p in entity_profiles]
        )
        relation_vectors = embedding_provider.embed_texts(
            [p.embedding_text or p.relation_type for p in relation_profiles]
        )
        chunk_vectors = None
        if opts.embed_chunks:
            chunk_vectors = embedding_provider.embed_texts([c.text for c in chunks])

    manifest = LightGraphBuildManifest(
        built_at=utc_now_iso(),
        source_hashes={s.source_id: s.content_hash for s in sources},
        chunking={
            "chunk_token_size": opts.chunk_token_size,
            "overlap_tokens": opts.overlap_tokens,
        },
        extraction_prompt_hash=extractor.prompt_hash,
        provider_identity=None,
        embedding_provider=getattr(embedding_provider, "model_name", "bm25"),
        embedding_model=getattr(embedding_provider, "model_name", "bm25-fallback"),
        embedding_dimension=int(getattr(embedding_provider, "dimension", 0)),
        embedding_tier=embedding_tier,
        embedding_tier_reason=embedding_tier_reason,
        extractor=extractor.name,
        index_schema_version=1,
        missing_sources=[
            {
                "source_id": cid,
                "status": "missing",
                "requires_review": True,
            }
            for cid in sorted(
                contributions[cid].source_id
                for cid in contributions
                if contributions[cid].status == "missing"
            )
        ],
    )

    index = LightGraphIndex(
        built_at=manifest.built_at,
        chunks=chunks,
        entities=entity_profiles,
        relations=relation_profiles,
        contributions=sorted(contributions.values(), key=lambda c: c.source_id),
        manifest=manifest,
    )

    artifacts: list[str] = []
    if store is not None:
        artifacts = store.save(
            index,
            entity_vectors=serialize_vectors(entity_profiles, entity_vectors),
            relation_vectors=serialize_vectors(relation_profiles, relation_vectors),
            chunk_vectors=(
                serialize_vectors(chunks, chunk_vectors)
                if chunk_vectors is not None
                else None
            ),
        )

    entity_dedupe_rate = (
        1.0 - (len(entity_profiles) / extracted_entity_count)
        if extracted_entity_count
        else 0.0
    )
    relation_dedupe_rate = (
        1.0 - (len(relation_profiles) / extracted_relation_count)
        if extracted_relation_count
        else 0.0
    )

    warnings: list[str] = []
    if not sources:
        warnings.append("no source records supplied; LightGraph is empty")
    if missing_count > 0:
        warnings.append(
            f"{missing_count} source(s) from the previous index are missing "
            "from the current manifest; marked as 'missing' for review."
        )

    report = LightGraphBuildReport(
        built_at=manifest.built_at,
        chunk_count=len(chunks),
        entity_count=len(entity_profiles),
        relation_count=len(relation_profiles),
        source_count=len(sources),
        missing_source_count=missing_count,
        extracted_entity_count=extracted_entity_count,
        extracted_relation_count=extracted_relation_count,
        entity_dedupe_rate=round(entity_dedupe_rate, 4),
        relation_dedupe_rate=round(relation_dedupe_rate, 4),
        extractor=extractor.name,
        embedding_provider=manifest.embedding_provider,
        embedding_model=manifest.embedding_model,
        embedding_tier=embedding_tier,
        embedding_tier_reason=embedding_tier_reason,
        incremental=incremental,
        reused_source_count=len(reused_source_ids),
        reprocessed_source_count=len(reprocessed_source_ids),
        artifacts=artifacts,
        warnings=warnings,
    )
    return index, report


def _fit_default_embedder(corpus: list[str]) -> EmbeddingProvider:
    provider = BM25SparseEmbeddingProvider()
    provider.fit(corpus or [""])
    return provider


def _replay_extraction_from_previous_index(
    *,
    previous_index: LightGraphIndex,
    reused_chunks: list,
) -> list[LightExtractionResult]:
    """Reconstruct per-chunk extraction results from a previous index.

    For each reused chunk we synthesise one :class:`ExtractedEntity`
    per :class:`EntityProfile` that references the chunk (and similarly
    for relations). This is structurally equivalent to what the
    extractor would have produced and lets the deduper rebuild the
    canonical id set without re-running the extractor on unchanged
    sources — the incremental-update guarantee the LightRAG paper
    emphasises (avoid full index rebuilds when new data arrives).
    """
    reused_chunk_ids = {chunk.id for chunk in reused_chunks}
    if not reused_chunk_ids:
        return []
    entities_by_chunk: dict[str, list[ExtractedEntity]] = {
        cid: [] for cid in reused_chunk_ids
    }
    for profile in previous_index.entities:
        relevant = [cid for cid in profile.chunk_ids if cid in reused_chunk_ids]
        for chunk_id in relevant:
            entities_by_chunk[chunk_id].append(
                ExtractedEntity(
                    name=profile.canonical_name,
                    type=profile.type,
                    description=profile.description,
                    aliases=list(profile.aliases),
                    chunk_ids=[chunk_id],
                    source_ids=list(profile.source_ids),
                    confidence=0.99,
                )
            )
    relations_by_chunk: dict[str, list[ExtractedRelation]] = {
        cid: [] for cid in reused_chunk_ids
    }
    entity_id_to_name = {
        profile.id: profile.canonical_name for profile in previous_index.entities
    }
    for profile in previous_index.relations:
        relevant = [cid for cid in profile.chunk_ids if cid in reused_chunk_ids]
        source_name = entity_id_to_name.get(
            profile.source_entity_id, profile.source_entity_id
        )
        target_name = entity_id_to_name.get(
            profile.target_entity_id, profile.target_entity_id
        )
        for chunk_id in relevant:
            relations_by_chunk[chunk_id].append(
                ExtractedRelation(
                    source=source_name,
                    target=target_name,
                    relation_type=profile.relation_type,
                    description=profile.description,
                    keywords=list(profile.keywords),
                    chunk_ids=[chunk_id],
                    source_ids=list(profile.source_ids),
                    weight=profile.weight,
                    confidence=0.99,
                )
            )
    return [
        LightExtractionResult(
            chunk_id=chunk_id,
            entities=entities_by_chunk.get(chunk_id, []),
            relations=relations_by_chunk.get(chunk_id, []),
            warnings=[],
            extractor="replay-previous-index",
        )
        for chunk_id in sorted(reused_chunk_ids)
    ]
