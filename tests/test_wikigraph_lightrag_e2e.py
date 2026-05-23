"""End-to-end tests for the LightRAG-style index builder and query engine."""

from __future__ import annotations

from pathlib import Path

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.config_service import (
    DEFAULT_CONFIG,
    resolve_wikigraph_config,
)
from graphwiki_kb.services.manifest_service import ManifestService
from graphwiki_kb.services.project_service import build_project_paths
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryService
from graphwiki_kb.wikigraph.light_index_builder import (
    build_lightgraph_index,
)
from graphwiki_kb.wikigraph.light_query_service import (
    LightAnswerService,
    LightGraphQueryEngine,
    render_answer_prompt,
)


def _source(
    slug: str, *, source_id: str, content_hash: str | None = None
) -> RawSourceRecord:
    content_hash = content_hash or f"hash-{source_id}"
    return RawSourceRecord(
        source_id=source_id,
        slug=slug,
        title=slug.replace("-", " ").title(),
        origin="local",
        source_type="paper",
        raw_path=f"raw/sources/{slug}.pdf",
        content_hash=content_hash,
        ingested_at="2024-01-01T00:00:00Z",
        normalized_path=f"raw/normalized/{slug}.md",
    )


def _seed_project(tmp_path: Path) -> tuple[Path, list[RawSourceRecord]]:
    (tmp_path / "raw" / "normalized").mkdir(parents=True)
    (tmp_path / "graph").mkdir()
    (tmp_path / "wiki" / "sources").mkdir(parents=True)
    (tmp_path / "raw" / "normalized" / "dpr.md").write_text(
        "Dense Passage Retrieval (DPR) is a dual-encoder retriever for "
        "open-domain QA. DPR was evaluated on Natural Questions and TriviaQA. "
        "DPR uses BERT as the underlying encoder."
    )
    (tmp_path / "raw" / "normalized" / "rag.md").write_text(
        "Retrieval-Augmented Generation (RAG) combines a seq2seq generator "
        "with a non-parametric retriever. RAG uses DPR for retrieval. "
        "RAG was evaluated on Natural Questions and TriviaQA."
    )
    return tmp_path, [_source("dpr", source_id="s1"), _source("rag", source_id="s2")]


def test_build_lightgraph_index_produces_chunks_entities_relations(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    index, report = build_lightgraph_index(paths, sources)
    assert report.chunk_count >= 2
    assert report.entity_count > 0
    assert report.relation_count > 0
    assert report.extractor == "deterministic"
    assert report.embedding_provider == "bm25-sparse"
    # Source contributions exist for each source.
    contribution_ids = {c.source_id for c in index.contributions}
    assert contribution_ids == {"s1", "s2"}
    # Manifest captures source hashes for freshness detection.
    assert set(index.manifest.source_hashes) == {"s1", "s2"}


def test_lightgraph_query_engine_local_returns_entity_contexts(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    index, _ = build_lightgraph_index(paths, sources)
    engine = LightGraphQueryEngine(index=index)
    result = engine.find("What is DPR?", method="local")
    assert result.method == "local"
    assert result.contexts, "expected non-empty contexts"
    assert any(c.node_kind == "entity" for c in result.contexts)


def test_lightgraph_query_engine_hybrid_fuses_results(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    index, _ = build_lightgraph_index(paths, sources)
    engine = LightGraphQueryEngine(index=index)
    result = engine.find("Compare RAG and DPR", method="hybrid")
    assert result.method == "drift-lite"  # hybrid maps to classic drift-lite shape.
    diagnostics_blob = " ".join(result.diagnostics)
    assert "lightrag_method=hybrid" in diagnostics_blob


def test_lightgraph_query_engine_basic_returns_chunk_contexts(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    index, _ = build_lightgraph_index(paths, sources)
    engine = LightGraphQueryEngine(index=index)
    result = engine.find("dual encoder", method="basic")
    assert result.method == "basic"
    chunk_contexts = [c for c in result.contexts if c.node_kind == "chunk"]
    assert chunk_contexts, "basic mode should return chunk contexts"
    # LightRAG-converted contexts use the ``chunk`` node kind so the
    # citation_ref renders as ``path#chunk-N`` end-to-end (matching
    # what the answer prompt shows the model).
    for ctx in chunk_contexts:
        assert "#chunk-" in ctx.citation_ref


def test_lightanswer_service_provider_free_emits_citations(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    index, _ = build_lightgraph_index(paths, sources)
    engine = LightGraphQueryEngine(index=index)
    service = LightAnswerService(engine=engine, provider=None)
    answer = service.ask("Compare RAG and DPR", method="hybrid")
    assert answer.provider_status.get("mode") == "provider-free"
    assert answer.contexts
    assert "Evidence summary" in answer.answer
    # Citations point at the LightRAG-canonical refs returned by the
    # retrieval bundle. They may be entity / relation ids when the
    # hybrid retriever returns no chunk-kind contexts; the important
    # invariant is that every citation matches a returned context.
    refs = {c["ref"] for c in answer.citations}
    known_refs = {ctx.citation_ref for ctx in answer.contexts}
    assert refs.issubset(known_refs)
    assert refs


def test_render_answer_prompt_contains_sections(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    index, _ = build_lightgraph_index(paths, sources)
    engine = LightGraphQueryEngine(index=index)
    bundle = engine.retrieve_bundle("How does RAG use DPR?", method="hybrid")
    prompt = render_answer_prompt("How does RAG use DPR?", bundle)
    assert "# Retrieved entities" in prompt
    assert "# Retrieved relationships" in prompt
    assert "# Source excerpts" in prompt
    assert "[C" in prompt or "Source excerpts" in prompt


def test_incremental_update_keeps_unchanged_sources(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    index_v1, _ = build_lightgraph_index(paths, sources)

    # Modify source s2's content and hash.
    (tmp_path / "raw" / "normalized" / "rag.md").write_text(
        "Retrieval-Augmented Generation (RAG) now also evaluates on WebQuestions. "
        "RAG still uses DPR for retrieval."
    )
    sources_v2 = [
        sources[0],
        _source("rag", source_id="s2", content_hash="h-changed"),
    ]
    index_v2, report = build_lightgraph_index(
        paths, sources_v2, previous_index=index_v1
    )
    assert report.incremental is True
    # Source s1 hash unchanged.
    assert index_v2.manifest.source_hashes["s1"] == "hash-s1"
    assert index_v2.manifest.source_hashes["s2"] == "h-changed"


def test_missing_sources_marked_for_review_not_deleted(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    index_v1, _ = build_lightgraph_index(paths, sources)
    # Drop source s2 entirely.
    sources_v2 = [sources[0]]
    index_v2, report = build_lightgraph_index(
        paths, sources_v2, previous_index=index_v1
    )
    assert report.missing_source_count == 1
    statuses = {c.source_id: c.status for c in index_v2.contributions}
    assert statuses["s2"] == "missing"


def _seed_manifest(paths, sources) -> ManifestService:
    """Persist ``sources`` to the manifest and return a manifest service."""
    import json

    from graphwiki_kb.services.file_lock import file_lock
    from graphwiki_kb.services.project_service import atomic_write_text, utc_now_iso

    manifest_service = ManifestService(paths)
    payload = {
        "version": 1,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "sources": [s.to_dict() for s in sources],
    }
    paths.raw_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(paths.raw_manifest_file):
        atomic_write_text(paths.raw_manifest_file, json.dumps(payload, indent=2))
    return manifest_service


def test_wikigraph_index_service_dispatches_lightrag_mode(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    manifest_service = _seed_manifest(paths, sources)

    config = {"wikigraph": {**DEFAULT_CONFIG["wikigraph"], "mode": "lightrag"}}
    runtime = resolve_wikigraph_config(config)
    assert runtime.mode == "lightrag"

    service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest_service
    )
    report = service.build()
    assert report.chunk_count >= 1
    assert report.entity_count >= 1
    # Status payload exposes the LightRAG block.
    status = service.status()
    assert status["mode"] == "lightrag"
    light_block = status["lightrag"]
    assert isinstance(light_block, dict)
    assert light_block.get("initialized") is True
    assert light_block.get("chunk_count", 0) >= 1


def test_wikigraph_query_service_routes_lightrag(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    manifest_service = _seed_manifest(paths, sources)

    config = {"wikigraph": {**DEFAULT_CONFIG["wikigraph"], "mode": "lightrag"}}
    index_service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest_service
    )
    index_service.build()
    query_service = WikiGraphQueryService(
        paths=paths,
        index_service=index_service,
        provider=None,
        config=config,
    )
    result = query_service.find("Compare RAG and DPR", method="drift-lite")
    assert result.contexts, "expected non-empty contexts from lightrag find"
    answer = query_service.ask("What is DPR?", method="local")
    assert answer.provider_status.get("mode") == "provider-free"
