"""Unit tests for LightRAG-style WikiGraphRAG models, chunker, store, and embedder."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.project_service import build_project_paths
from graphwiki_kb.wikigraph.light_chunker import (
    LightChunkerOptions,
    build_light_chunks,
    whitespace_tokenize,
)
from graphwiki_kb.wikigraph.light_embeddings import (
    BM25SparseEmbeddingProvider,
    HashingEmbeddingProvider,
)
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
    serialize_vectors,
)
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphBuildManifest,
    LightGraphIndex,
    LightRetrievedContext,
    RelationProfile,
    SourceContribution,
)
from graphwiki_kb.wikigraph.light_vector_store import LightVectorStore


def _make_record(
    *,
    source_id: str = "s1",
    slug: str = "doc",
    normalized_path: str = "raw/normalized/doc.md",
    content_hash: str = "h",
) -> RawSourceRecord:
    return RawSourceRecord(
        source_id=source_id,
        slug=slug,
        title=slug.replace("-", " ").title(),
        origin="local",
        source_type="paper",
        raw_path=f"raw/sources/{slug}.pdf",
        content_hash=content_hash,
        ingested_at="2024-01-01T00:00:00Z",
        normalized_path=normalized_path,
    )


def test_whitespace_tokenize_counts_words():
    assert whitespace_tokenize("Hello world, hello.") == [
        "Hello",
        "world,",
        "hello.",
    ]


def test_build_light_chunks_respects_token_size(tmp_path: Path):
    paths = build_project_paths(tmp_path)
    (tmp_path / "raw" / "normalized").mkdir(parents=True)
    body = " ".join(f"word{i}" for i in range(120))
    (tmp_path / "raw" / "normalized" / "doc.md").write_text(body)
    record = _make_record()
    chunks = build_light_chunks(
        root=paths.root,
        sources=[record],
        options=LightChunkerOptions(
            chunk_token_size=40, overlap_tokens=5, min_tokens=1
        ),
    )
    assert len(chunks) >= 3
    assert all(isinstance(c, LightChunk) for c in chunks)
    # Deterministic ids derived from content hash.
    ids = [c.id for c in chunks]
    assert ids == sorted(set(ids), key=ids.index)  # unique, insertion-ordered.
    # Each chunk references the same source.
    assert {c.source_id for c in chunks} == {"s1"}
    # First chunk has start_char==0, last chunk ends at len(body).
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(body)


def test_build_light_chunks_skips_missing_normalized(tmp_path: Path):
    paths = build_project_paths(tmp_path)
    record = _make_record(normalized_path="raw/normalized/missing.md")
    chunks = build_light_chunks(root=paths.root, sources=[record])
    assert chunks == []


def test_hashing_embedding_provider_is_normalized():
    provider = HashingEmbeddingProvider(dimension=128)
    vectors = provider.embed_texts(["hello world", "another doc"])
    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == 128
        norm = sum(v * v for v in vec) ** 0.5
        assert pytest.approx(norm, rel=1e-3) == 1.0


def test_bm25_embedding_provider_ranks_similar_docs_higher():
    provider = BM25SparseEmbeddingProvider()
    corpus = [
        "Dense Passage Retrieval uses a dual encoder",
        "Retrieval Augmented Generation combines a generator",
        "The Eiffel Tower is in Paris",
    ]
    provider.fit(corpus)
    vectors = provider.embed_texts([*corpus, "dense retrieval encoder"])
    store = LightVectorStore()
    for idx, vec in enumerate(vectors[:3]):
        store.add(f"doc{idx}", vec)
    hits = store.search(vectors[3], top_k=3)
    assert hits[0].id == "doc0"
    assert hits[0].score > hits[2].score


def test_vector_store_rejects_dimension_mismatch():
    store = LightVectorStore()
    store.add("a", [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        store.add("b", [0.1, 0.2])
    with pytest.raises(ValueError):
        store.search([0.1], top_k=1)


def test_light_graph_store_roundtrip(tmp_path: Path):
    store_root = tmp_path / "graph" / "wikigraph" / "lightrag"
    store = LightGraphStore(LightGraphStorePaths(store_root))
    assert store.exists() is False
    manifest = LightGraphBuildManifest(
        built_at="2024-01-01T00:00:00Z",
        source_hashes={"s1": "h1"},
        chunking={"chunk_token_size": 1200, "overlap_tokens": 100},
        extraction_prompt_hash="prompt-hash",
        embedding_provider="bm25",
        embedding_model="bm25-fallback",
        embedding_dimension=4,
        extractor="deterministic",
    )
    chunk = LightChunk(
        id="chunk:s1:0:abc",
        source_id="s1",
        source_slug="doc",
        source_title="Doc",
        normalized_path="raw/normalized/doc.md",
        chunk_index=0,
        token_count=10,
        text="hello world",
        content_hash="abc",
    )
    entity = EntityProfile(
        id="entity:doc:1",
        canonical_name="Doc",
        type="PAPER",
        chunk_ids=[chunk.id],
        source_ids=["s1"],
        embedding_text="Doc PAPER",
    )
    relation = RelationProfile(
        id="relation:1",
        source_entity_id="entity:doc:1",
        target_entity_id="entity:doc:1",
        relation_type="SUPPORTS",
        chunk_ids=[chunk.id],
        source_ids=["s1"],
        embedding_text="SUPPORTS",
    )
    contribution = SourceContribution(
        source_id="s1",
        source_hash="h1",
        chunk_ids=[chunk.id],
        entity_ids=[entity.id],
        relation_ids=[relation.id],
    )
    index = LightGraphIndex(
        built_at=manifest.built_at,
        chunks=[chunk],
        entities=[entity],
        relations=[relation],
        contributions=[contribution],
        manifest=manifest,
    )
    artifacts = store.save(
        index,
        entity_vectors=[(entity.id, [1.0, 0.0, 0.0, 0.0])],
        relation_vectors=[(relation.id, [0.0, 1.0, 0.0, 0.0])],
    )
    assert artifacts, "expected store.save() to report at least one artifact"
    assert store.exists() is True
    loaded = store.load()
    assert loaded is not None
    assert loaded.chunk_count == 1
    assert loaded.entity_count == 1
    assert loaded.relation_count == 1
    assert loaded.manifest.extraction_prompt_hash == "prompt-hash"
    # Vectors persisted and loadable by kind.
    entity_vectors = store.load_vectors("entity")
    assert entity_vectors == [(entity.id, [1.0, 0.0, 0.0, 0.0])]
    relation_vectors = store.load_vectors("relation")
    assert relation_vectors[0][0] == relation.id


def test_serialize_vectors_pairs_ids():
    e1 = EntityProfile(
        id="entity:a",
        canonical_name="A",
        type="MODEL",
        embedding_text="a",
    )
    e2 = EntityProfile(
        id="entity:b",
        canonical_name="B",
        type="MODEL",
        embedding_text="b",
    )
    paired = serialize_vectors([e1, e2], [[1.0, 0.0], [0.0, 1.0]])
    assert paired == [("entity:a", [1.0, 0.0]), ("entity:b", [0.0, 1.0])]


def test_light_retrieved_context_citation_ref_for_chunks():
    ctx = LightRetrievedContext(
        kind="chunk",
        id="chunk:s1:3:hash",
        title="Doc",
        score=0.5,
        path="raw/normalized/doc.md",
        chunk_index=3,
        source_ids=["s1"],
    )
    assert ctx.citation_ref == "raw/normalized/doc.md#chunk-3"
    entity_ctx = LightRetrievedContext(
        kind="entity",
        id="entity:doc",
        title="Doc",
        score=0.5,
    )
    assert entity_ctx.citation_ref == "entity:doc"
