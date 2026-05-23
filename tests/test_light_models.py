"""Unit tests for LightRAG Pydantic models."""

from __future__ import annotations

from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphIndex,
)


def test_light_graph_index_counts() -> None:
    chunk = LightChunk(
        id="chunk:src:0:abc",
        source_id="src",
        source_slug="src",
        normalized_path="raw/normalized/src.md",
        chunk_index=0,
        token_count=10,
        text="REALM improves retrieval.",
        content_hash="abc",
    )
    entity = EntityProfile(
        id="entity:realm",
        canonical_name="REALM",
        type="METHOD",
        aliases=[],
        description="Retriever pretraining.",
        profile_text="Entity: REALM",
        keywords=[],
        chunk_ids=[chunk.id],
        source_ids=["src"],
        relation_ids=[],
        embedding_text="REALM METHOD",
        updated_at="2026-01-01T00:00:00Z",
    )
    index = LightGraphIndex(
        built_at="2026-01-01T00:00:00Z",
        chunks=[chunk],
        entities=[entity],
        relations=[],
        source_hashes={"src": "hash"},
        extraction_prompt_hash="prompt",
        embedding_model="bm25",
        embedding_dimension=0,
        chunk_count=1,
        entity_count=1,
        relation_count=0,
    )
    assert index.chunk_count == 1
    assert index.entity_count == 1
