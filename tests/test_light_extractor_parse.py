"""Tests for deterministic LightRAG extraction parsing."""

from __future__ import annotations

from graphwiki_kb.wikigraph.light_extractor import (
    LightExtractionConfig,
    extract_entities_and_relations,
)
from graphwiki_kb.wikigraph.light_models import LightChunk


def test_deterministic_extractor_finds_capitalized_entities() -> None:
    chunk = LightChunk(
        id="chunk:src:0:abc",
        source_id="src",
        source_slug="src",
        normalized_path="raw/normalized/src.md",
        chunk_index=0,
        token_count=20,
        text="REALM improves retrieval. RAG uses dense retrievers.",
        content_hash="abc",
    )
    config = LightExtractionConfig(
        entity_types=("METHOD", "MODEL"),
        relation_types=("USES",),
    )
    result = extract_entities_and_relations(chunk, provider=None, config=config)
    names = {entity.name for entity in result.entities}
    assert "REALM" in names
    assert "RAG" in names
