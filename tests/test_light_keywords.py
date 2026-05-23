"""Unit tests for provider-free keyword extraction."""

from __future__ import annotations

from graphwiki_kb.wikigraph.light_keywords import extract_query_keywords
from graphwiki_kb.wikigraph.light_models import EntityProfile


def test_fallback_keywords_detect_entity_and_theme() -> None:
    catalog = [
        EntityProfile(
            id="entity:rag",
            canonical_name="RAG",
            type="METHOD",
            aliases=["Retrieval-Augmented Generation"],
            description="",
            profile_text="",
            keywords=[],
            chunk_ids=[],
            source_ids=[],
            relation_ids=[],
            embedding_text="RAG",
            updated_at="",
        )
    ]
    keywords = extract_query_keywords(
        "How does RAG trade off latency and factuality?",
        entity_catalog=catalog,
    )
    assert "RAG" in keywords.low_level_keywords
    assert keywords.high_level_keywords
