"""Unit tests for LightRAG vector helpers."""

from __future__ import annotations

from graphwiki_kb.wikigraph.light_vector_store import cosine_top_k


def test_cosine_top_k_orders_by_similarity() -> None:
    matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.0],
    ]
    hits = cosine_top_k([1.0, 0.0, 0.0], matrix, k=2)
    assert hits[0][0] == 0
    assert hits[1][0] == 2
