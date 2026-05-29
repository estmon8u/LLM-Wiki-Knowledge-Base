"""Tests for the LightRAG local vector store."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphwiki_kb.wikigraph import light_vector_store
from graphwiki_kb.wikigraph.light_vector_store import (
    LightVectorStore,
    cosine_top_k,
)


def test_cosine_top_k_orders_by_similarity() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    # Normalize as the store would.
    store = LightVectorStore.from_embeddings(
        ["a", "b", "c"], vectors, model="m", dimension=2
    )
    hits = store.search([1.0, 0.0], k=3)
    assert hits[0][0] == "a"
    assert hits[0][1] == pytest.approx(1.0)
    # "c" (45 deg) ranks above "b" (90 deg).
    assert [hit[0] for hit in hits] == ["a", "c", "b"]


def test_cosine_top_k_empty_and_zero_k() -> None:
    assert cosine_top_k([1.0], [], 3) == []
    assert cosine_top_k([1.0], [[1.0]], 0) == []


def test_cosine_top_k_pure_python_matches_numpy() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    store = LightVectorStore.from_embeddings(
        ["a", "b", "c"], vectors, model="m", dimension=2
    )
    # Force pure-python path by passing a sentinel "no numpy".
    py_hits = cosine_top_k([1.0, 0.2], store.vectors, 3, numpy_module=None)
    # And the default path (numpy if installed).
    default_hits = cosine_top_k([1.0, 0.2], store.vectors, 3)
    assert [i for i, _ in py_hits] == [i for i, _ in default_hits]


def test_normalization_of_zero_vector() -> None:
    store = LightVectorStore.from_embeddings(
        ["z"], [[0.0, 0.0]], model="m", dimension=2
    )
    assert store.vectors == [[0.0, 0.0]]
    assert store.search([1.0, 0.0], k=1) == [("z", pytest.approx(0.0))]


def test_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        LightVectorStore.from_embeddings(
            ["a"], [[1.0, 2.0, 3.0]], model="m", dimension=2
        )


def test_ids_vectors_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        LightVectorStore.from_embeddings(["a", "b"], [[1.0]], model="m", dimension=1)


def test_dimension_inferred_when_zero() -> None:
    store = LightVectorStore.from_embeddings(
        ["a"], [[3.0, 4.0]], model="m", dimension=0
    )
    assert store.dimension == 2
    assert store.vectors[0] == pytest.approx([0.6, 0.8])


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = LightVectorStore.from_embeddings(
        ["a", "b"],
        [[1.0, 0.0], [0.0, 2.0]],
        model="text-embedding-3-large",
        dimension=2,
    )
    path = tmp_path / "vectors.json"
    store.save(path)
    loaded = LightVectorStore.load(path)
    assert loaded is not None
    assert loaded.model == "text-embedding-3-large"
    assert loaded.dimension == 2
    assert loaded.ids == ["a", "b"]
    assert len(loaded) == 2
    assert loaded.search([0.0, 1.0], k=1)[0][0] == "b"


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert LightVectorStore.load(tmp_path / "nope.json") is None


def test_load_corrupt_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert LightVectorStore.load(path) is None


def test_get_numpy_returns_module_or_none() -> None:
    # Whatever the environment, the helper must not raise.
    result = light_vector_store._get_numpy()
    assert result is None or hasattr(result, "asarray")
