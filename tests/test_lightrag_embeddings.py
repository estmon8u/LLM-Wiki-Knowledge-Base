"""Tests for the LightRAG embedding-provider layer (no network)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from graphwiki_kb.providers import embedding_base, openai_embedding
from graphwiki_kb.providers.embedding_base import (
    EmbeddingConfigurationError,
    EmbeddingExecutionError,
    LazyEmbeddingProvider,
    UnavailableEmbeddingProvider,
)
from graphwiki_kb.services.config_service import DEFAULT_CONFIG
from graphwiki_kb.services.embedding_service import build_embedding_provider

# --------------------------------------------------------------------------- #
# Fakes                                                                       #
# --------------------------------------------------------------------------- #


class _FakeEmbeddingItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, data: list[_FakeEmbeddingItem]) -> None:
        self.data = data


class _FakeEmbeddings:
    def __init__(self, recorder: dict[str, Any]) -> None:
        self._recorder = recorder

    def create(self, **kwargs: Any) -> _FakeEmbeddingResponse:
        self._recorder.update(kwargs)
        texts = list(kwargs["input"])
        # Return vectors out of order to verify the provider re-sorts by index.
        items = [
            _FakeEmbeddingItem(index=i, embedding=[float(i), float(len(text))])
            for i, text in enumerate(texts)
        ]
        return _FakeEmbeddingResponse(list(reversed(items)))


class _FakeOpenAIClient:
    last_recorder: dict[str, Any] = {}

    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key
        _FakeOpenAIClient.last_recorder = {}
        self.embeddings = _FakeEmbeddings(_FakeOpenAIClient.last_recorder)


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def test_build_returns_lazy_openai_provider() -> None:
    provider = build_embedding_provider(deepcopy(DEFAULT_CONFIG))
    assert isinstance(provider, LazyEmbeddingProvider)
    assert provider.name == "openai"
    assert provider.model_name == "text-embedding-3-large"
    assert provider.dimension == 3072


def test_build_returns_none_for_unsupported_provider() -> None:
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["embeddings"]["provider"] = "anthropic"
    assert build_embedding_provider(cfg) is None


def test_build_returns_lazy_gemini_provider() -> None:
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["embeddings"]["provider"] = "gemini"
    cfg["embeddings"]["model"] = "text-embedding-004"
    provider = build_embedding_provider(cfg)
    assert isinstance(provider, LazyEmbeddingProvider)
    assert provider.name == "gemini"


# --------------------------------------------------------------------------- #
# OpenAI embedding behavior                                                   #
# --------------------------------------------------------------------------- #


def test_openai_embed_texts_orders_and_forwards_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_embedding, "OpenAI", _FakeOpenAIClient)
    provider = build_embedding_provider(deepcopy(DEFAULT_CONFIG))
    assert provider is not None

    vectors = provider.embed_texts(["alpha", "bb"])
    # Re-sorted by index even though the fake returned reversed order.
    assert vectors == [[0.0, 5.0], [1.0, 2.0]]
    assert _FakeOpenAIClient.last_recorder["dimensions"] == 3072
    assert _FakeOpenAIClient.last_recorder["model"] == "text-embedding-3-large"


def test_openai_embed_empty_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_embedding, "OpenAI", _FakeOpenAIClient)
    provider = build_embedding_provider(deepcopy(DEFAULT_CONFIG))
    assert provider is not None
    assert provider.embed_texts([]) == []


def test_openai_count_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BadEmbeddings:
        def create(self, **kwargs: Any) -> _FakeEmbeddingResponse:
            return _FakeEmbeddingResponse([_FakeEmbeddingItem(0, [0.1])])

    class _BadClient:
        def __init__(self, *, api_key: str) -> None:
            self.embeddings = _BadEmbeddings()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_embedding, "OpenAI", _BadClient)
    provider = build_embedding_provider(deepcopy(DEFAULT_CONFIG))
    assert provider is not None
    with pytest.raises(EmbeddingExecutionError):
        provider.embed_texts(["a", "b"])


def test_missing_api_key_makes_provider_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = build_embedding_provider(deepcopy(DEFAULT_CONFIG))
    assert isinstance(provider, LazyEmbeddingProvider)
    with pytest.raises(EmbeddingConfigurationError):
        provider.ensure_available()
    with pytest.raises(EmbeddingConfigurationError):
        provider.embed_texts(["x"])


# --------------------------------------------------------------------------- #
# Base classes                                                                #
# --------------------------------------------------------------------------- #


def test_unavailable_embedding_provider_raises() -> None:
    provider = UnavailableEmbeddingProvider("no key", provider_name="openai")
    assert provider.dimension == 0
    with pytest.raises(EmbeddingConfigurationError):
        provider.embed_texts(["x"])
    with pytest.raises(EmbeddingConfigurationError):
        provider.ensure_available()


def test_lazy_provider_updates_metadata_on_resolve() -> None:
    class _Concrete:
        name = "fake"
        model_name = "fake-model"
        dimension = 4

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0, 3.0, 4.0] for _ in texts]

    lazy = LazyEmbeddingProvider(
        _Concrete,
        provider_name="placeholder",
        model_name="placeholder-model",
        dimension=0,
    )
    out = lazy.embed_texts(["a"])
    assert out == [[1.0, 2.0, 3.0, 4.0]]
    assert lazy.name == "fake"
    assert lazy.model_name == "fake-model"
    assert lazy.dimension == 4


def test_embedding_provider_protocol_runtime_checkable() -> None:
    class _Concrete:
        name = "x"
        model_name = "m"
        dimension = 2

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return []

    assert isinstance(_Concrete(), embedding_base.EmbeddingProvider)
