"""Resolve embedding providers from project config."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from graphwiki_kb.providers.embedding_base import EmbeddingProvider
from graphwiki_kb.providers.gemini_embedding import GeminiEmbeddingProvider
from graphwiki_kb.providers.openai_embedding import OpenAIEmbeddingProvider


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
    """Resolved embedding settings."""

    provider: str
    model: str
    dimension: int
    api_key_env: str | None = None


def resolve_embedding_config(config: dict[str, Any]) -> EmbeddingRuntimeConfig:
    """Resolve top-level ``embeddings`` config with graph fallbacks."""
    embeddings = config.get("embeddings")
    if isinstance(embeddings, dict):
        provider = str(embeddings.get("provider", "openai")).strip().lower()
        model = str(embeddings.get("model", "text-embedding-3-large")).strip()
        dimension = int(embeddings.get("dimension", 3072))
        api_key_env = embeddings.get("api_key_env")
        if isinstance(api_key_env, str) and api_key_env.strip():
            key_env = api_key_env.strip()
        else:
            key_env = _default_api_key_env(provider)
        return EmbeddingRuntimeConfig(
            provider=provider,
            model=model,
            dimension=dimension,
            api_key_env=key_env,
        )
    graph = config.get("graph", {})
    if isinstance(graph, dict):
        provider = str(graph.get("embedding_provider", "openai")).strip().lower()
        model = str(graph.get("embedding_model", "text-embedding-3-large")).strip()
        return EmbeddingRuntimeConfig(
            provider=provider,
            model=model,
            dimension=3072,
            api_key_env=_default_api_key_env(provider),
        )
    return EmbeddingRuntimeConfig(
        provider="openai",
        model="text-embedding-3-large",
        dimension=3072,
        api_key_env="OPENAI_API_KEY",
    )


def build_embedding_provider(
    runtime: EmbeddingRuntimeConfig,
) -> EmbeddingProvider | None:
    """Instantiate an embedding provider when credentials are available."""
    if runtime.api_key_env and not os.environ.get(runtime.api_key_env, "").strip():
        return None
    if runtime.provider == "openai":
        return OpenAIEmbeddingProvider(
            model_name=runtime.model,
            dimension=runtime.dimension,
            api_key_env=runtime.api_key_env or "OPENAI_API_KEY",
        )
    if runtime.provider in {"gemini", "google"}:
        return GeminiEmbeddingProvider(
            model_name=runtime.model,
            dimension=runtime.dimension,
            api_key_env=runtime.api_key_env or "GEMINI_API_KEY",
        )
    return None


def _default_api_key_env(provider: str) -> str:
    if provider in {"gemini", "google"}:
        return "GEMINI_API_KEY"
    return "OPENAI_API_KEY"
