"""Resolve LightRAG embedding providers from project config.

Bridges the strict ``wikigraph.lightrag.embeddings`` config block with
the concrete provider classes under :mod:`graphwiki_kb.providers`.

The resolver intentionally has two outcomes, not three:

* **Strict provider available** — credentials are present and the
  configured ``provider`` resolves to a real backend. Returns a
  :class:`ResolvedEmbedding` with ``tier == "strict"``.
* **Fallback** — either the configured provider is ``bm25``/``hashing``,
  the provider is unknown, or credentials are missing. Returns a
  :class:`ResolvedEmbedding` with ``tier == "fallback"`` and a
  :attr:`reason` string suitable for surfacing in CLI diagnostics and
  the LightGraph build manifest.

Callers are expected to label fallback runs honestly (per project
recommendation §6 / §24 Tier C "fallback diagnostic mode"). The build
report and ``kb status`` ``wikigraph.lightrag`` block both include the
chosen tier so a strict-vs-diagnostic comparison cannot accidentally
be presented as strict LightRAG.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from graphwiki_kb.providers.embedding_base import EmbeddingProvider
from graphwiki_kb.providers.gemini_embedding import GeminiEmbeddingProvider
from graphwiki_kb.providers.openai_embedding import OpenAIEmbeddingProvider

EmbeddingTier = Literal["strict", "fallback"]

_FALLBACK_PROVIDERS = {"bm25", "hashing", "none", ""}


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
    """Resolved embedding settings derived from ``wikigraph.lightrag.embeddings``."""

    provider: str
    model: str
    dimension: int
    local_fallback: str
    api_key_env: str | None = None


@dataclass(frozen=True)
class ResolvedEmbedding:
    """A resolved embedding decision the LightGraph builder can act on.

    Attributes:
        provider: An instantiated :class:`EmbeddingProvider` when
            ``tier == "strict"``; ``None`` when ``tier == "fallback"``
            (callers should build the BM25/hashing fallback themselves).
        tier: ``"strict"`` for provider-backed embeddings,
            ``"fallback"`` otherwise.
        runtime: The :class:`EmbeddingRuntimeConfig` that was resolved.
        reason: Short human-readable explanation for why ``tier`` was
            chosen. Always populated.
    """

    provider: EmbeddingProvider | None
    tier: EmbeddingTier
    runtime: EmbeddingRuntimeConfig
    reason: str


def resolve_lightrag_embedding_config(
    config: dict[str, Any],
) -> EmbeddingRuntimeConfig:
    """Resolve the embedding sub-section of ``wikigraph.lightrag``.

    Falls back to BM25 defaults when ``wikigraph.lightrag.embeddings``
    is missing or malformed so the LightGraph builder never raises
    purely because of an under-configured project.
    """
    wikigraph = config.get("wikigraph") if isinstance(config, dict) else None
    lightrag = wikigraph.get("lightrag") if isinstance(wikigraph, dict) else None
    embeddings = lightrag.get("embeddings") if isinstance(lightrag, dict) else None
    if not isinstance(embeddings, dict):
        return EmbeddingRuntimeConfig(
            provider="bm25",
            model="bm25-fallback",
            dimension=0,
            local_fallback="bm25",
        )
    provider = str(embeddings.get("provider", "bm25")).strip().lower()
    model = str(embeddings.get("model", "bm25-fallback")).strip()
    dimension = int(embeddings.get("dimension", 0) or 0)
    local_fallback = str(embeddings.get("local_fallback", "bm25")).strip().lower()
    api_key_env_raw = embeddings.get("api_key_env")
    if isinstance(api_key_env_raw, str) and api_key_env_raw.strip():
        api_key_env: str | None = api_key_env_raw.strip()
    else:
        api_key_env = _default_api_key_env(provider)
    return EmbeddingRuntimeConfig(
        provider=provider,
        model=model,
        dimension=dimension,
        local_fallback=local_fallback,
        api_key_env=api_key_env,
    )


def build_embedding_provider(
    runtime: EmbeddingRuntimeConfig,
    *,
    environ: dict[str, str] | None = None,
) -> ResolvedEmbedding:
    """Instantiate the configured embedding provider, with fallback rules.

    Args:
        runtime: The resolved embedding runtime config.
        environ: Optional environment mapping. When ``None``, uses
            :mod:`os.environ`. Useful for tests.
    """
    env = environ if environ is not None else os.environ
    provider_name = runtime.provider
    if provider_name in _FALLBACK_PROVIDERS:
        return ResolvedEmbedding(
            provider=None,
            tier="fallback",
            runtime=runtime,
            reason=f"provider '{provider_name}' is a local fallback",
        )
    if runtime.api_key_env:
        api_key = env.get(runtime.api_key_env, "").strip()
        if not api_key:
            return ResolvedEmbedding(
                provider=None,
                tier="fallback",
                runtime=runtime,
                reason=(
                    f"missing API key in env {runtime.api_key_env!r}; "
                    "falling back to BM25"
                ),
            )
    if provider_name == "openai":
        return ResolvedEmbedding(
            provider=OpenAIEmbeddingProvider(
                model_name=runtime.model,
                api_key_env=runtime.api_key_env or "OPENAI_API_KEY",
                expected_dimension=runtime.dimension,
            ),
            tier="strict",
            runtime=runtime,
            reason=f"openai provider configured ({runtime.model})",
        )
    if provider_name in {"gemini", "google"}:
        return ResolvedEmbedding(
            provider=GeminiEmbeddingProvider(
                model_name=runtime.model,
                api_key_env=runtime.api_key_env or "GEMINI_API_KEY",
                expected_dimension=runtime.dimension,
            ),
            tier="strict",
            runtime=runtime,
            reason=f"gemini provider configured ({runtime.model})",
        )
    return ResolvedEmbedding(
        provider=None,
        tier="fallback",
        runtime=runtime,
        reason=f"unknown embedding provider {provider_name!r}; falling back to BM25",
    )


def _default_api_key_env(provider: str) -> str | None:
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider in {"gemini", "google"}:
        return "GEMINI_API_KEY"
    return None
