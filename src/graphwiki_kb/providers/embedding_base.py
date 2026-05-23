"""Embedding provider abstractions for LightRAG-style retrieval.

Lives next to the LLM completion providers (``openai_provider.py``,
``gemini_provider.py``, ``anthropic_provider.py``) because embedding
providers are an orthogonal capability, not a property of every
completion provider — see project recommendation §6 / §24.

The protocol mirrors the structural type used by
:mod:`graphwiki_kb.wikigraph.light_embeddings` so the BM25 / hashing
fallbacks can satisfy it without inheriting from anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal interface every LightRAG embedding provider must implement.

    ``model_name`` and ``dimension`` are used in the LightGraph build
    manifest and in status payloads so callers can verify that the
    persisted vectors match the active provider before serving queries.
    """

    model_name: str

    @property
    def dimension(self) -> int:
        """Vector dimension produced by :meth:`embed_texts`."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text in the input order."""
        ...
