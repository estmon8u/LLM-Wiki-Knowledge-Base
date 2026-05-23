"""Embedding provider protocol for LightRAG-style retrieval."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Protocol for text embedding backends."""

    model_name: str
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        ...
