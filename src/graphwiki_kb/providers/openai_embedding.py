"""OpenAI embedding provider for the LightRAG-style backend.

Wraps the ``openai`` Python SDK's ``embeddings.create`` endpoint. Defers
the SDK import so projects that do not configure an OpenAI provider
never need to install it. Raises a clear :class:`RuntimeError` when the
API key env var is missing — callers (e.g. the LightGraph builder)
should catch this and fall back to BM25 with a labeled diagnostic, per
project recommendation §6.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class OpenAIEmbeddingProvider:
    """Embed texts via the OpenAI embeddings API.

    Attributes:
        model_name: OpenAI embedding model id (e.g.
            ``text-embedding-3-large``).
        api_key_env: Name of the environment variable holding the API
            key. Defaults to ``OPENAI_API_KEY``.
        expected_dimension: Optional expected vector dimension. When set
            and non-zero, the provider raises if the API returns a
            different dimension (defends against silent model swaps).
    """

    model_name: str
    api_key_env: str = "OPENAI_API_KEY"
    expected_dimension: int = 0

    def __post_init__(self) -> None:
        # Cached dimension is learned on first successful call when the
        # caller did not configure ``expected_dimension``.
        self._observed_dimension: int = int(self.expected_dimension)

    @property
    def dimension(self) -> int:
        """Vector dimension produced by :meth:`embed_texts`."""
        return self._observed_dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Call the OpenAI embeddings endpoint for ``texts``.

        Returns the vectors in the same order as the input. Raises
        :class:`RuntimeError` when the API key env is unset or the
        ``openai`` SDK is not installed.
        """
        if not texts:
            return []
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing API key in environment variable {self.api_key_env}"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "openai package is required for OpenAI embeddings. "
                "Add it to your environment to use this provider."
            ) from exc
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=self.model_name, input=list(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in ordered]
        if vectors:
            observed = len(vectors[0])
            if self.expected_dimension and observed != self.expected_dimension:
                raise RuntimeError(
                    "OpenAI embedding dimension mismatch: expected "
                    f"{self.expected_dimension}, got {observed}"
                )
            self._observed_dimension = observed
        return vectors
