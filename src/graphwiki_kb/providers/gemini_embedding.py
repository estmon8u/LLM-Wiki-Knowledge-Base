"""Gemini embedding provider for the LightRAG-style backend.

Wraps the ``google-genai`` SDK's ``models.embed_content`` endpoint.
Defers the SDK import so projects that do not configure a Gemini
provider never need to install it. See
:mod:`graphwiki_kb.providers.openai_embedding` for the matching
behaviour around missing API keys and dimension validation.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class GeminiEmbeddingProvider:
    """Embed texts via the Google GenAI embedding endpoint.

    Attributes:
        model_name: Gemini embedding model id (e.g.
            ``text-embedding-004``).
        api_key_env: Name of the environment variable holding the API
            key. Defaults to ``GEMINI_API_KEY``.
        expected_dimension: Optional expected vector dimension. When set
            and non-zero, the provider raises if the API returns a
            different dimension.
    """

    model_name: str
    api_key_env: str = "GEMINI_API_KEY"
    expected_dimension: int = 0

    def __post_init__(self) -> None:
        self._observed_dimension: int = int(self.expected_dimension)

    @property
    def dimension(self) -> int:
        """Vector dimension produced by :meth:`embed_texts`."""
        return self._observed_dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Call the Gemini embedding endpoint once per input text."""
        if not texts:
            return []
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing API key in environment variable {self.api_key_env}"
            )
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "google-genai package is required for Gemini embeddings. "
                "Add it to your environment to use this provider."
            ) from exc
        client = genai.Client(api_key=api_key)
        vectors: list[list[float]] = []
        for text in texts:
            result = client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            embedding = list(result.embeddings[0].values)
            vectors.append(embedding)
        if vectors:
            observed = len(vectors[0])
            if self.expected_dimension and observed != self.expected_dimension:
                raise RuntimeError(
                    "Gemini embedding dimension mismatch: expected "
                    f"{self.expected_dimension}, got {observed}"
                )
            self._observed_dimension = observed
        return vectors
