from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProviderRequest:
    prompt: str
    system_prompt: str = ""
    max_tokens: int = 1024


@dataclass
class ProviderResponse:
    text: str
    model_name: str


class TextProvider:
    """Base class for LLM text-generation providers."""

    name: str = "base"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Return a normalized text completion."""
        raise NotImplementedError
