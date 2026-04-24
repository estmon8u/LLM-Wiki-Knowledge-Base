from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderRequest:
    prompt: str
    system_prompt: str = ""
    max_tokens: int = 1024
    response_schema: dict[str, Any] | None = None
    response_schema_name: str = "response"
    reasoning_effort: str | None = None


@dataclass
class ProviderResponse:
    text: str
    model_name: str
    provider: str = ""
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw: Any | None = None


class TextProvider:
    """Base class for LLM text-generation providers."""

    name: str = "base"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Return a normalized text completion."""
        raise NotImplementedError
