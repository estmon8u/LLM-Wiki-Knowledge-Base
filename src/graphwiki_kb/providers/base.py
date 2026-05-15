"""Provider integration helpers for base.

This module belongs to `graphwiki_kb.providers.base` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    """Represents provider request behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    prompt: str
    system_prompt: str = ""
    max_tokens: int = 1024
    response_schema: dict[str, Any] | None = None
    response_schema_name: str = "response"
    reasoning_effort: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Represents provider response behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

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
