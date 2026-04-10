from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderRequest:
    prompt: str
    system_prompt: str = ""


@dataclass
class ProviderResponse:
    text: str
    model_name: str


class TextProvider:
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Return a normalized text completion."""
        raise NotImplementedError
