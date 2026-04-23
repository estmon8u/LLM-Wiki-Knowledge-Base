from __future__ import annotations

import json
import os

from anthropic import Anthropic

from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.providers.retry import provider_retry


class AnthropicProvider(TextProvider):
    """Anthropic messages provider."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key_env: str = "ANTHROPIC_API_KEY",
        thinking_budget: int = 10_000,
    ) -> None:
        self.model = model
        self._thinking_budget = thinking_budget
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env} is not set. "
                "Set it to your Anthropic API key to use the Anthropic provider."
            )
        self._client = Anthropic(api_key=api_key)

    @provider_retry()
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        max_tokens = max(request.max_tokens, self._thinking_budget + 4096)
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if self._thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget,
            }
        system_prompt = request.system_prompt
        if request.response_schema:
            schema_text = json.dumps(request.response_schema, indent=2, sort_keys=True)
            system_prompt = (
                f"{system_prompt}\n\n" if system_prompt else ""
            ) + f"Return only JSON matching this schema:\n{schema_text}"
        if system_prompt:
            kwargs["system"] = system_prompt
        message = self._client.messages.create(**kwargs)
        text = next((b.text for b in message.content if b.type == "text"), "")
        return ProviderResponse(text=text.strip(), model_name=self.model)
