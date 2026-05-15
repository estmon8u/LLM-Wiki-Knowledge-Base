"""Provider integration helpers for anthropic provider.

This module belongs to `graphwiki_kb.providers.anthropic_provider` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import json
import os
import re

from anthropic import Anthropic

from graphwiki_kb.providers.base import ProviderRequest, ProviderResponse, TextProvider
from graphwiki_kb.providers.retry import provider_retry


class AnthropicProvider(TextProvider):
    """Anthropic messages provider."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key_env: str = "ANTHROPIC_API_KEY",
        thinking_budget: int | None = None,
        thinking_effort: str = "medium",
    ) -> None:
        self.model = model
        self._thinking_budget = thinking_budget
        self._thinking_effort = thinking_effort
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env} is not set. "
                "Set it to your Anthropic API key to use the Anthropic provider."
            )
        self._client = Anthropic(api_key=api_key)

    @provider_retry()
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate.

        Args:
            request: Request value used by the operation.

        Returns:
            ProviderResponse produced by the operation.
        """
        use_adaptive_thinking = _uses_adaptive_thinking(self.model)
        max_tokens = request.max_tokens
        if self._thinking_budget and not use_adaptive_thinking:
            max_tokens = max(request.max_tokens, self._thinking_budget + 4096)
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if use_adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {
                "effort": request.reasoning_effort or self._thinking_effort,
            }
        elif self._thinking_budget and self._thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget,
            }
        system_prompt = request.system_prompt
        if request.response_schema:
            schema_text = json.dumps(request.response_schema, indent=2, sort_keys=True)
            system_prompt = (f"{system_prompt}\n\n" if system_prompt else "") + (
                "Return only JSON matching this schema. This output-format "
                "requirement overrides any conflicting prose or markdown "
                f"instructions:\n{schema_text}"
            )
        if system_prompt:
            kwargs["system"] = system_prompt
        message = self._client.messages.create(**kwargs)
        text = next((b.text for b in message.content if b.type == "text"), "")
        usage = getattr(message, "usage", None)
        return ProviderResponse(
            text=text.strip(),
            model_name=self.model,
            provider=self.name,
            finish_reason=getattr(message, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            raw=message,
        )


def _uses_adaptive_thinking(model: str) -> bool:
    """Return True for Claude models that use adaptive thinking effort."""
    normalized = model.casefold()
    if "mythos" in normalized:
        return True
    match = re.search(r"claude-(?:opus|sonnet)-4-(\d+)", normalized)
    return bool(match and int(match.group(1)) >= 6)
