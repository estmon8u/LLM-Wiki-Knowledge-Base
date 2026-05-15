"""Provider integration helpers for gemini provider.

This module belongs to `src.providers.gemini_provider` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import json
import logging
import os

from google import genai
from google.genai import types

from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.providers.retry import provider_retry


logger = logging.getLogger(__name__)


class GeminiProvider(TextProvider):
    """Google Gemini provider via the google-genai SDK."""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        reasoning_effort: str = "high",
    ) -> None:
        self.model = model
        self._reasoning_effort = reasoning_effort
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env} is not set. "
                "Set it to your Gemini API key to use the Gemini provider."
            )
        self._client = genai.Client(api_key=api_key)

    _EFFORT_TO_LEVEL = {
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    _EFFORT_TO_25_BUDGET = {
        "none": 0,
        "minimal": 1024,
        "low": 1024,
        "medium": 8192,
        "high": 24576,
    }

    @provider_retry()
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate.

        Args:
            request: Request value used by the operation.

        Returns:
            ProviderResponse produced by the operation.
        """
        effort = request.reasoning_effort or self._reasoning_effort
        config_kwargs: dict = {
            "max_output_tokens": request.max_tokens,
            "thinking_config": self._thinking_config(effort),
        }
        if request.system_prompt:
            config_kwargs["system_instruction"] = request.system_prompt
        if request.response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            if _schema_uses_additional_properties(request.response_schema):
                logger.warning(
                    "Gemini response schema does not support additionalProperties; "
                    "removing it before provider call."
                )
            config_kwargs["response_schema"] = _gemini_response_schema(
                request.response_schema
            )
        response = self._client.models.generate_content(
            model=self.model,
            contents=request.prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = response.text or ""
        candidates = getattr(response, "candidates", None) or []
        candidate = candidates[0] if candidates else None
        usage = getattr(response, "usage_metadata", None)
        return ProviderResponse(
            text=text.strip(),
            model_name=self.model,
            provider=self.name,
            finish_reason=(
                str(getattr(candidate, "finish_reason", ""))
                if candidate is not None
                else None
            ),
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            raw=response,
        )

    def _thinking_config(self, effort: str) -> types.ThinkingConfig:
        if _uses_thinking_budget(self.model):
            budget = self._EFFORT_TO_25_BUDGET.get(effort, 24576)
            return types.ThinkingConfig(thinking_budget=budget)
        level = self._EFFORT_TO_LEVEL.get(effort, "high")
        return types.ThinkingConfig(thinking_level=level)


def _gemini_response_schema(schema: object) -> object:
    """Return a Gemini-compatible subset of the JSON schema payload."""
    if isinstance(schema, dict):
        return {
            key: _gemini_response_schema(value)
            for key, value in schema.items()
            if key != "additionalProperties"
        }
    if isinstance(schema, list):
        return [_gemini_response_schema(item) for item in schema]
    return schema


def _schema_uses_additional_properties(schema: object) -> bool:
    try:
        encoded = json.dumps(schema)
    except TypeError:
        return False
    return '"additionalProperties"' in encoded


def _uses_thinking_budget(model: str) -> bool:
    return model.casefold().startswith("gemini-2.5")
