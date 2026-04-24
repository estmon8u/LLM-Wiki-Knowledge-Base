from __future__ import annotations

import os

from google import genai
from google.genai import types

from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.providers.retry import provider_retry


class GeminiProvider(TextProvider):
    """Google Gemini provider via the google-genai SDK."""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite-preview",
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

    _EFFORT_TO_LEVEL = {"low": "low", "medium": "medium", "high": "high"}

    @provider_retry()
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        effort = request.reasoning_effort or self._reasoning_effort
        thinking_level = self._EFFORT_TO_LEVEL.get(effort, "high")
        config_kwargs: dict = {
            "max_output_tokens": request.max_tokens,
            "thinking_config": types.ThinkingConfig(thinking_level=thinking_level),
        }
        if request.system_prompt:
            config_kwargs["system_instruction"] = request.system_prompt
        if request.response_schema:
            config_kwargs["response_mime_type"] = "application/json"
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
