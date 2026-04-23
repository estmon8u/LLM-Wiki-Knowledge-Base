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
        thinking_level = self._EFFORT_TO_LEVEL.get(self._reasoning_effort, "high")
        config_kwargs: dict = {
            "max_output_tokens": request.max_tokens,
            "thinking_config": types.ThinkingConfig(thinking_level=thinking_level),
        }
        if request.system_prompt:
            config_kwargs["system_instruction"] = request.system_prompt
        if request.response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = request.response_schema
        response = self._client.models.generate_content(
            model=self.model,
            contents=request.prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = response.text or ""
        return ProviderResponse(text=text.strip(), model_name=self.model)
