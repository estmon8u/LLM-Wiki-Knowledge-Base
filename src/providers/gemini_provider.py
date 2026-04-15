from __future__ import annotations

import os

from google import genai
from google.genai import types

from src.providers.base import ProviderRequest, ProviderResponse, TextProvider


class GeminiProvider(TextProvider):
    """Google Gemini provider via the google-genai SDK."""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite-preview",
        api_key_env: str = "GEMINI_API_KEY",
    ) -> None:
        self.model = model
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env} is not set. "
                "Set it to your Gemini API key to use the Gemini provider."
            )
        self._client = genai.Client(api_key=api_key)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        config_kwargs: dict = {"max_output_tokens": request.max_tokens}
        if request.system_prompt:
            config_kwargs["system_instruction"] = request.system_prompt
        response = self._client.models.generate_content(
            model=self.model,
            contents=request.prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = response.text or ""
        return ProviderResponse(text=text.strip(), model_name=self.model)
