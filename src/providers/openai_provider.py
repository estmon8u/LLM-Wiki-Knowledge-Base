from __future__ import annotations

import os

from openai import OpenAI

from src.providers.base import ProviderRequest, ProviderResponse, TextProvider


class OpenAIProvider(TextProvider):
    """OpenAI chat-completions provider."""

    name = "openai"

    def __init__(
        self, model: str = "gpt-5.4-mini", api_key_env: str = "OPENAI_API_KEY"
    ) -> None:
        self.model = model
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env} is not set. "
                "Set it to your OpenAI API key to use the OpenAI provider."
            )
        self._client = OpenAI(api_key=api_key)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "developer", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=request.max_tokens,
            reasoning_effort="high",
        )
        text = completion.choices[0].message.content or ""
        return ProviderResponse(text=text.strip(), model_name=self.model)
