"""OpenAI provider integration helpers."""

from __future__ import annotations

import os
from typing import Any, Literal

from openai import OpenAI

from graphwiki_kb.providers.base import (
    ProviderCapabilities,
    ProviderRequest,
    ProviderResponse,
    TextProvider,
)
from graphwiki_kb.providers.retry import provider_retry


class OpenAIProvider(TextProvider):
    """OpenAI provider using Responses API by default."""

    name = "openai"
    _SUPPORTED_APIS = {"responses", "chat_completions"}
    _SUPPORTED_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
    capabilities = ProviderCapabilities(
        strict_json_schema=True,
        native_structured_output=True,
        reasoning_effort_values=tuple(sorted(_SUPPORTED_REASONING_EFFORTS)),
        supports_store_false=True,
    )

    def __init__(
        self,
        model: str = "gpt-5.4-nano",
        api_key_env: str = "OPENAI_API_KEY",
        reasoning_effort: str = "high",
        api: Literal["responses", "chat_completions"] = "responses",
        store_responses: bool = False,
    ) -> None:
        normalized_api = api.strip().lower()
        if normalized_api not in self._SUPPORTED_APIS:
            supported = ", ".join(sorted(self._SUPPORTED_APIS))
            raise ValueError(f"Unsupported OpenAI API mode {api!r}; use {supported}.")
        self.model = model
        self._reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
        self._api = normalized_api
        self._store_responses = bool(store_responses)
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env} is not set. "
                "Set it to your OpenAI API key to use the OpenAI provider."
            )
        self._client = OpenAI(api_key=api_key)

    @provider_retry()
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate a normalized provider response."""
        if self._api == "chat_completions":
            return self._generate_chat_completion(request)
        return self._generate_response(request)

    def _generate_response(self, request: ProviderRequest) -> ProviderResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": request.prompt}],
            "max_output_tokens": request.max_tokens,
            "store": self._store_responses,
        }
        reasoning_effort = _request_reasoning_effort(
            request.reasoning_effort,
            self._reasoning_effort,
        )
        if _supports_reasoning_effort(self.model):
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if request.system_prompt:
            kwargs["instructions"] = request.system_prompt
        if request.response_schema:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.response_schema_name,
                    "schema": request.response_schema,
                    "strict": True,
                }
            }
        response = self._client.responses.create(**kwargs)
        text = _response_output_text(response)
        usage = getattr(response, "usage", None)
        return ProviderResponse(
            text=text.strip(),
            model_name=getattr(response, "model", None) or self.model,
            provider=self.name,
            finish_reason=_response_finish_reason(response),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            raw=response,
        )

    def _generate_chat_completion(self, request: ProviderRequest) -> ProviderResponse:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "developer", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": request.max_tokens,
            "store": self._store_responses,
        }
        reasoning_effort = _request_reasoning_effort(
            request.reasoning_effort,
            self._reasoning_effort,
        )
        if _supports_reasoning_effort(self.model):
            kwargs["reasoning_effort"] = reasoning_effort
        if request.response_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_schema_name,
                    "schema": request.response_schema,
                    "strict": True,
                },
            }
        completion = self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = getattr(completion, "usage", None)
        return ProviderResponse(
            text=text.strip(),
            model_name=self.model,
            provider=self.name,
            finish_reason=getattr(choice, "finish_reason", None),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            raw=completion,
        )


def _response_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    parts: list[str] = []
    for item in _iter_response_items(getattr(response, "output", None)):
        item_type = _get_value(item, "type")
        if item_type == "message":
            for content in _iter_response_items(_get_value(item, "content")):
                text = _get_value(content, "text")
                if isinstance(text, str):
                    parts.append(text)
        elif isinstance(_get_value(item, "text"), str):
            parts.append(_get_value(item, "text"))
    return "\n".join(part for part in parts if part).strip()


def _response_finish_reason(response: Any) -> str | None:
    status = getattr(response, "status", None)
    if isinstance(status, str) and status:
        return status
    for item in _iter_response_items(getattr(response, "output", None)):
        finish_reason = _get_value(item, "finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            return finish_reason
    return None


def _iter_response_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _get_value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _normalize_reasoning_effort(effort: str) -> str:
    normalized = effort.strip().lower()
    if normalized not in OpenAIProvider._SUPPORTED_REASONING_EFFORTS:
        supported = ", ".join(sorted(OpenAIProvider._SUPPORTED_REASONING_EFFORTS))
        raise ValueError(
            f"Unsupported OpenAI reasoning effort {effort!r}; use {supported}."
        )
    return normalized


def _request_reasoning_effort(
    request_effort: str | None,
    default_effort: str,
) -> str:
    if request_effort is None:
        return default_effort
    return _normalize_reasoning_effort(request_effort)


def _supports_reasoning_effort(model: str) -> bool:
    normalized = model.casefold()
    reasoning_prefixes = (
        "gpt-5",
        "gpt-6",
        "o1",
        "o3",
        "o4",
    )
    return normalized.startswith(reasoning_prefixes)
