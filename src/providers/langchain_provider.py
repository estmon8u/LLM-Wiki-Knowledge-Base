"""LangChain-backed provider adapter.

When ``ecosystem.providers.backend`` is ``"langchain"`` in ``kb.config.yaml``,
this adapter wraps the LangChain chat-model integrations (``langchain-openai``,
``langchain-anthropic``, ``langchain-google-genai``) behind the same
:class:`~src.providers.base.TextProvider` contract used by the direct SDK
providers.
"""

from __future__ import annotations

import os
from typing import Any

from src.providers.base import ProviderRequest, ProviderResponse, TextProvider


class LangChainProvider(TextProvider):
    """Thin adapter that delegates to a LangChain chat model."""

    def __init__(
        self,
        provider_name: str,
        model: str,
        api_key_env: str,
        reasoning_effort: str = "high",
        thinking_budget: int = 0,
    ) -> None:
        self.name = provider_name
        self.model = model
        self._provider_name = provider_name
        self._reasoning_effort = reasoning_effort
        self._thinking_budget = thinking_budget

        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env} is not set. "
                f"Set it to use the {provider_name} provider via LangChain."
            )

        self._chat_model = self._build_chat_model(provider_name, model, api_key)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_chat_model(self, provider_name: str, model: str, api_key: str) -> Any:
        if provider_name == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                api_key=api_key,
                reasoning_effort=self._reasoning_effort,
            )
        if provider_name == "anthropic":
            from langchain_anthropic import ChatAnthropic

            kwargs: dict[str, Any] = {
                "model": model,
                "api_key": api_key,
            }
            if self._thinking_budget > 0:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self._thinking_budget,
                }
            return ChatAnthropic(**kwargs)
        if provider_name == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                thinking_level=self._reasoning_effort,
            )
        raise ValueError(
            f"Unsupported provider for LangChain backend: {provider_name!r}"
        )

    # ------------------------------------------------------------------
    # TextProvider interface
    # ------------------------------------------------------------------

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages: list = []
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        messages.append(HumanMessage(content=request.prompt))

        max_tokens = request.max_tokens
        if self._provider_name == "anthropic" and self._thinking_budget > 0:
            max_tokens = max(max_tokens, self._thinking_budget + 4096)

        response = self._chat_model.invoke(messages, max_tokens=max_tokens)
        text = self._extract_text(response)
        return ProviderResponse(text=text.strip(), model_name=self.model)

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull plain text from an AIMessage whose content may be str or blocks."""
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts) if parts else ""
        return str(content)
