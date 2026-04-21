"""Provider abstractions and factory for LLM-backed services."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.models.provider_models import ResolvedProviderConfig
from src.providers.base import ProviderRequest, ProviderResponse, TextProvider

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Base error for provider configuration and execution failures."""


class ProviderConfigurationError(ProviderError):
    """Raised when a provider is required but not configured correctly."""


class ProviderExecutionError(ProviderError):
    """Raised when a configured provider fails during generation."""


class UnavailableProvider(TextProvider):
    name = "unavailable"

    def __init__(self, message: str, *, provider_name: str = "unavailable") -> None:
        self.name = provider_name
        self._message = message

    def ensure_available(self) -> None:
        raise ProviderConfigurationError(self._message)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.ensure_available()


_DEFAULT_API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

_DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-3.1-flash-lite-preview",
}


def build_provider(
    config: dict[str, Any],
    *,
    resolved: ResolvedProviderConfig | None = None,
) -> Optional[TextProvider]:
    """Build a provider from the ``provider`` section of kb config.

    When *resolved* is supplied (from the model registry), its model,
    reasoning-effort, and thinking-budget values take precedence over the
    raw config dict.

    Returns ``None`` when no provider is configured, so deterministic
    commands can proceed without one.  Generation commands (compile, query,
    review) should check for ``None`` and raise a clear
    ``ProviderConfigurationError`` instead of silently falling back.
    """
    provider_cfg = config.get("provider") or {}
    name = provider_cfg.get("name", "")
    if not name:
        return None

    if resolved:
        model = resolved.model
        api_key_env = resolved.api_key_env
        reasoning_effort = resolved.reasoning_effort
        thinking_budget = resolved.thinking_budget
    else:
        model = provider_cfg.get("model", _DEFAULT_MODELS.get(name, ""))
        api_key_env = provider_cfg.get(
            "api_key_env", _DEFAULT_API_KEY_ENVS.get(name, "")
        )
        reasoning_effort = "high"
        thinking_budget = 10_000 if name == "anthropic" else 0

    # --- LangChain backend path ---
    ecosystem = config.get("ecosystem") or {}
    providers_cfg = ecosystem.get("providers") or {}
    backend = providers_cfg.get("backend", "direct")

    if backend == "langchain":
        try:
            from src.providers.langchain_provider import LangChainProvider

            return LangChainProvider(
                provider_name=name,
                model=model,
                api_key_env=api_key_env,
                reasoning_effort=reasoning_effort,
                thinking_budget=thinking_budget,
            )
        except (ImportError, ValueError) as exc:
            logger.warning("LangChain backend for %r unavailable: %s", name, exc)
            return UnavailableProvider(str(exc), provider_name=name)

    # --- Direct SDK backend path (default) ---
    try:
        if name == "openai":
            from src.providers.openai_provider import OpenAIProvider

            return OpenAIProvider(
                model=model,
                api_key_env=api_key_env,
                reasoning_effort=reasoning_effort,
            )
        if name == "anthropic":
            from src.providers.anthropic_provider import AnthropicProvider

            return AnthropicProvider(
                model=model,
                api_key_env=api_key_env,
                thinking_budget=thinking_budget,
            )
        if name == "gemini":
            from src.providers.gemini_provider import GeminiProvider

            return GeminiProvider(
                model=model,
                api_key_env=api_key_env,
                reasoning_effort=reasoning_effort,
            )
        message = (
            f"Unknown provider name {name!r}. Supported providers are: "
            "openai, anthropic, gemini."
        )
        logger.warning(message)
        return UnavailableProvider(message, provider_name=name)
    except ValueError as exc:
        logger.warning("Provider %r unavailable: %s", name, exc)
        return UnavailableProvider(str(exc), provider_name=name)
