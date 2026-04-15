"""Provider abstractions and factory for LLM-backed services."""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.providers.base import ProviderRequest, ProviderResponse, TextProvider

logger = logging.getLogger(__name__)

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


def build_provider(config: dict[str, Any]) -> Optional[TextProvider]:
    """Build a provider from the ``provider`` section of kb config.

    Returns ``None`` when no provider is configured or the API key is missing,
    so callers can fall back to deterministic behaviour.
    """
    provider_cfg = config.get("provider") or {}
    name = provider_cfg.get("name", "")
    if not name:
        return None

    model = provider_cfg.get("model", _DEFAULT_MODELS.get(name, ""))
    api_key_env = provider_cfg.get("api_key_env", _DEFAULT_API_KEY_ENVS.get(name, ""))

    try:
        if name == "openai":
            from src.providers.openai_provider import OpenAIProvider

            return OpenAIProvider(model=model, api_key_env=api_key_env)
        if name == "anthropic":
            from src.providers.anthropic_provider import AnthropicProvider

            return AnthropicProvider(model=model, api_key_env=api_key_env)
        if name == "gemini":
            from src.providers.gemini_provider import GeminiProvider

            return GeminiProvider(model=model, api_key_env=api_key_env)
        logger.warning(
            "Unknown provider name %r — falling back to heuristic mode.", name
        )
        return None
    except ValueError as exc:
        logger.warning(
            "Provider %r unavailable: %s — falling back to heuristic mode.", name, exc
        )
        return None
