"""Provider abstractions and factory for LLM-backed services."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from src.providers.base import ProviderRequest, ProviderResponse, TextProvider

logger = logging.getLogger(__name__)

ProviderCatalog = Mapping[str, Mapping[str, Any]]


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


_FALLBACK_PROVIDER_CATALOG = {
    "openai": {
        "model": "gpt-5.4-mini",
        "api_key_env": "OPENAI_API_KEY",
        "reasoning_effort": "high",
    },
    "anthropic": {
        "model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "thinking_budget": 10_000,
    },
    "gemini": {
        "model": "gemini-3.1-flash-lite-preview",
        "api_key_env": "GEMINI_API_KEY",
        "reasoning_effort": "high",
    },
}


def supported_provider_names(
    provider_catalog: ProviderCatalog | None = None,
) -> tuple[str, ...]:
    catalog = provider_catalog or _FALLBACK_PROVIDER_CATALOG
    return tuple(sorted(catalog))


def describe_supported_providers(
    provider_catalog: ProviderCatalog | None = None,
) -> str:
    return ", ".join(supported_provider_names(provider_catalog))


def validate_provider_name(
    name: str,
    provider_catalog: ProviderCatalog | None = None,
) -> str:
    normalized_name = str(name).strip().lower()
    if normalized_name not in supported_provider_names(provider_catalog):
        raise ValueError(
            f"Unknown provider {name!r}. Supported providers: "
            f"{describe_supported_providers(provider_catalog)}."
        )
    return normalized_name


def resolve_provider_settings(
    config: dict[str, Any],
    provider_catalog: ProviderCatalog | None = None,
) -> tuple[str, dict[str, Any]] | None:
    provider_cfg = config.get("provider") or {}
    name = provider_cfg.get("name", "")
    normalized_name = str(name).strip().lower()
    if not normalized_name:
        return None

    catalog = provider_catalog or config.get("providers") or _FALLBACK_PROVIDER_CATALOG
    resolved = dict(catalog.get(normalized_name, {}))
    use_legacy_overrides = int(config.get("version", 0) or 0) < 3
    if use_legacy_overrides:
        for key, value in provider_cfg.items():
            if key == "name" or value in (None, ""):
                continue
            resolved[key] = value
    return normalized_name, resolved


def build_provider(
    config: dict[str, Any],
    provider_catalog: ProviderCatalog | None = None,
) -> Optional[TextProvider]:
    """Build a provider from the ``provider`` section of kb config.

    Returns ``None`` when no provider is configured, so deterministic
    commands can proceed without one.  Generation commands (update, ask,
    review) should check for ``None`` and raise a clear
    ``ProviderConfigurationError`` instead of silently falling back.
    """
    resolved = resolve_provider_settings(config, provider_catalog=provider_catalog)
    if resolved is None:
        return None
    name, provider_cfg = resolved

    model = provider_cfg.get("model", "")
    api_key_env = provider_cfg.get("api_key_env", "")
    reasoning_effort = provider_cfg.get("reasoning_effort", "high")
    thinking_budget = provider_cfg.get("thinking_budget", 0)

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
            f"{describe_supported_providers(provider_catalog)}."
        )
        logger.warning(message)
        return UnavailableProvider(message, provider_name=name)
    except ValueError as exc:
        logger.warning("Provider %r unavailable: %s", name, exc)
        return UnavailableProvider(str(exc), provider_name=name)
