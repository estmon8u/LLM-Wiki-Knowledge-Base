"""Provider abstractions and factory for LLM-backed services."""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional

from graphwiki_kb.providers.base import ProviderRequest, ProviderResponse, TextProvider

logger = logging.getLogger(__name__)

ProviderCatalog = Mapping[str, Mapping[str, Any]]


class ProviderError(RuntimeError):
    """Base error for provider configuration and execution failures."""


class ProviderConfigurationError(ProviderError):
    """Raised when a provider is required but not configured correctly."""


class ProviderExecutionError(ProviderError):
    """Raised when a configured provider fails during generation."""


class UnavailableProvider(TextProvider):
    """Represents unavailable provider behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    name = "unavailable"

    def __init__(self, message: str, *, provider_name: str = "unavailable") -> None:
        self.name = provider_name
        self._message = message

    def ensure_available(self) -> None:
        """Ensure available."""
        raise ProviderConfigurationError(self._message)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate.

        Args:
            request: Request value used by the operation.

        Returns:
            ProviderResponse produced by the operation.
        """
        raise ProviderConfigurationError(self._message)


class LazyProvider(TextProvider):
    """Build a concrete provider only when a provider-backed command needs it."""

    def __init__(
        self,
        factory: Callable[[], Optional[TextProvider]],
        *,
        provider_name: str,
    ) -> None:
        self._factory = factory
        self._provider: Optional[TextProvider] = None
        self.name = provider_name

    def _resolve(self) -> TextProvider:
        if self._provider is None:
            provider = self._factory()
            if provider is None:
                raise ProviderConfigurationError("Provider is not configured.")
            self._provider = provider
            self.name = provider.name
        return self._provider

    def ensure_available(self) -> None:
        provider = self._resolve()
        ensure_available = getattr(provider, "ensure_available", None)
        if callable(ensure_available):
            ensure_available()

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return self._resolve().generate(request)


_FALLBACK_PROVIDER_CATALOG = {
    "openai": {
        "model": "gpt-5.4-nano",
        "api_key_env": "OPENAI_API_KEY",
        "reasoning_effort": "high",
        "api": "responses",
    },
    "anthropic": {
        "model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "thinking_effort": "medium",
    },
    "gemini": {
        "model": "gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "reasoning_effort": "high",
    },
}


def supported_provider_names(
    provider_catalog: ProviderCatalog | None = None,
) -> tuple[str, ...]:
    """Supported provider names.

    Args:
        provider_catalog: Provider catalog value used by the operation.

    Returns:
        tuple[str, ...] produced by the operation.
    """
    catalog = provider_catalog or _FALLBACK_PROVIDER_CATALOG
    return tuple(sorted(catalog))


def describe_supported_providers(
    provider_catalog: ProviderCatalog | None = None,
) -> str:
    """Describe supported providers.

    Args:
        provider_catalog: Provider catalog value used by the operation.

    Returns:
        str produced by the operation.
    """
    return ", ".join(supported_provider_names(provider_catalog))


def validate_provider_name(
    name: str,
    provider_catalog: ProviderCatalog | None = None,
) -> str:
    """Validate provider name.

    Args:
        name: Name value used for lookup or display.
        provider_catalog: Provider catalog value used by the operation.

    Returns:
        str produced by the operation.
    """
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
    """Resolve provider settings.

    Args:
        config: Loaded knowledge-base configuration mapping.
        provider_catalog: Provider catalog value used by the operation.

    Returns:
        tuple[str, dict[str, Any]] | None produced by the operation.
    """
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
    thinking_budget = provider_cfg.get("thinking_budget")
    thinking_effort = provider_cfg.get("thinking_effort", "medium")
    api = provider_cfg.get("api", "responses")

    try:
        if name == "openai":
            from graphwiki_kb.providers.openai_provider import OpenAIProvider

            return OpenAIProvider(
                model=model,
                api_key_env=api_key_env,
                reasoning_effort=reasoning_effort,
                api=api,
            )
        if name == "anthropic":
            from graphwiki_kb.providers.anthropic_provider import AnthropicProvider

            return AnthropicProvider(
                model=model,
                api_key_env=api_key_env,
                thinking_budget=thinking_budget,
                thinking_effort=thinking_effort,
            )
        if name == "gemini":
            from graphwiki_kb.providers.gemini_provider import GeminiProvider

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


def build_lazy_provider(
    config: dict[str, Any],
    provider_catalog: ProviderCatalog | None = None,
    provider_builder: (
        Callable[[dict[str, Any], ProviderCatalog | None], Optional[TextProvider]]
        | None
    ) = None,
) -> Optional[TextProvider]:
    """Return a provider proxy without importing SDKs or checking env eagerly."""
    resolved = resolve_provider_settings(config, provider_catalog=provider_catalog)
    if resolved is None:
        return None
    name, _ = resolved
    builder = provider_builder or build_provider
    return LazyProvider(
        lambda: builder(config, provider_catalog),
        provider_name=name,
    )
