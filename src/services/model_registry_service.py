from __future__ import annotations

from typing import Any

from src.models.provider_models import ModelProfile, ResolvedProviderConfig


# Built-in tier → model mapping per provider.
_PROFILES: dict[str, dict[str, ModelProfile]] = {
    "openai": {
        "fast": ModelProfile(
            provider="openai",
            model="gpt-5.4-nano",
            tier="fast",
            reasoning_effort="low",
            thinking_budget=0,
        ),
        "balanced": ModelProfile(
            provider="openai",
            model="gpt-5.4-mini",
            tier="balanced",
            reasoning_effort="medium",
            thinking_budget=0,
        ),
        "deep": ModelProfile(
            provider="openai",
            model="gpt-5.4",
            tier="deep",
            reasoning_effort="high",
            thinking_budget=0,
        ),
    },
    "anthropic": {
        "fast": ModelProfile(
            provider="anthropic",
            model="claude-haiku-4-5",
            tier="fast",
            reasoning_effort="low",
            thinking_budget=5_000,
        ),
        "balanced": ModelProfile(
            provider="anthropic",
            model="claude-sonnet-4-6",
            tier="balanced",
            reasoning_effort="high",
            thinking_budget=10_000,
        ),
        "deep": ModelProfile(
            provider="anthropic",
            model="claude-opus-4-6",
            tier="deep",
            reasoning_effort="high",
            thinking_budget=20_000,
        ),
    },
    "gemini": {
        "fast": ModelProfile(
            provider="gemini",
            model="gemini-3.1-flash-lite-preview",
            tier="fast",
            reasoning_effort="low",
            thinking_budget=0,
        ),
        "balanced": ModelProfile(
            provider="gemini",
            model="gemini-2.5-flash",
            tier="balanced",
            reasoning_effort="medium",
            thinking_budget=0,
        ),
        "deep": ModelProfile(
            provider="gemini",
            model="gemini-3.1-pro-preview",
            tier="deep",
            reasoning_effort="high",
            thinking_budget=0,
        ),
    },
}

TIERS = ("fast", "balanced", "deep")

PROVIDERS = tuple(_PROFILES.keys())

# Task-specific default tiers
TASK_TIER_DEFAULTS: dict[str, str] = {
    "update": "fast",
    "ask": "balanced",
    "review": "balanced",
}

_DEFAULT_API_KEY_ENVS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class ModelRegistryService:
    """Resolves a (provider, tier, model) triple into a concrete profile."""

    def list_profiles(self, provider: str) -> list[ModelProfile]:
        """Return all built-in profiles for a provider, ordered fast → deep."""
        tiers = _PROFILES.get(provider, {})
        return [tiers[t] for t in TIERS if t in tiers]

    def default_tier_for_task(self, task: str) -> str:
        return TASK_TIER_DEFAULTS.get(task, "balanced")

    def resolve(
        self,
        *,
        config: dict[str, Any],
        tier: str | None = None,
        model: str | None = None,
        provider_override: str | None = None,
        task: str | None = None,
    ) -> ResolvedProviderConfig:
        """Build a fully-resolved provider config from overlapping inputs.

        Priority (highest → lowest):
          1. Explicit ``--model`` flag  (pins both model and provider)
          2. Explicit ``--tier`` flag
          3. Config-file tier (persisted via ``config provider set --tier``)
          4. Config-file model (if set, look up its tier)
          5. Task-specific default tier
          6. Balanced fallback
        """
        provider_cfg = config.get("provider") or {}
        provider_name = provider_override or provider_cfg.get("name", "")
        if not provider_name:
            raise ValueError("No provider configured.")

        api_key_env = provider_cfg.get(
            "api_key_env",
            _DEFAULT_API_KEY_ENVS.get(
                provider_name, f"{provider_name.upper()}_API_KEY"
            ),
        )

        # Explicit model wins — look it up or use as-is.
        if model:
            profile = self._find_profile_by_model(provider_name, model)
            if profile:
                return ResolvedProviderConfig(
                    provider_name=provider_name,
                    model=model,
                    tier=profile.tier,
                    api_key_env=api_key_env,
                    reasoning_effort=profile.reasoning_effort,
                    thinking_budget=profile.thinking_budget,
                )
            # Unknown model — use balanced defaults for effort/budget.
            return ResolvedProviderConfig(
                provider_name=provider_name,
                model=model,
                tier="balanced",
                api_key_env=api_key_env,
                reasoning_effort="high",
                thinking_budget=10_000 if provider_name == "anthropic" else 0,
            )

        # Tier resolution: explicit > config tier > config model > task default > balanced
        effective_tier = tier
        if not effective_tier:
            cfg_tier = provider_cfg.get("tier", "")
            if cfg_tier and cfg_tier in TIERS:
                effective_tier = cfg_tier
        if not effective_tier:
            cfg_model = provider_cfg.get("model", "")
            if cfg_model:
                profile = self._find_profile_by_model(provider_name, cfg_model)
                if profile:
                    return ResolvedProviderConfig(
                        provider_name=provider_name,
                        model=cfg_model,
                        tier=profile.tier,
                        api_key_env=api_key_env,
                        reasoning_effort=profile.reasoning_effort,
                        thinking_budget=profile.thinking_budget,
                    )
                # Config model set but not in registry — balanced defaults
                return ResolvedProviderConfig(
                    provider_name=provider_name,
                    model=cfg_model,
                    tier="balanced",
                    api_key_env=api_key_env,
                    reasoning_effort="high",
                    thinking_budget=10_000 if provider_name == "anthropic" else 0,
                )
        if not effective_tier and task:
            effective_tier = self.default_tier_for_task(task)
        if not effective_tier:
            effective_tier = "balanced"

        # Look up the tier profile
        provider_tiers = _PROFILES.get(provider_name, {})
        profile = provider_tiers.get(effective_tier)
        if not profile:
            raise ValueError(
                f"No {effective_tier!r} tier profile for provider {provider_name!r}."
            )
        return ResolvedProviderConfig(
            provider_name=provider_name,
            model=profile.model,
            tier=profile.tier,
            api_key_env=api_key_env,
            reasoning_effort=profile.reasoning_effort,
            thinking_budget=profile.thinking_budget,
        )

    @staticmethod
    def _find_profile_by_model(provider_name: str, model: str) -> ModelProfile | None:
        for profile in _PROFILES.get(provider_name, {}).values():
            if profile.model == model:
                return profile
        return None
