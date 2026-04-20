"""Tests for model registry, update service, and related features."""

from __future__ import annotations

import pytest

from src.models.provider_models import ModelProfile, ResolvedProviderConfig
from src.services.model_registry_service import (
    PROVIDERS,
    TIERS,
    TASK_TIER_DEFAULTS,
    ModelRegistryService,
)


# ── ModelRegistryService ─────────────────────────────────────────────


class TestModelRegistryService:
    def setup_method(self):
        self.registry = ModelRegistryService()

    def test_list_profiles_returns_three_tiers(self):
        for provider in PROVIDERS:
            profiles = self.registry.list_profiles(provider)
            assert len(profiles) == 3
            assert [p.tier for p in profiles] == ["fast", "balanced", "deep"]

    def test_list_profiles_unknown_provider_returns_empty(self):
        assert self.registry.list_profiles("unknown") == []

    def test_default_tier_for_task(self):
        assert self.registry.default_tier_for_task("compile") == "fast"
        assert self.registry.default_tier_for_task("ask") == "balanced"
        assert self.registry.default_tier_for_task("review") == "balanced"
        assert self.registry.default_tier_for_task("unknown") == "balanced"

    def test_resolve_basic_balanced(self):
        config = {"provider": {"name": "openai"}}
        resolved = self.registry.resolve(config=config)
        assert resolved.provider_name == "openai"
        assert resolved.tier == "balanced"
        assert resolved.model == "gpt-5.4-mini"

    def test_resolve_explicit_tier(self):
        config = {"provider": {"name": "anthropic"}}
        resolved = self.registry.resolve(config=config, tier="deep")
        assert resolved.tier == "deep"
        assert resolved.model == "claude-opus-4-6"

    def test_resolve_explicit_model(self):
        config = {"provider": {"name": "openai"}}
        resolved = self.registry.resolve(config=config, model="gpt-5.4")
        assert resolved.model == "gpt-5.4"
        assert resolved.tier == "deep"

    def test_resolve_unknown_explicit_model(self):
        config = {"provider": {"name": "openai"}}
        resolved = self.registry.resolve(config=config, model="custom-model")
        assert resolved.model == "custom-model"
        assert resolved.tier == "balanced"

    def test_resolve_task_default_tier(self):
        config = {"provider": {"name": "gemini"}}
        resolved = self.registry.resolve(config=config, task="compile")
        assert resolved.tier == "fast"
        assert resolved.model == "gemini-3.1-flash-lite-preview"

    def test_resolve_tier_overrides_task_default(self):
        config = {"provider": {"name": "gemini"}}
        resolved = self.registry.resolve(config=config, tier="deep", task="compile")
        assert resolved.tier == "deep"

    def test_resolve_config_model_preserved_when_no_tier_or_task(self):
        config = {"provider": {"name": "openai", "model": "gpt-5.4"}}
        resolved = self.registry.resolve(config=config)
        assert resolved.model == "gpt-5.4"
        assert resolved.tier == "deep"

    def test_resolve_config_model_unknown_falls_to_balanced(self):
        config = {"provider": {"name": "openai", "model": "custom-fine-tuned"}}
        resolved = self.registry.resolve(config=config)
        assert resolved.model == "custom-fine-tuned"
        assert resolved.tier == "balanced"

    def test_resolve_provider_override(self):
        config = {"provider": {"name": "openai"}}
        resolved = self.registry.resolve(config=config, provider_override="anthropic")
        assert resolved.provider_name == "anthropic"

    def test_resolve_no_provider_raises(self):
        with pytest.raises(ValueError, match="No provider configured"):
            self.registry.resolve(config={})

    def test_resolve_unknown_tier_raises(self):
        config = {"provider": {"name": "openai"}}
        with pytest.raises(ValueError, match="No.*tier profile"):
            self.registry.resolve(config=config, tier="extreme")

    def test_resolve_api_key_env_from_config(self):
        config = {"provider": {"name": "openai", "api_key_env": "MY_KEY"}}
        resolved = self.registry.resolve(config=config, tier="fast")
        assert resolved.api_key_env == "MY_KEY"

    def test_resolve_anthropic_thinking_budget(self):
        config = {"provider": {"name": "anthropic"}}
        fast = self.registry.resolve(config=config, tier="fast")
        deep = self.registry.resolve(config=config, tier="deep")
        assert fast.thinking_budget < deep.thinking_budget

    def test_all_task_defaults_are_valid_tiers(self):
        for task, tier in TASK_TIER_DEFAULTS.items():
            assert tier in TIERS, f"Task {task} has invalid default tier {tier}"


# ── ResolvedProviderConfig frozen ────────────────────────────────────


def test_resolved_provider_config_is_frozen():
    cfg = ResolvedProviderConfig(
        provider_name="openai",
        model="gpt-5.4-mini",
        tier="fast",
        api_key_env="OPENAI_API_KEY",
        reasoning_effort="low",
        thinking_budget=0,
    )
    with pytest.raises(AttributeError):
        cfg.model = "changed"


# ── ModelProfile frozen ──────────────────────────────────────────────


def test_model_profile_is_frozen():
    p = ModelProfile(
        provider="openai",
        model="gpt-5.4-mini",
        tier="fast",
        reasoning_effort="low",
        thinking_budget=0,
    )
    with pytest.raises(AttributeError):
        p.model = "changed"
