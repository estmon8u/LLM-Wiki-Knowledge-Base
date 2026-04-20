from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    """Describes a concrete model within a provider/tier combination."""

    provider: str
    model: str
    tier: str  # "fast", "balanced", "deep"
    reasoning_effort: str  # provider-native hint: "low"/"high" etc.
    thinking_budget: int  # 0 = disabled


@dataclass(frozen=True)
class ResolvedProviderConfig:
    """Fully resolved provider + model details ready for construction."""

    provider_name: str
    model: str
    tier: str
    api_key_env: str
    reasoning_effort: str
    thinking_budget: int
