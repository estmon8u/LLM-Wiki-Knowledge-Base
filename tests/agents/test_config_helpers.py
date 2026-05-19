"""Tests for agent config helpers."""

from __future__ import annotations

from graphwiki_kb.agents.config_helpers import config_section


def test_config_section_returns_mapping() -> None:
    config = {"agent": {"enabled": True, "model": "gpt-5.4-nano"}}
    assert config_section(config, "agent")["model"] == "gpt-5.4-nano"


def test_config_section_missing_or_invalid() -> None:
    assert config_section({}, "agent") == {}
    assert config_section({"agent": "bad"}, "agent") == {}
