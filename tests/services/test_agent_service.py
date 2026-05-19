"""Tests for AgentService with mocked Runner."""

from __future__ import annotations

from dataclasses import dataclass

from graphwiki_kb.agents.models import AgentRunResult
from graphwiki_kb.services.agent_service import AgentService


@dataclass
class _FakeResult:
    final_output: str
    interruptions: list[object]

    def to_state(self) -> object:
        return self


def test_agent_service_run_once_mocked(test_project, monkeypatch) -> None:
    monkeypatch.setattr(
        "graphwiki_kb.services.agent_service.agents_sdk_available",
        lambda: True,
    )

    def _fake_turn(runtime, prompt, *, approval_callback=None):
        return "Local KB answer: test.", []

    monkeypatch.setattr(
        "graphwiki_kb.services.agent_service.run_agent_turn",
        _fake_turn,
    )
    service = AgentService(test_project.config, test_project.services)
    result = service.run_once(
        "What does my KB say about GraphRAG?",
        command_context=test_project.command_context,
        show_plan=True,
    )
    assert isinstance(result, AgentRunResult)
    assert "GraphRAG" in result.final_output or "test" in result.final_output
    assert "ask_kb" in result.planned_tools
