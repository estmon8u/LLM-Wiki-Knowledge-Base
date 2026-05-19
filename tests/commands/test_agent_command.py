"""Tests for kb agent command registration and CLI behavior."""

from __future__ import annotations

from click.testing import CliRunner

from graphwiki_kb.cli import main
from graphwiki_kb.engine.command_registry import get_click_command, list_command_names


def test_agent_command_registered() -> None:
    assert "agent" in list_command_names()
    command = get_click_command("agent")
    assert command is not None
    assert command.name == "agent"


def test_agent_help_lists_command() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "agent" in result.output


def test_agent_one_shot_mocked(monkeypatch) -> None:
    from graphwiki_kb.agents.models import AgentRunResult

    def _fake_run_once(self, prompt, **kwargs):
        return AgentRunResult(
            run_id="t",
            final_output=f"Echo: {prompt}",
            planned_tools=["ask_kb"] if kwargs.get("show_plan") else [],
        )

    monkeypatch.setattr(
        "graphwiki_kb.commands.agent.AgentService.run_once",
        _fake_run_once,
    )
    monkeypatch.setattr(
        "graphwiki_kb.commands.agent.AgentService.ensure_available",
        lambda self: None,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        result = runner.invoke(main, ["agent", "--show-plan", "hello"])
        assert result.exit_code == 0
        assert "Echo: hello" in result.output
        assert "ask_kb" in result.output


def test_agent_requires_init() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["agent", "what is in my kb"])
        assert result.exit_code != 0
        assert "Project not initialized" in result.output
