"""Tests for the kb agent click command."""

from __future__ import annotations

import json

from click.testing import CliRunner

from graphwiki_kb.agents.models import (
    AgentRunRecord,
    AgentToolResult,
    PendingApproval,
)
from graphwiki_kb.engine.command_registry import (
    get_click_command,
    list_command_names,
)
from graphwiki_kb.services.agent_service import AgentRunResult


def test_agent_command_is_registered() -> None:
    assert "agent" in list_command_names()
    command = get_click_command("agent")
    assert command is not None
    assert command.name == "agent"


def test_agent_help_lists_options() -> None:
    runner = CliRunner()
    command = get_click_command("agent")
    result = runner.invoke(command, ["--help"])
    assert result.exit_code == 0
    assert "--yes" in result.output
    assert "--show-plan" in result.output
    assert "--json" in result.output


def _stub_run(
    monkeypatch,
    *,
    record: AgentRunRecord,
    approvals: list[PendingApproval] | None = None,
):
    captured: dict = {}

    def _fake_run_once(self, prompt, *, command_context, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return AgentRunResult(record=record, pending_approvals=approvals or [])

    monkeypatch.setattr(
        "graphwiki_kb.services.agent_service.AgentService.run_once",
        _fake_run_once,
    )
    monkeypatch.setattr(
        "graphwiki_kb.commands.agent.is_agents_sdk_available",
        lambda: True,
    )
    return captured


def test_agent_one_shot_prints_final_output(test_project, monkeypatch) -> None:
    record = AgentRunRecord(
        run_id="r1",
        prompt="hi",
        created_at="2026-05-19T00:00:00+00:00",
        tool_results=[],
        final_output="**KB summary**",
    )
    captured = _stub_run(monkeypatch, record=record)
    command = get_click_command("agent")

    runner = CliRunner()
    result = runner.invoke(
        command,
        ["What", "is", "in", "my", "KB?"],
        obj=test_project.command_context,
    )

    assert result.exit_code == 0, result.output
    assert "KB summary" in result.output
    assert captured["prompt"] == "What is in my KB?"


def test_agent_one_shot_json_emits_run_payload(test_project, monkeypatch) -> None:
    record = AgentRunRecord(
        run_id="r1",
        prompt="hi",
        created_at="2026-05-19T00:00:00+00:00",
        tool_results=[
            AgentToolResult(tool_name="status", ok=True, summary="ok"),
        ],
        final_output="ok",
    )
    _stub_run(monkeypatch, record=record)

    runner = CliRunner()
    command = get_click_command("agent")
    result = runner.invoke(
        command, ["--json", "ping"], obj=test_project.command_context
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["run_id"] == "r1"
    assert payload["tool_results"][0]["tool_name"] == "status"


def test_agent_one_shot_show_plan_prints_tool_trace(test_project, monkeypatch) -> None:
    record = AgentRunRecord(
        run_id="r2",
        prompt="hi",
        created_at="2026-05-19T00:00:00+00:00",
        tool_results=[
            AgentToolResult(tool_name="status", ok=True, summary="ok"),
            AgentToolResult(tool_name="ask_kb", ok=False, summary="fail"),
        ],
        final_output="done",
    )
    _stub_run(monkeypatch, record=record)

    runner = CliRunner()
    command = get_click_command("agent")
    result = runner.invoke(
        command,
        ["--show-plan", "ping"],
        obj=test_project.command_context,
    )

    assert result.exit_code == 0, result.output
    assert "status [ok]" in result.output
    assert "ask_kb [fail]" in result.output


def test_agent_prints_pending_approval(test_project, monkeypatch) -> None:
    record = AgentRunRecord(
        run_id="r3",
        prompt="add rec",
        created_at="2026-05-19T00:00:00+00:00",
        tool_results=[],
        final_output="awaiting approval",
    )
    approval = PendingApproval(
        tool_name="ingest_recommendation",
        summary="Ingest 1 recommendation(s) from latest research run.",
        payload={"ids": [1]},
    )
    _stub_run(monkeypatch, record=record, approvals=[approval])

    runner = CliRunner()
    command = get_click_command("agent")
    result = runner.invoke(
        command, ["add", "recommendation", "1"], obj=test_project.command_context
    )

    assert result.exit_code == 0, result.output
    assert "Approval required" in result.output
    assert "ingest_recommendation" in result.output


def test_agent_one_shot_errors_when_sdk_missing(test_project, monkeypatch) -> None:
    monkeypatch.setattr(
        "graphwiki_kb.commands.agent.is_agents_sdk_available",
        lambda: False,
    )
    runner = CliRunner()
    command = get_click_command("agent")
    result = runner.invoke(command, ["ping"], obj=test_project.command_context)
    assert result.exit_code != 0
    assert "openai-agents" in result.output


def test_agent_uninitialized_project_raises(uninitialized_project) -> None:
    runner = CliRunner()
    command = get_click_command("agent")
    result = runner.invoke(command, ["ping"], obj=uninitialized_project.command_context)
    assert result.exit_code != 0
    assert "init" in result.output.lower()


def test_agent_json_without_prompt_rejected(test_project, monkeypatch) -> None:
    monkeypatch.setattr(
        "graphwiki_kb.commands.agent.is_agents_sdk_available",
        lambda: True,
    )
    runner = CliRunner()
    command = get_click_command("agent")
    result = runner.invoke(command, ["--json"], obj=test_project.command_context)
    assert result.exit_code != 0
    assert "one-shot" in result.output.lower()


def test_agent_json_sdk_missing_emits_json_error(test_project, monkeypatch) -> None:
    monkeypatch.setattr(
        "graphwiki_kb.commands.agent.is_agents_sdk_available",
        lambda: False,
    )
    runner = CliRunner()
    command = get_click_command("agent")
    result = runner.invoke(
        command, ["--json", "ping"], obj=test_project.command_context
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert "openai-agents" in payload["error"]


def test_agent_interactive_exit(test_project, monkeypatch) -> None:
    monkeypatch.setattr(
        "graphwiki_kb.commands.agent.is_agents_sdk_available",
        lambda: True,
    )
    runner = CliRunner()
    command = get_click_command("agent")
    result = runner.invoke(
        command,
        [],
        input="exit\n",
        obj=test_project.command_context,
    )
    assert result.exit_code == 0
    assert "GraphWiki KB agent" in result.output


def test_agent_interactive_runs_one_prompt_then_exits(
    test_project, monkeypatch
) -> None:
    record = AgentRunRecord(
        run_id="r1",
        prompt="hi",
        created_at="2026-05-19T00:00:00+00:00",
        tool_results=[],
        final_output="hello from agent",
    )
    _stub_run(monkeypatch, record=record)

    runner = CliRunner()
    command = get_click_command("agent")
    result = runner.invoke(
        command,
        [],
        input="ping\nquit\n",
        obj=test_project.command_context,
    )
    assert result.exit_code == 0
    assert "hello from agent" in result.output


def test_agent_interactive_handles_runtime_error(test_project, monkeypatch) -> None:
    monkeypatch.setattr(
        "graphwiki_kb.commands.agent.is_agents_sdk_available",
        lambda: True,
    )
    from graphwiki_kb.agents.runtime import AgentRuntimeError

    def _raise(self, prompt, **kwargs):
        raise AgentRuntimeError("simulated outage")

    monkeypatch.setattr(
        "graphwiki_kb.services.agent_service.AgentService.run_once",
        _raise,
    )

    runner = CliRunner()
    command = get_click_command("agent")
    result = runner.invoke(
        command,
        [],
        input="ping\nexit\n",
        obj=test_project.command_context,
    )
    assert result.exit_code == 0
    assert "simulated outage" in result.output


def test_agent_interactive_sdk_missing_prints_warning(
    test_project, monkeypatch
) -> None:
    monkeypatch.setattr(
        "graphwiki_kb.commands.agent.is_agents_sdk_available",
        lambda: False,
    )
    runner = CliRunner()
    command = get_click_command("agent")
    result = runner.invoke(
        command,
        [],
        input="\n",
        obj=test_project.command_context,
    )
    assert result.exit_code == 0
    assert "openai-agents is not installed" in result.output
