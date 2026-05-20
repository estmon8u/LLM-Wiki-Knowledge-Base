"""Tests for the high-level AgentService and its orchestration helpers."""

from __future__ import annotations

from typing import Any

import pytest

from graphwiki_kb.agents.models import AgentRunRecord, AgentToolResult, PendingApproval
from graphwiki_kb.services.agent_service import AgentService


class _StubRunner:
    def __init__(self, *, final_output: str = "stub final") -> None:
        self.final_output = final_output
        self.calls: list[dict[str, Any]] = []

    async def run(self, agent, prompt, *, context, session=None, max_turns=8):
        self.calls.append(
            {
                "prompt": prompt,
                "context": context,
                "session": session,
                "max_turns": max_turns,
            }
        )
        return type("Result", (), {"final_output": self.final_output})()


def _patch_runtime(monkeypatch, *, final_output: str = "ok") -> _StubRunner:
    runner = _StubRunner(final_output=final_output)

    async def _run_agent(*, agent, prompt, runtime, session=None, max_turns=8):
        return await runner.run(
            agent, prompt, context=runtime, session=session, max_turns=max_turns
        )

    monkeypatch.setattr("graphwiki_kb.services.agent_service.run_agent", _run_agent)
    monkeypatch.setattr(
        "graphwiki_kb.services.agent_service.build_kb_agent",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "graphwiki_kb.services.agent_service.build_session",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "graphwiki_kb.services.agent_service.is_agents_sdk_available",
        lambda: True,
    )
    return runner


def test_agent_service_runs_one_shot_and_persists_record(test_project, monkeypatch):
    runner = _patch_runtime(monkeypatch, final_output="hello")
    service = AgentService(test_project.config, test_project.services)

    result = service.run_once(
        "ping",
        command_context=test_project.command_context,
        session_id="s1",
    )

    assert result.record.final_output == "hello"
    assert result.record.session_id == "s1"
    assert result.saved_run_path is not None
    assert (test_project.paths.root / result.saved_run_path).exists()
    assert runner.calls and runner.calls[0]["prompt"] == "ping"


def test_agent_service_propagates_pending_approvals(test_project, monkeypatch):
    _patch_runtime(monkeypatch)
    service = AgentService(test_project.config, test_project.services)

    # Capture the runtime that the (stubbed) Runner receives so we can simulate
    # a tool registering an approval.
    captured: dict[str, Any] = {}

    async def _run_agent(*, agent, prompt, runtime, session=None, max_turns=8):
        captured["runtime"] = runtime
        runtime.tool_results.append(
            AgentToolResult(tool_name="status", ok=True, summary="ok"),
        )
        runtime.pending_approvals.append(
            PendingApproval(
                tool_name="update_kb",
                summary="Run kb update",
                payload={"force": False},
            )
        )
        return type("Result", (), {"final_output": "review approval"})()

    monkeypatch.setattr("graphwiki_kb.services.agent_service.run_agent", _run_agent)

    result = service.run_once(
        "update",
        command_context=test_project.command_context,
    )

    assert len(result.pending_approvals) == 1
    assert result.pending_approvals[0].tool_name == "update_kb"
    assert result.record.tool_results[0].tool_name == "status"
    assert isinstance(result.record, AgentRunRecord)


def test_agent_service_build_runtime_injects_research_dependencies(test_project):
    service = AgentService(test_project.config, test_project.services)
    runtime = service.build_runtime(command_context=test_project.command_context)
    assert "research_service" in runtime.metadata
    assert "web_source_acquisition" in runtime.metadata


def test_agent_service_save_runs_disabled_skips_writing(test_project, monkeypatch):
    _patch_runtime(monkeypatch)
    config = dict(test_project.config)
    config["agent"] = dict(config.get("agent", {}))
    config["agent"]["save_runs"] = False
    service = AgentService(config, test_project.services)

    result = service.run_once(
        "ping",
        command_context=test_project.command_context,
    )
    assert result.saved_run_path is None


def test_agent_service_raises_when_sdk_missing(test_project, monkeypatch):
    monkeypatch.setattr(
        "graphwiki_kb.services.agent_service.is_agents_sdk_available",
        lambda: False,
    )
    service = AgentService(test_project.config, test_project.services)

    with pytest.raises(Exception):
        service.run_once("ping", command_context=test_project.command_context)


def test_agent_service_disabling_session_backend_skips_storage(
    test_project, monkeypatch
):
    """When session_backend is not sqlite, no storage path is produced."""
    runner = _patch_runtime(monkeypatch)
    config = dict(test_project.config)
    config["agent"] = dict(config.get("agent", {}))
    config["agent"]["session_backend"] = "memory"
    service = AgentService(config, test_project.services)

    assert service._session_storage_path() is None  # type: ignore[attr-defined]

    result = service.run_once(
        "ping",
        command_context=test_project.command_context,
        session_id="s",
    )
    # session should be None for non-sqlite backend
    assert runner.calls[0]["session"] is None
    assert result.saved_run_path is not None


def test_agent_service_disabled_web_research_skips_web_service(test_project):
    config = dict(test_project.config)
    config["research"] = dict(config.get("research", {}))
    config["research"]["web_enabled"] = False
    service = AgentService(config, test_project.services)
    assert service.web_research_service is None
    # Research service should be wired but with no web service
    assert service.research_service.web_service is None


def test_agent_service_build_run_record_includes_session_id(test_project, monkeypatch):
    service = AgentService(test_project.config, test_project.services)
    runtime = service.build_runtime(
        command_context=test_project.command_context,
        session_id="my-session",
    )
    runtime.tool_results.append(
        AgentToolResult(tool_name="status", ok=True, summary="ok"),
    )
    record = service.build_run_record(
        prompt="hi",
        runtime=runtime,
        final_output="bye",
        session_id="my-session",
    )
    assert record.session_id == "my-session"
    assert record.final_output == "bye"
    assert record.tool_results[0].tool_name == "status"


def test_agent_service_extract_final_output_handles_dicts(test_project, monkeypatch):
    from graphwiki_kb.services.agent_service import _extract_final_output

    class _Result:
        final_output = {"a": 1}

    assert "a" in _extract_final_output(_Result())

    class _ResultStr:
        final_output = "hello"

    assert _extract_final_output(_ResultStr()) == "hello"

    class _ResultNone:
        final_output = None
        output_text = "fallback"

    assert _extract_final_output(_ResultNone()) == "fallback"

    assert _extract_final_output(None) == ""
