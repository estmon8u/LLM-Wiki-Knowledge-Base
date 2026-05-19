"""Tests for update_kb subprocess fallback in agent worker threads."""

from __future__ import annotations

import json

import pytest

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import UpdateKbInput
from graphwiki_kb.agents.tools import update as update_tool


def test_update_kb_uses_subprocess_off_main_thread(test_project, monkeypatch) -> None:
    runtime = AgentRuntimeContext(
        command_context=test_project.command_context,
        services=test_project.services,
    )
    monkeypatch.setattr(
        "graphwiki_kb.agents.tools.main_thread.is_main_thread",
        lambda: False,
    )
    monkeypatch.setattr(
        update_tool,
        "_run_update_inprocess",
        lambda *_args, **_kwargs: pytest.fail("in-process update should not run"),
    )

    class _Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(
        update_tool.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(),
    )
    payload = json.loads(
        update_tool.run_update_kb(
            runtime,
            UpdateKbInput(graph_method="auto", graph_only=True),
        )
    )
    assert payload["ok"] is True
    assert "subprocess" in payload["summary"].lower()
