"""Tests for update_kb tool."""

from __future__ import annotations

import json

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import UpdateKbInput
from graphwiki_kb.agents.tools.update import run_update_kb
from graphwiki_kb.services.update_service import UpdateResult


def test_update_kb_tool_reports_result(test_project, monkeypatch) -> None:
    class _UpdateService:
        def run(self, options, **kwargs: object) -> UpdateResult:
            return UpdateResult()

    monkeypatch.setattr(
        "graphwiki_kb.agents.tools.update.UpdateService",
        lambda **kwargs: _UpdateService(),
    )
    runtime = AgentRuntimeContext(
        command_context=test_project.command_context,
        services=test_project.services,
    )
    payload = json.loads(run_update_kb(runtime, UpdateKbInput(graph_method="auto")))
    assert "summary" in payload
