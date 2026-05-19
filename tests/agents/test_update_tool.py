"""Tests for the update_kb agent tool."""

from __future__ import annotations

from dataclasses import dataclass, field

from graphwiki_kb.agents.models import UpdateInput
from graphwiki_kb.agents.tools.update import run_update_kb
from graphwiki_kb.services.graphrag_sync_service import (
    GraphRAGSyncDecision,
    GraphRAGSyncError,
)


@dataclass
class _FakeInputSync:
    source_count: int = 0


@dataclass
class _FakeSyncResult:
    decision: GraphRAGSyncDecision
    input_sync: _FakeInputSync = field(default_factory=_FakeInputSync)


def _decision(action: str = "noop", method: str | None = None) -> GraphRAGSyncDecision:
    return GraphRAGSyncDecision(
        action=action,
        method=method,
        reason="reason",
        output_state="present",
        input_digest="d",
        config_digest="c",
        input_changed=False,
        config_changed=False,
        changed_source_count=None,
    )


def test_update_requires_approval_by_default(runtime) -> None:
    class _StubSync:
        def sync(self, **kwargs):
            raise AssertionError("sync should not run without approval")

    runtime.services.graphrag_sync = _StubSync()  # type: ignore[assignment]

    output = run_update_kb(runtime, UpdateInput())

    assert output.ok is False
    assert "approval" in output.summary.lower()
    assert len(runtime.pending_approvals) == 1
    assert runtime.pending_approvals[0].tool_name == "update_kb"


def test_update_with_auto_approve_runs_sync(runtime) -> None:
    class _StubSync:
        def sync(self, **kwargs):
            return _FakeSyncResult(
                decision=_decision(action="incremental", method="fast")
            )

    runtime.services.graphrag_sync = _StubSync()  # type: ignore[assignment]
    runtime.auto_approve = True

    output = run_update_kb(runtime, UpdateInput())

    assert output.ok is True
    assert output.method == "fast"
    assert "incremental" in output.summary
    assert runtime.tool_results[-1].ok is True


def test_update_returns_diagnostics_on_sync_error(runtime) -> None:
    class _StubSync:
        def sync(self, **kwargs):
            raise GraphRAGSyncError("missing input")

    runtime.services.graphrag_sync = _StubSync()  # type: ignore[assignment]
    runtime.auto_approve = True

    output = run_update_kb(runtime, UpdateInput())

    assert output.ok is False
    assert "missing input" in output.summary
    assert runtime.tool_results[-1].ok is False


def test_update_dry_run_runs_sync_without_approval(runtime) -> None:
    captured: dict[str, object] = {}

    class _StubSync:
        def sync(self, **kwargs):
            captured.update(kwargs)
            return _FakeSyncResult(decision=_decision(action="plan", method=None))

    runtime.services.graphrag_sync = _StubSync()  # type: ignore[assignment]

    output = run_update_kb(runtime, UpdateInput(dry_run=True))

    assert output.ok is True
    assert captured["dry_run"] is True
    assert captured["run_index"] is False
