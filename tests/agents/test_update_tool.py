"""Tests for the update_kb agent tool.

The tool is now backed by :class:`UpdateService` (the canonical ``kb update``
pipeline) with a subprocess fallback for off-main-thread runs. These tests
patch the in-process / subprocess entrypoints to keep the unit tests fast and
provider-free, while still exercising approval gating, dry-run behavior, and
error paths.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphwiki_kb.agents import tools as tools_pkg  # noqa: F401 - import side-effect ok
from graphwiki_kb.agents.models import UpdateInput, UpdateOutput
from graphwiki_kb.agents.tools import update as update_tool
from graphwiki_kb.agents.tools.update import run_update_kb


class _FakeGraphStatus:
    graph_freshness_state = "fresh"
    graph_stale_reasons: list[str] = []


def _patch_status(runtime: Any) -> None:
    class _Stub:
        def status(self) -> _FakeGraphStatus:
            return _FakeGraphStatus()

    runtime.services.graphrag_status = _Stub()  # type: ignore[assignment]


def test_update_requires_approval_by_default(runtime: Any) -> None:
    _patch_status(runtime)
    output = run_update_kb(runtime, UpdateInput())

    assert output.ok is False
    assert "approval" in output.summary.lower()
    assert len(runtime.pending_approvals) == 1
    pending = runtime.pending_approvals[0]
    assert pending.tool_name == "update_kb"
    # The approval payload exposes the requested update knobs.
    assert pending.payload["graph_method"] == "auto"
    assert pending.payload["no_graph"] is False
    assert pending.payload["graph_only"] is False
    assert pending.payload["no_wikigraph"] is False
    assert pending.payload["export_wikigraph_artifacts"] is False


def test_update_with_auto_approve_runs_update_service(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_status(runtime)
    runtime.auto_approve = True

    called: dict[str, Any] = {}

    def _fake_inprocess(rt: Any, payload: UpdateInput) -> UpdateOutput:
        called["inprocess"] = payload
        return UpdateOutput(
            ok=True,
            summary="Update completed: compile, graph(fast).",
            method="fast",
            next_action="kb status",
            graph_freshness="fresh",
            details={"compile": True, "graph": True},
        )

    def _fake_subprocess(
        rt: Any, payload: UpdateInput
    ) -> UpdateOutput:  # pragma: no cover - guard rail
        raise AssertionError("Subprocess fallback should not run on main thread")

    monkeypatch.setattr(update_tool, "_run_inprocess", _fake_inprocess)
    monkeypatch.setattr(update_tool, "_run_subprocess", _fake_subprocess)

    output = run_update_kb(runtime, UpdateInput(graph_method="fast"))

    assert output.ok is True
    assert output.method == "fast"
    assert "Update completed" in output.summary
    assert called["inprocess"].graph_method == "fast"
    assert runtime.tool_results[-1].ok is True
    assert runtime.pending_approvals == []


def test_update_falls_back_to_subprocess_off_main_thread(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_status(runtime)
    runtime.auto_approve = True

    monkeypatch.setattr(update_tool, "is_main_thread", lambda: False, raising=False)
    # `run_on_main_thread` is imported into the module's namespace.
    monkeypatch.setattr(
        update_tool,
        "run_on_main_thread",
        lambda func, *, fallback: fallback(),
    )

    def _fake_subprocess(rt: Any, payload: UpdateInput) -> UpdateOutput:
        return UpdateOutput(
            ok=True,
            summary="kb update completed via subprocess.",
            next_action="kb status",
            details={"mode": "subprocess"},
        )

    monkeypatch.setattr(update_tool, "_run_subprocess", _fake_subprocess)

    output = run_update_kb(runtime, UpdateInput())

    assert output.ok is True
    assert "subprocess" in output.summary
    assert output.details.get("mode") == "subprocess"


def test_update_dry_run_skips_approval_and_does_not_mutate(runtime: Any) -> None:
    _patch_status(runtime)

    output = run_update_kb(runtime, UpdateInput(dry_run=True))

    assert output.ok is True
    assert "Dry run" in output.summary
    assert runtime.pending_approvals == []
    assert runtime.tool_results[-1].ok is True
    assert runtime.tool_results[-1].data.get("details", {}).get("dry_run") is True


def test_update_returns_diagnostics_on_exception(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_status(runtime)
    runtime.auto_approve = True

    def _boom(rt: Any, payload: UpdateInput) -> UpdateOutput:
        raise RuntimeError("compile blew up")

    monkeypatch.setattr(update_tool, "_run_inprocess", _boom)

    output = run_update_kb(runtime, UpdateInput())

    assert output.ok is False
    assert "compile blew up" in output.summary
    assert output.diagnostics == ["compile blew up"]
    assert runtime.tool_results[-1].ok is False


# ---------------------------------------------------------------------------
# Helpers: keep _summarize_result/_graph_freshness/_run_subprocess covered
# without needing a real provider, GraphRAG workspace, or child process.
# ---------------------------------------------------------------------------


class _DummyDecision:
    def __init__(self, method: str | None, action: str = "index") -> None:
        self.method = method
        self.action = action


class _DummySync:
    def __init__(self, method: str) -> None:
        self.decision = _DummyDecision(method=method)


class _DummyGraphResult:
    def __init__(
        self,
        *,
        skipped: bool = False,
        skip_reason: str = "",
        method: str | None = "fast",
        with_sync: bool = True,
        with_preflight: bool = False,
    ) -> None:
        self.skipped = skipped
        self.skip_reason = skip_reason
        self.sync_result = _DummySync(method) if with_sync else None
        self.preflight_result = _DummySync(method) if with_preflight else None


class _DummyWikiGraphResult:
    def __init__(
        self,
        *,
        skipped: bool = False,
        skip_reason: str = "",
        with_build: bool = True,
    ) -> None:
        self.skipped = skipped
        self.skip_reason = skip_reason
        self.build = object() if with_build else None


class _DummyUpdateResult:
    def __init__(
        self,
        *,
        compile_result: Any = None,
        search_refreshed: bool = False,
        graph_result: Any = None,
        wikigraph_result: Any = None,
    ) -> None:
        self.compile_result = compile_result
        self.search_refreshed = search_refreshed
        self.graph_result = graph_result
        self.wikigraph_result = wikigraph_result

    @property
    def ok(self) -> bool:
        return (
            self.compile_result is not None
            or self.graph_result is not None
            or self.wikigraph_result is not None
        )


def test_summarize_result_reports_empty_when_no_changes() -> None:
    summary, method, details = update_tool._summarize_result(_DummyUpdateResult())
    assert summary == "Update produced no changes."
    assert method == ""
    assert details["compile"] is False
    assert details["graph"] is False


def test_summarize_result_reports_combined_phases() -> None:
    summary, method, details = update_tool._summarize_result(
        _DummyUpdateResult(
            compile_result=object(),
            search_refreshed=True,
            graph_result=_DummyGraphResult(method="fast"),
        )
    )
    assert "compile" in summary
    assert "search" in summary
    assert "graph(fast)" in summary
    assert method == "fast"
    assert details["graph_skipped"] is False


def test_summarize_result_reports_wikigraph() -> None:
    summary, _, details = update_tool._summarize_result(
        _DummyUpdateResult(
            compile_result=object(),
            wikigraph_result=_DummyWikiGraphResult(),
        )
    )
    assert "wikigraph" in summary
    assert details["wikigraph"] is True
    assert details["wikigraph_skipped"] is False


def test_summarize_result_reports_wikigraph_no_build() -> None:
    summary, _, _ = update_tool._summarize_result(
        _DummyUpdateResult(
            wikigraph_result=_DummyWikiGraphResult(with_build=False),
        )
    )
    assert "wikigraph(no-build)" in summary


def test_summarize_result_reports_skipped_wikigraph() -> None:
    summary, _, details = update_tool._summarize_result(
        _DummyUpdateResult(
            wikigraph_result=_DummyWikiGraphResult(
                skipped=True,
                skip_reason="disabled",
                with_build=False,
            ),
        )
    )
    assert "wikigraph(skipped:disabled)" in summary
    assert details["wikigraph_skipped"] is True


def test_summarize_result_reports_skipped_graph() -> None:
    summary, _, details = update_tool._summarize_result(
        _DummyUpdateResult(
            compile_result=object(),
            graph_result=_DummyGraphResult(
                skipped=True,
                skip_reason="missing-credentials",
                with_sync=False,
            ),
        )
    )
    assert "graph(skipped:missing-credentials)" in summary
    assert details["graph_skipped"] is True


def test_summarize_result_falls_back_to_preflight_method() -> None:
    _, method, _ = update_tool._summarize_result(
        _DummyUpdateResult(
            graph_result=_DummyGraphResult(
                method="standard",
                with_sync=False,
                with_preflight=True,
            ),
        )
    )
    assert method == "standard"


def test_graph_freshness_returns_warnings_when_stale(runtime: Any) -> None:
    class _StaleStatus:
        graph_freshness_state = "stale"
        graph_stale_reasons = ["input hash changed"]

    runtime.services.graphrag_status = type(
        "_S", (), {"status": staticmethod(lambda: _StaleStatus())}
    )()

    freshness, warnings = update_tool._graph_freshness(runtime)
    assert freshness == "stale"
    assert "input hash changed" in warnings
    assert any("kb status" in w for w in warnings)


def test_graph_freshness_returns_empty_on_status_failure(runtime: Any) -> None:
    class _Boom:
        def status(self) -> None:
            raise RuntimeError("status broken")

    runtime.services.graphrag_status = _Boom()  # type: ignore[assignment]
    assert update_tool._graph_freshness(runtime) == (None, [])


def test_run_subprocess_success_branch(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_status(runtime)
    captured: dict[str, Any] = {}

    class _CompletedOk:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def _fake_run(command: list[str], **kwargs: Any) -> _CompletedOk:
        captured["command"] = command
        captured["cwd"] = kwargs.get("cwd")
        return _CompletedOk()

    monkeypatch.setattr(update_tool.subprocess, "run", _fake_run)

    output = update_tool._run_subprocess(
        runtime,
        UpdateInput(graph_method="fast", force=True, graph_only=True),
    )
    assert output.ok is True
    assert output.details["mode"] == "subprocess"
    # Argument propagation
    assert "update" in captured["command"]
    assert "--graph-only" in captured["command"]
    assert "--force" in captured["command"]
    assert "--graph-method" in captured["command"]
    assert captured["cwd"] == str(runtime.command_context.project_root)


def test_run_subprocess_passes_wikigraph_flags(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_status(runtime)
    captured: dict[str, Any] = {}

    class _CompletedOk:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(command: list[str], **kwargs: Any) -> _CompletedOk:
        captured["command"] = command
        return _CompletedOk()

    monkeypatch.setattr(update_tool.subprocess, "run", _fake_run)

    update_tool._run_subprocess(
        runtime,
        UpdateInput(
            no_wikigraph=True,
            wikigraph_include_graphrag_export_pages=True,
            export_wikigraph_artifacts=True,
        ),
    )
    assert "--no-wikigraph" in captured["command"]
    assert "--wikigraph-include-graphrag-export-pages" in captured["command"]
    assert "--export-wikigraph-artifacts" in captured["command"]


def test_run_subprocess_failure_branch(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_status(runtime)

    class _Failed:
        returncode = 3
        stdout = ""
        stderr = "bad config"

    monkeypatch.setattr(update_tool.subprocess, "run", lambda *a, **kw: _Failed())

    output = update_tool._run_subprocess(runtime, UpdateInput(no_graph=True))
    assert output.ok is False
    assert "bad config" in output.diagnostics[0]
    assert output.details["exit_code"] == 3
