"""Tests for project file locking helpers."""

from __future__ import annotations

from graphwiki_kb.services.file_lock import file_lock, workspace_lock


def test_file_lock_is_reentrant_for_same_state_file(tmp_path) -> None:
    """Verifies nested same-file locks do not deadlock in one process."""
    state_file = tmp_path / "state.json"

    with file_lock(state_file):
        with file_lock(state_file):
            state_file.write_text("ok", encoding="utf-8")

    assert state_file.read_text(encoding="utf-8") == "ok"


def test_workspace_lock_is_reentrant(tmp_path) -> None:
    """Verifies workspace-level GraphRAG locks compose with nested operations."""
    workspace = tmp_path / "graph" / "graphrag"

    with workspace_lock(workspace):
        with workspace_lock(workspace):
            (workspace / "marker.txt").write_text("locked", encoding="utf-8")

    assert (workspace / "marker.txt").read_text(encoding="utf-8") == "locked"
