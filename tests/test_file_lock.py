"""Tests for project file locking helpers."""

from __future__ import annotations

import threading

from graphwiki_kb.services import file_lock as file_lock_module
from graphwiki_kb.services.file_lock import file_lock, workspace_lock


def test_file_lock_is_reentrant_for_same_state_file(tmp_path) -> None:
    """Verifies nested same-file locks do not deadlock in one process."""
    state_file = tmp_path / "state.json"

    with file_lock(state_file):
        with file_lock(state_file):
            state_file.write_text("ok", encoding="utf-8")

    assert state_file.read_text(encoding="utf-8") == "ok"


def test_file_lock_reentrancy_is_thread_owner_scoped(tmp_path) -> None:
    """Regression: one thread must not inherit another thread's reentrant lock."""
    lock_path = (tmp_path / ".state.json.lock").resolve()
    other_thread_result = []

    file_lock_module._mark_lock_held(lock_path)
    try:
        assert file_lock_module._enter_reentrant_lock(lock_path) is True

        def attempt_reentrant_enter() -> None:
            other_thread_result.append(
                file_lock_module._enter_reentrant_lock(lock_path)
            )

        thread = threading.Thread(target=attempt_reentrant_enter)
        thread.start()
        thread.join(timeout=5)

        assert other_thread_result == [False]
    finally:
        file_lock_module._leave_reentrant_lock(lock_path)
        file_lock_module._leave_reentrant_lock(lock_path)


def test_workspace_lock_is_reentrant(tmp_path) -> None:
    """Verifies workspace-level GraphRAG locks compose with nested operations."""
    workspace = tmp_path / "graph" / "graphrag"

    with workspace_lock(workspace):
        with workspace_lock(workspace):
            (workspace / "marker.txt").write_text("locked", encoding="utf-8")

    assert (workspace / "marker.txt").read_text(encoding="utf-8") == "locked"
