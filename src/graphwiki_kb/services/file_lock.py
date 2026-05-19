"""Small cross-process file lock helpers for project state files."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_HELD_LOCKS: dict[tuple[Path, int], int] = {}
_HELD_LOCKS_GUARD = threading.Lock()


@contextmanager
def file_lock(target_path: Path) -> Iterator[None]:
    """Lock a sibling ``.lock`` file while a state file is read and rewritten."""
    lock_path = target_path.with_name(f".{target_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_lock_path = lock_path.resolve()
    if _enter_reentrant_lock(resolved_lock_path):
        try:
            yield
        finally:
            _leave_reentrant_lock(resolved_lock_path)
        return
    with lock_path.open("a+b") as handle:
        _acquire_lock(handle)
        _mark_lock_held(resolved_lock_path)
        try:
            yield
        finally:
            _unmark_lock_held(resolved_lock_path)
            _release_lock(handle)


@contextmanager
def workspace_lock(workspace_dir: Path) -> Iterator[None]:
    """Serialize multi-file GraphRAG workspace reads and writes."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(workspace_dir / "workspace.state"):
        yield


def _enter_reentrant_lock(lock_path: Path) -> bool:
    key = _thread_lock_key(lock_path)
    with _HELD_LOCKS_GUARD:
        count = _HELD_LOCKS.get(key)
        if count is None:
            return False
        _HELD_LOCKS[key] = count + 1
        return True


def _leave_reentrant_lock(lock_path: Path) -> None:
    key = _thread_lock_key(lock_path)
    with _HELD_LOCKS_GUARD:
        count = _HELD_LOCKS.get(key, 0)
        if count <= 1:
            _HELD_LOCKS.pop(key, None)
        else:
            _HELD_LOCKS[key] = count - 1


def _mark_lock_held(lock_path: Path) -> None:
    key = _thread_lock_key(lock_path)
    with _HELD_LOCKS_GUARD:
        _HELD_LOCKS[key] = _HELD_LOCKS.get(key, 0) + 1


def _unmark_lock_held(lock_path: Path) -> None:
    _leave_reentrant_lock(lock_path)


def _thread_lock_key(lock_path: Path) -> tuple[Path, int]:
    return lock_path, threading.get_ident()


def _acquire_lock(handle) -> None:
    try:
        import msvcrt
    except ImportError:  # pragma: no cover - POSIX fallback
        import fcntl as fcntl_module

        fcntl: Any = fcntl_module

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]


def _release_lock(handle) -> None:
    try:
        import msvcrt
    except ImportError:  # pragma: no cover - POSIX fallback
        import fcntl as fcntl_module

        fcntl: Any = fcntl_module

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
