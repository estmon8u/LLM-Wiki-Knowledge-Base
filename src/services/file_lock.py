"""Small cross-process file lock helpers for project state files."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def file_lock(target_path: Path) -> Iterator[None]:
    """Lock a sibling ``.lock`` file while a state file is read and rewritten."""
    lock_path = target_path.with_name(f".{target_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        _acquire_lock(handle)
        try:
            yield
        finally:
            _release_lock(handle)


def _acquire_lock(handle) -> None:
    try:
        import msvcrt
    except ImportError:  # pragma: no cover - POSIX fallback
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)


def _release_lock(handle) -> None:
    try:
        import msvcrt
    except ImportError:  # pragma: no cover - POSIX fallback
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
