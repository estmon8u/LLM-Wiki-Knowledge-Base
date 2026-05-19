"""Helpers for running GraphRAG-backed work outside agent worker threads."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def is_main_thread() -> bool:
    """Return whether the current thread is the interpreter main thread."""
    return threading.current_thread() is threading.main_thread()


def run_on_main_thread(func: Callable[[], T], *, fallback: Callable[[], T]) -> T:
    """Run *func* inline on the main thread, otherwise use *fallback*."""
    if is_main_thread():
        return func()
    return fallback()
