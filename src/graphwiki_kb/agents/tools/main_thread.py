"""Helpers for running GraphRAG-backed work outside agent worker threads.

This module belongs to ``graphwiki_kb.agents.tools.main_thread`` and keeps
related behavior close to the command, service, model, provider, storage,
script, or test surface that uses it.

GraphRAG indexing registers POSIX signal handlers during its run, and the
``signal`` module only accepts handler registration from the interpreter's
main thread. The OpenAI Agents SDK invokes function tools from its own
async runtime, which is usually on the main thread for a CLI process but
may move to a worker thread under test runners or future SDK versions.
``run_on_main_thread`` lets a tool execute heavy in-process work when it
can, and delegate to a safer subprocess fallback when it cannot.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def is_main_thread() -> bool:
    """Return ``True`` if the current thread is the interpreter main thread."""
    return threading.current_thread() is threading.main_thread()


def run_on_main_thread(func: Callable[[], T], *, fallback: Callable[[], T]) -> T:
    """Invoke ``func`` inline on the main thread, otherwise use ``fallback``."""
    if is_main_thread():
        return func()
    return fallback()
