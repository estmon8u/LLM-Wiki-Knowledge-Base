"""Optional dependency helpers for WikiGraphRAG.

The WikiGraphRAG backend depends on NetworkX (always) and BM25S (optional).
Both are exposed only through the ``wikigraph`` Poetry extra; this module
keeps the imports lazy so that a base install (no `-E wikigraph`) can still
import :mod:`graphwiki_kb.services` and run unrelated commands such as
``kb init``, ``kb status``, or ``kb find --engine wiki``.
"""

from __future__ import annotations

from typing import Any

WIKIGRAPH_EXTRA_HINT = "poetry install -E wikigraph"


def wikigraph_extra_hint() -> str:
    """Return the install command that enables WikiGraphRAG."""
    return WIKIGRAPH_EXTRA_HINT


def require_networkx() -> Any:
    """Import NetworkX or raise a clear install error.

    Returns:
        The imported :mod:`networkx` module.

    Raises:
        ImportError: When NetworkX is not installed, with an actionable
            install hint.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise ImportError(
            "WikiGraphRAG requires NetworkX. Install with: " f"{WIKIGRAPH_EXTRA_HINT}"
        ) from exc
    return nx


def try_import_bm25s() -> Any | None:
    """Return the :mod:`bm25s` module when installed, otherwise ``None``."""
    try:
        import bm25s
    except Exception:  # pragma: no cover - optional dependency
        return None
    return bm25s
