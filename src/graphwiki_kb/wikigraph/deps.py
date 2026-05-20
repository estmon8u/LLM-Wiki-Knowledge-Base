"""Optional dependency helpers for WikiGraphRAG."""

from __future__ import annotations

WIKIGRAPH_EXTRA_HINT = "poetry install -E wikigraph"


def wikigraph_extra_hint() -> str:
    """Return install instructions for WikiGraphRAG extras."""
    return WIKIGRAPH_EXTRA_HINT


def require_networkx() -> object:
    """Import NetworkX or raise a clear install error."""
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "WikiGraphRAG requires NetworkX. Install with: " f"{WIKIGRAPH_EXTRA_HINT}"
        ) from exc
    return nx


def try_import_bm25s() -> object | None:
    """Return the bm25s module when installed."""
    try:
        import bm25s
    except ImportError:
        return None
    return bm25s
