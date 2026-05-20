"""Custom WikiGraphRAG backend built from inspectable wiki artifacts."""

from graphwiki_kb.wikigraph.deps import require_networkx, wikigraph_extra_hint
from graphwiki_kb.wikigraph.models import (
    WikiGraphAnswer,
    WikiGraphEdge,
    WikiGraphNode,
    WikiGraphRetrievedContext,
)

__all__ = [
    "WikiGraphAnswer",
    "WikiGraphEdge",
    "WikiGraphNode",
    "WikiGraphRetrievedContext",
    "require_networkx",
    "wikigraph_extra_hint",
]
