"""Community detection over the WikiGraphRAG graph."""

from __future__ import annotations

from dataclasses import dataclass

from graphwiki_kb.wikigraph.graph_store import WikiGraphStore
from graphwiki_kb.wikigraph.models import WikiGraphNode


@dataclass(frozen=True)
class WikiCommunity:
    """One detected community cluster."""

    community_id: str
    title: str
    summary: str
    member_ids: tuple[str, ...]
    representative_chunks: tuple[str, ...]


def detect_communities(
    store: WikiGraphStore,
    nodes: list[WikiGraphNode],
    *,
    algorithm: str = "louvain",
) -> list[WikiCommunity]:
    """Detect communities and build summary nodes."""
    if store.node_count == 0:
        return []
    undirected = store.graph.to_undirected()
    if algorithm != "louvain":
        raise ValueError(f"Unsupported community algorithm: {algorithm}")
    communities = store._nx.community.louvain_communities(  # type: ignore[attr-defined]
        undirected,
        seed=42,
    )
    node_by_id = {node.id: node for node in nodes}
    results: list[WikiCommunity] = []
    for index, member_set in enumerate(sorted(communities, key=len, reverse=True)):
        member_ids = tuple(sorted(str(node_id) for node_id in member_set))
        titles = [
            node_by_id[node_id].title
            for node_id in member_ids
            if node_id in node_by_id and node_by_id[node_id].kind != "chunk"
        ]
        chunk_ids = [
            node_id
            for node_id in member_ids
            if node_id in node_by_id and node_by_id[node_id].kind == "chunk"
        ]
        label = ", ".join(titles[:4]) or f"Community {index + 1}"
        summary_parts = [
            node_by_id[node_id].text[:240]
            for node_id in member_ids
            if node_id in node_by_id and node_by_id[node_id].text
        ]
        summary = " ".join(summary_parts)[:1200]
        results.append(
            WikiCommunity(
                community_id=f"community:{index}",
                title=label,
                summary=summary,
                member_ids=member_ids,
                representative_chunks=tuple(chunk_ids[:6]),
            )
        )
    return results
