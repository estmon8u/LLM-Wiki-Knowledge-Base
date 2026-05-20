"""Community detection helpers for the WikiGraphRAG pipeline.

NetworkX ships :func:`networkx.algorithms.community.louvain_communities` for
modularity-optimized community detection. We use it when available and fall
back to connected components otherwise so the pipeline still produces
``community`` nodes even on tiny graphs where Louvain would degenerate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from graphwiki_kb.wikigraph.deps import require_networkx
from graphwiki_kb.wikigraph.models import WikiGraphCommunity, WikiGraphNode

if TYPE_CHECKING:
    import networkx as nx


@dataclass
class CommunityDetectionResult:
    """Detection output prior to ``WikiGraphCommunity`` construction."""

    algorithm: str
    member_lists: list[list[str]]


def detect_communities(
    graph: nx.MultiGraph,
    *,
    seed: int = 42,
) -> CommunityDetectionResult:
    """Detect communities on ``graph``.

    Tries Louvain first; falls back to connected components on errors or on
    graphs that are too small for Louvain to produce stable results.
    """
    if graph.number_of_nodes() == 0:
        return CommunityDetectionResult(algorithm="empty", member_lists=[])
    nx = require_networkx()
    simple = nx.Graph()
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += weight
        else:
            simple.add_edge(u, v, weight=weight)
    for node_id in graph.nodes:
        if node_id not in simple:
            simple.add_node(node_id)
    try:
        partitions = nx.community.louvain_communities(
            simple, weight="weight", seed=seed
        )
        algorithm = "louvain"
    except Exception:
        partitions = list(nx.connected_components(simple))
        algorithm = "connected_components"
    member_lists = [sorted(community) for community in partitions if community]
    member_lists.sort(
        key=lambda members: (-len(members), members[0] if members else "")
    )
    return CommunityDetectionResult(algorithm=algorithm, member_lists=member_lists)


def build_community_records(
    detection: CommunityDetectionResult,
    *,
    nodes_by_id: Mapping[str, WikiGraphNode],
    min_size: int = 1,
) -> list[WikiGraphCommunity]:
    """Convert raw community member lists into :class:`WikiGraphCommunity`.

    Each community gets:

    * a deterministic id ``community-<index>``;
    * a short summary listing top entity titles and source ids;
    * a level (always ``0`` for the flat partition we produce).
    """
    records: list[WikiGraphCommunity] = []
    for index, members in enumerate(detection.member_lists):
        if len(members) < min_size:
            continue
        entity_titles: list[str] = []
        source_ids: list[str] = []
        page_titles: list[str] = []
        for member in members:
            node = nodes_by_id.get(member)
            if node is None:
                continue
            if node.kind == "entity":
                entity_titles.append(node.title)
            if node.kind in {"source_page", "concept_page", "analysis_page"}:
                page_titles.append(node.title)
            for sid in node.source_ids:
                if sid not in source_ids:
                    source_ids.append(sid)
        top_entities = entity_titles[:8]
        community_title = "Community " + str(index + 1)
        if top_entities:
            community_title += ": " + ", ".join(top_entities[:3])
        elif page_titles:
            community_title += ": " + ", ".join(page_titles[:3])
        summary_parts: list[str] = []
        if top_entities:
            summary_parts.append("Entities: " + ", ".join(top_entities))
        if page_titles:
            summary_parts.append("Pages: " + ", ".join(page_titles[:8]))
        if source_ids:
            summary_parts.append("Sources: " + ", ".join(source_ids[:8]))
        summary = (
            " | ".join(summary_parts) if summary_parts else "(no annotated members)"
        )
        records.append(
            WikiGraphCommunity(
                id=f"community-{index + 1}",
                level=0,
                title=community_title,
                members=members,
                summary=summary,
                top_entities=top_entities,
                source_ids=source_ids,
            )
        )
    return records
