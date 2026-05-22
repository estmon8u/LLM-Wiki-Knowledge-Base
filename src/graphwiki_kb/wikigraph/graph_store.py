"""Persistent JSON store for wiki graph nodes, edges, and communities.

The store uses NetworkX as the in-memory graph workhorse but always writes
plain JSON artifacts (``nodes.json``, ``edges.json``, ``communities.json``,
``index.json``) so the WikiGraphRAG state remains fully inspectable without
any extra tooling.

NetworkX is imported lazily via
:func:`graphwiki_kb.wikigraph.deps.require_networkx` so a base install
without the ``wikigraph`` extra can still import this module's pure-Python
helpers (``WikiGraphStore.save`` / ``WikiGraphStore.load``) without
triggering an ``ImportError`` at import time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graphwiki_kb.services.project_service import atomic_write_text
from graphwiki_kb.wikigraph.deps import require_networkx
from graphwiki_kb.wikigraph.models import (
    WikiGraphCommunity,
    WikiGraphEdge,
    WikiGraphIndex,
    WikiGraphNode,
)

if TYPE_CHECKING:
    import networkx as nx


@dataclass
class WikiGraphStorePaths:
    """Filesystem layout for persisted WikiGraphRAG artifacts."""

    root: Path

    @property
    def index_dir(self) -> Path:
        """Top-level wikigraph artifact directory."""
        return self.root

    @property
    def nodes_file(self) -> Path:
        """JSON file holding every :class:`WikiGraphNode`."""
        return self.root / "nodes.json"

    @property
    def edges_file(self) -> Path:
        """JSON file holding every :class:`WikiGraphEdge`."""
        return self.root / "edges.json"

    @property
    def communities_file(self) -> Path:
        """JSON file holding every :class:`WikiGraphCommunity`."""
        return self.root / "communities.json"

    @property
    def chunks_file(self) -> Path:
        """JSON file holding chunk metadata (a slice of ``nodes.json``)."""
        return self.root / "chunks.json"

    @property
    def text_units_file(self) -> Path:
        """JSON file holding source-derived TextUnit metadata.

        TextUnits are also present in ``nodes.json``; this dedicated file
        makes inspection / evaluator slicing easier without re-filtering
        the full node list.
        """
        return self.root / "text_units.json"

    @property
    def documents_file(self) -> Path:
        """JSON file holding ``source_document`` nodes."""
        return self.root / "documents.json"

    @property
    def index_file(self) -> Path:
        """JSON file holding top-level build metadata for the index."""
        return self.root / "index.json"

    @property
    def runs_dir(self) -> Path:
        """Directory holding per-build run JSON files."""
        return self.root / "runs"


class WikiGraphStore:
    """Loads, persists, and exposes the wiki graph index as a NetworkX graph."""

    def __init__(self, paths: WikiGraphStorePaths) -> None:
        self.paths = paths

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, index: WikiGraphIndex) -> list[str]:
        """Persist ``index`` as JSON artifacts. Returns written file paths."""
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.runs_dir.mkdir(parents=True, exist_ok=True)

        nodes_payload = [node.model_dump() for node in index.nodes]
        edges_payload = [edge.model_dump() for edge in index.edges]
        communities_payload = [
            community.model_dump() for community in index.communities
        ]
        chunks_payload = [
            node.model_dump() for node in index.nodes if node.kind == "chunk"
        ]
        text_units_payload = [
            node.model_dump() for node in index.nodes if node.kind == "text_unit"
        ]
        documents_payload = [
            node.model_dump() for node in index.nodes if node.kind == "source_document"
        ]
        index_payload = {
            "built_at": index.built_at,
            "node_count": len(index.nodes),
            "edge_count": len(index.edges),
            "chunk_count": index.chunk_count,
            "text_unit_count": index.text_unit_count,
            "document_count": index.document_count,
            "entity_count": index.entity_count,
            "source_count": index.source_count,
            "community_count": len(index.communities),
            "include_graphrag_export_pages": index.include_graphrag_export_pages,
            "include_normalized_text_units": index.include_normalized_text_units,
        }

        atomic_write_text(
            self.paths.nodes_file, json.dumps(nodes_payload, indent=2, default=str)
        )
        atomic_write_text(
            self.paths.edges_file, json.dumps(edges_payload, indent=2, default=str)
        )
        atomic_write_text(
            self.paths.communities_file,
            json.dumps(communities_payload, indent=2, default=str),
        )
        atomic_write_text(
            self.paths.chunks_file, json.dumps(chunks_payload, indent=2, default=str)
        )
        atomic_write_text(
            self.paths.text_units_file,
            json.dumps(text_units_payload, indent=2, default=str),
        )
        atomic_write_text(
            self.paths.documents_file,
            json.dumps(documents_payload, indent=2, default=str),
        )
        atomic_write_text(
            self.paths.index_file, json.dumps(index_payload, indent=2, default=str)
        )

        run_file = self.paths.runs_dir / f"build-{_safe_timestamp(index.built_at)}.json"
        atomic_write_text(run_file, json.dumps(index_payload, indent=2, default=str))
        latest_file = self.paths.runs_dir / "latest.json"
        atomic_write_text(latest_file, json.dumps(index_payload, indent=2, default=str))

        return [
            str(self.paths.nodes_file),
            str(self.paths.edges_file),
            str(self.paths.communities_file),
            str(self.paths.chunks_file),
            str(self.paths.text_units_file),
            str(self.paths.documents_file),
            str(self.paths.index_file),
            str(run_file),
            str(latest_file),
        ]

    def exists(self) -> bool:
        """Return ``True`` when a persisted index is present on disk."""
        return self.paths.index_file.exists() and self.paths.nodes_file.exists()

    def load(self) -> WikiGraphIndex | None:
        """Load the persisted index from disk, returning ``None`` if missing."""
        if not self.exists():
            return None
        try:
            nodes_raw = json.loads(self.paths.nodes_file.read_text(encoding="utf-8"))
            edges_raw = json.loads(self.paths.edges_file.read_text(encoding="utf-8"))
            index_raw = json.loads(self.paths.index_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            communities_raw = json.loads(
                self.paths.communities_file.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            communities_raw = []
        nodes = [WikiGraphNode.model_validate(item) for item in nodes_raw]
        edges = [WikiGraphEdge.model_validate(item) for item in edges_raw]
        communities = [
            WikiGraphCommunity.model_validate(item) for item in communities_raw
        ]
        return WikiGraphIndex(
            nodes=nodes,
            edges=edges,
            communities=communities,
            built_at=str(index_raw.get("built_at", "")),
            include_graphrag_export_pages=bool(
                index_raw.get("include_graphrag_export_pages", False)
            ),
            include_normalized_text_units=bool(
                index_raw.get("include_normalized_text_units", False)
            ),
            source_count=int(index_raw.get("source_count", 0)),
            document_count=int(index_raw.get("document_count", 0)),
            chunk_count=int(index_raw.get("chunk_count", 0)),
            text_unit_count=int(index_raw.get("text_unit_count", 0)),
            entity_count=int(index_raw.get("entity_count", 0)),
        )

    # ------------------------------------------------------------------ #
    # NetworkX adapters                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_networkx(index: WikiGraphIndex) -> nx.MultiGraph:
        """Build an undirected MultiGraph view of ``index``.

        Edges of every kind are added with their ``kind`` and ``weight``
        preserved so that centrality, PageRank, and community detection see
        a consistent topology.
        """
        nx = require_networkx()
        graph = nx.MultiGraph()
        for node in index.nodes:
            graph.add_node(
                node.id,
                kind=node.kind,
                title=node.title,
                path=node.path,
                source_ids=tuple(node.source_ids),
            )
        for edge in index.edges:
            graph.add_edge(
                edge.source,
                edge.target,
                kind=edge.kind,
                weight=edge.weight,
            )
        return graph


def _safe_timestamp(timestamp: str) -> str:
    if not timestamp:
        return "unknown"
    return "".join(c if c.isalnum() else "-" for c in timestamp).strip("-") or "unknown"


def collect_neighbors(
    graph: nx.MultiGraph,
    seed: str,
    *,
    max_hops: int,
    edge_kinds: Iterable[str] | None = None,
) -> list[tuple[str, int, str]]:
    """Return ``(node_id, distance, via_edge_kind)`` tuples within ``max_hops``."""
    if seed not in graph:
        return []
    allowed = set(edge_kinds) if edge_kinds is not None else None
    visited: dict[str, tuple[int, str]] = {seed: (0, "")}
    frontier: list[tuple[str, int]] = [(seed, 0)]
    results: list[tuple[str, int, str]] = []
    while frontier:
        node_id, distance = frontier.pop(0)
        if distance >= max_hops:
            continue
        for neighbor in graph.neighbors(node_id):
            edges = graph.get_edge_data(node_id, neighbor) or {}
            edge_kind = ""
            for data in edges.values():
                kind = str(data.get("kind", ""))
                if allowed is None or kind in allowed:
                    edge_kind = kind
                    break
            else:
                continue
            if neighbor in visited:
                continue
            visited[neighbor] = (distance + 1, edge_kind)
            results.append((neighbor, distance + 1, edge_kind))
            frontier.append((neighbor, distance + 1))
    return results


def node_pagerank(graph: nx.MultiGraph, *, max_iter: int = 50) -> dict[str, float]:
    """Compute PageRank scores on a simplified graph.

    NetworkX's ``pagerank`` does not accept ``MultiGraph`` directly, so we
    collapse parallel edges by summing their weights before running the
    algorithm.
    """
    if graph.number_of_nodes() == 0:
        return {}
    nx = require_networkx()
    simple = nx.Graph()
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += weight
        else:
            simple.add_edge(u, v, weight=weight)
    for node_id, data in graph.nodes(data=True):
        if node_id not in simple:
            simple.add_node(node_id, **data)
    try:
        return nx.pagerank(simple, max_iter=max_iter, weight="weight")
    except nx.PowerIterationFailedConvergence:
        return {node_id: 1.0 / simple.number_of_nodes() for node_id in simple.nodes}


def to_node_link_json(index: WikiGraphIndex) -> dict[str, Any]:
    """Return a NetworkX node-link representation, useful for visualization."""
    nx = require_networkx()
    graph = WikiGraphStore.to_networkx(index)
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        if simple.has_edge(u, v):
            simple[u][v]["weight"] = simple[u][v].get("weight", 1.0) + float(
                data.get("weight", 1.0)
            )
        else:
            simple.add_edge(u, v, **data)
    return nx.node_link_data(simple, edges="links")
