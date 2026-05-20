"""NetworkX-backed graph storage and traversal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphwiki_kb.wikigraph.deps import require_networkx
from graphwiki_kb.wikigraph.models import WikiGraphEdge, WikiGraphNode


class WikiGraphStore:
    """Load, save, and traverse the WikiGraphRAG graph."""

    def __init__(self) -> None:
        self._nx = require_networkx()
        self.graph = self._nx.DiGraph()

    @property
    def node_count(self) -> int:
        return int(self.graph.number_of_nodes())

    @property
    def edge_count(self) -> int:
        return int(self.graph.number_of_edges())

    def add_node(self, node: WikiGraphNode) -> None:
        self.graph.add_node(
            node.id,
            kind=node.kind,
            title=node.title,
            path=node.path,
            text=node.text,
            metadata=node.metadata,
        )

    def add_edge(self, edge: WikiGraphEdge) -> None:
        self.graph.add_edge(
            edge.source,
            edge.target,
            kind=edge.kind,
            weight=edge.weight,
            evidence=edge.evidence,
        )

    def neighbors(
        self,
        node_id: str,
        *,
        hops: int = 1,
        kinds: set[str] | None = None,
    ) -> list[tuple[str, list[str]]]:
        """Return reachable node ids with edge-kind traces."""
        if node_id not in self.graph:
            return []
        visited: dict[str, list[str]] = {node_id: [node_id]}
        frontier = [node_id]
        for _ in range(hops):
            next_frontier: list[str] = []
            for current in frontier:
                for _, target, data in self.graph.out_edges(current, data=True):
                    edge_kind = str(data.get("kind", "related_to"))
                    if kinds and edge_kind not in kinds:
                        continue
                    trace = visited[current] + [f"{edge_kind}->{target}"]
                    if target not in visited or len(trace) < len(visited[target]):
                        visited[target] = trace
                        next_frontier.append(target)
                for source, _, data in self.graph.in_edges(current, data=True):
                    edge_kind = str(data.get("kind", "related_to"))
                    if kinds and edge_kind not in kinds:
                        continue
                    trace = visited[current] + [f"{edge_kind}<-{source}"]
                    if source not in visited or len(trace) < len(visited[source]):
                        visited[source] = trace
                        next_frontier.append(source)
            frontier = next_frontier
        return [(node, trace) for node, trace in visited.items() if node != node_id]

    def pagerank_scores(self) -> dict[str, float]:
        if self.graph.number_of_nodes() == 0:
            return {}
        scores = self._nx.pagerank(self.graph, alpha=0.85, max_iter=100)
        return {str(node): float(score) for node, score in scores.items()}

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        node_link = self._nx.node_link_data(self.graph)
        (output_dir / "graph_node_link.json").write_text(
            json.dumps(node_link, indent=2),
            encoding="utf-8",
        )

    def load_node_link(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.graph = self._nx.node_link_graph(payload, directed=True)

    @staticmethod
    def write_nodes(path: Path, nodes: list[WikiGraphNode]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([node.model_dump() for node in nodes], indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def write_edges(path: Path, edges: list[WikiGraphEdge]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([edge.model_dump() for edge in edges], indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def read_nodes(path: Path) -> list[WikiGraphNode]:
        payload: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return [WikiGraphNode.model_validate(item) for item in payload]

    @staticmethod
    def read_edges(path: Path) -> list[WikiGraphEdge]:
        payload: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return [WikiGraphEdge.model_validate(item) for item in payload]
