"""Build ranked retrieval contexts from graph and lexical signals."""

from __future__ import annotations

from graphwiki_kb.wikigraph.graph_store import WikiGraphStore
from graphwiki_kb.wikigraph.lexical_index import LexicalHit, LexicalIndex
from graphwiki_kb.wikigraph.markdown_parser import ParsedChunk
from graphwiki_kb.wikigraph.models import WikiGraphNode, WikiGraphRetrievedContext


def contexts_from_lexical_hits(
    hits: list[LexicalHit],
    *,
    chunk_nodes: dict[str, WikiGraphNode],
    chunk_records: dict[str, ParsedChunk],
    limit: int,
) -> list[WikiGraphRetrievedContext]:
    """Convert lexical hits into retrieval contexts."""
    contexts: list[WikiGraphRetrievedContext] = []
    for hit in hits[:limit]:
        node_id = f"chunk:{hit.chunk_id}"
        node = chunk_nodes.get(node_id)
        record = chunk_records.get(hit.chunk_id)
        if node is None or record is None:
            continue
        source_ids = [record.source_id] if record.source_id else []
        contexts.append(
            WikiGraphRetrievedContext(
                node_id=node_id,
                node_kind="chunk",
                title=node.title,
                path=node.path,
                text=node.text,
                score=hit.score,
                source_ids=source_ids,
                trace=[f"lexical:{hit.chunk_id}"],
            )
        )
    return contexts


def expand_contexts(
    store: WikiGraphStore,
    seed_node_ids: list[str],
    *,
    node_by_id: dict[str, WikiGraphNode],
    max_hops: int,
    limit: int,
) -> list[WikiGraphRetrievedContext]:
    """Expand seed nodes through the graph."""
    contexts: list[WikiGraphRetrievedContext] = []
    seen: set[str] = set()
    for seed in seed_node_ids:
        for node_id, trace in store.neighbors(seed, hops=max_hops):
            if node_id in seen:
                continue
            node = node_by_id.get(node_id)
            if node is None or not node.text.strip():
                continue
            seen.add(node_id)
            source_ids = [
                str(item)
                for item in node.metadata.get("source_ids", [])
                if str(item).strip()
            ]
            if node.metadata.get("source_id"):
                source_ids.append(str(node.metadata["source_id"]))
            contexts.append(
                WikiGraphRetrievedContext(
                    node_id=node_id,
                    node_kind=node.kind,
                    title=node.title,
                    path=node.path,
                    text=node.text[:2000],
                    score=0.5,
                    source_ids=sorted(set(source_ids)),
                    trace=trace,
                )
            )
            if len(contexts) >= limit:
                return contexts
    return contexts


def merge_contexts(
    *groups: list[WikiGraphRetrievedContext],
    limit: int,
) -> list[WikiGraphRetrievedContext]:
    """Merge context groups, keeping the highest score per node."""
    merged: dict[str, WikiGraphRetrievedContext] = {}
    for group in groups:
        for context in group:
            existing = merged.get(context.node_id)
            if existing is None or context.score > existing.score:
                merged[context.node_id] = context
    ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    return ordered[:limit]


def rank_with_pagerank(
    contexts: list[WikiGraphRetrievedContext],
    pagerank: dict[str, float],
) -> list[WikiGraphRetrievedContext]:
    """Boost contexts using PageRank scores."""
    boosted: list[WikiGraphRetrievedContext] = []
    for context in contexts:
        boost = pagerank.get(context.node_id, 0.0)
        boosted.append(
            WikiGraphRetrievedContext(
                **{
                    **context.model_dump(),
                    "score": context.score + boost,
                }
            )
        )
    boosted.sort(key=lambda item: item.score, reverse=True)
    return boosted


def build_chunk_maps(
    build_chunks: list[ParsedChunk],
    chunk_nodes: list[WikiGraphNode],
) -> tuple[dict[str, ParsedChunk], dict[str, WikiGraphNode], LexicalIndex | None]:
    chunk_records = {chunk.chunk_id: chunk for chunk in build_chunks}
    chunk_nodes_by_id = {node.id: node for node in chunk_nodes if node.kind == "chunk"}
    return chunk_records, chunk_nodes_by_id, None
