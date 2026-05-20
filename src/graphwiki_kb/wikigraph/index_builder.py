"""Build WikiGraphRAG index artifacts from wiki directories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from graphwiki_kb.services.project_service import ProjectPaths, slugify, utc_now_iso
from graphwiki_kb.wikigraph.community_builder import WikiCommunity, detect_communities
from graphwiki_kb.wikigraph.entity_extractor import (
    build_entity_nodes,
    co_mentioned_entities,
)
from graphwiki_kb.wikigraph.graph_store import WikiGraphStore
from graphwiki_kb.wikigraph.lexical_index import LexicalIndex
from graphwiki_kb.wikigraph.markdown_parser import (
    ParsedChunk,
    ParsedWikiPage,
    chunks_from_page,
    parse_wiki_page,
)
from graphwiki_kb.wikigraph.models import (
    WikiGraphEdge,
    WikiGraphIndexSnapshot,
    WikiGraphNode,
)


@dataclass(frozen=True)
class WikiGraphBuildResult:
    """Result of a WikiGraphRAG index build."""

    snapshot: WikiGraphIndexSnapshot
    nodes: list[WikiGraphNode]
    edges: list[WikiGraphEdge]
    chunks: list[ParsedChunk]
    communities: list[WikiCommunity]
    output_dir: Path


def wikigraph_output_dir(paths: ProjectPaths) -> Path:
    return paths.graph_dir / "wikigraph"


def wiki_generated_dir(paths: ProjectPaths) -> Path:
    return paths.wiki_dir / "wikigraph"


def collect_wiki_dirs(
    paths: ProjectPaths,
    *,
    include_graphrag_export_pages: bool,
) -> list[Path]:
    """Return wiki directories included in the default WikiGraphRAG scope."""
    dirs = [
        paths.wiki_sources_dir,
        paths.wiki_concepts_dir,
        paths.wiki_analysis_dir,
    ]
    if include_graphrag_export_pages:
        graph_export = paths.wiki_dir / "graph"
        if graph_export.exists():
            dirs.append(graph_export)
    return [directory for directory in dirs if directory.exists()]


def build_wikigraph_index(
    paths: ProjectPaths,
    *,
    include_graphrag_export_pages: bool,
    lexical_backend: str,
    community_algorithm: str,
) -> WikiGraphBuildResult:
    """Parse wiki artifacts and write graph/wikigraph index files."""
    pages = _load_pages(
        paths, include_graphrag_export_pages=include_graphrag_export_pages
    )
    if not pages:
        raise ValueError(
            "No wiki pages found for WikiGraphRAG indexing. Add sources and run "
            "`kb update` after compile."
        )
    chunks = [chunk for page in pages for chunk in chunks_from_page(page)]
    entity_nodes = build_entity_nodes(pages, chunks)
    page_nodes = [_page_node(page) for page in pages]
    chunk_nodes = [_chunk_node(chunk) for chunk in chunks]
    nodes = page_nodes + chunk_nodes + entity_nodes
    edges = _build_edges(pages, chunks, entity_nodes)
    store = WikiGraphStore()
    for node in nodes:
        store.add_node(node)
    for edge in edges:
        store.add_edge(edge)
    communities = detect_communities(store, nodes, algorithm=community_algorithm)
    community_nodes = [
        WikiGraphNode(
            id=community.community_id,
            kind="community",
            title=community.title,
            path=None,
            text=community.summary,
            metadata={
                "member_count": len(community.member_ids),
                "representative_chunks": list(community.representative_chunks),
            },
        )
        for community in communities
    ]
    for node in community_nodes:
        store.add_node(node)
    nodes.extend(community_nodes)
    for community in communities:
        for member_id in community.member_ids:
            edges.append(
                WikiGraphEdge(
                    source=member_id,
                    target=community.community_id,
                    kind="member_of",
                    weight=1.0,
                    evidence=[community.community_id],
                )
            )
    output_dir = wikigraph_output_dir(paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    WikiGraphStore.write_nodes(output_dir / "nodes.json", nodes)
    WikiGraphStore.write_edges(output_dir / "edges.json", edges)
    _write_chunks(output_dir / "chunks.json", chunks)
    _write_communities(output_dir / "communities.json", communities)
    store.save(output_dir)
    lexical = LexicalIndex(
        backend=lexical_backend,
        chunks=chunks,
        index_dir=output_dir / "lexical",
    )
    lexical.save()
    source_dirs = [
        directory.relative_to(paths.root).as_posix()
        for directory in collect_wiki_dirs(
            paths, include_graphrag_export_pages=include_graphrag_export_pages
        )
    ]
    snapshot = WikiGraphIndexSnapshot(
        built_at=utc_now_iso(),
        node_count=len(nodes),
        edge_count=len(edges),
        chunk_count=len(chunks),
        community_count=len(communities),
        include_graphrag_export_pages=include_graphrag_export_pages,
        lexical_backend=lexical.backend,
        community_algorithm=community_algorithm,
        source_dirs=source_dirs,
    )
    index_payload = snapshot.model_dump()
    (output_dir / "index.json").write_text(
        json.dumps(index_payload, indent=2),
        encoding="utf-8",
    )
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"build-{snapshot.built_at.replace(':', '').replace('+', '')}.json"
    (runs_dir / run_name).write_text(
        json.dumps(index_payload, indent=2),
        encoding="utf-8",
    )
    (runs_dir / "latest.json").write_text(
        json.dumps(index_payload, indent=2),
        encoding="utf-8",
    )
    return WikiGraphBuildResult(
        snapshot=snapshot,
        nodes=nodes,
        edges=edges,
        chunks=chunks,
        communities=communities,
        output_dir=output_dir,
    )


def load_built_index(paths: ProjectPaths) -> WikiGraphBuildResult | None:
    """Load a previously built WikiGraphRAG index from disk."""
    output_dir = wikigraph_output_dir(paths)
    index_file = output_dir / "index.json"
    if not index_file.exists():
        return None
    import json

    snapshot = WikiGraphIndexSnapshot.model_validate(
        json.loads(index_file.read_text(encoding="utf-8"))
    )
    nodes = WikiGraphStore.read_nodes(output_dir / "nodes.json")
    edges = WikiGraphStore.read_edges(output_dir / "edges.json")
    chunks = _read_chunks(output_dir / "chunks.json")
    communities = _read_communities(output_dir / "communities.json")
    return WikiGraphBuildResult(
        snapshot=snapshot,
        nodes=nodes,
        edges=edges,
        chunks=chunks,
        communities=communities,
        output_dir=output_dir,
    )


def _load_pages(
    paths: ProjectPaths,
    *,
    include_graphrag_export_pages: bool,
) -> list[ParsedWikiPage]:
    pages: list[ParsedWikiPage] = []
    for directory in collect_wiki_dirs(
        paths, include_graphrag_export_pages=include_graphrag_export_pages
    ):
        for file_path in sorted(directory.rglob("*.md")):
            if file_path.name.startswith("_"):
                continue
            if "/wikigraph/" in file_path.as_posix():
                continue
            parsed = parse_wiki_page(file_path, paths.root)
            if parsed is not None:
                pages.append(parsed)
    return pages


def _page_node(page: ParsedWikiPage) -> WikiGraphNode:
    return WikiGraphNode(
        id=f"page:{page.path}",
        kind=page.page_kind,
        title=page.title,
        path=page.path,
        text=page.summary or page.body[:1200],
        metadata={
            "source_id": page.source_id,
            "aliases": list(page.aliases),
            "tags": list(page.tags),
        },
    )


def _chunk_node(chunk: ParsedChunk) -> WikiGraphNode:
    return WikiGraphNode(
        id=f"chunk:{chunk.chunk_id}",
        kind="chunk",
        title=f"{chunk.title} — {chunk.heading}",
        path=chunk.page_path,
        text=chunk.text,
        metadata={"source_id": chunk.source_id, "chunk_id": chunk.chunk_id},
    )


def _build_edges(
    pages: list[ParsedWikiPage],
    chunks: list[ParsedChunk],
    entities: list[WikiGraphNode],
) -> list[WikiGraphEdge]:
    edges: list[WikiGraphEdge] = []
    title_to_entity = {entity.title.lower(): entity.id for entity in entities}
    co_mentions = co_mentioned_entities(chunks)
    for page in pages:
        page_id = f"page:{page.path}"
        for chunk in chunks_from_page(page):
            chunk_id = f"chunk:{chunk.chunk_id}"
            edges.append(
                WikiGraphEdge(
                    source=page_id,
                    target=chunk_id,
                    kind="contains",
                    weight=1.0,
                    evidence=[page.path],
                )
            )
        for link in page.wikilinks:
            target = f"page:wiki/{_resolve_wikilink_path(link)}"
            edges.append(
                WikiGraphEdge(
                    source=page_id,
                    target=target,
                    kind="links_to",
                    weight=1.0,
                    evidence=[link],
                )
            )
        for alias in page.aliases:
            entity_id = title_to_entity.get(alias.lower())
            if entity_id:
                edges.append(
                    WikiGraphEdge(
                        source=page_id,
                        target=entity_id,
                        kind="mentions",
                        weight=1.0,
                        evidence=[alias],
                    )
                )
    for left, right_set in co_mentions.items():
        left_id = title_to_entity.get(left.lower())
        if not left_id:
            continue
        for right in right_set:
            right_id = title_to_entity.get(right.lower())
            if right_id and left_id != right_id:
                edges.append(
                    WikiGraphEdge(
                        source=left_id,
                        target=right_id,
                        kind="related_to",
                        weight=0.6,
                        evidence=[left, right],
                    )
                )
    return edges


def _resolve_wikilink_path(target: str) -> str:
    cleaned = target.strip().strip("/")
    if cleaned.startswith("wiki/"):
        return f"{cleaned}.md" if not cleaned.endswith(".md") else cleaned
    slug = slugify(cleaned)
    for prefix in ("sources", "concepts", "analysis"):
        candidate = f"wiki/{prefix}/{slug}.md"
        return candidate
    return f"wiki/sources/{slug}.md"


def _write_chunks(path: Path, chunks: list[ParsedChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "chunk_id": chunk.chunk_id,
            "page_path": chunk.page_path,
            "page_kind": chunk.page_kind,
            "title": chunk.title,
            "heading": chunk.heading,
            "text": chunk.text,
            "source_id": chunk.source_id,
            "aliases": list(chunk.aliases),
        }
        for chunk in chunks
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_chunks(path: Path) -> list[ParsedChunk]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        ParsedChunk(
            chunk_id=str(item["chunk_id"]),
            page_path=str(item["page_path"]),
            page_kind=str(item["page_kind"]),
            title=str(item["title"]),
            heading=str(item["heading"]),
            text=str(item["text"]),
            source_id=item.get("source_id"),
            aliases=tuple(str(alias) for alias in item.get("aliases", [])),
        )
        for item in payload
    ]


def _write_communities(path: Path, communities: list[WikiCommunity]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "community_id": community.community_id,
            "title": community.title,
            "summary": community.summary,
            "member_ids": list(community.member_ids),
            "representative_chunks": list(community.representative_chunks),
        }
        for community in communities
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_communities(path: Path) -> list[WikiCommunity]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        WikiCommunity(
            community_id=str(item["community_id"]),
            title=str(item["title"]),
            summary=str(item["summary"]),
            member_ids=tuple(str(node_id) for node_id in item.get("member_ids", [])),
            representative_chunks=tuple(
                str(chunk_id) for chunk_id in item.get("representative_chunks", [])
            ),
        )
        for item in payload
    ]
