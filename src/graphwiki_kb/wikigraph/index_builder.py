"""Build a wiki graph index from the maintained wiki artifacts.

The builder walks ``wiki/sources``, ``wiki/concepts``, and ``wiki/analysis``
(optionally including ``wiki/graph`` for an ablation) and produces:

* Nodes for pages, chunks, entities, claims, and communities.
* Edges expressing ``contains``, ``mentions``, ``links_to``, ``supports``,
  ``co_mentions``, ``related_to``, and ``member_of`` relationships.
* A community partition produced by NetworkX Louvain.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from graphwiki_kb.services.project_service import (
    ProjectPaths,
    slugify,
    utc_now_iso,
)
from graphwiki_kb.wikigraph.community_builder import (
    build_community_records,
    detect_communities,
)
from graphwiki_kb.wikigraph.entity_extractor import (
    EntityCatalog,
    ExtractedClaim,
    build_entity_catalog,
    extract_page_claims,
)
from graphwiki_kb.wikigraph.graph_store import WikiGraphStore
from graphwiki_kb.wikigraph.markdown_parser import WikiPage, parse_wiki_page
from graphwiki_kb.wikigraph.models import (
    WikiGraphEdge,
    WikiGraphIndex,
    WikiGraphNode,
)


@dataclass(frozen=True)
class BuildOptions:
    """Tunable knobs for :func:`build_wikigraph_index`."""

    chunk_char_limit: int = 1200
    include_graphrag_export_pages: bool = False
    fuzzy_entity_match_threshold: int = 88
    min_community_size: int = 1


_DEFAULT_INCLUDE_DIRS: tuple[str, ...] = (
    "wiki/sources",
    "wiki/concepts",
    "wiki/analysis",
)
_GRAPHRAG_PAGES_DIR = "wiki/graph"


# --------------------------------------------------------------------------- #
# ID helpers                                                                  #
# --------------------------------------------------------------------------- #


def page_node_id(page: WikiPage) -> str:
    """Stable node id for a wiki page."""
    return f"page::{page.relative_path}"


def chunk_node_id(page: WikiPage, chunk_index: int) -> str:
    """Stable node id for a section-level chunk."""
    return f"chunk::{page.relative_path}#chunk-{chunk_index}"


def entity_node_id(name: str) -> str:
    """Stable node id for an entity, using a slugified name."""
    return f"entity::{slugify(name)}"


def claim_node_id(claim: ExtractedClaim, ordinal: int) -> str:
    """Stable node id for a claim."""
    base = slugify(claim.text[:64]) or "claim"
    return f"claim::{slugify(claim.page_path)}#{ordinal}-{base[:32]}"


# --------------------------------------------------------------------------- #
# File discovery                                                              #
# --------------------------------------------------------------------------- #


def _iter_wiki_pages(
    paths: ProjectPaths,
    *,
    include_graphrag_export_pages: bool,
    chunk_char_limit: int,
) -> list[WikiPage]:
    pages: list[WikiPage] = []
    include_dirs = list(_DEFAULT_INCLUDE_DIRS)
    if include_graphrag_export_pages:
        include_dirs.append(_GRAPHRAG_PAGES_DIR)
    for include_dir in include_dirs:
        target = paths.root / include_dir
        if not target.exists():
            continue
        for file_path in sorted(target.rglob("*.md")):
            relative_path = file_path.relative_to(paths.root).as_posix()
            page = parse_wiki_page(
                file_path,
                relative_path,
                chunk_char_limit=chunk_char_limit,
            )
            if page is None:
                continue
            pages.append(page)
    return pages


# --------------------------------------------------------------------------- #
# Node and edge construction                                                  #
# --------------------------------------------------------------------------- #


def _build_page_node(page: WikiPage) -> WikiGraphNode:
    kind_map = {
        "source": "source_page",
        "concept": "concept_page",
        "analysis": "analysis_page",
        "graph": "graph_page",
    }
    node_kind = kind_map.get(page.page_type, "source_page")
    metadata: dict[str, object] = {"page_type": page.page_type}
    if page.frontmatter.get("summary"):
        metadata["summary"] = str(page.frontmatter["summary"])
    return WikiGraphNode(
        id=page_node_id(page),
        kind=node_kind,
        title=page.title,
        path=page.relative_path,
        text=str(page.frontmatter.get("summary", ""))[:600],
        aliases=list(page.aliases),
        tags=list(page.tags),
        source_ids=list(page.source_ids),
        metadata=metadata,
    )


def _build_chunk_nodes_and_edges(
    page: WikiPage,
    page_node: WikiGraphNode,
) -> tuple[list[WikiGraphNode], list[WikiGraphEdge]]:
    nodes: list[WikiGraphNode] = []
    edges: list[WikiGraphEdge] = []
    for chunk in page.chunks:
        node_id = chunk_node_id(page, chunk.chunk_index)
        nodes.append(
            WikiGraphNode(
                id=node_id,
                kind="chunk",
                title=chunk.section or page.title,
                path=page.relative_path,
                text=chunk.body,
                source_ids=list(page.source_ids),
                metadata={
                    "chunk_index": chunk.chunk_index,
                    "section": chunk.section,
                    "page_id": page_node.id,
                },
            )
        )
        edges.append(
            WikiGraphEdge(
                source=page_node.id,
                target=node_id,
                kind="contains",
                weight=1.0,
                evidence=[page.relative_path],
            )
        )
    return nodes, edges


def _build_entity_nodes(
    catalog: EntityCatalog,
) -> dict[str, WikiGraphNode]:
    by_id: dict[str, WikiGraphNode] = {}
    for entity in catalog.iter_entities():
        node_id = entity_node_id(entity.name)
        if node_id in by_id:
            existing = by_id[node_id]
            merged_sources = list(
                dict.fromkeys([*existing.source_ids, *entity.source_ids])
            )
            merged_aliases = list(dict.fromkeys([*existing.aliases, *entity.aliases]))
            by_id[node_id] = existing.model_copy(
                update={
                    "source_ids": merged_sources,
                    "aliases": merged_aliases,
                }
            )
            continue
        by_id[node_id] = WikiGraphNode(
            id=node_id,
            kind="entity",
            title=entity.name,
            path=entity.page_path,
            text=f"Entity surface form: {entity.name}",
            aliases=list(entity.aliases),
            source_ids=list(entity.source_ids),
            metadata={
                "first_seen_page": entity.page_path,
                "first_seen_title": entity.page_title,
                "occurrences": entity.occurrences,
            },
        )
    return by_id


def _build_mention_edges(
    pages: list[WikiPage],
    catalog: EntityCatalog,
    entity_nodes: dict[str, WikiGraphNode],
    *,
    fuzzy_threshold: int,
) -> list[WikiGraphEdge]:
    edges: list[WikiGraphEdge] = []
    entity_names = [entity.name for entity in catalog.iter_entities()]
    for page in pages:
        page_id = page_node_id(page)
        page_text = page.body.lower()
        for entity_name in entity_names:
            normalized = entity_name.lower()
            mentioned = False
            occurrences = 0
            if len(normalized) >= 4 and normalized in page_text:
                mentioned = True
                occurrences = page_text.count(normalized)
            else:
                ratio = fuzz.token_set_ratio(entity_name, page.title)
                if ratio >= fuzzy_threshold:
                    mentioned = True
                    occurrences = 1
            if not mentioned:
                continue
            entity_id = entity_node_id(entity_name)
            if entity_id not in entity_nodes:
                continue
            if entity_nodes[entity_id].path == page.relative_path:
                continue
            edges.append(
                WikiGraphEdge(
                    source=page_id,
                    target=entity_id,
                    kind="mentions",
                    weight=float(min(occurrences, 5)),
                    evidence=[page.relative_path],
                )
            )
    return edges


def _build_wikilink_edges(
    pages: list[WikiPage],
    pages_by_title: dict[str, WikiPage],
) -> list[WikiGraphEdge]:
    edges: list[WikiGraphEdge] = []
    for page in pages:
        for link in page.wikilinks:
            target_page = pages_by_title.get(link.target.casefold())
            if target_page is None or target_page.relative_path == page.relative_path:
                continue
            edges.append(
                WikiGraphEdge(
                    source=page_node_id(page),
                    target=page_node_id(target_page),
                    kind="links_to",
                    weight=1.0,
                    evidence=[page.relative_path],
                )
            )
    return edges


def _build_claim_nodes_and_edges(
    pages: list[WikiPage],
) -> tuple[list[WikiGraphNode], list[WikiGraphEdge]]:
    nodes: list[WikiGraphNode] = []
    edges: list[WikiGraphEdge] = []
    for page in pages:
        page_id = page_node_id(page)
        claims = extract_page_claims(page)
        for ordinal, claim in enumerate(claims):
            node_id = claim_node_id(claim, ordinal)
            nodes.append(
                WikiGraphNode(
                    id=node_id,
                    kind="claim",
                    title=claim.text[:80],
                    path=claim.page_path,
                    text=claim.text,
                    source_ids=list(claim.source_ids),
                    metadata={
                        "section": claim.section,
                        "chunk_index": claim.chunk_index,
                        "page_id": page_id,
                    },
                )
            )
            edges.append(
                WikiGraphEdge(
                    source=page_id,
                    target=node_id,
                    kind="supports",
                    weight=1.0,
                    evidence=[page.relative_path],
                )
            )
            if claim.chunk_index is not None:
                chunk_id = chunk_node_id(page, claim.chunk_index)
                edges.append(
                    WikiGraphEdge(
                        source=chunk_id,
                        target=node_id,
                        kind="supports",
                        weight=0.5,
                        evidence=[page.relative_path],
                    )
                )
    return nodes, edges


def _build_co_mention_edges(
    pages: list[WikiPage],
    catalog: EntityCatalog,
    entity_nodes: dict[str, WikiGraphNode],
) -> list[WikiGraphEdge]:
    edges: list[WikiGraphEdge] = []
    for page in pages:
        page_text_lower = page.body.lower()
        present_entities: list[str] = []
        for entity in catalog.iter_entities():
            normalized = entity.name.lower()
            if len(normalized) >= 4 and normalized in page_text_lower:
                present_entities.append(entity.name)
        for index_a, name_a in enumerate(present_entities):
            for name_b in present_entities[index_a + 1 :]:
                edges.append(
                    WikiGraphEdge(
                        source=entity_node_id(name_a),
                        target=entity_node_id(name_b),
                        kind="co_mentions",
                        weight=0.5,
                        evidence=[page.relative_path],
                    )
                )
    return _dedupe_undirected(edges)


def _dedupe_undirected(edges: list[WikiGraphEdge]) -> list[WikiGraphEdge]:
    merged: dict[tuple[str, str, str], WikiGraphEdge] = {}
    for edge in edges:
        if edge.source <= edge.target:
            key = (edge.source, edge.target, edge.kind)
        else:
            key = (edge.target, edge.source, edge.kind)
        existing = merged.get(key)
        if existing is None:
            merged[key] = edge
            continue
        new_weight = existing.weight + edge.weight
        new_evidence = list(dict.fromkeys([*existing.evidence, *edge.evidence]))
        merged[key] = existing.model_copy(
            update={"weight": new_weight, "evidence": new_evidence}
        )
    return list(merged.values())


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def build_wikigraph_index(
    paths: ProjectPaths,
    *,
    options: BuildOptions | None = None,
) -> WikiGraphIndex:
    """Build a :class:`WikiGraphIndex` from the maintained wiki artifacts."""
    opts = options or BuildOptions()
    pages = _iter_wiki_pages(
        paths,
        include_graphrag_export_pages=opts.include_graphrag_export_pages,
        chunk_char_limit=opts.chunk_char_limit,
    )

    page_nodes: list[WikiGraphNode] = []
    chunk_nodes: list[WikiGraphNode] = []
    chunk_edges: list[WikiGraphEdge] = []
    for page in pages:
        page_node = _build_page_node(page)
        page_nodes.append(page_node)
        nodes, edges = _build_chunk_nodes_and_edges(page, page_node)
        chunk_nodes.extend(nodes)
        chunk_edges.extend(edges)

    pages_by_title = {page.title.casefold(): page for page in pages}
    for page in pages:
        for alias in page.aliases:
            pages_by_title.setdefault(alias.casefold(), page)

    catalog = build_entity_catalog(pages)
    entity_nodes = _build_entity_nodes(catalog)
    mention_edges = _build_mention_edges(
        pages,
        catalog,
        entity_nodes,
        fuzzy_threshold=opts.fuzzy_entity_match_threshold,
    )
    wikilink_edges = _build_wikilink_edges(pages, pages_by_title)
    claim_nodes, claim_edges = _build_claim_nodes_and_edges(pages)
    co_mention_edges = _build_co_mention_edges(pages, catalog, entity_nodes)

    all_nodes: list[WikiGraphNode] = [
        *page_nodes,
        *chunk_nodes,
        *entity_nodes.values(),
        *claim_nodes,
    ]
    all_edges: list[WikiGraphEdge] = [
        *chunk_edges,
        *mention_edges,
        *wikilink_edges,
        *claim_edges,
        *co_mention_edges,
    ]

    nodes_by_id = {node.id: node for node in all_nodes}
    networkx_graph = WikiGraphStore.to_networkx(
        WikiGraphIndex(nodes=all_nodes, edges=all_edges)
    )
    detection = detect_communities(networkx_graph)
    communities = build_community_records(
        detection,
        nodes_by_id=nodes_by_id,
        min_size=opts.min_community_size,
    )

    member_edges: list[WikiGraphEdge] = []
    community_nodes: list[WikiGraphNode] = []
    for community in communities:
        community_node = WikiGraphNode(
            id=community.id,
            kind="community",
            title=community.title,
            path=None,
            text=community.summary,
            source_ids=list(community.source_ids),
            metadata={
                "level": community.level,
                "member_count": len(community.members),
                "top_entities": list(community.top_entities),
            },
        )
        community_nodes.append(community_node)
        for member in community.members:
            member_edges.append(
                WikiGraphEdge(
                    source=member,
                    target=community.id,
                    kind="member_of",
                    weight=0.25,
                    evidence=[community.id],
                )
            )

    all_nodes.extend(community_nodes)
    all_edges.extend(member_edges)

    return WikiGraphIndex(
        nodes=all_nodes,
        edges=all_edges,
        communities=communities,
        built_at=utc_now_iso(),
        include_graphrag_export_pages=opts.include_graphrag_export_pages,
        source_count=sum(1 for node in page_nodes if node.kind == "source_page"),
        chunk_count=sum(1 for node in chunk_nodes if node.kind == "chunk"),
        entity_count=len(entity_nodes),
    )


def iter_wiki_pages(
    paths: ProjectPaths,
    *,
    include_graphrag_export_pages: bool = False,
    chunk_char_limit: int = 1200,
) -> Iterable[WikiPage]:
    """Convenience iterator over parsed wiki pages (for tests and tools)."""
    return _iter_wiki_pages(
        paths,
        include_graphrag_export_pages=include_graphrag_export_pages,
        chunk_char_limit=chunk_char_limit,
    )


def wiki_paths_under_root(root: Path) -> list[Path]:
    """Return the canonical wiki directories under ``root`` for diagnostics."""
    return [root / directory for directory in _DEFAULT_INCLUDE_DIRS]
