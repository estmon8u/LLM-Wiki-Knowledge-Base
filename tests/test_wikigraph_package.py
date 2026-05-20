"""Unit tests for the WikiGraphRAG package and high-level services."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
from graphwiki_kb.services.wikigraph_query_service import (
    WikiGraphQueryError,
    WikiGraphQueryService,
)
from graphwiki_kb.wikigraph.context_builder import (
    WikiGraphContextBuilder,
    merge_contexts,
)
from graphwiki_kb.wikigraph.entity_extractor import (
    build_entity_catalog,
    extract_page_claims,
)
from graphwiki_kb.wikigraph.graph_store import (
    WikiGraphStore,
    WikiGraphStorePaths,
    collect_neighbors,
    node_pagerank,
    to_node_link_json,
)
from graphwiki_kb.wikigraph.index_builder import (
    BuildOptions,
    build_wikigraph_index,
)
from graphwiki_kb.wikigraph.lexical_index import (
    LexicalDocument,
    LexicalIndex,
    tokenize,
)
from graphwiki_kb.wikigraph.markdown_parser import parse_wiki_page
from graphwiki_kb.wikigraph.models import (
    WikiGraphAnswer,
    WikiGraphIndex,
    WikiGraphNode,
    WikiGraphRetrievedContext,
)
from graphwiki_kb.wikigraph.query_service import WikiGraphQueryEngine

REALM_PAGE = textwrap.dedent(
    """\
---
title: REALM
type: source
source_id: realm
aliases:
  - Retrieval-Augmented Language Model
tags:
  - retrieval
  - pretraining
summary: REALM pretrains a language model alongside a learned retriever.
---

# REALM

## Summary

REALM is a retrieval-augmented language model.

## Key Points

- REALM jointly trains a retriever and a masked language model.
- REALM beats T5 on open-domain QA.

## Methods

REALM backpropagates through retrieval, which is expensive. See [[RAG]] for a related approach.
"""
)

RAG_PAGE = textwrap.dedent(
    """\
---
title: RAG
type: source
source_id: rag
aliases:
  - Retrieval-Augmented Generation
tags:
  - retrieval
  - generation
summary: RAG augments a seq2seq model with retrieved Wikipedia passages.
---

# RAG

## Summary

RAG augments a generator with retrieved passages from Wikipedia.

## Key Points

- RAG uses a frozen retriever plus a seq2seq generator.
- RAG matches REALM on open-domain QA without backprop through retrieval.

## Methods

RAG decouples retrieval and generation, making it cheaper than [[REALM]].
"""
)


@pytest.fixture
def populated_project(test_project):
    """Seed ``test_project`` with two minimal wiki source pages."""
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(
        REALM_PAGE, encoding="utf-8"
    )
    (test_project.paths.wiki_sources_dir / "rag.md").write_text(
        RAG_PAGE, encoding="utf-8"
    )
    return test_project


# --------------------------------------------------------------------------- #
# Markdown parsing                                                            #
# --------------------------------------------------------------------------- #


def test_parse_wiki_page_extracts_metadata_and_chunks(tmp_path: Path) -> None:
    path = tmp_path / "wiki" / "sources" / "realm.md"
    path.parent.mkdir(parents=True)
    path.write_text(REALM_PAGE, encoding="utf-8")

    page = parse_wiki_page(path, "wiki/sources/realm.md")

    assert page is not None
    assert page.title == "REALM"
    assert "Retrieval-Augmented Language Model" in page.aliases
    assert page.source_ids == ["realm"]
    assert page.page_type == "source"
    assert any("REALM" in chunk.body for chunk in page.chunks)
    assert any(link.target == "RAG" for link in page.wikilinks)


def test_parse_wiki_page_skips_maintenance(tmp_path: Path) -> None:
    path = tmp_path / "wiki" / "index.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Index", encoding="utf-8")
    assert parse_wiki_page(path, "wiki/index.md") is None


def test_parse_wiki_page_handles_unreadable(tmp_path: Path) -> None:
    page = parse_wiki_page(
        tmp_path / "does" / "not" / "exist.md", "wiki/sources/missing.md"
    )
    assert page is None


# --------------------------------------------------------------------------- #
# Entity / claim extraction                                                   #
# --------------------------------------------------------------------------- #


def test_extract_entities_and_claims(tmp_path: Path) -> None:
    realm_path = tmp_path / "wiki" / "sources" / "realm.md"
    rag_path = tmp_path / "wiki" / "sources" / "rag.md"
    realm_path.parent.mkdir(parents=True)
    realm_path.write_text(REALM_PAGE, encoding="utf-8")
    rag_path.write_text(RAG_PAGE, encoding="utf-8")

    realm_page = parse_wiki_page(realm_path, "wiki/sources/realm.md")
    rag_page = parse_wiki_page(rag_path, "wiki/sources/rag.md")
    assert realm_page is not None and rag_page is not None

    catalog = build_entity_catalog([realm_page, rag_page])
    names = {entry.name for entry in catalog.iter_entities()}
    assert "REALM" in names
    assert "RAG" in names

    claims = extract_page_claims(realm_page)
    texts = [claim.text.lower() for claim in claims]
    assert any("realm" in text for text in texts)


# --------------------------------------------------------------------------- #
# Lexical index                                                               #
# --------------------------------------------------------------------------- #


def test_lexical_index_returns_ranked_hits() -> None:
    index = LexicalIndex()
    index.add(
        LexicalDocument(
            doc_id="a",
            text="REALM jointly trains a retriever and a masked language model.",
        )
    )
    index.add(
        LexicalDocument(doc_id="b", text="RAG decouples retrieval and generation.")
    )
    index.fit()
    hits = index.search("retriever language model", limit=2)
    assert hits
    assert hits[0].doc_id == "a"


def test_lexical_index_handles_empty_query() -> None:
    index = LexicalIndex()
    index.add(LexicalDocument(doc_id="a", text="hello"))
    assert index.search("", limit=5) == []


def test_tokenize_drops_stopwords() -> None:
    assert "the" not in tokenize("the dog ran")
    assert "dog" in tokenize("the dog ran")


# --------------------------------------------------------------------------- #
# Graph store + builder                                                       #
# --------------------------------------------------------------------------- #


def test_build_wikigraph_index_creates_nodes_and_communities(populated_project) -> None:
    index = build_wikigraph_index(populated_project.paths, options=BuildOptions())
    assert index.source_count == 2
    assert index.chunk_count >= 4
    assert index.entity_count >= 2
    assert index.nodes
    assert index.edges
    assert any(node.kind == "community" for node in index.nodes)


def test_wikigraph_store_roundtrip(tmp_path: Path) -> None:
    store = WikiGraphStore(WikiGraphStorePaths(tmp_path / "wiki_store"))
    index = WikiGraphIndex(
        nodes=[
            WikiGraphNode(id="page::a", kind="source_page", title="A"),
            WikiGraphNode(id="chunk::a-0", kind="chunk", title="A1", text="hello"),
        ],
        edges=[],
        built_at="2026-01-01T00:00:00+00:00",
        source_count=1,
        chunk_count=1,
        entity_count=0,
    )
    written = store.save(index)
    assert written
    loaded = store.load()
    assert loaded is not None
    assert {n.id for n in loaded.nodes} == {"page::a", "chunk::a-0"}


def test_node_pagerank_and_neighbors() -> None:
    index = WikiGraphIndex(
        nodes=[
            WikiGraphNode(id="a", kind="entity", title="A"),
            WikiGraphNode(id="b", kind="entity", title="B"),
            WikiGraphNode(id="c", kind="entity", title="C"),
        ],
        edges=[],
        built_at="now",
    )
    from graphwiki_kb.wikigraph.graph_store import WikiGraphStore
    from graphwiki_kb.wikigraph.models import WikiGraphEdge

    index = index.model_copy(
        update={
            "edges": [
                WikiGraphEdge(source="a", target="b", kind="related_to", weight=1.0),
                WikiGraphEdge(source="b", target="c", kind="related_to", weight=1.0),
            ]
        }
    )
    graph = WikiGraphStore.to_networkx(index)
    ranks = node_pagerank(graph)
    assert pytest.approx(sum(ranks.values()), abs=1e-6) == 1.0
    neighbors = collect_neighbors(graph, "a", max_hops=2)
    assert {neighbor for neighbor, _, _ in neighbors} == {"b", "c"}
    assert "nodes" in to_node_link_json(index)


# --------------------------------------------------------------------------- #
# Query engine                                                                #
# --------------------------------------------------------------------------- #


def test_query_engine_local_and_basic(populated_project) -> None:
    index = build_wikigraph_index(populated_project.paths, options=BuildOptions())
    engine = WikiGraphQueryEngine(index=index)

    result = engine.find("How does REALM differ from RAG?", method="auto")
    assert result.method == "local"
    assert result.entities
    assert result.contexts

    basic = engine.find("retrieval generation", method="basic")
    assert basic.method == "basic"

    global_result = engine.find("overall corpus themes", method="global")
    assert global_result.method == "global"
    assert global_result.contexts or global_result.diagnostics

    drift = engine.find("How does REALM compare to RAG?", method="drift-lite")
    assert drift.method == "drift-lite"


def test_query_engine_rejects_unknown_method(populated_project) -> None:
    index = build_wikigraph_index(populated_project.paths)
    engine = WikiGraphQueryEngine(index=index)
    with pytest.raises(ValueError):
        engine.find("question", method="not-a-method")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Answer service (provider-free)                                              #
# --------------------------------------------------------------------------- #


def test_answer_service_provider_free(populated_project) -> None:
    index = build_wikigraph_index(populated_project.paths)
    engine = WikiGraphQueryEngine(index=index)
    from graphwiki_kb.wikigraph.answer_service import WikiGraphAnswerService

    service = WikiGraphAnswerService(engine=engine, provider=None)
    answer = service.ask("How does REALM differ from RAG?", method="local")
    assert isinstance(answer, WikiGraphAnswer)
    assert "Provider-free" in answer.answer
    assert answer.contexts


# --------------------------------------------------------------------------- #
# Query service + persistence                                                 #
# --------------------------------------------------------------------------- #


def test_query_service_requires_index(populated_project) -> None:
    service = WikiGraphQueryService(
        paths=populated_project.paths,
        index_service=WikiGraphIndexService(paths=populated_project.paths),
    )
    with pytest.raises(WikiGraphQueryError):
        service.find("anything")


def test_query_service_end_to_end(populated_project) -> None:
    index_service = WikiGraphIndexService(paths=populated_project.paths)
    report = index_service.build()
    assert report.node_count > 0
    snapshot = index_service.status()
    assert snapshot["initialized"] is True

    query_service = WikiGraphQueryService(
        paths=populated_project.paths,
        index_service=index_service,
    )
    answer = query_service.ask(
        "How does REALM differ from RAG?",
        method="auto",
        save=True,
        save_as="realm-vs-rag",
    )
    assert answer.saved_path
    saved = populated_project.paths.root / answer.saved_path
    assert saved.exists()
    assert "REALM" in saved.read_text(encoding="utf-8")


def test_query_service_save_rejects_empty_answer(populated_project) -> None:
    index_service = WikiGraphIndexService(paths=populated_project.paths)
    index_service.build()
    query_service = WikiGraphQueryService(
        paths=populated_project.paths,
        index_service=index_service,
    )
    empty = WikiGraphAnswer(
        method="basic", question="x", answer="   ", contexts=[], citations=[]
    )
    with pytest.raises(WikiGraphQueryError):
        query_service.save_answer("x", empty)


# --------------------------------------------------------------------------- #
# Context utilities                                                           #
# --------------------------------------------------------------------------- #


def test_merge_contexts_rrf() -> None:
    a = [
        WikiGraphRetrievedContext(
            node_id="a", node_kind="chunk", title="A", path="p", text="t", score=1.0
        ),
        WikiGraphRetrievedContext(
            node_id="b", node_kind="chunk", title="B", path="p", text="t", score=0.5
        ),
    ]
    b = [
        WikiGraphRetrievedContext(
            node_id="b", node_kind="chunk", title="B", path="p", text="t", score=2.0
        ),
        WikiGraphRetrievedContext(
            node_id="c", node_kind="chunk", title="C", path="p", text="t", score=0.1
        ),
    ]
    merged = merge_contexts(a, b, limit=3)
    ids = [ctx.node_id for ctx in merged]
    assert set(ids) == {"a", "b", "c"}
    assert ids[0] == "b"


def test_context_builder_handles_empty_index() -> None:
    empty = WikiGraphIndex(nodes=[], edges=[], built_at="now")
    builder = WikiGraphContextBuilder(empty)
    assert builder.basic_search("anything") == []
    local, seeds = builder.local_search("anything")
    assert local == [] and seeds == []
    glob, communities = builder.global_search("anything")
    assert glob == [] and communities == []
    drift, seeds, subs = builder.drift_lite("anything")
    assert drift == [] and seeds == [] and subs == []
