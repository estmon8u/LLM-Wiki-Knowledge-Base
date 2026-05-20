"""Internal-coverage tests for the WikiGraphRAG package."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from graphwiki_kb.providers import (
    ProviderConfigurationError,
    UnavailableProvider,
)
from graphwiki_kb.providers.base import ProviderRequest, ProviderResponse, TextProvider
from graphwiki_kb.wikigraph import lexical_index as lexical_module
from graphwiki_kb.wikigraph.answer_service import WikiGraphAnswerService
from graphwiki_kb.wikigraph.community_builder import (
    build_community_records,
    detect_communities,
)
from graphwiki_kb.wikigraph.context_builder import (
    ContextBuilderConfig,
    WikiGraphContextBuilder,
)
from graphwiki_kb.wikigraph.entity_extractor import (
    EntityCatalog,
    ExtractedClaim,
    ExtractedEntity,
)
from graphwiki_kb.wikigraph.graph_store import (
    WikiGraphStore,
    WikiGraphStorePaths,
    node_pagerank,
)
from graphwiki_kb.wikigraph.index_builder import (
    BuildOptions,
    build_wikigraph_index,
    chunk_node_id,
    iter_wiki_pages,
    page_node_id,
    wiki_paths_under_root,
)
from graphwiki_kb.wikigraph.lexical_index import (
    LexicalDocument,
    LexicalIndex,
    tokenize,
)
from graphwiki_kb.wikigraph.markdown_parser import (
    _coerce_string_list,
    _source_ids_from_frontmatter,
    page_type_from_path,
    parse_wiki_page,
)
from graphwiki_kb.wikigraph.models import (
    WikiGraphAnswer,
    WikiGraphEdge,
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
summary: REALM pretrains a retriever and a language model.
---

# REALM

## Summary

REALM is a retrieval-augmented language model.

## Key Points

- REALM jointly trains a retriever and a masked language model.

## Methods

REALM backpropagates through retrieval. See [[RAG]].
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
summary: RAG augments a generator with retrieved passages.
---

# RAG

## Summary

RAG combines a frozen retriever and a seq2seq generator.

## Key Points

- RAG matches REALM on open-domain QA.

## Methods

RAG decouples retrieval and generation. See [[REALM]].
"""
)


# --------------------------------------------------------------------------- #
# Markdown parser helpers                                                     #
# --------------------------------------------------------------------------- #


def test_page_type_from_path_variants() -> None:
    assert page_type_from_path("wiki/sources/x.md") == "source"
    assert page_type_from_path("wiki/concepts/x.md") == "concept"
    assert page_type_from_path("wiki/analysis/x.md") == "analysis"
    assert page_type_from_path("wiki/graph/x.md") == "graph"
    assert page_type_from_path("wiki/wikigraph/entities/x.md") == "wikigraph_generated"
    assert page_type_from_path("other/x.md") == ""


def test_coerce_string_list_variants() -> None:
    assert _coerce_string_list(None) == []
    assert _coerce_string_list("hello ") == ["hello"]
    assert _coerce_string_list("") == []
    assert _coerce_string_list(["a", "", " b "]) == ["a", "b"]
    assert _coerce_string_list({"a", "b"}) == sorted({"a", "b"}) or sorted(
        _coerce_string_list({"a", "b"})
    ) == ["a", "b"]
    assert _coerce_string_list(42) == ["42"]


def test_source_ids_dedup_case_insensitive() -> None:
    out = _source_ids_from_frontmatter(
        {"source_id": "Realm", "source_ids": ["realm", "rag"]}
    )
    assert "Realm" in out
    assert "rag" in out
    assert len(out) == 2


def test_parse_wiki_page_falls_back_to_filename_title(tmp_path: Path) -> None:
    path = tmp_path / "wiki" / "sources" / "no-fm.md"
    path.parent.mkdir(parents=True)
    path.write_text("Just a single line of body text.\n", encoding="utf-8")
    page = parse_wiki_page(path, "wiki/sources/no-fm.md")
    assert page is not None
    assert page.title == "No Fm"


def test_parse_wiki_page_empty_aliases(tmp_path: Path) -> None:
    path = tmp_path / "wiki" / "sources" / "x.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\ntitle: X\ntype: source\nsource_id: x\n---\n\n# X\n\nBody.\n",
        encoding="utf-8",
    )
    page = parse_wiki_page(path, "wiki/sources/x.md")
    assert page is not None and page.aliases == []


# --------------------------------------------------------------------------- #
# Entity catalog merging                                                      #
# --------------------------------------------------------------------------- #


def test_entity_catalog_merge_aliases_and_sources() -> None:
    catalog = EntityCatalog()
    catalog.add(
        ExtractedEntity(
            name="REALM",
            aliases=("Retrieval-Augmented Language Model",),
            page_path="wiki/sources/realm.md",
            page_title="REALM",
            occurrences=3,
            source_ids=("realm",),
        )
    )
    catalog.add(
        ExtractedEntity(
            name="REALM",
            aliases=("REALM Model",),
            page_path="wiki/sources/realm.md",
            page_title="REALM",
            occurrences=2,
            source_ids=("realm_v2",),
        )
    )
    merged = catalog.find("realm")
    assert merged is not None
    assert merged.occurrences == 5
    assert "realm_v2" in merged.source_ids
    assert catalog.find("REALM Model") is not None


# --------------------------------------------------------------------------- #
# Lexical pure-python fallback                                                #
# --------------------------------------------------------------------------- #


def test_lexical_index_pure_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lexical_module, "_BM25S_AVAILABLE", False)
    monkeypatch.setattr(lexical_module, "bm25s", None)
    index = LexicalIndex()
    index.add(
        LexicalDocument(
            doc_id="a", text="REALM trains a retriever and a language model."
        )
    )
    index.add(
        LexicalDocument(doc_id="b", text="RAG decouples retrieval and generation.")
    )
    index.add(LexicalDocument(doc_id="c", text="Unrelated text about cats."))
    index.fit()
    hits = index.search("retriever language", limit=2)
    assert hits
    assert hits[0].doc_id in {"a", "b"}
    assert index.backend == "pure-python-bm25"


def test_lexical_index_double_fit_no_op() -> None:
    index = LexicalIndex()
    index.add(LexicalDocument(doc_id="a", text="hello world"))
    index.fit()
    index.fit()
    with pytest.raises(RuntimeError):
        index.add(LexicalDocument(doc_id="b", text="x"))


def test_tokenize_handles_punctuation() -> None:
    tokens = tokenize("Hello, world! Hello again.")
    assert "hello" in tokens
    assert "world" in tokens


# --------------------------------------------------------------------------- #
# Community detection edge cases                                              #
# --------------------------------------------------------------------------- #


def test_detect_communities_on_empty_graph() -> None:
    import networkx as nx

    result = detect_communities(nx.MultiGraph())
    assert result.algorithm == "empty"
    assert result.member_lists == []


def test_build_community_records_skips_too_small() -> None:
    from graphwiki_kb.wikigraph.community_builder import CommunityDetectionResult

    detection = CommunityDetectionResult(
        algorithm="connected_components",
        member_lists=[["a"], ["b", "c"]],
    )
    records = build_community_records(detection, nodes_by_id={}, min_size=2)
    assert len(records) == 1
    assert "Community" in records[0].title


# --------------------------------------------------------------------------- #
# Index builder helpers                                                       #
# --------------------------------------------------------------------------- #


def test_index_builder_helpers(tmp_path: Path, test_project) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    pages = list(iter_wiki_pages(test_project.paths))
    assert pages and pages[0].title == "REALM"
    assert page_node_id(pages[0]).startswith("page::wiki/sources/")
    assert "#chunk-0" in chunk_node_id(pages[0], 0)
    paths_under = wiki_paths_under_root(test_project.paths.root)
    assert paths_under
    assert any("sources" in str(p) for p in paths_under)


def test_build_with_graphrag_export_pages(tmp_path: Path, test_project) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    graph_dir = test_project.paths.wiki_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "communities.md").write_text(
        "---\ntitle: Community A\ntype: graph\n---\n\nGraph community body.\n"
    )
    index = build_wikigraph_index(
        test_project.paths,
        options=BuildOptions(include_graphrag_export_pages=True),
    )
    assert any(node.kind == "graph_page" for node in index.nodes)


# --------------------------------------------------------------------------- #
# Answer service: provider-backed paths                                       #
# --------------------------------------------------------------------------- #


class _StaticProvider(TextProvider):
    name = "stub-static"

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(text=self._payload, model_name="stub-static")


class _ExplodingProvider(TextProvider):
    name = "stub-explode"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        raise RuntimeError("simulated provider failure")


def _build_engine(test_project) -> WikiGraphQueryEngine:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    (test_project.paths.wiki_sources_dir / "rag.md").write_text(RAG_PAGE)
    index = build_wikigraph_index(test_project.paths)
    return WikiGraphQueryEngine(index=index)


def test_answer_service_provider_backed_success(test_project) -> None:
    engine = _build_engine(test_project)
    find = engine.find("REALM and RAG", method="basic")
    refs = [ctx.citation_ref for ctx in find.contexts[:2]]
    payload = (
        '{"answer_markdown": "REALM and RAG both use retrieval.",'
        ' "claims": [{"text": "REALM uses retrieval.", "citation_refs": '
        + f'["{refs[0]}"]}}],'
        + ' "citations": [{"ref": "'
        + refs[0]
        + '", "title": "Summary"}], "insufficient_evidence": false}'
    )
    service = WikiGraphAnswerService(engine=engine, provider=_StaticProvider(payload))
    answer = service.ask("REALM and RAG", method="basic")
    assert isinstance(answer, WikiGraphAnswer)
    assert "REALM" in answer.answer
    assert not answer.insufficient_evidence
    assert answer.provider_status.get("mode") == "provider"


def test_answer_service_provider_accepts_same_path_unit(test_project) -> None:
    """Citation pointing to a different TextUnit of a *retrieved* doc.

    The LLM sometimes cites neighbor TextUnits of the document we
    actually retrieved (e.g. ``...md#text-unit-3`` when we returned
    ``...md#text-unit-7``). Because the retrieved context already
    contains the body text the model is grounding on, the answer
    service should accept the cite by normalizing to the canonical
    retrieved ref instead of marking the whole answer insufficient.
    """
    engine = _build_engine(test_project)
    find = engine.find("REALM and RAG", method="basic")
    assert find.contexts, "fixture should retrieve at least one context"
    canonical = find.contexts[0].citation_ref
    path_only = canonical.split("#", 1)[0]
    neighbor = f"{path_only}#text-unit-9999"

    payload = (
        '{"answer_markdown": "REALM uses retrieval.",'
        ' "claims": [{"text": "REALM uses retrieval.",'
        f' "citation_refs": ["{neighbor}"]}}],'
        f' "citations": [{{"ref": "{neighbor}", "title": "Summary"}}],'
        ' "insufficient_evidence": false}'
    )
    service = WikiGraphAnswerService(engine=engine, provider=_StaticProvider(payload))
    answer = service.ask("REALM and RAG", method="basic")

    assert not answer.insufficient_evidence
    assert answer.citations, "neighbor-cite should be normalized, not dropped"
    assert answer.citations[0]["ref"] == canonical


def test_answer_service_provider_invalid_payload(test_project) -> None:
    engine = _build_engine(test_project)
    service = WikiGraphAnswerService(
        engine=engine, provider=_StaticProvider("not json")
    )
    answer = service.ask("REALM and RAG", method="basic")
    assert "provider-parse-error" in answer.warnings


def test_answer_service_provider_exception(test_project) -> None:
    engine = _build_engine(test_project)
    service = WikiGraphAnswerService(engine=engine, provider=_ExplodingProvider())
    answer = service.ask("REALM and RAG", method="basic")
    assert "provider-error" in answer.warnings
    assert answer.insufficient_evidence is True


def test_answer_service_require_provider_without_one(test_project) -> None:
    engine = _build_engine(test_project)
    service = WikiGraphAnswerService(engine=engine, provider=None)
    with pytest.raises(ProviderConfigurationError):
        service.ask("REALM", method="basic", require_provider=True)


def test_answer_service_unavailable_provider_falls_back(test_project) -> None:
    engine = _build_engine(test_project)
    service = WikiGraphAnswerService(
        engine=engine,
        provider=UnavailableProvider("no key", provider_name="openai"),
    )
    answer = service.ask("REALM", method="basic")
    assert "provider-free" in answer.warnings


def test_answer_service_no_contexts_returns_warning(test_project) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    empty_index = WikiGraphIndex(
        nodes=[
            WikiGraphNode(id="page::a", kind="source_page", title="A"),
        ],
        edges=[],
        built_at="now",
    )
    engine = WikiGraphQueryEngine(index=empty_index)
    service = WikiGraphAnswerService(engine=engine, provider=None)
    answer = service.ask("anything", method="basic")
    assert "no_context" in answer.warnings


# --------------------------------------------------------------------------- #
# Context builder corner cases                                                #
# --------------------------------------------------------------------------- #


def test_context_builder_uses_communities_for_global(test_project) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    (test_project.paths.wiki_sources_dir / "rag.md").write_text(RAG_PAGE)
    index = build_wikigraph_index(test_project.paths)
    builder = WikiGraphContextBuilder(
        index, config=ContextBuilderConfig(max_context_chunks=4, max_hops=2)
    )
    contexts, community_ids = builder.global_search("retrieval methods")
    assert contexts
    assert community_ids


def test_graph_store_load_returns_none_when_missing(tmp_path: Path) -> None:
    store = WikiGraphStore(WikiGraphStorePaths(tmp_path / "missing"))
    assert store.load() is None


def test_node_pagerank_handles_empty() -> None:
    import networkx as nx

    assert node_pagerank(nx.MultiGraph()) == {}


def test_context_builder_drift_lite_returns_subquestions(test_project) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    (test_project.paths.wiki_sources_dir / "rag.md").write_text(RAG_PAGE)
    index = build_wikigraph_index(test_project.paths)
    builder = WikiGraphContextBuilder(index)
    contexts, seeds, subs = builder.drift_lite("How does REALM differ from RAG?")
    assert contexts
    assert seeds  # Entities matched
    assert subs  # At least one sub-question generated


def test_models_default_warnings_lists() -> None:
    answer = WikiGraphAnswer(method="basic", question="x", answer="y")
    assert answer.warnings == []
    assert answer.contexts == []


def test_retrieved_context_citation_ref_without_chunk_index() -> None:
    ctx = WikiGraphRetrievedContext(
        node_id="x", node_kind="entity", title="X", path=None, text="t", score=0.1
    )
    assert ctx.citation_ref == "x"


def test_chunk_citation_ref_with_index() -> None:
    ctx = WikiGraphRetrievedContext(
        node_id="x",
        node_kind="chunk",
        title="X",
        path="wiki/sources/a.md",
        text="t",
        score=0.1,
        chunk_index=2,
    )
    assert ctx.citation_ref == "wiki/sources/a.md#chunk-2"


def test_extracted_claim_dataclass() -> None:
    claim = ExtractedClaim(
        text="some claim",
        page_path="wiki/sources/a.md",
        page_title="A",
        section="Key Points",
        chunk_index=0,
        source_ids=("a",),
    )
    assert claim.section == "Key Points"


def test_entity_node_dedup_when_slug_collides(test_project) -> None:
    # Two entity names that slugify to the same id should be merged.
    page_a = test_project.paths.wiki_sources_dir / "a.md"
    page_b = test_project.paths.wiki_sources_dir / "b.md"
    page_a.write_text(
        "---\ntitle: REALM\ntype: source\nsource_id: a\n---\n\n# REALM\n\n"
        "## Key Points\n\n- The REALM paper studies retrieval.\n",
        encoding="utf-8",
    )
    page_b.write_text(
        "---\ntitle: realm\ntype: source\nsource_id: b\n---\n\n# realm\n\n"
        "## Key Points\n\n- The realm paper studies retrieval again.\n",
        encoding="utf-8",
    )
    index = build_wikigraph_index(test_project.paths)
    realm_nodes = [
        node
        for node in index.nodes
        if node.kind == "entity" and node.title.lower() == "realm"
    ]
    assert len(realm_nodes) == 1
    assert "a" in realm_nodes[0].source_ids or "b" in realm_nodes[0].source_ids


def test_dedupe_undirected_merges_reverse_edges() -> None:
    from graphwiki_kb.wikigraph.index_builder import _dedupe_undirected

    edges = [
        WikiGraphEdge(
            source="z", target="a", kind="related_to", weight=1.0, evidence=["p"]
        ),
        WikiGraphEdge(
            source="a", target="z", kind="related_to", weight=1.5, evidence=["q"]
        ),
    ]
    merged = _dedupe_undirected(edges)
    assert len(merged) == 1
    assert merged[0].weight == pytest.approx(2.5)
    assert set(merged[0].evidence) == {"p", "q"}


def test_graph_store_to_node_link_handles_parallel_edges() -> None:
    from graphwiki_kb.wikigraph.graph_store import to_node_link_json

    index = WikiGraphIndex(
        nodes=[
            WikiGraphNode(id="a", kind="entity", title="A"),
            WikiGraphNode(id="b", kind="entity", title="B"),
        ],
        edges=[
            WikiGraphEdge(source="a", target="b", kind="related_to", weight=1.0),
            WikiGraphEdge(source="a", target="b", kind="co_mentions", weight=0.5),
        ],
        built_at="now",
    )
    payload = to_node_link_json(index)
    assert {n["id"] for n in payload["nodes"]} == {"a", "b"}
