"""Phase 4 retrieval-improvement tests for WikiGraphRAG.

Covers reciprocal-rank fusion in local/drift_lite search, alias-aware
query expansion, and the section-title overlap boost added to
basic_search and local_search.
"""

from __future__ import annotations

import textwrap

import pytest

from graphwiki_kb.wikigraph.context_builder import (
    ContextBuilderConfig,
    WikiGraphContextBuilder,
    _reciprocal_rank_fusion,
)
from graphwiki_kb.wikigraph.index_builder import build_wikigraph_index


def _write(project, relpath: str, body: str) -> None:
    target = project.paths.root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)


@pytest.fixture
def acronym_project(test_project):
    """Seed two source pages, one with an acronym alias."""
    _write(
        test_project,
        "wiki/sources/dpr.md",
        textwrap.dedent(
            """\
            ---
            title: Dense Passage Retrieval
            type: source
            source_id: dpr
            aliases:
              - DPR
            tags:
              - retrieval
            summary: Dense Passage Retrieval encodes queries and passages with BERT.
            ---

            # Dense Passage Retrieval

            ## Summary

            Dense passage retrieval encodes queries and passages into dense vectors.

            ## Methods

            DPR trains a bi-encoder with in-batch negatives. Inference uses MIPS.
            """
        ),
    )
    _write(
        test_project,
        "wiki/sources/orqa.md",
        textwrap.dedent(
            """\
            ---
            title: Latent Retrieval for Open-Domain QA
            type: source
            source_id: orqa
            aliases:
              - ORQA
            tags:
              - retrieval
            summary: ORQA pretrains the retriever with the inverse cloze task.
            ---

            # ORQA

            ## Summary

            ORQA pretrains the retriever with the inverse cloze task.

            ## Methods

            ORQA jointly optimizes question answering with latent retrieval.
            """
        ),
    )
    return test_project


def test_reciprocal_rank_fusion_basic() -> None:
    bundles = [
        ["a", "b", "c"],
        ["b", "c", "d"],
        ["c", "d", "e"],
    ]
    fused = _reciprocal_rank_fusion(bundles, k=60)
    order = [node_id for node_id, _ in fused]
    # 'c' appears in all three bundles at ranks 3, 2, 1 -> highest.
    assert order[0] == "c"
    # 'b' has ranks 2 + 1 across two bundles, 'd' has 3 + 2 across two.
    assert order.index("b") < order.index("d")


def test_reciprocal_rank_fusion_empty() -> None:
    assert _reciprocal_rank_fusion([], k=60) == []
    assert _reciprocal_rank_fusion([[]], k=60) == []


def test_alias_query_expansion_recovers_acronym_match(acronym_project) -> None:
    """Query 'DPR' should retrieve dense-passage-retrieval body via alias expansion."""
    index = build_wikigraph_index(acronym_project.paths)
    builder = WikiGraphContextBuilder(
        index, config=ContextBuilderConfig(retrieval_improvements_enabled=True)
    )
    contexts, seed_entities = builder.local_search("Explain DPR", limit=4)
    assert contexts
    paths = [c.path for c in contexts if c.path]
    assert any("dpr.md" in p for p in paths)
    # The seed entity title is the spelled-out form.
    assert any("Dense Passage Retrieval" in title for title in seed_entities)


def test_section_title_boost_lifts_methods_match(test_project) -> None:
    """A chunk whose section matches a question token should outrank a non-matching peer."""
    _write(
        test_project,
        "wiki/sources/fid.md",
        textwrap.dedent(
            """\
            ---
            title: FiD
            type: source
            source_id: fid
            aliases:
              - Fusion-in-Decoder
            tags:
              - generation
            summary: FiD combines retrieved passages with a generator.
            ---

            # FiD

            ## Background

            Retrieval and generation are two long-standing fields.

            ## Methods

            FiD encodes each passage and fuses them in the decoder.
            """
        ),
    )
    index = build_wikigraph_index(test_project.paths)
    config = ContextBuilderConfig(
        retrieval_improvements_enabled=True,
        section_title_overlap_boost=0.5,
    )
    builder = WikiGraphContextBuilder(index, config=config)
    contexts = builder.basic_search("describe the methods of FiD", limit=5)
    assert contexts
    methods_position = next(
        (i for i, c in enumerate(contexts) if c.section == "Methods"), None
    )
    assert methods_position is not None
    # Methods chunk must outrank or match the Background chunk for the
    # same paper because the section matches the question token.
    background_position = next(
        (i for i, c in enumerate(contexts) if c.section == "Background"), None
    )
    if background_position is not None:
        assert methods_position < background_position


def test_disabled_improvements_use_baseline_paths(acronym_project) -> None:
    """With improvements disabled, local_search falls back to the BM25-boost path."""
    index = build_wikigraph_index(acronym_project.paths)
    builder = WikiGraphContextBuilder(
        index,
        config=ContextBuilderConfig(retrieval_improvements_enabled=False),
    )
    contexts, _ = builder.local_search("DPR retrieval", limit=4)
    assert contexts
    # No assertion on order; the test ensures the disabled path runs
    # without exceptions, which protects the A/B ablation flow.
