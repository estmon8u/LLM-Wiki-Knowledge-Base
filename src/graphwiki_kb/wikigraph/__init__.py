"""WikiGraphRAG backend package.

A custom, inspectable retrieval pipeline built directly from the maintained
wiki artifacts (``wiki/sources``, ``wiki/concepts``, ``wiki/analysis``).

The package is intentionally narrow:

* :mod:`graphwiki_kb.wikigraph.models` -- strict Pydantic models.
* :mod:`graphwiki_kb.wikigraph.markdown_parser` -- wiki-page parsing helpers.
* :mod:`graphwiki_kb.wikigraph.entity_extractor` -- entity/alias extraction.
* :mod:`graphwiki_kb.wikigraph.graph_store` -- node/edge store with JSON IO.
* :mod:`graphwiki_kb.wikigraph.index_builder` -- builds the wiki graph.
* :mod:`graphwiki_kb.wikigraph.lexical_index` -- lexical retrieval (BM25 or
  pure-python fallback).
* :mod:`graphwiki_kb.wikigraph.community_builder` -- Louvain or fallback
  community detection over the wiki graph.
* :mod:`graphwiki_kb.wikigraph.context_builder` -- assemble retrieved
  context bundles from the graph.
* :mod:`graphwiki_kb.wikigraph.query_service` -- low-level wikigraph query
  pipeline (no provider calls).
* :mod:`graphwiki_kb.wikigraph.answer_service` -- thin synthesis layer that
  can run provider-free or provider-backed.

The high-level CLI wiring lives in
:mod:`graphwiki_kb.services.wikigraph_index_service`,
:mod:`graphwiki_kb.services.wikigraph_query_service`,
:mod:`graphwiki_kb.services.wikigraph_status_service`, and
:mod:`graphwiki_kb.commands.wikigraph`.
"""

from __future__ import annotations

from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphBuildManifest,
    LightGraphBuildReport,
    LightGraphIndex,
    LightQueryMethod,
    LightRetrievedBundle,
    LightRetrievedContext,
    RelationProfile,
    SourceContribution,
)
from graphwiki_kb.wikigraph.light_models import (
    ExtractedEntity as LightExtractedEntity,
)
from graphwiki_kb.wikigraph.light_models import (
    ExtractedRelation as LightExtractedRelation,
)
from graphwiki_kb.wikigraph.models import (
    EVIDENCE_NODE_KINDS,
    STRUCTURAL_NODE_KINDS,
    WikiGraphAnswer,
    WikiGraphBuildReport,
    WikiGraphCommunity,
    WikiGraphEdge,
    WikiGraphFindResult,
    WikiGraphIndex,
    WikiGraphNode,
    WikiGraphRetrievedContext,
)

__all__ = [
    "EVIDENCE_NODE_KINDS",
    "STRUCTURAL_NODE_KINDS",
    "EntityProfile",
    "LightChunk",
    "LightExtractedEntity",
    "LightExtractedRelation",
    "LightGraphBuildManifest",
    "LightGraphBuildReport",
    "LightGraphIndex",
    "LightQueryMethod",
    "LightRetrievedBundle",
    "LightRetrievedContext",
    "RelationProfile",
    "SourceContribution",
    "WikiGraphAnswer",
    "WikiGraphBuildReport",
    "WikiGraphCommunity",
    "WikiGraphEdge",
    "WikiGraphFindResult",
    "WikiGraphIndex",
    "WikiGraphNode",
    "WikiGraphRetrievedContext",
]
