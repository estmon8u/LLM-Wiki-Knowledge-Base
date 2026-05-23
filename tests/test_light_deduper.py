"""Unit tests for LightRAG entity/relation deduplication."""

from __future__ import annotations

from graphwiki_kb.wikigraph.light_deduper import dedupe_and_profile
from graphwiki_kb.wikigraph.light_extractor import LightExtractionResult
from graphwiki_kb.wikigraph.light_models import ExtractedEntity, ExtractedRelation


def test_dedupe_merges_alias_entities() -> None:
    result = LightExtractionResult(
        entities=[
            ExtractedEntity(
                name="Dense Passage Retrieval",
                type="METHOD",
                description="Dual encoder retriever.",
                aliases=["DPR"],
                chunk_ids=["chunk:a:0:x"],
                source_ids=["dpr"],
            ),
            ExtractedEntity(
                name="DPR",
                type="METHOD",
                description="Alias mention.",
                aliases=[],
                chunk_ids=["chunk:a:1:y"],
                source_ids=["dpr"],
            ),
        ],
        relations=[],
    )
    entities, relations = dedupe_and_profile([("chunk:a:0:x", result)])
    assert len(entities) == 1
    assert (
        "DPR" in entities[0].aliases
        or entities[0].canonical_name == "Dense Passage Retrieval"
    )
    assert relations == []
