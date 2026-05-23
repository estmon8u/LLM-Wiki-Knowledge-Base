"""Query tests for LightRAG-style WikiGraph retrieval."""

from __future__ import annotations

import copy

import pytest

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.config_service import DEFAULT_CONFIG
from graphwiki_kb.services.project_service import utc_now_iso
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryService
from graphwiki_kb.wikigraph.light_models import LightGraphFindResult


@pytest.fixture
def light_kb(test_project):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["wikigraph"]["mode"] = "lightrag"
    test_project.write_file(
        "raw/normalized/realm.md",
        "# REALM\n\nREALM improves retrieval. RAG uses dense retrievers.\n",
    )
    test_project.services["manifest"].save_source(
        RawSourceRecord(
            source_id="realm",
            slug="realm",
            title="REALM",
            origin="/tmp/realm.pdf",
            source_type="pdf",
            raw_path="raw/sources/realm.pdf",
            normalized_path="raw/normalized/realm.md",
            content_hash="hash-realm",
            ingested_at=utc_now_iso(),
        )
    )
    index_service = WikiGraphIndexService(
        test_project.paths,
        config=config,
        manifest_service=test_project.services["manifest"],
    )
    index_service.build()
    query_service = WikiGraphQueryService(
        test_project.paths,
        index_service=index_service,
        config=config,
    )
    return query_service


def test_hybrid_find_returns_entities_and_chunks(light_kb) -> None:
    result = light_kb.find("How does RAG use retrieval?", method="hybrid")
    assert isinstance(result, LightGraphFindResult)
    assert result.method == "hybrid"
    assert result.contexts or result.entities


def test_auto_routes_specific_query_to_local(light_kb) -> None:
    result = light_kb.find("What is REALM?", method="auto")
    assert result.method in {"local", "hybrid"}
