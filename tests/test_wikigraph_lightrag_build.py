"""Integration tests for LightRAG-style WikiGraph index builds."""

from __future__ import annotations

import copy

import pytest

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.config_service import DEFAULT_CONFIG
from graphwiki_kb.services.project_service import utc_now_iso
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
)


REALM_NORMALIZED = """\
# REALM

REALM pretrains a retriever with masked language modeling.
RAG combines retrieval with generation.
"""


@pytest.fixture
def light_config() -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["wikigraph"]["mode"] = "lightrag"
    config["wikigraph"]["export_generated_artifacts"] = True
    return config


def test_lightgraph_build_from_normalized_source(test_project, light_config) -> None:
    test_project.write_file("raw/normalized/realm.md", REALM_NORMALIZED)
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
    service = WikiGraphIndexService(
        test_project.paths,
        config=light_config,
        manifest_service=test_project.services["manifest"],
    )
    report = service.build()
    assert report.chunk_count >= 1
    assert report.entity_count >= 1
    store = LightGraphStore(
        LightGraphStorePaths(test_project.paths.graph_dir / "wikigraph" / "lightrag")
    )
    index = store.load()
    assert index is not None
    assert index.chunks
