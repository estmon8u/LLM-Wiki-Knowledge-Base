"""Tests for test query router service.

This module belongs to `tests.test_query_router_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from graphwiki_kb.services.graphrag_status_service import GraphRAGStatusService
from graphwiki_kb.services.query_router_service import (
    QueryRouterError,
    QueryRouterService,
    _read_term_columns,
    _term_in_question,
)


def test_router_routes_global_drift_and_basic_without_graph_terms() -> None:
    """Verifies that router routes global drift and basic without graph terms."""
    router = QueryRouterService()

    assert (
        router.route("What are the main themes across the corpus?").method == "global"
    )
    assert router.route("How does REALM differ from RAG?").method == "drift"
    assert router.route("Compare REALM and RAG.").method == "drift"
    assert router.route("How does REALM relate to RAG?").method == "drift"
    assert router.route("What is retrieval used for?").method == "basic"
    assert router.route("Where is REALM mentioned?").method == "basic"
    assert router.route("Summarize the whole corpus.").method == "global"
    assert router.route("What is REALM?").method == "basic"


def test_router_uses_explicit_method_override() -> None:
    """Verifies that router uses explicit method override."""
    router = QueryRouterService()

    route = router.route("What patterns appear across the corpus?", method="local")

    assert route.method == "local"
    assert route.reason == "explicit method override"


def test_router_rejects_unknown_method() -> None:
    """Verifies that router rejects unknown method."""
    router = QueryRouterService()

    with pytest.raises(QueryRouterError, match="Unsupported GraphRAG method"):
        router.route("What is RAG?", method="lexical")


def test_router_routes_known_entity_questions_to_local(test_project) -> None:
    """Verifies that router routes known entity questions to local.

    Args:
        test_project: Test project value used by the operation.
    """
    output_dir = test_project.paths.graph_dir / "graphrag" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"id": "entity-1", "title": "REALM"},
            {"id": "entity-2", "title": "Dense Passage Retrieval"},
        ]
    ).to_parquet(output_dir / "create_final_entities.parquet")
    router = QueryRouterService(GraphRAGStatusService(test_project.paths))

    assert router.route("What is REALM?").method == "local"
    assert router.route("Explain dense passage retrieval.").method == "local"


def test_router_uses_configured_aliases_for_local_routing() -> None:
    """Verifies corpus aliases are explicit config, not hard-coded benchmark terms."""
    router = QueryRouterService(
        routing_aliases={"dense passage retrieval": ["dpr", "dual encoder"]}
    )

    route = router.route("What does DPR retrieve?")

    assert route.method == "local"
    assert route.reason == "configured graph routing alias"


def test_router_helpers_handle_missing_invalid_and_empty_terms(tmp_path) -> None:
    """Verifies that router helpers handle missing invalid and empty terms.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    unmatched_table = output_dir / "unmatched.parquet"
    unmatched_table.write_text("not parquet", encoding="utf-8")
    assert list(_read_term_columns(unmatched_table)) == []
    assert not _term_in_question("", "what is rag?")


def test_router_term_reader_caps_large_parquet_scans(tmp_path) -> None:
    """Verifies graph term discovery does not load unbounded Parquet rows."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    table_path = output_dir / "entities.parquet"
    pd.DataFrame(
        [{"id": f"entity-{index}", "title": f"Entity {index}"} for index in range(5)]
    ).to_parquet(table_path)

    assert list(_read_term_columns(table_path, max_rows=2)) == [
        "Entity 0",
        "Entity 1",
        "entity-0",
        "entity-1",
    ]
