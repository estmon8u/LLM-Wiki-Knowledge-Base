"""Tests for direct GraphRAG artifact search helpers."""

from __future__ import annotations

import logging

from graphwiki_kb.services.graphrag_find_service import _read_parquet_records


def test_read_parquet_records_logs_debug_on_invalid_table(tmp_path, caplog) -> None:
    """Verifies unreadable GraphRAG parquet tables leave debug breadcrumbs."""
    table_path = tmp_path / "entities.parquet"
    table_path.write_text("not a parquet table", encoding="utf-8")

    caplog.set_level(
        logging.DEBUG, logger="graphwiki_kb.services.graphrag_find_service"
    )

    records = _read_parquet_records(table_path)

    assert records == []
    assert "Unable to read GraphRAG parquet table" in caplog.text
    assert str(table_path) in caplog.text
