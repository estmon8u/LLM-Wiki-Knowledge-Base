"""Tests for retrieval engine dispatch helpers."""

from __future__ import annotations

import click
import pytest

from graphwiki_kb.commands.retrieval_engines import (
    normalize_ask_engine,
    normalize_find_engine,
    normalize_wikigraph_method,
    validate_ask_method_for_engine,
)


def test_normalize_engines() -> None:
    assert normalize_ask_engine("wikigraph") == "wikigraph"
    assert normalize_find_engine("graph") == "graph"
    assert normalize_wikigraph_method("drift") == "drift-lite"


def test_invalid_engines_raise() -> None:
    with pytest.raises(click.ClickException, match="Unsupported ask engine"):
        normalize_ask_engine("legacy")
    with pytest.raises(click.ClickException, match="Unsupported find engine"):
        normalize_find_engine("graphrag")


def test_validate_ask_method_for_engine() -> None:
    validate_ask_method_for_engine("wikigraph", "local")
    with pytest.raises(click.ClickException, match="drift-lite"):
        validate_ask_method_for_engine("graphrag", "drift-lite")


def test_run_wikigraph_find_and_ask_human_output(test_project) -> None:
    from graphwiki_kb.commands.retrieval_engines import (
        run_wikigraph_ask,
        run_wikigraph_find,
    )
    from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
    from graphwiki_kb.wikigraph.deps import require_networkx

    require_networkx()
    test_project.write_file(
        "wiki/sources/realm.md",
        """---
title: REALM
type: source
source_id: realm
summary: REALM retrieval.
---
# REALM

## Overview

REALM uses retrieval.
""",
    )
    WikiGraphIndexService(test_project.paths, test_project.config).build()
    run_wikigraph_find(
        test_project.command_context,
        "REALM",
        method="local",
        limit=5,
        as_json=False,
    )
    answer = run_wikigraph_ask(
        test_project.command_context,
        "What is REALM?",
        method="local",
        save_answer=False,
    )
    assert answer.answer
