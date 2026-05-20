"""Tests for the kb wikigraph command group."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from graphwiki_kb.cli import main as cli_main
from graphwiki_kb.engine.command_registry import (
    get_click_command,
    list_command_names,
)

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
summary: REALM pretrains a language model alongside a learned retriever.
---

# REALM

## Summary

REALM is a retrieval-augmented language model.

## Key Points

- REALM trains a retriever and a masked language model jointly.

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
tags:
  - retrieval
summary: RAG augments a seq2seq generator with retrieved passages.
---

# RAG

## Summary

RAG augments a generator with retrieved passages.

## Key Points

- RAG uses a frozen retriever and a seq2seq generator.

## Methods

RAG decouples retrieval and generation, unlike [[REALM]].
"""
)


@pytest.fixture
def runner() -> CliRunner:
    """Click test runner with a wide terminal."""
    return CliRunner(mix_stderr=False)


def _seed_project(test_project) -> Path:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(
        REALM_PAGE, encoding="utf-8"
    )
    (test_project.paths.wiki_sources_dir / "rag.md").write_text(
        RAG_PAGE, encoding="utf-8"
    )
    return test_project.paths.root


def test_wikigraph_is_registered() -> None:
    assert "wikigraph" in list_command_names()
    assert get_click_command("wikigraph") is not None


def test_wikigraph_build_find_ask_flow(runner: CliRunner, test_project) -> None:
    project_root = _seed_project(test_project)

    build = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(project_root),
            "wikigraph",
            "build",
            "--json",
        ],
    )
    assert build.exit_code == 0, build.output + build.stderr
    payload = json.loads(build.output)
    assert payload["node_count"] > 0
    assert payload["source_count"] == 2

    status = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(project_root),
            "wikigraph",
            "status",
            "--json",
        ],
    )
    assert status.exit_code == 0, status.output + status.stderr
    status_payload = json.loads(status.output)
    assert status_payload["initialized"] is True

    find = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(project_root),
            "wikigraph",
            "find",
            "How does REALM differ from RAG?",
            "--json",
        ],
    )
    assert find.exit_code == 0, find.output + find.stderr
    find_payload = json.loads(find.output)
    assert find_payload["method"] in {"local", "basic"}
    assert find_payload["contexts"]

    ask = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(project_root),
            "wikigraph",
            "ask",
            "How does REALM differ from RAG?",
            "--json",
        ],
    )
    assert ask.exit_code == 0, ask.output + ask.stderr
    ask_payload = json.loads(ask.output)
    assert ask_payload["engine"] == "wikigraph"
    assert "answer" in ask_payload
    assert ask_payload["citation_count"] >= 1


def test_wikigraph_find_requires_terms(runner: CliRunner, test_project) -> None:
    project_root = _seed_project(test_project)
    runner.invoke(cli_main, ["--project-root", str(project_root), "wikigraph", "build"])
    result = runner.invoke(
        cli_main, ["--project-root", str(project_root), "wikigraph", "find"]
    )
    assert result.exit_code != 0
    assert "Provide" in result.stderr or "Provide" in result.output


def test_wikigraph_ask_without_index_errors(runner: CliRunner, test_project) -> None:
    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "wikigraph",
            "ask",
            "anything",
        ],
    )
    assert result.exit_code != 0


def test_wikigraph_status_uninitialized(runner: CliRunner, test_project) -> None:
    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "wikigraph",
            "status",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["initialized"] is False


def test_wikigraph_ask_save(runner: CliRunner, test_project) -> None:
    project_root = _seed_project(test_project)
    runner.invoke(cli_main, ["--project-root", str(project_root), "wikigraph", "build"])
    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(project_root),
            "wikigraph",
            "ask",
            "How does REALM differ from RAG?",
            "--save-as",
            "realm-vs-rag-cli",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    assert payload["saved_path"]
    saved = test_project.paths.root / payload["saved_path"]
    assert saved.exists()
    assert "REALM" in saved.read_text(encoding="utf-8")
