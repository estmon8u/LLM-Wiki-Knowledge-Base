"""Tests for the WikiGraphRAG backend folded into kb update/find/ask."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from graphwiki_kb.cli import main as cli_main
from graphwiki_kb.engine.command_registry import list_command_names
from graphwiki_kb.providers.base import ProviderRequest, ProviderResponse, TextProvider

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
    """Click test runner."""
    return CliRunner(mix_stderr=False)


def _seed_project(test_project) -> Path:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(
        REALM_PAGE, encoding="utf-8"
    )
    (test_project.paths.wiki_sources_dir / "rag.md").write_text(
        RAG_PAGE, encoding="utf-8"
    )
    return test_project.paths.root


def test_wikigraph_command_group_removed() -> None:
    """The standalone ``kb wikigraph`` group is folded into other commands."""
    assert "wikigraph" not in list_command_names()


def test_kb_update_builds_wikigraph_index(test_project) -> None:
    """``kb update``'s service path builds the WikiGraphRAG index by default."""
    _seed_project(test_project)
    service = test_project.services.wikigraph_index
    assert service.status()["initialized"] is False

    from graphwiki_kb.services.update_service import UpdateOptions, UpdateService

    update_service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=test_project.config,
        wikigraph_index_service=service,
    )
    # Use a stub provider via wikigraph-only path: bypass preflight by disabling
    # the compile phase and going straight to the wikigraph build.
    options = UpdateOptions(no_graph=True, wikigraph=True)
    # Bypass the preflight check (no provider configured in tests) by simulating
    # the post-compile step directly.
    update_service._maybe_build_wikigraph(options, _DummyResult())
    snapshot = service.status()
    assert snapshot["initialized"] is True
    assert snapshot["source_count"] == 2


class _DummyResult:
    """Lightweight stand-in for ``UpdateResult`` used by the helper."""

    wikigraph_skipped = False
    wikigraph_skip_reason = ""
    wikigraph_result = None


def test_kb_update_skips_wikigraph_when_disabled(test_project) -> None:
    _seed_project(test_project)
    service = test_project.services.wikigraph_index

    from graphwiki_kb.services.update_service import UpdateOptions, UpdateService

    update_service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=test_project.config,
        wikigraph_index_service=service,
    )
    result = _DummyResult()
    update_service._maybe_build_wikigraph(UpdateOptions(wikigraph=False), result)
    assert result.wikigraph_skipped is True
    assert "--no-wikigraph" in result.wikigraph_skip_reason


def test_kb_update_skips_wikigraph_when_service_missing(test_project) -> None:
    from graphwiki_kb.services.update_service import UpdateOptions, UpdateService

    update_service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=test_project.config,
        wikigraph_index_service=None,
    )
    result = _DummyResult()
    update_service._maybe_build_wikigraph(UpdateOptions(wikigraph=True), result)
    assert result.wikigraph_skipped is True
    assert "unavailable" in result.wikigraph_skip_reason


def test_kb_update_allow_partial_swallows_wikigraph_error(test_project) -> None:
    from graphwiki_kb.services.update_service import UpdateOptions, UpdateService

    class _ExplodingIndex:
        def build(self, **_kwargs: object) -> object:
            raise RuntimeError("boom")

    update_service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=test_project.config,
        wikigraph_index_service=_ExplodingIndex(),
    )
    result = _DummyResult()
    update_service._maybe_build_wikigraph(
        UpdateOptions(wikigraph=True, allow_partial=True), result
    )
    assert result.wikigraph_skipped is True
    assert "WikiGraphRAG build failed" in result.wikigraph_skip_reason


def test_kb_update_propagates_wikigraph_error_without_allow_partial(
    test_project,
) -> None:
    from graphwiki_kb.services.update_service import UpdateOptions, UpdateService

    class _ExplodingIndex:
        def build(self, **_kwargs: object) -> object:
            raise RuntimeError("boom")

    update_service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=test_project.config,
        wikigraph_index_service=_ExplodingIndex(),
    )
    result = _DummyResult()
    with pytest.raises(RuntimeError, match="boom"):
        update_service._maybe_build_wikigraph(UpdateOptions(wikigraph=True), result)


# --------------------------------------------------------------------------- #
# kb find --engine                                                            #
# --------------------------------------------------------------------------- #


def test_kb_find_engine_auto_includes_wikigraph(
    runner: CliRunner, test_project
) -> None:
    _seed_project(test_project)
    test_project.services.wikigraph_index.build()

    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "find",
            "REALM",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    retrievers = {item["retriever"] for item in payload["results"]}
    assert "wikigraph" in retrievers
    assert payload["wikigraph"]["contexts"]


def test_kb_find_engine_wikigraph_only(runner: CliRunner, test_project) -> None:
    _seed_project(test_project)
    test_project.services.wikigraph_index.build()

    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "find",
            "REALM",
            "--engine",
            "wikigraph",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    assert payload["engine"] == "wikigraph"
    retrievers = {item["retriever"] for item in payload["results"]}
    assert retrievers == {"wikigraph"}


def test_kb_find_engine_wikigraph_errors_without_index(
    runner: CliRunner, test_project
) -> None:
    _seed_project(test_project)
    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "find",
            "REALM",
            "--engine",
            "wikigraph",
        ],
    )
    assert result.exit_code != 0


def test_kb_find_auto_warns_when_wikigraph_missing(
    runner: CliRunner, test_project
) -> None:
    _seed_project(test_project)
    # Do not build the wikigraph index — auto mode should warn but not fail.
    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "find",
            "REALM",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    assert any("WikiGraphRAG unavailable" in diag for diag in payload["diagnostics"])


# --------------------------------------------------------------------------- #
# kb ask --engine wikigraph                                                   #
# --------------------------------------------------------------------------- #


class _StaticProvider(TextProvider):
    """Returns a fixed structured-output payload from generate()."""

    name = "stub-static"

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(text=self._payload, model_name="stub-static")


def test_kb_ask_engine_wikigraph_provider_free(runner: CliRunner, test_project) -> None:
    _seed_project(test_project)
    test_project.services.wikigraph_index.build()
    # Ensure the wikigraph query service uses no provider so the provider-free
    # synthesis path runs even though the test_project has a stub provider on
    # other services.
    test_project.services.wikigraph_query.provider = None

    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "ask",
            "How does REALM differ from RAG?",
            "--engine",
            "wikigraph",
            "--method",
            "local",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    assert payload["engine"] == "wikigraph"
    assert payload["method"] == "local"
    assert "REALM" in payload["answer"] or "RAG" in payload["answer"]


def test_kb_ask_engine_wikigraph_with_provider(runner: CliRunner, test_project) -> None:
    _seed_project(test_project)
    test_project.services.wikigraph_index.build()

    refs_payload = (
        '{"answer_markdown": "REALM differs from RAG by training the retriever '
        'jointly.",'
        ' "claims": [{"text": "REALM trains a retriever and a masked language '
        'model jointly.", "citation_refs": '
        '["wiki/sources/realm.md#chunk-0"]}],'
        ' "citations": [{"ref": "wiki/sources/realm.md#chunk-0", "title": '
        '"Summary"}], "insufficient_evidence": false}'
    )
    test_project.services.wikigraph_query.provider = _StaticProvider(refs_payload)

    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "ask",
            "How does REALM differ from RAG?",
            "--engine",
            "wikigraph",
            "--method",
            "local",
            "--save-as",
            "ask-folded-realm-vs-rag",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    assert payload["engine"] == "wikigraph"
    assert payload["saved_path"]
    saved = test_project.paths.root / payload["saved_path"]
    assert saved.exists()
    assert "REALM" in saved.read_text(encoding="utf-8")


def test_kb_ask_engine_wikigraph_invalid_method(
    runner: CliRunner, test_project
) -> None:
    _seed_project(test_project)
    test_project.services.wikigraph_index.build()

    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "ask",
            "test",
            "--engine",
            "wikigraph",
            "--method",
            "drift",
        ],
    )
    assert result.exit_code != 0
    assert "wikigraph" in (result.output + result.stderr)


def test_kb_ask_engine_graphrag_rejects_drift_lite(
    runner: CliRunner, test_project
) -> None:
    _seed_project(test_project)

    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "ask",
            "test",
            "--engine",
            "graphrag",
            "--method",
            "drift-lite",
        ],
    )
    assert result.exit_code != 0


def test_kb_ask_engine_wikigraph_without_index_errors(
    runner: CliRunner, test_project
) -> None:
    _seed_project(test_project)
    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "ask",
            "REALM",
            "--engine",
            "wikigraph",
        ],
    )
    assert result.exit_code != 0
