"""Tests for the WikiGraphRAG backend folded into kb update/find/ask."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
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


def _enable_stub_provider(project_root: Path) -> None:
    """Write a stub provider into ``kb.config.yaml`` for update preflight."""
    config_path = project_root / "kb.config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg["provider"] = {"name": "stub"}
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


class _StubProvider(TextProvider):
    """Minimal stub used to satisfy update preflight."""

    name = "stub"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        if request.response_schema_name == "kb_review_report":
            return ProviderResponse(text='{"issues": []}', model_name="stub-1")
        return ProviderResponse(text="Stub summary", model_name="stub-1")


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


def test_kb_ask_engine_wikigraph_show_source_trace(
    runner: CliRunner, test_project
) -> None:
    """``--show-source-trace`` renders the WikiGraphRAG trace section."""
    project_root = _seed_project(test_project)
    test_project.services.wikigraph_index.build()  # bypass update preflight
    test_project.services.wikigraph_query.provider = None  # provider-free

    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(project_root),
            "ask",
            "How does REALM differ from RAG?",
            "--engine",
            "wikigraph",
            "--method",
            "local",
            "--show-source-trace",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    assert "WikiGraphRAG Source Trace" in result.output
    assert "Retrieved contexts:" in result.output


def test_kb_update_no_wikigraph_flag_via_cli(runner: CliRunner, test_project) -> None:
    """`--no-wikigraph` should win even when config says enabled=true."""
    _seed_project(test_project)
    _enable_stub_provider(test_project.paths.root)
    with patch("graphwiki_kb.services.build_provider", return_value=_StubProvider()):
        result = runner.invoke(
            cli_main,
            [
                "--project-root",
                str(test_project.paths.root),
                "update",
                "--no-graph",
                "--no-wikigraph",
            ],
        )
    assert result.exit_code == 0, result.output + result.stderr
    assert "--no-wikigraph requested" in result.output


def test_kb_update_artifact_types_flag(runner: CliRunner, test_project) -> None:
    _seed_project(test_project)
    _enable_stub_provider(test_project.paths.root)
    with patch("graphwiki_kb.services.build_provider", return_value=_StubProvider()):
        result = runner.invoke(
            cli_main,
            [
                "--project-root",
                str(test_project.paths.root),
                "update",
                "--no-graph",
                "--export-wikigraph-artifacts",
                "--artifact-types",
                "entities,communities",
            ],
        )
    assert result.exit_code == 0, result.output + result.stderr
    assert "entities card(s)" in result.output
    assert "communities card(s)" in result.output
    assert "chunks card(s)" not in result.output


def test_kb_ask_default_engine_is_wikigraph(runner: CliRunner, test_project) -> None:
    """``kb ask`` (no flag) now routes through WikiGraphRAG."""
    _seed_project(test_project)
    test_project.services.wikigraph_index.build()
    test_project.services.wikigraph_query.provider = None

    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "ask",
            "How does REALM differ from RAG?",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    assert payload["engine"] == "wikigraph"


def test_kb_ask_engine_all_runs_each_backend(runner: CliRunner, test_project) -> None:
    """``--engine all`` runs wikigraph + graphrag + legacy and merges output."""
    _seed_project(test_project)
    test_project.services.wikigraph_index.build()
    test_project.services.wikigraph_query.provider = None

    # Stub the GraphRAG controller so we do not need a real workspace.
    class _StubAnswer:
        retriever = "graph"
        method = "auto"
        planner = "auto"
        route_reason = "stub"
        route_confidence = "low"
        route_matched_terms = ()
        claim_support = "graph-index-answer"
        source_trace = {"input_path": "stub", "output_dir": "stub"}
        staleness_warnings: list[str] = []
        answer = "Stub GraphRAG answer."
        saved_path = None
        index_run_id = "stub-run"
        graph_data_references: list[dict] = []

        def to_dict(self) -> dict:
            return {
                "retriever": self.retriever,
                "method": self.method,
                "answer": self.answer,
            }

    test_project.services.graph_ask_controller.ask = lambda *a, **kw: _StubAnswer()  # type: ignore[assignment]

    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "ask",
            "How does REALM differ from RAG?",
            "--engine",
            "all",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    assert set(payload["engines"]) == {"wikigraph", "graphrag", "legacy"}
    assert payload["results"]["wikigraph"]["engine"] == "wikigraph"
    assert payload["results"]["graphrag"]["engine"] == "graphrag"
    assert payload["results"]["legacy"]["engine"] == "legacy"


def test_kb_ask_engine_csv_runs_only_selected(runner: CliRunner, test_project) -> None:
    """``--engine wikigraph,legacy`` runs exactly those two backends."""
    _seed_project(test_project)
    test_project.services.wikigraph_index.build()
    test_project.services.wikigraph_query.provider = None

    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "ask",
            "How does REALM differ from RAG?",
            "--engine",
            "wikigraph,legacy",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    assert payload["engines"] == ["wikigraph", "legacy"]


def test_kb_ask_engine_unknown_value_errors(runner: CliRunner, test_project) -> None:
    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "ask",
            "anything",
            "--engine",
            "mystery",
        ],
    )
    assert result.exit_code != 0
    assert "Unknown --engine" in (result.output + result.stderr)


def test_kb_find_engine_legacy_prints_deprecation_note(
    runner: CliRunner, test_project
) -> None:
    _seed_project(test_project)
    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "find",
            "--engine",
            "legacy",
            "REALM",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    assert "Deprecated" in result.output


def test_kb_find_engine_legacy_json_carries_deprecated_flag(
    runner: CliRunner, test_project
) -> None:
    _seed_project(test_project)
    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(test_project.paths.root),
            "find",
            "--engine",
            "legacy",
            "REALM",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    assert payload["engine"] == "legacy"
    assert payload["deprecated"] is True
    assert "Deprecated" in payload["warning"]


def test_kb_legacy_group_is_removed() -> None:
    from graphwiki_kb.engine.command_registry import (
        get_click_command,
        list_command_names,
    )

    assert "legacy" not in list_command_names()
    assert get_click_command("legacy") is None


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
