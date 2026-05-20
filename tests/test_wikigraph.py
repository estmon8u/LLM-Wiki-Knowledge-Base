"""Tests for WikiGraphRAG backend integrated into kb ask/find/update."""

from __future__ import annotations

import json

import pytest
import yaml
from click.testing import CliRunner

from graphwiki_kb.commands.ask import create_command as create_ask_command
from graphwiki_kb.commands.find import create_command as create_find_command
from graphwiki_kb.commands.update import create_command as create_update_command
from graphwiki_kb.services.config_service import CURRENT_CONFIG_VERSION, ConfigService
from graphwiki_kb.services.update_service import UpdateService
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryFacade
from graphwiki_kb.wikigraph.deps import require_networkx

pytest.importorskip("networkx")


def _ensure_provider_config(project) -> None:
    config = yaml.safe_load(project.paths.config_file.read_text(encoding="utf-8"))
    config["provider"] = {"name": "openai"}
    project.paths.config_file.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    project.config["provider"] = {"name": "openai"}


def _write_source_page(project, slug: str, title: str, body: str) -> None:
    project.write_file(
        f"wiki/sources/{slug}.md",
        f"""---
title: "{title}"
type: source
source_id: {slug}
summary: "{body[:120]}"
---
# {title}

## Overview

{body}

## Related Work

See [[RAG]] and [[REALM]] for comparison.
""",
    )


def test_wikigraph_build_via_update_and_query(test_project) -> None:
    require_networkx()
    _ensure_provider_config(test_project)
    _write_source_page(
        test_project,
        "realm",
        "REALM",
        "REALM augments language model pretraining with retrieval.",
    )
    _write_source_page(
        test_project,
        "rag",
        "RAG",
        "RAG conditions generation on retrieved passages.",
    )
    runner = CliRunner()
    update = runner.invoke(
        create_update_command(),
        ["--no-graph", "--no-wikigraph"],
        obj=test_project.command_context,
    )
    assert update.exit_code == 0, update.output
    index_service = WikiGraphIndexService(test_project.paths, test_project.config)
    index_service.build()
    assert (test_project.root / "graph/wikigraph/nodes.json").exists()
    facade = WikiGraphQueryFacade(test_project.paths, test_project.config)
    find_payload = facade.find("How does REALM differ from RAG?", method="local")
    assert find_payload["engine"] == "wikigraph"
    assert find_payload["contexts"]
    answer = facade.ask("How does REALM differ from RAG?", method="local")
    assert answer.engine == "wikigraph"
    assert "REALM" in answer.answer or "RAG" in answer.answer


def test_kb_find_and_ask_engine_wikigraph(test_project) -> None:
    require_networkx()
    _write_source_page(
        test_project,
        "realm",
        "REALM",
        "REALM uses retrieval during pretraining.",
    )
    WikiGraphIndexService(test_project.paths, test_project.config).build()
    runner = CliRunner()
    find = runner.invoke(
        create_find_command(),
        ["REALM", "RAG", "--engine", "wikigraph", "--json"],
        obj=test_project.command_context,
    )
    assert find.exit_code == 0, find.output
    json_text = find.output[find.output.find("{") :]
    payload = json.loads(json_text)
    assert payload["engine"] == "wikigraph"
    assert payload["contexts"]
    ask = runner.invoke(
        create_ask_command(),
        [
            "What is REALM?",
            "--engine",
            "wikigraph",
            "--method",
            "local",
            "--json",
        ],
        obj=test_project.command_context,
    )
    assert ask.exit_code == 0, ask.output
    answer = json.loads(ask.output[ask.output.find("{") :])
    assert answer["engine"] == "wikigraph"


def test_kb_update_builds_wikigraph_index(test_project) -> None:
    require_networkx()
    _ensure_provider_config(test_project)
    _write_source_page(test_project, "realm", "REALM", "REALM retrieval memory.")
    runner = CliRunner()
    result = runner.invoke(
        create_update_command(),
        ["--no-graph"],
        obj=test_project.command_context,
    )
    assert result.exit_code == 0, result.output
    assert (test_project.root / "graph/wikigraph/index.json").exists()
    assert "WikiGraphRAG" in result.output


def test_config_migrates_to_version_9(test_project) -> None:
    import yaml

    config = yaml.safe_load(test_project.paths.config_file.read_text(encoding="utf-8"))
    config["version"] = 8
    config.pop("wikigraph", None)
    test_project.paths.config_file.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    loaded = ConfigService(test_project.paths).load()
    assert loaded["version"] == CURRENT_CONFIG_VERSION
    assert "wikigraph" in loaded


def test_wikigraph_query_modes(test_project) -> None:
    require_networkx()
    _write_source_page(
        test_project,
        "realm",
        "REALM",
        "REALM augments language model pretraining with retrieval.",
    )
    _write_source_page(
        test_project,
        "rag",
        "RAG",
        "RAG conditions generation on retrieved passages.",
    )
    WikiGraphIndexService(test_project.paths, test_project.config).build()
    facade = WikiGraphQueryFacade(test_project.paths, test_project.config)
    for method in ("basic", "local", "global", "drift-lite", "auto"):
        payload = facade.find(
            "What are the main retrieval patterns across the corpus?",
            method=method,  # type: ignore[arg-type]
        )
        assert "contexts" in payload
    global_answer = facade.ask(
        "What are the main retrieval patterns across the corpus?",
        method="global",
    )
    assert global_answer.method == "global"


def test_wikigraph_status_and_missing_index(test_project) -> None:
    from graphwiki_kb.wikigraph.status_service import wikigraph_status

    status = wikigraph_status(test_project.paths)
    assert status.built is False
    facade = WikiGraphQueryFacade(test_project.paths, test_project.config)
    with pytest.raises(FileNotFoundError, match="not built"):
        facade.find("REALM")


def test_ask_rejects_drift_lite_for_graphrag(test_project) -> None:
    runner = CliRunner()
    result = runner.invoke(
        create_ask_command(),
        ["What is REALM?", "--method", "drift-lite"],
        obj=test_project.command_context,
    )
    assert result.exit_code != 0
    assert "drift-lite" in result.output


def test_find_rejects_method_for_graph_engine(test_project) -> None:
    runner = CliRunner()
    result = runner.invoke(
        create_find_command(),
        ["REALM", "--method", "local"],
        obj=test_project.command_context,
    )
    assert result.exit_code != 0
    assert "--engine wikigraph" in result.output


def test_export_wikigraph_artifacts_flag(test_project) -> None:
    from graphwiki_kb.commands.export_cmd import create_command

    require_networkx()
    _write_source_page(test_project, "realm", "REALM", "REALM retrieval.")
    WikiGraphIndexService(test_project.paths, test_project.config).build()
    runner = CliRunner()
    result = runner.invoke(
        create_command(),
        ["--wikigraph-artifacts"],
        obj=test_project.command_context,
    )
    assert result.exit_code == 0, result.output
    assert "wiki/wikigraph/" in result.output


def test_wikigraph_export_artifacts(test_project) -> None:
    require_networkx()
    _write_source_page(test_project, "realm", "REALM", "REALM retrieval memory.")
    service = WikiGraphIndexService(test_project.paths, test_project.config)
    service.build()
    created = service.export_artifacts()
    assert created
    assert (test_project.root / "wiki/wikigraph/entities").exists()


def test_update_skips_wikigraph_without_wiki_pages(test_project) -> None:
    _ensure_provider_config(test_project)
    runner = CliRunner()
    result = runner.invoke(
        create_update_command(),
        ["--no-graph"],
        obj=test_project.command_context,
    )
    assert result.exit_code == 0, result.output
    assert "WikiGraphRAG skipped" in result.output
    assert "No wiki pages found" in result.output


def test_update_service_builds_and_exports_wikigraph(test_project) -> None:
    from graphwiki_kb.services.update_service import UpdateOptions

    require_networkx()
    _ensure_provider_config(test_project)
    test_project.config["provider"] = {"name": "openai"}
    _write_source_page(test_project, "realm", "REALM", "REALM body.")
    service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=test_project.config,
        wikigraph_index_service=test_project.services.wikigraph_index,
    )
    result = service.run(
        UpdateOptions(
            no_graph=True,
            export_wikigraph_artifacts=True,
        ),
    )
    assert result.wikigraph_result is not None
    assert result.wikigraph_result.build is not None
    assert result.wikigraph_result.exported_artifacts


def test_kb_status_includes_wikigraph(test_project) -> None:
    from graphwiki_kb.commands.status import create_command

    require_networkx()
    _write_source_page(test_project, "realm", "REALM", "REALM.")
    WikiGraphIndexService(test_project.paths, test_project.config).build()
    runner = CliRunner()
    result = runner.invoke(
        create_command(),
        ["--json"],
        obj=test_project.command_context,
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "wikigraph_status" in payload
    assert payload["wikigraph_status"]["built"] is True


def test_update_skips_wikigraph_when_disabled_in_config(test_project) -> None:
    from graphwiki_kb.services.update_service import UpdateOptions

    _ensure_provider_config(test_project)
    test_project.config["wikigraph"] = {
        **test_project.config.get("wikigraph", {}),
        "enabled": False,
    }
    _write_source_page(test_project, "realm", "REALM", "body")
    service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=test_project.config,
        wikigraph_index_service=test_project.services.wikigraph_index,
    )
    result = service.run(UpdateOptions(no_graph=True))
    assert result.wikigraph_result is not None
    assert result.wikigraph_result.skipped is True
    assert "enabled is false" in result.wikigraph_result.skip_reason


def test_update_service_no_wikigraph_flag(test_project) -> None:
    from graphwiki_kb.services.update_service import UpdateOptions

    require_networkx()
    _ensure_provider_config(test_project)
    test_project.config["provider"] = {"name": "openai"}
    _write_source_page(test_project, "realm", "REALM", "REALM body.")
    service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=test_project.config,
        wikigraph_index_service=test_project.services.wikigraph_index,
    )
    result = service.run(
        UpdateOptions(no_graph=True, no_wikigraph=True),
    )
    assert result.wikigraph_result is not None
    assert result.wikigraph_result.skipped is True
