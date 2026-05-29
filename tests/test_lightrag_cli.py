"""CLI-surface tests for LightRAG: ask --method hybrid, find JSON, update mode."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from click.testing import CliRunner

from graphwiki_kb.cli import main as cli_main
from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.config_service import ConfigService
from graphwiki_kb.services.manifest_service import ManifestService
from graphwiki_kb.services.project_service import ProjectService, build_project_paths
from graphwiki_kb.services.update_service import UpdateOptions, UpdateService
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService

REALM_PAGE = textwrap.dedent(
    """\
    ---
    title: REALM
    type: source
    source_id: realm
    aliases:
      - Retrieval-Augmented Language Model
    summary: REALM pretrains a language model alongside a learned retriever.
    ---

    # REALM

    ## Summary

    REALM is a retrieval-augmented language model that uses Dense Passage Retrieval.
    """
)


def _lightrag_project(tmp_path: Path):
    paths = build_project_paths(tmp_path)
    ProjectService(paths).ensure_structure()
    ConfigService(paths).ensure_files()
    config = ConfigService(paths).load()
    config["wikigraph"]["mode"] = "lightrag"
    config["embeddings"]["provider"] = "anthropic"  # force BM25, no network
    manifest = ManifestService(paths)
    manifest.ensure_manifest()
    normalized_rel = "raw/normalized/realm.md"
    (tmp_path / normalized_rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / normalized_rel).write_text(
        "REALM is a retrieval-augmented language model that uses Dense Passage "
        "Retrieval for open domain question answering.",
        encoding="utf-8",
    )
    manifest.save_source(
        RawSourceRecord(
            source_id="realm",
            slug="realm",
            title="REALM",
            origin="/tmp/realm.pdf",
            source_type="pdf",
            raw_path="raw/sources/realm.pdf",
            normalized_path=normalized_rel,
            content_hash="abc",
            ingested_at="2026-01-01T00:00:00Z",
        )
    )
    return paths, config, manifest


def test_update_service_mode_override_builds_lightrag(tmp_path: Path) -> None:
    """``--wikigraph-mode lightrag`` (via UpdateOptions) builds the LightRAG index."""
    paths, config, manifest = _lightrag_project(tmp_path)
    # Start from classic in config; the override should switch the build.
    config["wikigraph"]["mode"] = "classic"
    index_service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest
    )
    update_service = UpdateService(
        ingest_service=None,
        compile_service=None,
        concept_service=None,
        search_service=None,
        config=config,
        wikigraph_index_service=index_service,
    )

    class _Result:
        wikigraph_skipped = False
        wikigraph_skip_reason = ""
        wikigraph_result = None
        wikigraph_artifact_paths: list[str] = []

    update_service._maybe_build_wikigraph(
        UpdateOptions(wikigraph=True, wikigraph_mode="lightrag", no_graph=True),
        _Result(),
    )
    status = index_service.status()
    assert status["mode"] == "lightrag"
    assert status["initialized"] is True


def test_cli_ask_accepts_hybrid_method_classic(tmp_path: Path) -> None:
    """``kb ask --method hybrid`` is accepted and works on a classic index."""
    paths = build_project_paths(tmp_path)
    ProjectService(paths).ensure_structure()
    ConfigService(paths).ensure_files()
    (paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE, encoding="utf-8")
    # Build a classic wikigraph index on disk.
    config = ConfigService(paths).load()
    WikiGraphIndexService(paths=paths, config=config).build()

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(paths.root),
            "ask",
            "How does REALM use retrieval?",
            "--engine",
            "wikigraph",
            "--method",
            "hybrid",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    # Classic engine coerces hybrid -> drift-lite internally.
    assert payload["engine"] == "wikigraph"
    assert payload["method"] in {"hybrid", "drift-lite", "auto"}


def test_cli_find_json_has_wikigraph_mode(tmp_path: Path) -> None:
    """``kb find --json`` exposes the wikigraph mode block."""
    paths = build_project_paths(tmp_path)
    ProjectService(paths).ensure_structure()
    ConfigService(paths).ensure_files()
    (paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE, encoding="utf-8")
    config = ConfigService(paths).load()
    WikiGraphIndexService(paths=paths, config=config).build()

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        cli_main,
        ["--project-root", str(paths.root), "find", "REALM", "--json"],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    assert payload["wikigraph"]["mode"] == "classic"
    assert "low_level_keywords" in payload["wikigraph"]


def test_cli_find_json_lightrag_mode(tmp_path: Path) -> None:
    """``kb find --json`` reports lightrag mode + keywords when configured."""
    paths, config, manifest = _lightrag_project(tmp_path)
    # Persist lightrag mode + offline embeddings to kb.config.yaml so the CLI
    # container picks them up.
    ConfigService(paths).save(config)
    WikiGraphIndexService(paths=paths, config=config, manifest_service=manifest).build()

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(paths.root),
            "find",
            "Dense Passage Retrieval",
            "--engine",
            "wikigraph",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    assert payload["wikigraph"]["mode"] == "lightrag"


def test_cli_find_method_passthrough_lightrag(tmp_path: Path) -> None:
    """``kb find --method local`` is honored for the wikigraph engine."""
    paths, config, manifest = _lightrag_project(tmp_path)
    ConfigService(paths).save(config)
    WikiGraphIndexService(paths=paths, config=config, manifest_service=manifest).build()

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        cli_main,
        [
            "--project-root",
            str(paths.root),
            "find",
            "Dense Passage Retrieval",
            "--engine",
            "wikigraph",
            "--method",
            "local",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + result.stderr
    payload = json.loads(result.output)
    assert payload["wikigraph"]["method"] == "local"
