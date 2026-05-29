"""Tests that wikigraph-classic and wikigraph-lightrag are distinct backends."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.config_service import ConfigService
from graphwiki_kb.services.manifest_service import ManifestService
from graphwiki_kb.services.project_service import ProjectService, build_project_paths
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
from scripts.rag_eval.backends import WikiGraphBackend, build_command_context
from scripts.rag_eval.cli import main
from scripts.rag_eval.dataset import EvalQuestion

REALM_PAGE = textwrap.dedent(
    """\
    ---
    title: REALM
    type: source
    source_id: realm
    summary: REALM pretrains a retriever with a masked language model.
    ---

    # REALM

    ## Summary

    REALM jointly trains a neural retriever with a masked language model.
    """
)

TINY_BENCHMARK = textwrap.dedent(
    """\
    version: 4
    name: tiny
    questions:
      - id: realm_q
        question: How does REALM train its retriever?
        category: local
        expected_sources: [REALM]
        reference_answer: REALM jointly trains a retriever with an MLM.
    """
)


def _setup_both_modes(tmp_path: Path) -> Path:
    paths = build_project_paths(tmp_path)
    ProjectService(paths).ensure_structure()
    ConfigService(paths).ensure_files()
    config = ConfigService(paths).load()
    config["embeddings"]["provider"] = "anthropic"  # force BM25 everywhere (offline)
    ConfigService(paths).save(config)

    # Classic source page.
    (paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE, encoding="utf-8")
    # Lightrag normalized source + manifest.
    manifest = ManifestService(paths)
    manifest.ensure_manifest()
    normalized_rel = "raw/normalized/realm.md"
    (tmp_path / normalized_rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / normalized_rel).write_text(
        "REALM jointly trains a neural retriever with a masked language model "
        "for open domain question answering.",
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

    classic_cfg = ConfigService(paths).load()
    classic_cfg["wikigraph"]["mode"] = "classic"
    WikiGraphIndexService(
        paths=paths, config=classic_cfg, manifest_service=manifest
    ).build()

    light_cfg = ConfigService(paths).load()
    light_cfg["wikigraph"]["mode"] = "lightrag"
    light_cfg["embeddings"]["provider"] = "anthropic"
    WikiGraphIndexService(
        paths=paths, config=light_cfg, manifest_service=manifest
    ).build()
    return paths.root


def test_wikigraph_modes_are_distinct_backends(tmp_path: Path) -> None:
    root = _setup_both_modes(tmp_path)
    context = build_command_context(root)
    classic = WikiGraphBackend(
        context, method="auto", mode="classic", name="wikigraph-classic"
    )
    lightrag = WikiGraphBackend(
        context, method="hybrid", mode="lightrag", name="wikigraph-lightrag"
    )
    q = EvalQuestion(id="realm_q", question="How does REALM train its retriever?")
    classic_ctx = classic.retrieve(q)
    light_ctx = lightrag.retrieve(q)
    assert classic.name == "wikigraph-classic"
    assert lightrag.name == "wikigraph-lightrag"
    # Both modes return contexts from their own (distinct) indexes.
    assert classic_ctx
    assert light_ctx


def test_cli_four_methods_includes_both_wikigraph_modes(tmp_path: Path) -> None:
    root = _setup_both_modes(tmp_path)
    benchmark = tmp_path / "tiny.yaml"
    benchmark.write_text(TINY_BENCHMARK, encoding="utf-8")
    results = tmp_path / "results"
    code = main(
        [
            "--project-root",
            str(root),
            "--benchmark",
            str(benchmark),
            "--results-dir",
            str(results),
            "--methods",
            "wikigraph-classic",
            "wikigraph-lightrag",
            "--retrieval-only",
            "--bootstrap",
            "50",
        ]
    )
    assert code == 0
    payload = json.loads((results / "rag_eval_summary.json").read_text())
    backends = {s["backend"] for s in payload["summaries"]}
    assert "wikigraph-classic" in backends
    assert "wikigraph-lightrag" in backends
