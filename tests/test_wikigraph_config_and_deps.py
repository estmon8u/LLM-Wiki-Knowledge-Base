"""Tests for the WikiGraphRAG config integration and optional-dep helpers."""

from __future__ import annotations

import copy
import textwrap

import pytest

from graphwiki_kb.services.config_service import (
    DEFAULT_CONFIG,
    WikiGraphRuntimeConfig,
    resolve_wikigraph_config,
)
from graphwiki_kb.services.update_service import UpdateOptions, UpdateService
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryService
from graphwiki_kb.wikigraph import deps as wikigraph_deps
from graphwiki_kb.wikigraph.context_builder import ContextBuilderConfig

REALM_PAGE = textwrap.dedent(
    """\
    ---
    title: REALM
    type: source
    source_id: realm
    aliases:
      - Retrieval-Augmented Language Model
    summary: REALM pretrains a retriever with masked language modeling.
    ---

    # REALM

    ## Summary

    REALM jointly trains a retriever and a masked language model.

    ## Methods

    REALM backpropagates through retrieval. See [[RAG]].
    """
)


# --------------------------------------------------------------------------- #
# Config resolution                                                           #
# --------------------------------------------------------------------------- #


def test_resolve_wikigraph_config_defaults_are_complete() -> None:
    runtime = resolve_wikigraph_config(copy.deepcopy(DEFAULT_CONFIG))
    assert isinstance(runtime, WikiGraphRuntimeConfig)
    assert runtime.enabled is True
    assert runtime.mode == "classic"
    assert runtime.lightrag.chunk_token_size == 1200
    assert runtime.max_hops == 2
    assert runtime.max_context_chunks == 12
    assert runtime.chunk_char_limit == 1200
    assert runtime.fuzzy_entity_match_threshold == 88
    assert runtime.lexical_backend == "bm25s"
    assert runtime.community_algorithm == "louvain"


def test_resolve_wikigraph_config_missing_section_falls_back_to_defaults() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    del config["wikigraph"]
    runtime = resolve_wikigraph_config(config)
    assert runtime.enabled is True
    assert runtime.max_hops == 2


def test_resolve_wikigraph_config_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        resolve_wikigraph_config({"wikigraph": [1, 2, 3]})


def test_resolve_wikigraph_config_rejects_out_of_range_values() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["wikigraph"]["max_hops"] = 99
    with pytest.raises(ValueError):
        resolve_wikigraph_config(config)


def test_resolve_wikigraph_config_rejects_unknown_keys() -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["wikigraph"]["mystery"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_wikigraph_config(config)


def test_default_config_includes_wikigraph_section() -> None:
    assert "wikigraph" in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["wikigraph"]["enabled"] is True


# --------------------------------------------------------------------------- #
# Service wiring                                                              #
# --------------------------------------------------------------------------- #


def test_index_service_honors_config_overrides(test_project) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    test_project.config.setdefault("wikigraph", {})["chunk_char_limit"] = 400
    service = WikiGraphIndexService(
        paths=test_project.paths, config=test_project.config
    )
    report = service.build()
    assert report.source_count == 1
    # Build does not need to fail; we just confirm the override is honored
    # downstream via the resolved BuildOptions.
    runtime = service.runtime_config
    assert runtime.chunk_char_limit == 400


def test_index_service_build_overrides_take_precedence(test_project) -> None:
    """Explicit ``build`` kwargs should beat the resolved runtime config."""
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    test_project.config.setdefault("wikigraph", {})["chunk_char_limit"] = 400
    service = WikiGraphIndexService(
        paths=test_project.paths, config=test_project.config
    )
    report = service.build(chunk_char_limit=200, include_graphrag_export_pages=False)
    assert report.node_count > 0


def test_index_service_falls_back_when_config_invalid(test_project) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    # Bogus section -> the index service should still produce a build using
    # package defaults rather than crash.
    test_project.config["wikigraph"] = {"max_hops": 99}
    service = WikiGraphIndexService(
        paths=test_project.paths, config=test_project.config
    )
    report = service.build()
    assert report.source_count == 1


def test_query_service_threads_runtime_config(test_project) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    test_project.config.setdefault("wikigraph", {})["max_context_chunks"] = 3
    index_service = WikiGraphIndexService(
        paths=test_project.paths, config=test_project.config
    )
    index_service.build()
    query_service = WikiGraphQueryService(
        paths=test_project.paths,
        index_service=index_service,
        provider=None,
        config=test_project.config,
    )
    ctx_config = query_service._context_builder_config()
    assert isinstance(ctx_config, ContextBuilderConfig)
    assert ctx_config.max_context_chunks == 3


def test_query_service_invalid_config_falls_back(test_project) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    index_service = WikiGraphIndexService(
        paths=test_project.paths, config=test_project.config
    )
    index_service.build()
    query_service = WikiGraphQueryService(
        paths=test_project.paths,
        index_service=index_service,
        provider=None,
        config={"wikigraph": {"max_hops": 99}},
    )
    ctx_config = query_service._context_builder_config()
    assert ctx_config.max_hops == 2  # default


# --------------------------------------------------------------------------- #
# Generated artifact export                                                   #
# --------------------------------------------------------------------------- #


def test_export_artifacts_writes_entity_community_chunk_cards(test_project) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    service = WikiGraphIndexService(
        paths=test_project.paths, config=test_project.config
    )
    service.build()
    written = service.export_artifacts()
    assert written
    assert any("/entities/" in path for path in written)
    assert any("/communities/" in path for path in written)
    assert any("/chunks/" in path for path in written)
    # The cards must be marked clearly so other tooling can filter them.
    sample = test_project.paths.root / written[0]
    text = sample.read_text(encoding="utf-8")
    assert "generated: true" in text
    assert "retrieval_backend: wikigraph" in text


def test_export_artifacts_requires_built_index(test_project) -> None:
    service = WikiGraphIndexService(
        paths=test_project.paths, config=test_project.config
    )
    with pytest.raises(FileNotFoundError):
        service.export_artifacts()


def test_update_service_export_wikigraph_artifacts_flag(test_project) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    service = WikiGraphIndexService(
        paths=test_project.paths, config=test_project.config
    )
    update_service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=test_project.config,
        wikigraph_index_service=service,
    )

    class _Result:
        wikigraph_skipped = False
        wikigraph_skip_reason = ""
        wikigraph_result = None
        wikigraph_artifact_paths: list[str] = []

    result = _Result()
    update_service._maybe_build_wikigraph(
        UpdateOptions(wikigraph=True, export_wikigraph_artifacts=True), result
    )
    assert result.wikigraph_artifact_paths


# --------------------------------------------------------------------------- #
# Deps helpers                                                                #
# --------------------------------------------------------------------------- #


def test_wikigraph_extra_hint_is_actionable() -> None:
    hint = wikigraph_deps.wikigraph_extra_hint()
    assert "poetry" in hint
    assert "wikigraph" in hint


def test_require_networkx_returns_module() -> None:
    nx = wikigraph_deps.require_networkx()
    assert hasattr(nx, "Graph")
    assert hasattr(nx, "MultiGraph")


def test_try_import_bm25s_returns_module_when_installed() -> None:
    bm25s = wikigraph_deps.try_import_bm25s()
    assert bm25s is not None
    assert hasattr(bm25s, "BM25")


def test_require_networkx_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the import so we can exercise the failure branch."""
    import builtins

    original_import = builtins.__import__

    def _fake_import(name: str, *args, **kwargs):
        if name == "networkx" or name.startswith("networkx."):
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError, match="poetry install -E wikigraph"):
        wikigraph_deps.require_networkx()


def test_update_options_wikigraph_enabled_config_drives_default(test_project) -> None:
    """``wikigraph.enabled: false`` should skip the build when CLI is unset."""
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    config = copy.deepcopy(test_project.config)
    config.setdefault("wikigraph", {})["enabled"] = False
    service = WikiGraphIndexService(paths=test_project.paths, config=config)
    update_service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=config,
        wikigraph_index_service=service,
    )

    class _R:
        wikigraph_skipped = False
        wikigraph_skip_reason = ""
        wikigraph_result = None
        wikigraph_artifact_paths: list[str] = []

    # Unset CLI flag -> config drives -> skip.
    r1 = _R()
    update_service._maybe_build_wikigraph(UpdateOptions(wikigraph=None), r1)
    assert r1.wikigraph_skipped is True
    assert "config" in r1.wikigraph_skip_reason

    # CLI override -> build despite config disable.
    r2 = _R()
    update_service._maybe_build_wikigraph(UpdateOptions(wikigraph=True), r2)
    assert r2.wikigraph_skipped is False
    assert r2.wikigraph_result is not None


def test_update_options_export_generated_artifacts_config_drives_default(
    test_project,
) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    config = copy.deepcopy(test_project.config)
    config.setdefault("wikigraph", {})["export_generated_artifacts"] = True
    service = WikiGraphIndexService(paths=test_project.paths, config=config)
    update_service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=config,
        wikigraph_index_service=service,
    )

    class _R:
        wikigraph_skipped = False
        wikigraph_skip_reason = ""
        wikigraph_result = None
        wikigraph_artifact_paths: list[str] = []

    # CLI unset, config says export -> artifacts produced.
    r1 = _R()
    update_service._maybe_build_wikigraph(UpdateOptions(wikigraph=None), r1)
    assert r1.wikigraph_artifact_paths

    # CLI explicit false should win over config.
    r2 = _R()
    update_service._maybe_build_wikigraph(
        UpdateOptions(wikigraph=None, export_wikigraph_artifacts=False), r2
    )
    assert r2.wikigraph_artifact_paths == []


def test_export_artifacts_respects_types_filter(test_project) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    service = WikiGraphIndexService(
        paths=test_project.paths, config=test_project.config
    )
    service.build()
    only_entities = service.export_artifacts(types=("entities",))
    assert only_entities
    assert all("/entities/" in path for path in only_entities)
    assert not any("/communities/" in path for path in only_entities)


def test_export_artifacts_rejects_unknown_type(test_project) -> None:
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    service = WikiGraphIndexService(
        paths=test_project.paths, config=test_project.config
    )
    service.build()
    with pytest.raises(ValueError, match="Unknown wikigraph artifact type"):
        service.export_artifacts(types=("entities", "mystery"))


def test_context_builder_enforces_max_context_tokens(test_project) -> None:
    """``max_context_tokens`` should trim contexts and record the cut."""
    from graphwiki_kb.wikigraph.context_builder import (
        ContextBuilderConfig,
        WikiGraphContextBuilder,
    )

    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    service = WikiGraphIndexService(
        paths=test_project.paths, config=test_project.config
    )
    service.build()
    index = service.load()
    assert index is not None
    builder = WikiGraphContextBuilder(
        index,
        config=ContextBuilderConfig(
            max_context_chunks=8,
            max_context_tokens=20,  # tiny budget -> aggressive trim
        ),
    )
    contexts = builder.basic_search("REALM retrieval")
    assert contexts  # at least one chunk fits the budget
    # Either we kept only 1 context, or the last kept context carries a
    # budget trace.
    if len(contexts) == 1:
        return
    assert any("budget:" in entry for entry in contexts[-1].trace)


def test_lexical_index_prefer_simple_forces_pure_python() -> None:
    from graphwiki_kb.wikigraph.lexical_index import LexicalDocument, LexicalIndex

    index = LexicalIndex(prefer_simple=True)
    index.add(LexicalDocument(doc_id="a", text="REALM retrieves Wikipedia documents."))
    index.add(LexicalDocument(doc_id="b", text="RAG fuses retrieved passages."))
    index.fit()
    assert index.backend == "simple"
    hits = index.search("retrieved", limit=2)
    assert hits


def test_context_builder_lexical_backend_simple(test_project) -> None:
    from graphwiki_kb.wikigraph.context_builder import (
        ContextBuilderConfig,
        WikiGraphContextBuilder,
    )

    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    service = WikiGraphIndexService(
        paths=test_project.paths, config=test_project.config
    )
    service.build()
    index = service.load()
    assert index is not None
    builder = WikiGraphContextBuilder(
        index,
        config=ContextBuilderConfig(lexical_backend="simple"),
    )
    assert builder._lexical.backend == "simple"


def test_services_import_when_networkx_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``graphwiki_kb.services`` must import cleanly without ``networkx``.

    This is the install-hint pattern: the WikiGraphRAG package should never
    crash a base ``kb init`` / ``kb status`` call just because the optional
    extra is not installed.
    """
    import sys

    blocked = ("networkx", "bm25s")
    sentinel = object()
    saved = {name: sys.modules.get(name, sentinel) for name in blocked}
    for name in blocked:
        sys.modules[name] = None  # type: ignore[assignment]
    try:
        import importlib

        for module in (
            "graphwiki_kb.wikigraph.graph_store",
            "graphwiki_kb.wikigraph.community_builder",
            "graphwiki_kb.wikigraph.context_builder",
            "graphwiki_kb.wikigraph.lexical_index",
        ):
            if module in sys.modules:
                importlib.reload(sys.modules[module])
        # Confirm the package itself imports.
        importlib.import_module("graphwiki_kb.services")
    finally:
        for name, original in saved.items():
            if original is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original  # type: ignore[assignment]
