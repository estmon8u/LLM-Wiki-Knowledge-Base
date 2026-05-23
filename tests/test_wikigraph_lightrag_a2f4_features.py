"""Tests for the embedding service, export service, tier labeling, and true incremental update."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from graphwiki_kb.providers.embedding_base import EmbeddingProvider
from graphwiki_kb.providers.gemini_embedding import GeminiEmbeddingProvider
from graphwiki_kb.providers.openai_embedding import OpenAIEmbeddingProvider
from graphwiki_kb.services.embedding_service import (
    EmbeddingRuntimeConfig,
    build_embedding_provider,
    resolve_lightrag_embedding_config,
)
from graphwiki_kb.services.project_service import build_project_paths
from graphwiki_kb.services.wikigraph_light_export_service import (
    WikiGraphLightExportService,
    _group_chunks_by_slug,
)
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
)
from graphwiki_kb.wikigraph.light_index_builder import (
    build_lightgraph_index,
)
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphBuildManifest,
    LightGraphIndex,
    RelationProfile,
    SourceContribution,
)
from tests.test_wikigraph_lightrag_e2e import _seed_project, _source

# --------------------------------------------------------------------------- #
# Embedding service                                                           #
# --------------------------------------------------------------------------- #


def test_resolve_lightrag_embedding_config_uses_lightrag_block():
    config = {
        "wikigraph": {
            "lightrag": {
                "embeddings": {
                    "provider": "openai",
                    "model": "text-embedding-3-large",
                    "dimension": 3072,
                    "local_fallback": "bm25",
                    "api_key_env": "MY_KEY",
                }
            }
        }
    }
    runtime = resolve_lightrag_embedding_config(config)
    assert runtime.provider == "openai"
    assert runtime.model == "text-embedding-3-large"
    assert runtime.dimension == 3072
    assert runtime.api_key_env == "MY_KEY"


def test_resolve_lightrag_embedding_config_handles_missing_block():
    runtime = resolve_lightrag_embedding_config({})
    assert runtime.provider == "bm25"
    assert runtime.model == "bm25-fallback"
    assert runtime.api_key_env is None


def test_resolve_lightrag_embedding_config_assigns_default_api_key_env():
    config = {"wikigraph": {"lightrag": {"embeddings": {"provider": "openai"}}}}
    runtime = resolve_lightrag_embedding_config(config)
    assert runtime.api_key_env == "OPENAI_API_KEY"
    gemini_config = {"wikigraph": {"lightrag": {"embeddings": {"provider": "gemini"}}}}
    gemini_runtime = resolve_lightrag_embedding_config(gemini_config)
    assert gemini_runtime.api_key_env == "GEMINI_API_KEY"


def test_build_embedding_provider_returns_fallback_for_bm25():
    runtime = EmbeddingRuntimeConfig(
        provider="bm25", model="bm25-fallback", dimension=0, local_fallback="bm25"
    )
    resolved = build_embedding_provider(runtime, environ={})
    assert resolved.provider is None
    assert resolved.tier == "fallback"
    assert "fallback" in resolved.reason


def test_build_embedding_provider_falls_back_when_api_key_missing():
    runtime = EmbeddingRuntimeConfig(
        provider="openai",
        model="text-embedding-3-large",
        dimension=3072,
        local_fallback="bm25",
        api_key_env="OPENAI_API_KEY",
    )
    resolved = build_embedding_provider(runtime, environ={})
    assert resolved.tier == "fallback"
    assert "missing API key" in resolved.reason


def test_build_embedding_provider_returns_strict_when_credentials_present():
    runtime = EmbeddingRuntimeConfig(
        provider="openai",
        model="text-embedding-3-large",
        dimension=3072,
        local_fallback="bm25",
        api_key_env="OPENAI_API_KEY",
    )
    resolved = build_embedding_provider(runtime, environ={"OPENAI_API_KEY": "sk-test"})
    assert resolved.tier == "strict"
    assert isinstance(resolved.provider, OpenAIEmbeddingProvider)
    assert isinstance(resolved.provider, EmbeddingProvider)


def test_build_embedding_provider_supports_gemini():
    runtime = EmbeddingRuntimeConfig(
        provider="gemini",
        model="text-embedding-004",
        dimension=768,
        local_fallback="bm25",
        api_key_env="GEMINI_API_KEY",
    )
    resolved = build_embedding_provider(runtime, environ={"GEMINI_API_KEY": "g-test"})
    assert resolved.tier == "strict"
    assert isinstance(resolved.provider, GeminiEmbeddingProvider)


def test_build_embedding_provider_unknown_provider_falls_back():
    runtime = EmbeddingRuntimeConfig(
        provider="cohere",
        model="something",
        dimension=1024,
        local_fallback="bm25",
        api_key_env=None,
    )
    resolved = build_embedding_provider(runtime, environ={})
    assert resolved.tier == "fallback"
    assert "unknown" in resolved.reason


def test_openai_embedding_provider_raises_without_api_key(monkeypatch):
    provider = OpenAIEmbeddingProvider(
        model_name="text-embedding-3-small", api_key_env="MISSING_KEY"
    )
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Missing API key"):
        provider.embed_texts(["hello"])


def test_openai_embedding_provider_returns_empty_for_empty_input():
    provider = OpenAIEmbeddingProvider(model_name="text-embedding-3-small")
    assert provider.embed_texts([]) == []


def test_gemini_embedding_provider_raises_without_api_key(monkeypatch):
    provider = GeminiEmbeddingProvider(
        model_name="text-embedding-004", api_key_env="MISSING_KEY"
    )
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Missing API key"):
        provider.embed_texts(["hello"])


def test_gemini_embedding_provider_returns_empty_for_empty_input():
    provider = GeminiEmbeddingProvider(model_name="text-embedding-004")
    assert provider.embed_texts([]) == []


def test_openai_embedding_provider_invokes_sdk_with_stubbed_module(monkeypatch):
    """Exercise the OpenAI provider's happy path with a stubbed SDK module."""
    import sys
    import types

    captured: dict[str, Any] = {}

    class _StubResponseItem:
        def __init__(self, index: int, embedding: list[float]) -> None:
            self.index = index
            self.embedding = embedding

    class _StubResponse:
        def __init__(self, data: list[_StubResponseItem]) -> None:
            self.data = data

    class _StubEmbeddings:
        def create(self, *, model: str, input: list[str]) -> _StubResponse:
            captured["model"] = model
            captured["input"] = list(input)
            return _StubResponse(
                [
                    _StubResponseItem(idx, [float(idx), 0.5, 0.25, 0.125])
                    for idx in range(len(input))
                ]
            )

    class _StubClient:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key
            self.embeddings = _StubEmbeddings()

    stub_module = types.SimpleNamespace(OpenAI=_StubClient)
    monkeypatch.setitem(sys.modules, "openai", stub_module)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    provider = OpenAIEmbeddingProvider(
        model_name="text-embedding-3-small", expected_dimension=4
    )
    vectors = provider.embed_texts(["alpha", "beta"])
    assert vectors == [
        [0.0, 0.5, 0.25, 0.125],
        [1.0, 0.5, 0.25, 0.125],
    ]
    assert captured["api_key"] == "sk-test"
    assert captured["model"] == "text-embedding-3-small"
    assert captured["input"] == ["alpha", "beta"]
    assert provider.dimension == 4


def test_openai_embedding_provider_raises_on_dimension_mismatch(monkeypatch):
    import sys
    import types

    class _StubResponseItem:
        def __init__(self, index: int, embedding: list[float]) -> None:
            self.index = index
            self.embedding = embedding

    class _StubResponse:
        def __init__(self, data) -> None:
            self.data = data

    class _StubEmbeddings:
        def create(self, *, model, input):
            return _StubResponse([_StubResponseItem(0, [0.1, 0.2])])

    class _StubClient:
        def __init__(self, *, api_key: str) -> None:
            self.embeddings = _StubEmbeddings()

    monkeypatch.setitem(
        sys.modules, "openai", types.SimpleNamespace(OpenAI=_StubClient)
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = OpenAIEmbeddingProvider(
        model_name="text-embedding-3-small", expected_dimension=4
    )
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        provider.embed_texts(["foo"])


def test_gemini_embedding_provider_invokes_sdk_with_stubbed_module(monkeypatch):
    """Exercise the Gemini provider's happy path with a stubbed SDK module."""
    import sys
    import types

    captured: dict[str, Any] = {}

    class _StubEmbedding:
        def __init__(self, values: list[float]) -> None:
            self.values = values

    class _StubResult:
        def __init__(self, vec: list[float]) -> None:
            self.embeddings = [_StubEmbedding(vec)]

    class _StubModels:
        def embed_content(self, *, model: str, contents: str) -> _StubResult:
            captured.setdefault("calls", []).append((model, contents))
            return _StubResult([0.5, 0.5, 0.5])

    class _StubClient:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key
            self.models = _StubModels()

    stub_genai = types.SimpleNamespace(Client=_StubClient)
    stub_google = types.SimpleNamespace(genai=stub_genai)
    monkeypatch.setitem(sys.modules, "google", stub_google)
    monkeypatch.setitem(sys.modules, "google.genai", stub_genai)
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")

    provider = GeminiEmbeddingProvider(
        model_name="text-embedding-004", expected_dimension=3
    )
    vectors = provider.embed_texts(["alpha", "beta"])
    assert vectors == [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    assert captured["api_key"] == "g-test"
    assert captured["calls"] == [
        ("text-embedding-004", "alpha"),
        ("text-embedding-004", "beta"),
    ]
    assert provider.dimension == 3


def test_gemini_embedding_provider_raises_on_dimension_mismatch(monkeypatch):
    import sys
    import types

    class _StubEmbedding:
        def __init__(self, values: list[float]) -> None:
            self.values = values

    class _StubResult:
        def __init__(self, vec: list[float]) -> None:
            self.embeddings = [_StubEmbedding(vec)]

    class _StubModels:
        def embed_content(self, *, model, contents):
            return _StubResult([0.1, 0.2])

    class _StubClient:
        def __init__(self, *, api_key: str) -> None:
            self.models = _StubModels()

    stub_genai = types.SimpleNamespace(Client=_StubClient)
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=stub_genai))
    monkeypatch.setitem(sys.modules, "google.genai", stub_genai)
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    provider = GeminiEmbeddingProvider(
        model_name="text-embedding-004", expected_dimension=4
    )
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        provider.embed_texts(["foo"])


# --------------------------------------------------------------------------- #
# Build manifest tier labeling                                                #
# --------------------------------------------------------------------------- #


class _StubEmbeddingProvider:
    """Predictable strict-tier embedding provider for tests."""

    model_name = "stub-embed"

    def __init__(self) -> None:
        self._dim = 4
        self.calls: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [[float(idx), 0.0, 0.0, 0.0] for idx, _ in enumerate(texts)]


class _BrokenEmbeddingProvider:
    """Strict-tier provider that always raises at embed-time."""

    model_name = "broken-embed"

    @property
    def dimension(self) -> int:
        return 4

    def embed_texts(self, texts):
        raise RuntimeError("API rate limit exceeded")


def test_build_lightgraph_index_records_fallback_tier_by_default(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    _, report = build_lightgraph_index(paths, sources)
    assert report.embedding_tier == "fallback"
    assert "fallback" in report.embedding_tier_reason.lower()


def test_build_lightgraph_index_records_strict_tier_when_resolution_supplied(
    tmp_path: Path,
):
    from graphwiki_kb.services.embedding_service import ResolvedEmbedding

    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    provider = _StubEmbeddingProvider()
    resolved = ResolvedEmbedding(
        provider=provider,
        tier="strict",
        runtime=EmbeddingRuntimeConfig(
            provider="stub",
            model="stub-embed",
            dimension=4,
            local_fallback="bm25",
            api_key_env=None,
        ),
        reason="stub provider for tests",
    )
    index, report = build_lightgraph_index(
        paths, sources, embedding_resolution=resolved
    )
    assert report.embedding_tier == "strict"
    assert provider.calls, "stub embedding provider should have been called"
    assert index.manifest.embedding_tier == "strict"
    assert index.manifest.embedding_provider == "stub-embed"


def test_build_lightgraph_index_degrades_to_fallback_on_provider_error(
    tmp_path: Path,
):
    from graphwiki_kb.services.embedding_service import ResolvedEmbedding

    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    resolved = ResolvedEmbedding(
        provider=_BrokenEmbeddingProvider(),
        tier="strict",
        runtime=EmbeddingRuntimeConfig(
            provider="broken",
            model="broken-embed",
            dimension=4,
            local_fallback="bm25",
            api_key_env=None,
        ),
        reason="broken provider for tests",
    )
    _, report = build_lightgraph_index(paths, sources, embedding_resolution=resolved)
    assert report.embedding_tier == "fallback"
    assert "degraded" in report.embedding_tier_reason.lower()


# --------------------------------------------------------------------------- #
# True source-level incremental update                                        #
# --------------------------------------------------------------------------- #


class _CountingExtractor:
    """Wraps the deterministic extractor and counts extract() calls."""

    name = "deterministic-counter"

    def __init__(self) -> None:
        from graphwiki_kb.wikigraph.light_extractor import (
            DeterministicLightExtractor,
        )

        self._inner = DeterministicLightExtractor()
        self.prompt_hash = self._inner.prompt_hash
        self.calls: list[str] = []

    def extract(self, chunk):
        self.calls.append(chunk.id)
        return self._inner.extract(chunk)


def test_incremental_update_reuses_chunks_for_unchanged_sources(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    extractor = _CountingExtractor()
    index_v1, report_v1 = build_lightgraph_index(
        paths, sources, extractor=extractor, use_cache=False
    )
    assert report_v1.incremental is False
    initial_chunks = report_v1.chunk_count
    initial_calls = list(extractor.calls)
    assert initial_calls

    # Modify ONLY rag.md and bump only its hash. The rebuilder should
    # leave dpr.md chunks intact and only re-chunk + re-extract rag.md.
    (tmp_path / "raw" / "normalized" / "rag.md").write_text(
        "Retrieval-Augmented Generation (RAG) now also evaluates on WebQuestions. "
        "RAG still uses DPR for retrieval. RAG uses BERT as the encoder."
    )
    sources_v2 = [
        sources[0],
        _source("rag", source_id="s2", content_hash="h-rag-v2"),
    ]
    extractor.calls.clear()
    index_v2, report_v2 = build_lightgraph_index(
        paths,
        sources_v2,
        extractor=extractor,
        previous_index=index_v1,
        use_cache=False,
    )
    assert report_v2.incremental is True
    assert report_v2.reused_source_count == 1
    assert report_v2.reprocessed_source_count == 1
    # The previous chunks for s1 (dpr.md) are reused verbatim.
    s1_chunk_ids_before = {c.id for c in index_v1.chunks if c.source_id == "s1"}
    s1_chunk_ids_after = {c.id for c in index_v2.chunks if c.source_id == "s1"}
    assert s1_chunk_ids_before == s1_chunk_ids_after
    # The extractor was called only for the changed source's chunks.
    extracted_source_ids = {
        cid.split(":")[1] for cid in extractor.calls if cid.startswith("chunk:")
    }
    assert extracted_source_ids == {"s2"}
    assert len(extractor.calls) <= initial_chunks


def test_incremental_update_with_only_new_source_does_not_reprocess_existing(
    tmp_path: Path,
):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    extractor = _CountingExtractor()
    index_v1, _ = build_lightgraph_index(
        paths, sources, extractor=extractor, use_cache=False
    )

    # Add a brand-new source.
    (tmp_path / "raw" / "normalized" / "fid.md").write_text(
        "Fusion-in-Decoder (FiD) processes retrieved passages independently."
    )
    new_source = _source("fid", source_id="s3", content_hash="h-fid")
    extractor.calls.clear()
    _, report = build_lightgraph_index(
        paths,
        [*sources, new_source],
        extractor=extractor,
        previous_index=index_v1,
        use_cache=False,
    )
    assert report.incremental is True
    assert report.reprocessed_source_count == 1
    assert report.reused_source_count == 2
    extracted_source_ids = {
        cid.split(":")[1] for cid in extractor.calls if cid.startswith("chunk:")
    }
    assert extracted_source_ids == {"s3"}


# --------------------------------------------------------------------------- #
# Wiki light export service                                                   #
# --------------------------------------------------------------------------- #


def _build_minimal_index() -> LightGraphIndex:
    chunk = LightChunk(
        id="chunk:s1:0:abc",
        source_id="s1",
        source_slug="dpr",
        source_title="DPR",
        normalized_path="raw/normalized/dpr.md",
        compiled_page_path="wiki/sources/dpr.md",
        chunk_index=0,
        token_count=10,
        text="Dense passage retrieval uses a dual encoder.",
        content_hash="abc",
    )
    entity_a = EntityProfile(
        id="entity:dpr:001",
        canonical_name="Dense Passage Retrieval",
        type="METHOD",
        aliases=["DPR"],
        description="Dual-encoder retriever for open-domain QA.",
        profile_text="Entity: Dense Passage Retrieval\nType: METHOD",
        chunk_ids=[chunk.id],
        source_ids=["s1"],
        relation_ids=["relation:rag-uses-dpr"],
        embedding_text="dense passage retrieval method",
    )
    entity_b = EntityProfile(
        id="entity:rag:002",
        canonical_name="RAG",
        type="MODEL",
        description="Retrieval-Augmented Generation model.",
        profile_text="Entity: RAG\nType: MODEL",
        chunk_ids=[chunk.id],
        source_ids=["s1"],
        relation_ids=["relation:rag-uses-dpr"],
        embedding_text="rag retrieval-augmented generation model",
    )
    relation = RelationProfile(
        id="relation:rag-uses-dpr",
        source_entity_id=entity_b.id,
        target_entity_id=entity_a.id,
        relation_type="USES",
        description="RAG uses DPR for retrieval.",
        profile_text="Relation: RAG USES DPR",
        keywords=["retrieval", "dense"],
        chunk_ids=[chunk.id],
        source_ids=["s1"],
        embedding_text="rag uses dpr",
    )
    contributions = [
        SourceContribution(source_id="s1", source_hash="h-s1"),
        SourceContribution(source_id="s-old", status="missing", requires_review=True),
    ]
    manifest = LightGraphBuildManifest(built_at="2024-01-01T00:00:00Z")
    return LightGraphIndex(
        built_at=manifest.built_at,
        chunks=[chunk],
        entities=[entity_a, entity_b],
        relations=[relation],
        contributions=contributions,
        manifest=manifest,
    )


def test_export_service_writes_entity_relation_source_and_diagnostic_cards(
    tmp_path: Path,
):
    paths = build_project_paths(tmp_path)
    (tmp_path / "graph").mkdir()
    index = _build_minimal_index()
    exporter = WikiGraphLightExportService(paths=paths)
    written = exporter.export_cards(index=index)
    rels = set(written)
    assert any(r.startswith("wiki/wikigraph/entities/") for r in rels)
    assert any(r.startswith("wiki/wikigraph/relations/") for r in rels)
    assert any(r.startswith("wiki/wikigraph/sources/") for r in rels)
    diagnostic = "wiki/wikigraph/diagnostics/stale-sources.md"
    assert diagnostic in rels
    # Entity card contains the expected frontmatter + body.
    entity_card = (
        tmp_path / "wiki" / "wikigraph" / "entities" / "dense-passage-retrieval.md"
    ).read_text()
    assert "kind: wikigraph_entity" in entity_card
    assert "engine: wikigraph-lightrag" in entity_card
    assert "entity_id: entity:dpr:001" in entity_card
    # Relation card includes the source/target entity names.
    relation_card = next(
        (tmp_path / "wiki" / "wikigraph" / "relations").iterdir()
    ).read_text()
    assert "RAG" in relation_card
    assert "Dense Passage Retrieval" in relation_card
    # Stale-sources diagnostic surfaces the missing source.
    stale = (
        tmp_path / "wiki" / "wikigraph" / "diagnostics" / "stale-sources.md"
    ).read_text()
    assert "s-old" in stale
    assert "requires_review=true" in stale


def test_export_service_raises_when_index_missing(tmp_path: Path):
    paths = build_project_paths(tmp_path)
    (tmp_path / "graph").mkdir()
    exporter = WikiGraphLightExportService(paths=paths)
    with pytest.raises(FileNotFoundError):
        exporter.export_cards()


def test_export_service_loads_index_from_store(tmp_path: Path):
    paths = build_project_paths(tmp_path)
    (tmp_path / "graph").mkdir()
    store = LightGraphStore(
        LightGraphStorePaths(paths.graph_dir / "wikigraph" / "lightrag")
    )
    store.save(_build_minimal_index())
    exporter = WikiGraphLightExportService(paths=paths, store=store)
    written = exporter.export_cards()
    assert written


def test_group_chunks_by_slug_sorts_by_chunk_index():
    chunks = [
        LightChunk(
            id=f"chunk:{slug}:{idx}:x",
            source_id="s",
            source_slug=slug,
            source_title=slug,
            normalized_path=f"raw/normalized/{slug}.md",
            chunk_index=idx,
            token_count=1,
            text="x",
            content_hash="x",
        )
        for slug, idx in [("a", 2), ("a", 0), ("b", 1)]
    ]
    grouped = _group_chunks_by_slug(chunks)
    assert [c.chunk_index for c in grouped["a"]] == [0, 2]
    assert [c.chunk_index for c in grouped["b"]] == [1]


def test_index_service_build_with_export_writes_lightrag_cards(tmp_path: Path):
    """When wikigraph.export_generated_artifacts is true, build() exports cards."""
    from graphwiki_kb.services.config_service import DEFAULT_CONFIG
    from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
    from tests.test_wikigraph_lightrag_e2e import _seed_manifest, _seed_project

    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    manifest_service = _seed_manifest(paths, sources)
    config: dict[str, Any] = {
        "wikigraph": {
            **DEFAULT_CONFIG["wikigraph"],
            "mode": "lightrag",
            "export_generated_artifacts": True,
        }
    }
    service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest_service
    )
    report = service.build()
    assert any("wiki/wikigraph/entities/" in a for a in report.artifacts)
    assert any("embedding_tier=" in w for w in report.warnings)
    status = service.status()
    light_block = status["lightrag"]
    assert isinstance(light_block, dict)
    assert light_block["embedding_tier"] == "fallback"
