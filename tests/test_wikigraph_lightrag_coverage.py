"""Targeted coverage tests for the LightRAG query / answer / store paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphwiki_kb.providers.base import (
    ProviderRequest,
    ProviderResponse,
    TextProvider,
)
from graphwiki_kb.services.project_service import build_project_paths
from graphwiki_kb.services.wikigraph_query_service import (
    classic_method_to_light as _classic_method_to_light,
)
from graphwiki_kb.wikigraph.light_context_builder import (
    LightContextBuilder,
    LightContextBuilderConfig,
    _rrf_fuse,
)
from graphwiki_kb.wikigraph.light_embeddings import (
    BM25SparseEmbeddingProvider,
    HashingEmbeddingProvider,
    default_embedding_provider,
)
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
)
from graphwiki_kb.wikigraph.light_index_builder import (
    build_lightgraph_index,
)
from graphwiki_kb.wikigraph.light_keywords import RuleBasedKeywordProvider
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphBuildManifest,
    LightGraphIndex,
    LightRetrievedContext,
)
from graphwiki_kb.wikigraph.light_query_service import (
    LightAnswerService,
    LightGraphQueryEngine,
)
from graphwiki_kb.wikigraph.light_vector_store import LightVectorStore
from tests.test_wikigraph_lightrag_e2e import _seed_project


class StubProvider(TextProvider):
    """Minimal :class:`TextProvider` that echoes a fixed answer."""

    name = "stub"

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.requests: list[ProviderRequest] = []

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text=self._response_text,
            model_name="stub-model",
            provider="stub",
            input_tokens=10,
            output_tokens=20,
        )


def test_default_embedding_provider_falls_back_to_hashing():
    provider = default_embedding_provider()
    assert isinstance(provider, HashingEmbeddingProvider)
    vectors = provider.embed_texts(["hello"])
    assert len(vectors[0]) == provider.dimension


def test_default_embedding_provider_uses_bm25_with_corpus():
    provider = default_embedding_provider(corpus=["a b c", "b c d"])
    assert isinstance(provider, BM25SparseEmbeddingProvider)
    assert provider.dimension > 0


def test_hashing_embedding_provider_rejects_zero_dimension():
    with pytest.raises(ValueError):
        HashingEmbeddingProvider(dimension=0)


def test_bm25_provider_raises_when_fit_not_called():
    provider = BM25SparseEmbeddingProvider()
    with pytest.raises(RuntimeError):
        provider.embed_texts(["hello"])


def test_bm25_provider_handles_empty_corpus():
    provider = BM25SparseEmbeddingProvider()
    provider.fit([])
    assert provider.dimension == 1
    vectors = provider.embed_texts(["anything"])
    assert vectors == [[0.0]]


def test_bm25_provider_handles_corpus_of_empty_strings():
    provider = BM25SparseEmbeddingProvider()
    provider.fit([""])
    assert provider.dimension == 1
    assert provider.embed_texts([""]) == [[0.0]]


def test_vector_store_empty_search_returns_empty():
    store = LightVectorStore()
    assert store.search([0.1, 0.2], top_k=5) == []
    store.add("a", [1.0, 0.0])
    assert store.search([], top_k=5) == []
    assert store.search([1.0, 0.0], top_k=0) == []


def test_vector_store_dimension_and_ids_properties():
    store = LightVectorStore()
    assert store.dimension == 0
    store.add_many([("a", [1.0, 0.0]), ("b", [0.0, 1.0])])
    assert store.dimension == 2
    assert store.ids == ("a", "b")


def test_light_graph_store_load_returns_none_when_missing(tmp_path: Path):
    store = LightGraphStore(LightGraphStorePaths(tmp_path / "lr"))
    assert store.load() is None
    assert store.load_vectors("chunk") == []


def test_light_graph_store_load_returns_none_on_corrupt_file(tmp_path: Path):
    store = LightGraphStore(LightGraphStorePaths(tmp_path / "lr"))
    store.paths.root.mkdir(parents=True, exist_ok=True)
    store.paths.manifest_file.write_text("{not json")
    store.paths.index_file.write_text("{}")
    assert store.load() is None


def test_rrf_fuse_combines_ranks_across_lists():
    a = LightRetrievedContext(kind="entity", id="a", title="A", score=0.0)
    b = LightRetrievedContext(kind="entity", id="b", title="B", score=0.0)
    c = LightRetrievedContext(kind="entity", id="c", title="C", score=0.0)
    fused = _rrf_fuse(
        [[a, b, c], [a, c, b]],
        k=60,
        weights=[1.0, 1.0],
    )
    ids = [ctx.id for ctx in fused]
    assert set(ids) == {"a", "b", "c"}
    # ``a`` appears at rank 0 in both lists so it always wins.
    assert ids[0] == "a"
    # Returned contexts carry the fused score.
    assert fused[0].score > 0


def test_rrf_fuse_handles_uneven_weights():
    a = LightRetrievedContext(kind="entity", id="a", title="A", score=0.0)
    b = LightRetrievedContext(kind="entity", id="b", title="B", score=0.0)
    fused = _rrf_fuse(
        [[a], [b]],
        k=10,
        weights=[2.0, 1.0],
    )
    assert fused[0].id == "a"


def test_light_context_builder_routing_drift_lite(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    index, _ = build_lightgraph_index(paths, sources)
    builder = LightContextBuilder(
        index=index, config=LightContextBuilderConfig(top_k_chunks=3)
    )
    bundle = builder.retrieve("Explore DPR and RAG", method="drift-lite")
    assert bundle.method == "drift-lite"


def test_light_answer_service_provider_backed_validates_citations(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    index, _ = build_lightgraph_index(paths, sources)
    engine = LightGraphQueryEngine(index=index)

    response = "Answer body that cites [C1] and [C99]. The latter is invalid."
    provider = StubProvider(response)
    service = LightAnswerService(engine=engine, provider=provider)
    answer = service.ask("Compare RAG and DPR", method="hybrid")
    assert provider.requests, "expected provider.generate() to be called"
    assert answer.provider_status["mode"] == "provider"
    # Valid C1 was matched; invalid C99 triggered a warning.
    assert any("invalid_citations" in w for w in answer.warnings)
    assert any(c["ref"] for c in answer.citations)


def test_light_answer_service_no_provider_with_require_provider(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    index, _ = build_lightgraph_index(paths, sources)
    engine = LightGraphQueryEngine(index=index)
    service = LightAnswerService(engine=engine, provider=None)
    answer = service.ask("Anything", method="local", require_provider=True)
    assert answer.insufficient_evidence is True
    assert answer.provider_status["mode"] == "provider-required"


def test_classic_method_to_light_maps_drift_lite_to_hybrid():
    assert _classic_method_to_light("drift-lite") == "hybrid"
    assert _classic_method_to_light("local") == "local"
    assert _classic_method_to_light("auto") == "auto"


def test_light_answer_service_provider_free_with_empty_index(tmp_path: Path):
    """Provider-free path returns ``insufficient_evidence`` with an empty index."""
    paths = build_project_paths(tmp_path)
    (tmp_path / "raw" / "normalized").mkdir(parents=True)
    (tmp_path / "graph").mkdir()
    index, _ = build_lightgraph_index(paths, [])
    engine = LightGraphQueryEngine(index=index)
    service = LightAnswerService(engine=engine, provider=None)
    answer = service.ask("Anything", method="basic")
    assert answer.insufficient_evidence is True


def test_keyword_provider_handles_acronym_in_aliases():
    provider = RuleBasedKeywordProvider(known_aliases=("DPR",))
    result = provider.extract("Is DPR worth using?")
    assert "DPR" in result.low_level_keywords


def test_light_answer_service_provider_response_without_citations(tmp_path: Path):
    """Provider returning no [C#] markers must yield insufficient_evidence."""
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    index, _ = build_lightgraph_index(paths, sources)
    engine = LightGraphQueryEngine(index=index)
    provider = StubProvider("Plain answer without any citation markers.")
    service = LightAnswerService(engine=engine, provider=provider)
    answer = service.ask("What is DPR?", method="local")
    assert answer.insufficient_evidence is True
    assert answer.citations == []
    assert any(w == "no_valid_citations_in_answer" for w in answer.warnings)


def test_light_answer_service_provider_says_insufficient(tmp_path: Path):
    """When the model itself says 'insufficient', mark the answer accordingly."""
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    index, _ = build_lightgraph_index(paths, sources)
    engine = LightGraphQueryEngine(index=index)
    provider = StubProvider("The evidence is insufficient. [C1] not enough detail.")
    service = LightAnswerService(engine=engine, provider=provider)
    answer = service.ask("Anything", method="hybrid")
    assert answer.insufficient_evidence is True


def test_wikigraph_index_service_status_lightrag_block_absent(tmp_path: Path):
    """When LightRAG isn't built, status reports initialized=False for that block."""
    from graphwiki_kb.services.config_service import DEFAULT_CONFIG
    from graphwiki_kb.services.wikigraph_index_service import (
        WikiGraphIndexService,
    )

    paths = build_project_paths(tmp_path)
    (tmp_path / "graph").mkdir()
    service = WikiGraphIndexService(
        paths=paths, config=DEFAULT_CONFIG, manifest_service=None
    )
    status = service.status()
    assert status["mode"] == "classic"
    light_block = status["lightrag"]
    assert isinstance(light_block, dict)
    assert light_block["initialized"] is False


def test_light_context_builder_route_branches():
    """Exercise every branch of :meth:`LightContextBuilder.route`."""
    from graphwiki_kb.wikigraph.light_keywords import QueryKeywords

    builder = LightContextBuilder(index=LightGraphIndex())
    # Hybrid keyword: 'compare'.
    assert builder.route("compare A and B", QueryKeywords()) == "hybrid"
    # Global theme keyword.
    assert builder.route("what are the main themes", QueryKeywords()) == "global"
    # Low-level keywords -> local.
    assert (
        builder.route(
            "tell me",
            QueryKeywords(low_level_keywords=["DPR"]),
        )
        == "local"
    )
    # High-level only -> global.
    assert (
        builder.route(
            "tell me",
            QueryKeywords(high_level_keywords=["retrieval"]),
        )
        == "global"
    )
    # No signals -> basic.
    assert builder.route("blah blah", QueryKeywords()) == "basic"


def test_light_context_builder_apply_token_budget_disabled():
    from graphwiki_kb.wikigraph.light_context_builder import (
        _apply_token_budget,
    )

    contexts = [
        LightRetrievedContext(
            kind="chunk", id=f"c{i}", title="t", score=0.1, text="x" * 200
        )
        for i in range(3)
    ]
    # With budget <= 0, the function is a no-op.
    assert _apply_token_budget(contexts, max_total_tokens=0) == contexts
    # With a tiny budget, truncates after the first one.
    kept = _apply_token_budget(contexts, max_total_tokens=60)
    assert len(kept) < len(contexts)


def test_light_context_builder_vector_for_query_falls_back_on_runtime_error():
    builder = LightContextBuilder(index=LightGraphIndex())

    class Broken:
        model_name = "broken"
        dimension = 1

        def embed_texts(self, texts):
            raise RuntimeError("nope")

    builder.embedding_provider = Broken()
    assert builder._vector_for_query("hi") == []


def test_light_context_builder_basic_search_with_empty_index():
    builder = LightContextBuilder(index=LightGraphIndex())
    bundle = builder.retrieve("anything", method="basic")
    assert bundle.method == "basic"
    assert bundle.contexts == []


def test_is_acceptable_entity_helper_rejects_short_or_stopword():
    from graphwiki_kb.wikigraph.light_extractor import _is_acceptable_entity

    assert _is_acceptable_entity("BERT") is True
    assert _is_acceptable_entity("") is False
    assert _is_acceptable_entity("a") is False  # too short
    assert _is_acceptable_entity("1234") is False  # no letters
    assert _is_acceptable_entity("the") is False  # stopword


def test_short_quote_fallback_when_mention_not_found():
    from graphwiki_kb.wikigraph.light_extractor import _short_quote

    assert _short_quote("hello world", "missing") == "missing"


def test_extraction_cache_returns_none_on_corrupt_file(tmp_path: Path):
    from graphwiki_kb.wikigraph.light_extractor import (
        DeterministicLightExtractor,
        LightExtractionCache,
    )

    cache = LightExtractionCache(tmp_path / "cache")
    chunk = LightChunk(
        id="chunk:s:0:a",
        source_id="s",
        source_slug="d",
        source_title="D",
        normalized_path="n.md",
        chunk_index=0,
        token_count=1,
        text="hi",
        content_hash="a",
    )
    extractor = DeterministicLightExtractor()
    # Write garbage at the cache key.
    key = cache._key(chunk, extractor.prompt_hash)
    (tmp_path / "cache" / f"{key}.json").write_text("{not-json")
    assert cache.get(chunk, extractor.prompt_hash) is None


def test_deterministic_extractor_skips_unknown_entity_types():
    """When source_title type is not in the configured set, falls back."""
    from graphwiki_kb.wikigraph.light_extractor import (
        DeterministicLightExtractor,
        LightExtractorOptions,
    )

    chunk = LightChunk(
        id="chunk:s:0:x",
        source_id="s",
        source_slug="paper",
        source_title="MyPaper",
        normalized_path="n.md",
        chunk_index=0,
        token_count=2,
        text="MyPaper introduces FOO and BAR.",
        content_hash="x",
    )
    # Configure with no PAPER, no CLAIM; default falls back to first type.
    extractor = DeterministicLightExtractor(
        options=LightExtractorOptions(
            entity_types=("METHOD", "DATASET"),
            relation_types=("USES",),
        )
    )
    result = extractor.extract(chunk)
    types = {e.type for e in result.entities}
    assert types.issubset({"METHOD", "DATASET", "MODEL"})


def test_lightchunker_skips_unreadable_file(tmp_path: Path, monkeypatch):
    from graphwiki_kb.models.source_models import RawSourceRecord
    from graphwiki_kb.wikigraph.light_chunker import build_light_chunks

    paths = build_project_paths(tmp_path)
    (tmp_path / "raw" / "normalized").mkdir(parents=True)
    (tmp_path / "raw" / "normalized" / "doc.md").write_text("hello")

    record = RawSourceRecord(
        source_id="s",
        slug="doc",
        title="Doc",
        origin="local",
        source_type="paper",
        raw_path="raw/sources/doc.pdf",
        content_hash="h",
        ingested_at="2024-01-01T00:00:00Z",
        normalized_path="raw/normalized/doc.md",
    )

    real_read_text = Path.read_text

    def boom(self, *args, **kwargs):
        if "doc.md" in str(self):
            raise OSError("simulated")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", boom)
    chunks = build_light_chunks(root=paths.root, sources=[record])
    assert chunks == []


def test_lightchunker_skips_empty_text(tmp_path: Path):
    from graphwiki_kb.models.source_models import RawSourceRecord
    from graphwiki_kb.wikigraph.light_chunker import build_light_chunks

    paths = build_project_paths(tmp_path)
    (tmp_path / "raw" / "normalized").mkdir(parents=True)
    (tmp_path / "raw" / "normalized" / "doc.md").write_text("   \n\n   ")
    record = RawSourceRecord(
        source_id="s",
        slug="doc",
        title="Doc",
        origin="local",
        source_type="paper",
        raw_path="raw/sources/doc.pdf",
        content_hash="h",
        ingested_at="2024-01-01T00:00:00Z",
        normalized_path="raw/normalized/doc.md",
    )
    assert build_light_chunks(root=paths.root, sources=[record]) == []


def test_light_graph_store_saves_chunk_vectors_and_unknown_kind(tmp_path: Path):
    """Cover chunk-vector save + load_vectors(unknown_kind) + _rel fallback."""
    from graphwiki_kb.wikigraph.light_graph_store import (
        LightGraphStore,
        LightGraphStorePaths,
    )

    store = LightGraphStore(LightGraphStorePaths(tmp_path / "lr"))
    manifest = LightGraphBuildManifest(built_at="2024-01-01T00:00:00Z")
    index = LightGraphIndex(manifest=manifest)
    artifacts = store.save(
        index,
        entity_vectors=[],
        relation_vectors=[],
        chunk_vectors=[("c1", [0.1, 0.2])],
    )
    assert any("chunk_vectors" in a for a in artifacts)
    # Unknown kind returns [].
    assert store.load_vectors("unknown") == []
    # Loading the saved chunk vectors works.
    loaded = store.load_vectors("chunk")
    assert loaded[0][0] == "c1"


def test_light_graph_store_load_vectors_handles_corrupt_payload(tmp_path: Path):
    from graphwiki_kb.wikigraph.light_graph_store import (
        LightGraphStore,
        LightGraphStorePaths,
    )

    store = LightGraphStore(LightGraphStorePaths(tmp_path / "lr"))
    store.paths.root.mkdir(parents=True, exist_ok=True)
    store.paths.entity_vectors_file.write_text("{not-json")
    assert store.load_vectors("entity") == []


def test_light_context_builder_handles_missing_entity_during_lookup():
    """When entity/relation/chunk vector hits a stale id, the builder skips."""
    from graphwiki_kb.wikigraph.light_context_builder import (
        LightContextBuilder,
    )

    # Build a real index, then mutate it to remove entities but keep
    # vectors. This forces ``_entity_by_id.get(hit.id)`` to return None.
    chunk = LightChunk(
        id="chunk:s:0:x",
        source_id="s",
        source_slug="d",
        source_title="D",
        normalized_path="n.md",
        chunk_index=0,
        token_count=1,
        text="hello world",
        content_hash="x",
    )
    entity = EntityProfile(
        id="entity:doc",
        canonical_name="Doc",
        type="PAPER",
        chunk_ids=[chunk.id],
        embedding_text="Doc PAPER",
    )
    index = LightGraphIndex(
        chunks=[chunk],
        entities=[entity],
        manifest=LightGraphBuildManifest(built_at="2024-01-01T00:00:00Z"),
    )
    builder = LightContextBuilder(index=index)
    # Inject a stale id into the entity vector store.
    builder._entity_store.add("entity:missing", [0.0] * builder._entity_store.dimension)
    bundle = builder.retrieve("Doc", method="local")
    # The stale entry was skipped; original entity still returned.
    assert any(c.id == "entity:doc" for c in bundle.contexts)


def test_light_context_builder_global_query_with_no_relations(tmp_path: Path):
    """When no relations exist, global mode still returns a valid bundle."""
    from graphwiki_kb.wikigraph.light_context_builder import (
        LightContextBuilder,
    )

    index = LightGraphIndex(
        manifest=LightGraphBuildManifest(built_at="2024-01-01T00:00:00Z"),
    )
    builder = LightContextBuilder(index=index)
    bundle = builder.retrieve("main themes", method="global")
    assert bundle.method == "global"
    assert bundle.relations == []


def test_keyword_provider_skips_stopword_capitalized_phrases():
    from graphwiki_kb.wikigraph.light_keywords import RuleBasedKeywordProvider

    provider = RuleBasedKeywordProvider()
    # "The" is a stopword and should be filtered out of low-level keys.
    result = provider.extract("The")
    assert "The" not in result.low_level_keywords


def test_light_chunker_handles_very_short_text(tmp_path: Path):
    from graphwiki_kb.models.source_models import RawSourceRecord
    from graphwiki_kb.wikigraph.light_chunker import (
        LightChunkerOptions,
        build_light_chunks,
    )

    paths = build_project_paths(tmp_path)
    (tmp_path / "raw" / "normalized").mkdir(parents=True)
    (tmp_path / "raw" / "normalized" / "tiny.md").write_text("Hi.")
    record = RawSourceRecord(
        source_id="tiny",
        slug="tiny",
        title="Tiny",
        origin="local",
        source_type="paper",
        raw_path="raw/sources/tiny.pdf",
        content_hash="h",
        ingested_at="2024-01-01T00:00:00Z",
        normalized_path="raw/normalized/tiny.md",
    )
    chunks = build_light_chunks(
        root=paths.root,
        sources=[record],
        options=LightChunkerOptions(
            chunk_token_size=10, overlap_tokens=0, min_tokens=1
        ),
    )
    assert len(chunks) == 1
    assert chunks[0].text == "Hi."
