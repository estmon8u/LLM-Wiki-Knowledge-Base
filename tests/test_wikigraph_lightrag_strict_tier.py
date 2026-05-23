"""Tests for the LightRAG strict-tier wiring (vector reuse, LLM extractor)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from graphwiki_kb.providers.base import (
    ProviderRequest,
    ProviderResponse,
    TextProvider,
)
from graphwiki_kb.services.config_service import (
    DEFAULT_CONFIG,
    resolve_wikigraph_config,
)
from graphwiki_kb.services.project_service import build_project_paths
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryService
from graphwiki_kb.wikigraph.light_context_builder import (
    LightContextBuilder,
    LightContextBuilderConfig,
)
from graphwiki_kb.wikigraph.light_index_builder import build_lightgraph_index
from graphwiki_kb.wikigraph.light_llm_extractor import (
    EXTRACTION_SCHEMA,
    LLMLightExtractor,
    _parse_llm_response,
)
from graphwiki_kb.wikigraph.light_models import LightChunk
from tests.test_wikigraph_lightrag_e2e import (
    _seed_manifest,
    _seed_project,
)


class _ReplayProvider(TextProvider):
    """Predictable TextProvider that echoes a fixed JSON for every call."""

    name = "stub-llm"

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.requests: list[ProviderRequest] = []

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text=json.dumps(self._payload),
            model_name="stub-llm",
            provider="stub-llm",
            input_tokens=20,
            output_tokens=40,
        )


# --------------------------------------------------------------------------- #
# LLM extractor                                                               #
# --------------------------------------------------------------------------- #


def _make_chunk() -> LightChunk:
    return LightChunk(
        id="chunk:s1:0:abc",
        source_id="s1",
        source_slug="dpr",
        source_title="Dense Passage Retrieval",
        normalized_path="raw/normalized/dpr.md",
        chunk_index=0,
        token_count=20,
        text=(
            "Dense Passage Retrieval (DPR) uses a dual encoder. "
            "DPR was evaluated on Natural Questions. RAG uses DPR."
        ),
        content_hash="abc",
    )


def test_llm_extractor_round_trips_typed_entities_and_relations():
    payload = {
        "entities": [
            {
                "name": "Dense Passage Retrieval",
                "type": "METHOD",
                "aliases": ["DPR"],
                "description": "Dual-encoder retriever for open-domain QA.",
                "evidence_quote": "Dense Passage Retrieval (DPR) uses a dual encoder.",
            },
            {
                "name": "Natural Questions",
                "type": "DATASET",
                "aliases": [],
                "description": "Open-domain QA benchmark.",
                "evidence_quote": "DPR was evaluated on Natural Questions.",
            },
            {
                "name": "RAG",
                "type": "MODEL",
                "aliases": [],
                "description": "Retrieval-augmented generation model.",
                "evidence_quote": "RAG uses DPR.",
            },
        ],
        "relations": [
            {
                "source": "Dense Passage Retrieval",
                "target": "Natural Questions",
                "relation_type": "EVALUATES_ON",
                "keywords": ["open-domain QA"],
                "description": "DPR was evaluated on Natural Questions.",
                "evidence_quote": "DPR was evaluated on Natural Questions.",
            },
            {
                "source": "RAG",
                "target": "Dense Passage Retrieval",
                "relation_type": "USES",
                "keywords": ["retrieval"],
                "description": "RAG uses DPR for retrieval.",
                "evidence_quote": "RAG uses DPR.",
            },
        ],
    }
    provider = _ReplayProvider(payload)
    extractor = LLMLightExtractor(provider=provider)
    result = extractor.extract(_make_chunk())
    assert provider.requests, "LLM extractor should call the provider"
    last_request = provider.requests[-1]
    # The structured-output schema is forwarded to the provider.
    assert last_request.response_schema == EXTRACTION_SCHEMA
    assert "Dense Passage Retrieval" in last_request.prompt
    # Entity types preserved. The chunk's source title ("Dense Passage
    # Retrieval") is also auto-seeded as the PAPER anchor entity, so
    # the LLM-emitted "Dense Passage Retrieval"/METHOD is folded into
    # that single PAPER entity (deduper would later merge them anyway).
    names = {ent.name: ent.type for ent in result.entities}
    assert names == {
        "Dense Passage Retrieval": "PAPER",
        "Natural Questions": "DATASET",
        "RAG": "MODEL",
    }
    # Relations typed, not the fallback "SUPPORTS".
    rel_types = {(r.source, r.target): r.relation_type for r in result.relations}
    assert rel_types[("Dense Passage Retrieval", "Natural Questions")] == "EVALUATES_ON"
    assert rel_types[("RAG", "Dense Passage Retrieval")] == "USES"
    # Evidence quotes propagate.
    assert all(r.evidence_quote for r in result.relations)
    assert result.extractor == "llm"


def test_llm_extractor_drops_relation_with_unknown_endpoint():
    payload = {
        "entities": [
            {
                "name": "Alpha",
                "type": "METHOD",
                "aliases": [],
                "description": "",
                "evidence_quote": "",
            }
        ],
        "relations": [
            {
                "source": "Alpha",
                "target": "Beta",
                "relation_type": "USES",
                "keywords": [],
                "description": "",
                "evidence_quote": "",
            }
        ],
    }
    extractor = LLMLightExtractor(provider=_ReplayProvider(payload))
    result = extractor.extract(_make_chunk())
    assert result.relations == []
    assert any("relation_endpoint_missing" in w for w in result.warnings)


def test_llm_extractor_handles_provider_error():
    class _ExplodingProvider(TextProvider):
        name = "boom"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            raise RuntimeError("rate limit")

    extractor = LLMLightExtractor(provider=_ExplodingProvider())
    result = extractor.extract(_make_chunk())
    assert result.entities == []
    assert result.relations == []
    assert any(w.startswith("provider_error:") for w in result.warnings)


def test_llm_extractor_handles_invalid_json():
    class _GarbageProvider(TextProvider):
        name = "garbage"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse(
                text="this is not json at all",
                model_name="garbage",
                provider="garbage",
            )

    extractor = LLMLightExtractor(provider=_GarbageProvider())
    result = extractor.extract(_make_chunk())
    assert any(w.startswith("invalid_json:") for w in result.warnings)
    assert result.entities == []


def test_llm_extractor_strips_markdown_code_fence():
    payload = {"entities": [], "relations": []}

    class _FencedProvider(TextProvider):
        name = "fenced"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse(
                text=f"```json\n{json.dumps(payload)}\n```",
                model_name="fenced",
                provider="fenced",
            )

    extractor = LLMLightExtractor(provider=_FencedProvider())
    result = extractor.extract(_make_chunk())
    assert result.warnings == []


def test_llm_extractor_coerces_unknown_entity_type():
    payload = {
        "entities": [
            {
                "name": "BERT",
                "type": "TRANSFORMER",  # not in the allowed type set
                "aliases": [],
                "description": "",
                "evidence_quote": "",
            }
        ],
        "relations": [],
    }
    result = _parse_llm_response(
        _make_chunk(),
        json.dumps(payload),
        warnings=[],
        extractor_name="llm",
        entity_types=("MODEL", "METHOD", "CLAIM"),
        relation_types=("USES", "SUPPORTS"),
    )
    assert result.entities[0].type in {"CLAIM", "MODEL", "METHOD"}


def test_llm_extractor_prompt_hash_changes_when_options_change():
    a = LLMLightExtractor(provider=_ReplayProvider({"entities": [], "relations": []}))
    b = LLMLightExtractor(
        provider=_ReplayProvider({"entities": [], "relations": []}),
        max_chunk_chars=1234,
    )
    assert a.prompt_hash != b.prompt_hash


# --------------------------------------------------------------------------- #
# Strict-tier query path                                                      #
# --------------------------------------------------------------------------- #


class _DimEmbeddingProvider:
    """Deterministic strict-tier provider used to exercise vector reuse."""

    model_name = "dim-stub"

    def __init__(self, dimension: int = 6) -> None:
        self._dim = dimension
        self.calls: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        out: list[list[float]] = []
        for idx, text in enumerate(texts):
            base = [0.0] * self._dim
            slot = (idx + len(text)) % self._dim
            base[slot] = 1.0
            out.append(base)
        return out


def test_light_context_builder_skips_corpus_embed_with_precomputed_vectors():
    from graphwiki_kb.wikigraph.light_models import (
        EntityProfile,
        LightGraphBuildManifest,
        LightGraphIndex,
        RelationProfile,
    )

    chunk = _make_chunk()
    entity = EntityProfile(
        id="entity:dpr",
        canonical_name="DPR",
        type="METHOD",
        chunk_ids=[chunk.id],
        source_ids=[chunk.source_id],
        embedding_text="DPR METHOD",
    )
    relation = RelationProfile(
        id="relation:rag-uses-dpr",
        source_entity_id=entity.id,
        target_entity_id=entity.id,
        relation_type="USES",
        chunk_ids=[chunk.id],
        source_ids=[chunk.source_id],
        embedding_text="USES",
    )
    index = LightGraphIndex(
        built_at="2024-01-01T00:00:00Z",
        chunks=[chunk],
        entities=[entity],
        relations=[relation],
        manifest=LightGraphBuildManifest(built_at="2024-01-01T00:00:00Z"),
    )
    provider = _DimEmbeddingProvider(dimension=4)
    builder = LightContextBuilder(
        index=index,
        config=LightContextBuilderConfig(),
        embedding_provider=provider,
        precomputed_entity_vectors=[(entity.id, [1.0, 0.0, 0.0, 0.0])],
        precomputed_relation_vectors=[(relation.id, [0.0, 1.0, 0.0, 0.0])],
        precomputed_chunk_vectors=[(chunk.id, [0.0, 0.0, 1.0, 0.0])],
    )
    # No corpus embed calls — only the query embed call should happen
    # when retrieve() runs.
    assert provider.calls == []
    bundle = builder.retrieve("anything", method="basic")
    assert bundle.contexts, "should find at least one chunk via vector store"
    assert provider.calls, "query embedding must use the supplied provider"


def test_query_service_caches_light_engine_across_calls(tmp_path: Path):
    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    manifest_service = _seed_manifest(paths, sources)
    config = {"wikigraph": {**DEFAULT_CONFIG["wikigraph"], "mode": "lightrag"}}
    index_service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest_service
    )
    index_service.build()
    qs = WikiGraphQueryService(
        paths=paths, index_service=index_service, provider=None, config=config
    )
    engine_a = qs._ensure_light_engine()
    engine_b = qs._ensure_light_engine()
    assert engine_a is engine_b, "engine must be cached across find() calls"
    qs.invalidate_lightgraph_cache()
    engine_c = qs._ensure_light_engine()
    assert engine_c is not engine_a, "invalidate_lightgraph_cache should drop the cache"


def test_strict_tier_query_uses_persisted_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Index with a strict embedding provider, then verify the query service
    reuses the persisted vectors (no corpus refit at query time)."""
    from graphwiki_kb.services.embedding_service import ResolvedEmbedding

    root, sources = _seed_project(tmp_path)
    paths = build_project_paths(root)
    manifest_service = _seed_manifest(paths, sources)

    # Build the index with a strict stub embedder so vectors persist
    # at dimension 6 with a known model_name.
    stub = _DimEmbeddingProvider(dimension=6)
    from graphwiki_kb.services.embedding_service import EmbeddingRuntimeConfig

    resolution = ResolvedEmbedding(
        provider=stub,
        tier="strict",
        runtime=EmbeddingRuntimeConfig(
            provider="dim-stub",
            model="dim-stub",
            dimension=6,
            local_fallback="bm25",
        ),
        reason="dim-stub provider for tests",
    )
    _index, _ = build_lightgraph_index(
        paths,
        sources,
        embedding_resolution=resolution,
        store=None,  # we'll save through the index service to mimic prod
    )
    # Persist via the service so the manifest reflects the strict tier.
    config = {"wikigraph": {**DEFAULT_CONFIG["wikigraph"], "mode": "lightrag"}}
    index_service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest_service
    )

    # Monkeypatch the embedding resolver so the query service picks up
    # the same stub provider (so dimensions line up between persisted
    # vectors and the query vector). Otherwise build_embedding_provider
    # would return None (no API keys in tests).
    from graphwiki_kb.services import (
        wikigraph_index_service as wis,
    )
    from graphwiki_kb.services import (
        wikigraph_query_service as wqs,
    )

    captured_calls = {"count": 0}

    def fake_build(runtime):
        captured_calls["count"] += 1
        return resolution

    monkeypatch.setattr(wqs, "build_embedding_provider", fake_build)
    monkeypatch.setattr(wis, "build_embedding_provider", fake_build)

    # Build the lightrag index through the production path so it
    # persists with the same provider.
    index_service.build()
    # Sanity: the manifest should record the strict tier from the build.
    loaded = index_service.load_lightgraph()
    assert loaded is not None
    assert loaded.manifest.embedding_tier == "strict"

    qs = WikiGraphQueryService(
        paths=paths, index_service=index_service, provider=None, config=config
    )
    engine = qs._ensure_light_engine()
    assert engine.precomputed_entity_vectors, "should reuse persisted entity vectors"
    assert (
        engine.precomputed_relation_vectors
    ), "should reuse persisted relation vectors"
    assert (
        captured_calls["count"] >= 1
    ), "build_embedding_provider must be invoked at query time"


def test_light_context_builder_skips_query_when_dimension_mismatches():
    """Defensive: a dim-mismatched query vector returns no hits, not a crash."""
    from graphwiki_kb.wikigraph.light_models import (
        EntityProfile,
        LightGraphBuildManifest,
        LightGraphIndex,
    )

    entity = EntityProfile(
        id="entity:dpr",
        canonical_name="DPR",
        type="METHOD",
        embedding_text="DPR METHOD",
    )
    index = LightGraphIndex(
        built_at="2024-01-01T00:00:00Z",
        entities=[entity],
        manifest=LightGraphBuildManifest(built_at="2024-01-01T00:00:00Z"),
    )

    class _WrongDimProvider:
        model_name = "wrong"
        _dim = 2

        @property
        def dimension(self) -> int:
            return self._dim

        def embed_texts(self, texts):
            return [[1.0, 0.0] for _ in texts]

    builder = LightContextBuilder(
        index=index,
        config=LightContextBuilderConfig(),
        embedding_provider=_WrongDimProvider(),
        precomputed_entity_vectors=[("entity:dpr", [1.0, 0.0, 0.0, 0.0])],
    )
    bundle = builder.retrieve("anything", method="local")
    # Mismatched dim -> empty result, not an exception.
    assert bundle.entities == []


def test_extraction_config_round_trip_runtime():
    runtime = resolve_wikigraph_config(
        {
            "wikigraph": {
                **DEFAULT_CONFIG["wikigraph"],
                "lightrag": {
                    **DEFAULT_CONFIG["wikigraph"]["lightrag"],
                    "extraction": {
                        "extractor": "llm",
                        "provider": "openai",
                        "max_tokens": 4096,
                        "max_chunk_chars": 4000,
                    },
                },
            }
        }
    )
    assert runtime.lightrag.extraction.extractor == "llm"
    assert runtime.lightrag.extraction.provider == "openai"
    assert runtime.lightrag.extraction.max_tokens == 4096
    assert runtime.lightrag.extraction.max_chunk_chars == 4000


def test_extraction_config_default_is_deterministic():
    runtime = resolve_wikigraph_config(DEFAULT_CONFIG)
    assert runtime.lightrag.extraction.extractor == "deterministic"
    assert runtime.lightrag.extraction.provider is None


def test_v8_migration_includes_extraction_section(tmp_path: Path):
    from copy import deepcopy

    import yaml

    from graphwiki_kb.services.config_service import (
        CURRENT_CONFIG_VERSION,
        ConfigService,
    )

    legacy = deepcopy(DEFAULT_CONFIG)
    legacy["wikigraph"].pop("lightrag", None)
    legacy["wikigraph"].pop("mode", None)
    legacy["version"] = 8
    paths = build_project_paths(tmp_path)
    paths.config_file.write_text(yaml.safe_dump(legacy), encoding="utf-8")
    loaded = ConfigService(paths).load()
    assert loaded["version"] == CURRENT_CONFIG_VERSION
    extraction = loaded["wikigraph"]["lightrag"]["extraction"]
    assert extraction["extractor"] == "deterministic"
    assert extraction["max_tokens"] == 2048
