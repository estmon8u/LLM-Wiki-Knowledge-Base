"""Tests for LightRAG extraction, deduplication, and profiling."""

from __future__ import annotations

import json
from pathlib import Path

from graphwiki_kb.providers.base import ProviderResponse, TextProvider
from graphwiki_kb.wikigraph.light_deduper import (
    DedupeConfig,
    dedupe_entities_and_relations,
)
from graphwiki_kb.wikigraph.light_extractor import (
    ExtractionCache,
    ExtractionConfig,
    deterministic_extract_chunk,
    extraction_prompt_hash,
    run_extraction,
)
from graphwiki_kb.wikigraph.light_models import (
    ExtractedEntity,
    ExtractedRelation,
    LightChunk,
)
from graphwiki_kb.wikigraph.light_profiler import profile_index

CONFIG = ExtractionConfig(
    entity_types=("MODEL", "METHOD", "DATASET"),
    relation_types=("USES", "EVALUATES_ON"),
    max_gleaning=0,
)


def _chunk(
    text: str, *, chunk_id: str = "chunk:s1:0:abc", source_id: str = "s1"
) -> LightChunk:
    return LightChunk(
        id=chunk_id,
        source_id=source_id,
        source_slug="dpr",
        normalized_path="raw/normalized/dpr.md",
        compiled_page_path="wiki/sources/dpr.md",
        chunk_index=0,
        token_count=20,
        text=text,
        content_hash="hash-" + chunk_id,
    )


class _ScriptedProvider(TextProvider):
    """A fake provider returning canned JSON; records prompts."""

    name = "fake"

    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = payloads
        self._calls = 0
        self.prompts: list[str] = []

    def generate(self, request) -> ProviderResponse:  # type: ignore[override]
        self.prompts.append(request.prompt)
        payload = self._payloads[min(self._calls, len(self._payloads) - 1)]
        self._calls += 1
        return ProviderResponse(text=json.dumps(payload), model_name="fake-model")


# --------------------------------------------------------------------------- #
# Provider extraction                                                         #
# --------------------------------------------------------------------------- #


def test_provider_extraction_normalizes_and_validates() -> None:
    text = "RAG uses Dense Passage Retrieval to retrieve passages."
    provider = _ScriptedProvider(
        [
            {
                "entities": [
                    {
                        "name": "RAG",
                        "type": "model",
                        "aliases": ["Retrieval-Augmented Generation"],
                        "description": "A retrieval-augmented generator.",
                        "evidence_quote": "RAG uses Dense Passage Retrieval",
                    },
                    {
                        "name": "Dense Passage Retrieval",
                        "type": "METHOD",
                        "aliases": ["DPR"],
                        "description": "A dense retriever.",
                        "evidence_quote": "Dense Passage Retrieval to retrieve",
                    },
                ],
                "relations": [
                    {
                        "source": "RAG",
                        "target": "Dense Passage Retrieval",
                        "relation_type": "uses",
                        "keywords": ["retrieval"],
                        "description": "RAG uses DPR.",
                        "evidence_quote": "RAG uses Dense Passage Retrieval",
                    },
                    {
                        "source": "RAG",
                        "target": "Unknown Thing",
                        "relation_type": "USES",
                        "keywords": [],
                        "description": "bad endpoint",
                        "evidence_quote": "RAG uses Dense Passage Retrieval",
                    },
                ],
            }
        ]
    )
    run = run_extraction(
        [_chunk(text)], config=CONFIG, provider=provider, provider_identity="fake:model"
    )
    assert run.tier == "provider"
    assert {e.name for e in run.entities} == {"RAG", "Dense Passage Retrieval"}
    assert run.entities[0].type == "MODEL"
    # The relation with a non-extracted endpoint is dropped with a warning.
    assert len(run.relations) == 1
    assert run.relations[0].relation_type == "USES"
    assert any("dropped" in w for w in run.warnings)


def test_provider_extraction_warns_on_bad_evidence_quote() -> None:
    provider = _ScriptedProvider(
        [
            {
                "entities": [
                    {
                        "name": "REALM",
                        "type": "MODEL",
                        "aliases": [],
                        "description": "",
                        "evidence_quote": "a quote that is absent entirely xyz",
                    }
                ],
                "relations": [],
            }
        ]
    )
    run = run_extraction(
        [_chunk("REALM is a retrieval model.")],
        config=CONFIG,
        provider=provider,
        provider_identity="fake:model",
    )
    assert any("evidence quote not found" in w for w in run.warnings)


def test_provider_failure_falls_back_to_deterministic() -> None:
    class _BoomProvider(TextProvider):
        name = "boom"

        def generate(self, request):  # type: ignore[override]
            raise RuntimeError("provider down")

    run = run_extraction(
        [_chunk("Dense Passage Retrieval and REALM are retrieval methods.")],
        config=CONFIG,
        provider=_BoomProvider(),
        provider_identity="boom:model",
    )
    # Provider tier was attempted; fallback produced entities + a warning.
    assert run.entities
    assert any("provider extraction failed" in w for w in run.warnings)


def test_gleaning_makes_additional_calls() -> None:
    provider = _ScriptedProvider(
        [
            {
                "entities": [
                    {
                        "name": "RAG",
                        "type": "MODEL",
                        "aliases": [],
                        "description": "",
                        "evidence_quote": "RAG",
                    }
                ],
                "relations": [],
            },
            {
                "entities": [
                    {
                        "name": "REALM",
                        "type": "MODEL",
                        "aliases": [],
                        "description": "",
                        "evidence_quote": "REALM",
                    }
                ],
                "relations": [],
            },
        ]
    )
    config = ExtractionConfig(
        entity_types=("MODEL",), relation_types=("USES",), max_gleaning=1
    )
    run = run_extraction(
        [_chunk("RAG and REALM are models.")],
        config=config,
        provider=provider,
        provider_identity="fake:model",
    )
    assert {e.name for e in run.entities} == {"RAG", "REALM"}
    assert len(provider.prompts) == 2  # initial + one gleaning pass


# --------------------------------------------------------------------------- #
# Cache                                                                       #
# --------------------------------------------------------------------------- #


def test_extraction_cache_hit_skips_provider(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path / "extraction_cache")
    provider = _ScriptedProvider(
        [
            {
                "entities": [
                    {
                        "name": "RAG",
                        "type": "MODEL",
                        "aliases": [],
                        "description": "",
                        "evidence_quote": "RAG",
                    }
                ],
                "relations": [],
            }
        ]
    )
    chunk = _chunk("RAG is a model.")
    run1 = run_extraction(
        [chunk],
        config=CONFIG,
        provider=provider,
        provider_identity="fake:model",
        cache=cache,
    )
    assert run1.cache_misses == 1 and run1.cache_hits == 0
    calls_after_first = len(provider.prompts)

    run2 = run_extraction(
        [chunk],
        config=CONFIG,
        provider=provider,
        provider_identity="fake:model",
        cache=cache,
    )
    assert run2.cache_hits == 1
    # No new provider calls on the cached run.
    assert len(provider.prompts) == calls_after_first


def test_prompt_hash_changes_with_types() -> None:
    a = extraction_prompt_hash(CONFIG)
    b = extraction_prompt_hash(
        ExtractionConfig(entity_types=("MODEL",), relation_types=("USES",))
    )
    assert a != b


# --------------------------------------------------------------------------- #
# Deterministic fallback                                                      #
# --------------------------------------------------------------------------- #


def test_deterministic_extraction_finds_entities_and_comentions() -> None:
    result = deterministic_extract_chunk(
        _chunk("Dense Passage Retrieval and REALM improve Open Domain QA."),
        CONFIG,
    )
    names = {e.name for e in result.entities}
    assert "REALM" in names
    assert result.relations  # co-mention relations among detected entities
    assert all(r.relation_type == "RELATED_TO" for r in result.relations)


def test_run_extraction_without_provider_uses_fallback() -> None:
    run = run_extraction(
        [_chunk("Dense Passage Retrieval and REALM are methods.")],
        config=CONFIG,
        provider=None,
    )
    assert run.tier == "fallback"
    assert run.entities


# --------------------------------------------------------------------------- #
# Dedup + profiling                                                           #
# --------------------------------------------------------------------------- #


def test_dedupe_merges_aliases_acronyms_and_inverse_relations() -> None:
    entities = [
        ExtractedEntity(
            name="Retrieval-Augmented Generation",
            type="MODEL",
            aliases=["RAG"],
            chunk_ids=["c1"],
            source_ids=["s1"],
        ),
        ExtractedEntity(name="RAG", type="MODEL", chunk_ids=["c2"], source_ids=["s2"]),
        ExtractedEntity(
            name="Dense Passage Retrieval",
            type="METHOD",
            aliases=["DPR"],
            chunk_ids=["c1"],
            source_ids=["s1"],
        ),
        ExtractedEntity(name="DPR", type="METHOD", chunk_ids=["c3"], source_ids=["s3"]),
        ExtractedEntity(
            name="Fusion in Decoder", type="METHOD", chunk_ids=["c4"], source_ids=["s4"]
        ),
        ExtractedEntity(name="FID", type="METHOD", chunk_ids=["c5"], source_ids=["s5"]),
    ]
    relations = [
        ExtractedRelation(
            source="RAG",
            target="DPR",
            relation_type="USES",
            chunk_ids=["c1"],
            source_ids=["s1"],
        ),
        ExtractedRelation(
            source="DPR",
            target="RAG",
            relation_type="USED_BY",
            chunk_ids=["c2"],
            source_ids=["s2"],
        ),
    ]
    entity_profiles, relation_profiles = dedupe_entities_and_relations(
        entities, relations, config=DedupeConfig(fuzzy_threshold=88)
    )
    names = {e.canonical_name for e in entity_profiles}
    # RAG merged into the full name; DPR merged; Fusion in Decoder + FID merged.
    assert "Retrieval-Augmented Generation" in names
    assert "Dense Passage Retrieval" in names
    assert "Fusion in Decoder" in names
    assert len(entity_profiles) == 3
    # Inverse relation merged into a single canonical USES relation.
    assert len(relation_profiles) == 1
    rel = relation_profiles[0]
    assert rel.relation_type == "USES"
    assert set(rel.source_ids) == {"s1", "s2"}
    # The RAG entity merged provenance from both chunks.
    rag = next(e for e in entity_profiles if e.canonical_name.startswith("Retrieval"))
    assert set(rag.chunk_ids) == {"c1", "c2"}
    assert "RAG" in rag.aliases
    assert rel.id in rag.relation_ids


def test_profiler_fills_text_and_embeddings() -> None:
    chunks = [_chunk("Dense Passage Retrieval uses a dual encoder.", chunk_id="c1")]
    entities = [
        ExtractedEntity(
            name="Dense Passage Retrieval",
            type="METHOD",
            aliases=["DPR"],
            description="A dense retriever.",
            chunk_ids=["c1"],
            source_ids=["s1"],
        ),
        ExtractedEntity(
            name="Dual Encoder", type="METHOD", chunk_ids=["c1"], source_ids=["s1"]
        ),
    ]
    relations = [
        ExtractedRelation(
            source="Dense Passage Retrieval",
            target="Dual Encoder",
            relation_type="USES",
            keywords=["encoder"],
            chunk_ids=["c1"],
            source_ids=["s1"],
        )
    ]
    entity_profiles, relation_profiles = dedupe_entities_and_relations(
        entities, relations
    )
    profile_index(
        entity_profiles, relation_profiles, chunks, updated_at="2026-01-01T00:00:00Z"
    )
    dpr = next(
        e for e in entity_profiles if e.canonical_name == "Dense Passage Retrieval"
    )
    assert "Dense Passage Retrieval" in dpr.embedding_text
    assert "DPR" in dpr.embedding_text
    assert "Type: METHOD" in dpr.profile_text
    assert dpr.updated_at == "2026-01-01T00:00:00Z"
    assert relation_profiles[0].embedding_text
    assert "Evidence snippets" in relation_profiles[0].profile_text
