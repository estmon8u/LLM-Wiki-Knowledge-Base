"""Unit tests for the LightRAG deduper, extractor, and keyword extractor."""

from __future__ import annotations

from graphwiki_kb.wikigraph.light_deduper import (
    LightDeduper,
    LightDeduperOptions,
    LightRelationDeduper,
    dedupe_and_profile,
    normalize_relation_type,
)
from graphwiki_kb.wikigraph.light_extractor import (
    DeterministicLightExtractor,
    LightExtractionCache,
    LightExtractorOptions,
    extract_corpus,
)
from graphwiki_kb.wikigraph.light_keywords import (
    QueryKeywords,
    RuleBasedKeywordProvider,
)
from graphwiki_kb.wikigraph.light_models import (
    ExtractedEntity,
    ExtractedRelation,
    LightChunk,
)


def _chunk(text: str, *, chunk_id: str = "chunk:s:0:abc") -> LightChunk:
    return LightChunk(
        id=chunk_id,
        source_id="s",
        source_slug="doc",
        source_title="Doc",
        normalized_path="raw/normalized/doc.md",
        chunk_index=0,
        token_count=len(text.split()),
        text=text,
        content_hash="abc",
    )


def test_deterministic_extractor_emits_entities_and_relations():
    extractor = DeterministicLightExtractor(
        options=LightExtractorOptions(min_occurrences=1)
    )
    chunk = _chunk(
        "Dense Passage Retrieval (DPR) uses a dual encoder. "
        "DPR was evaluated on Natural Questions. "
        "DPR was introduced by Stanford NLP."
    )
    result = extractor.extract(chunk)
    assert result.extractor == "deterministic"
    names = {e.name for e in result.entities}
    assert any("DPR" in n or "Dense Passage" in n for n in names)
    assert "Natural Questions" in names
    assert len(result.relations) > 0
    # Co-occurrence relations attach the chunk id for provenance.
    assert all(chunk.id in r.chunk_ids for r in result.relations)


def test_extraction_cache_roundtrip(tmp_path):
    extractor = DeterministicLightExtractor()
    chunk = _chunk("ModelA improves over ModelB and ModelC.")
    cache = LightExtractionCache(tmp_path / "cache")
    results = extract_corpus([chunk], extractor, cache=cache)
    assert results[0].chunk_id == chunk.id
    # Second pass should hit the cache.
    cached = extract_corpus([chunk], extractor, cache=cache)
    assert cached[0].model_dump() == results[0].model_dump()
    # Cache file exists.
    cache_files = list((tmp_path / "cache").iterdir())
    assert len(cache_files) == 1


def test_deduper_merges_by_alias_and_acronym():
    deduper = LightDeduper(options=LightDeduperOptions(fuzzy_match_threshold=85))
    e1 = ExtractedEntity(
        name="REALM: Retrieval-Augmented Language Model Pre-Training",
        type="PAPER",
        chunk_ids=["c1"],
        source_ids=["s1"],
    )
    e2 = ExtractedEntity(
        name="REALM", type="PAPER", chunk_ids=["c2"], source_ids=["s2"]
    )
    e3 = ExtractedEntity(
        name="Retrieval-Augmented Language Model Pre-Training",
        type="PAPER",
        chunk_ids=["c3"],
        source_ids=["s3"],
    )
    deduper.add_entity(e1)
    deduper.add_entity(e2)
    deduper.add_entity(e3)
    assert deduper.entity_count == 1
    profile = deduper.build_entity_profiles()[0]
    assert set(profile.chunk_ids) == {"c1", "c2", "c3"}
    assert set(profile.source_ids) == {"s1", "s2", "s3"}
    assert "REALM" in profile.aliases or profile.canonical_name == "REALM"


def test_deduper_respects_type_compatibility():
    deduper = LightDeduper(options=LightDeduperOptions(fuzzy_match_threshold=70))
    deduper.add_entity(ExtractedEntity(name="BERT", type="MODEL"))
    deduper.add_entity(ExtractedEntity(name="BERT", type="DATASET"))
    assert deduper.entity_count == 2


def test_relation_deduper_canonicalizes_inverse_types():
    ent = LightDeduper()
    a = ent.add_entity(ExtractedEntity(name="RAG", type="MODEL"))
    b = ent.add_entity(ExtractedEntity(name="DPR", type="METHOD"))
    rel_deduper = LightRelationDeduper()
    rel_deduper.add_relation(
        ExtractedRelation(
            source="RAG", target="DPR", relation_type="USES", chunk_ids=["c1"]
        ),
        source_entity_id=a,
        target_entity_id=b,
    )
    rel_deduper.add_relation(
        ExtractedRelation(
            source="DPR", target="RAG", relation_type="USED_BY", chunk_ids=["c2"]
        ),
        source_entity_id=b,
        target_entity_id=a,
    )
    profiles = rel_deduper.build_relation_profiles()
    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.source_entity_id == a
    assert profile.target_entity_id == b
    assert profile.relation_type == "USES"
    assert set(profile.chunk_ids) == {"c1", "c2"}


def test_normalize_relation_type_handles_unknown():
    assert normalize_relation_type("uses") == "USES"
    assert normalize_relation_type("USED_BY") == "USES"
    assert normalize_relation_type("ABSORBS") == "ABSORBS"


def test_dedupe_and_profile_drops_relations_with_unknown_endpoints():
    entities = [
        ExtractedEntity(name="X", type="MODEL", chunk_ids=["c1"], source_ids=["s"])
    ]
    relations = [
        ExtractedRelation(
            source="X", target="Y", relation_type="USES", chunk_ids=["c1"]
        ),
    ]
    ent_profiles, rel_profiles = dedupe_and_profile([(entities, relations)])
    assert len(ent_profiles) == 1
    # Y is not in the entity set -> relation discarded.
    assert rel_profiles == []


def test_rule_based_keyword_provider_matches_aliases_and_themes():
    provider = RuleBasedKeywordProvider(known_aliases=("DPR", "RAG"))
    result = provider.extract("Compare DPR and RAG retrieval tradeoffs")
    assert isinstance(result, QueryKeywords)
    assert "DPR" in result.low_level_keywords
    assert "RAG" in result.low_level_keywords
    assert any(
        kw in {"tradeoff", "tradeoffs", "compare"} for kw in result.high_level_keywords
    )


def test_rule_based_keyword_provider_falls_back_to_nouns():
    provider = RuleBasedKeywordProvider()
    result = provider.extract("explain the indexing pipeline")
    # No themes match -> falls back to long lowercase nouns.
    assert result.high_level_keywords
