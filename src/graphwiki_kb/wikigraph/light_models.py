"""Strict Pydantic models for the LightRAG-style WikiGraphRAG backend.

These models describe a *source-chunk-first* graph index: normalized source
text is segmented into token-aware :class:`LightChunk`s, an LLM (or a
deterministic fallback) extracts :class:`ExtractedEntity` / :class:`ExtractedRelation`
tuples from each chunk, and those are deduplicated and profiled into
:class:`EntityProfile` / :class:`RelationProfile` records that drive dual-level
vector retrieval.

The retrieval/answer shapes (:class:`LightRetrievedBundle`,
:class:`LightAnswerPayload`) are kept separate from the classic
``wikigraph.models`` so the LightRAG engine can evolve without disturbing the
classic backend, while still re-using
:class:`graphwiki_kb.wikigraph.models.WikiGraphRetrievedContext` for CLI output
compatibility.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from graphwiki_kb.wikigraph.models import WikiGraphRetrievedContext

LightQueryMethod = Literal["local", "global", "hybrid", "basic", "drift-lite", "auto"]


class LightChunk(BaseModel):
    """A token-aware chunk of normalized source text with full provenance."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_id: str
    source_slug: str
    normalized_path: str
    compiled_page_path: str | None = None
    chunk_index: int
    token_count: int
    text: str
    content_hash: str
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def source_ref(self) -> str:
        """A citation reference anchored to the compiled source page.

        Falls back to the normalized path with a ``#text-unit-N`` anchor when
        no compiled wiki page is known yet.
        """
        if self.compiled_page_path:
            return f"{self.compiled_page_path}#chunk-{self.chunk_index}"
        return f"{self.normalized_path}#text-unit-{self.chunk_index}"


class ExtractedEntity(BaseModel):
    """A raw entity extracted from a single chunk (pre-dedup)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    evidence_quote: str = ""
    confidence: float | None = None


class ExtractedRelation(BaseModel):
    """A raw relation extracted from a single chunk (pre-dedup)."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    relation_type: str
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    evidence_quote: str = ""
    weight: float = 1.0
    confidence: float | None = None


class LightExtractionResult(BaseModel):
    """Structured output of a single-chunk extraction call."""

    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EntityProfile(BaseModel):
    """A deduplicated, canonical entity with retrieval profile + provenance."""

    model_config = ConfigDict(extra="forbid")

    id: str
    canonical_name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    profile_text: str = ""
    keywords: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    embedding_text: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationProfile(BaseModel):
    """A deduplicated, canonical relation with retrieval profile + provenance."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    description: str = ""
    profile_text: str = ""
    keywords: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    embedding_text: str = ""
    weight: float = 1.0
    updated_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LightGraphIndex(BaseModel):
    """In-memory representation of a built LightRAG-style index."""

    model_config = ConfigDict(extra="forbid")

    built_at: str = ""
    chunks: list[LightChunk] = Field(default_factory=list)
    entities: list[EntityProfile] = Field(default_factory=list)
    relations: list[RelationProfile] = Field(default_factory=list)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    extraction_prompt_hash: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 0
    provider_identity: str | None = None
    embedding_identity: str | None = None
    tier: str = "fallback"
    schema_version: int = 1

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the index."""
        return len(self.chunks)

    @property
    def entity_count(self) -> int:
        """Number of entity profiles in the index."""
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        """Number of relation profiles in the index."""
        return len(self.relations)


class MissingSourceRecord(BaseModel):
    """A source that disappeared from the manifest but remains in the index."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    status: Literal["missing"] = "missing"
    requires_review: bool = True
    detail: str = ""


class LightGraphBuildReport(BaseModel):
    """Summary returned from :func:`build_lightgraph_index`."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    tier: str
    mode: str = "lightrag"
    chunk_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    source_count: int = 0
    new_source_ids: list[str] = Field(default_factory=list)
    changed_source_ids: list[str] = Field(default_factory=list)
    missing_source_ids: list[str] = Field(default_factory=list)
    reused_source_count: int = 0
    reprocessed_source_count: int = 0
    embedding_model: str = ""
    embedding_dimension: int = 0
    incremental: bool = False
    extraction_cache_hits: int = 0
    extraction_cache_misses: int = 0
    artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QueryKeywords(BaseModel):
    """Low/high-level query keywords for dual-level retrieval."""

    model_config = ConfigDict(extra="forbid")

    low_level_keywords: list[str] = Field(default_factory=list)
    high_level_keywords: list[str] = Field(default_factory=list)


class LightRetrievedBundle(BaseModel):
    """Structured LightRAG retrieval result (entities + relations + chunks)."""

    model_config = ConfigDict(extra="forbid")

    question: str
    method: LightQueryMethod
    low_level_keywords: list[str] = Field(default_factory=list)
    high_level_keywords: list[str] = Field(default_factory=list)
    entities: list[EntityProfile] = Field(default_factory=list)
    relations: list[RelationProfile] = Field(default_factory=list)
    chunks: list[LightChunk] = Field(default_factory=list)
    contexts: list[WikiGraphRetrievedContext] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """A single citation reference resolved to a retrieved source chunk."""

    model_config = ConfigDict(extra="forbid")

    ref: str
    title: str = ""


class ClaimWithCitations(BaseModel):
    """An answer claim with at least one supporting source-chunk citation."""

    model_config = ConfigDict(extra="forbid")

    text: str
    citation_refs: list[str] = Field(default_factory=list)


class LightAnswerPayload(BaseModel):
    """Validated structured answer produced by the LightRAG answer service."""

    model_config = ConfigDict(extra="forbid")

    answer: str = ""
    claims: list[ClaimWithCitations] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    insufficient_evidence: bool = False
