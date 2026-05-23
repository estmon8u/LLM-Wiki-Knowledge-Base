"""Strict Pydantic data models for the LightRAG-style WikiGraphRAG backend.

These models complement :mod:`graphwiki_kb.wikigraph.models` (the classic
backend) and intentionally separate the LightRAG world from generic
:class:`WikiGraphNode` / :class:`WikiGraphEdge` to keep the two modes
clearly distinguishable on disk and in code.

See ``docs/wikigraph_lightrag.md`` (and PR description) for the
conceptual mapping between the LightRAG paper and these models.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr

LightQueryMethod = Literal["local", "global", "hybrid", "basic", "drift-lite", "auto"]


class LightChunk(BaseModel):
    """A token-aware chunk of normalized source text.

    Chunks are the *grounding* layer for LightRAG: every answer claim
    must trace back to one or more of these chunks. They are produced
    once at index time from ``raw/normalized/<slug>.md`` and never
    re-derived from raw binary files at retrieval time.
    """

    model_config = ConfigDict(extra="forbid")

    id: StrictStr
    source_id: StrictStr
    source_slug: StrictStr
    source_title: StrictStr = ""
    normalized_path: StrictStr
    compiled_page_path: StrictStr | None = None
    chunk_index: int
    token_count: int = 0
    text: StrictStr
    content_hash: StrictStr
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedEntity(BaseModel):
    """A single entity returned by the extractor for one chunk."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr
    type: StrictStr
    description: StrictStr = ""
    aliases: list[StrictStr] = Field(default_factory=list)
    chunk_ids: list[StrictStr] = Field(default_factory=list)
    source_ids: list[StrictStr] = Field(default_factory=list)
    evidence_quote: StrictStr = ""
    confidence: float | None = None


class ExtractedRelation(BaseModel):
    """A single relation (subject/predicate/object) returned for one chunk."""

    model_config = ConfigDict(extra="forbid")

    source: StrictStr
    target: StrictStr
    relation_type: StrictStr
    description: StrictStr = ""
    keywords: list[StrictStr] = Field(default_factory=list)
    chunk_ids: list[StrictStr] = Field(default_factory=list)
    source_ids: list[StrictStr] = Field(default_factory=list)
    evidence_quote: StrictStr = ""
    weight: float = 1.0
    confidence: float | None = None


class LightExtractionResult(BaseModel):
    """Per-chunk extractor output (cacheable, deterministic shape)."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: StrictStr
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    warnings: list[StrictStr] = Field(default_factory=list)
    extractor: StrictStr = "deterministic"


class EntityProfile(BaseModel):
    """A deduplicated, canonical entity ready for retrieval.

    ``embedding_text`` is the concise text fed to the embedding model;
    ``profile_text`` is the human-readable card content used by the
    wiki exporter and by answer prompts.
    """

    model_config = ConfigDict(extra="forbid")

    id: StrictStr
    canonical_name: StrictStr
    type: StrictStr
    aliases: list[StrictStr] = Field(default_factory=list)
    description: StrictStr = ""
    profile_text: StrictStr = ""
    keywords: list[StrictStr] = Field(default_factory=list)
    chunk_ids: list[StrictStr] = Field(default_factory=list)
    source_ids: list[StrictStr] = Field(default_factory=list)
    relation_ids: list[StrictStr] = Field(default_factory=list)
    embedding_text: StrictStr = ""
    updated_at: StrictStr = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationProfile(BaseModel):
    """A deduplicated, canonical relation between two :class:`EntityProfile`."""

    model_config = ConfigDict(extra="forbid")

    id: StrictStr
    source_entity_id: StrictStr
    target_entity_id: StrictStr
    relation_type: StrictStr
    description: StrictStr = ""
    profile_text: StrictStr = ""
    keywords: list[StrictStr] = Field(default_factory=list)
    chunk_ids: list[StrictStr] = Field(default_factory=list)
    source_ids: list[StrictStr] = Field(default_factory=list)
    embedding_text: StrictStr = ""
    weight: float = 1.0
    updated_at: StrictStr = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceContribution(BaseModel):
    """Per-source provenance accounting for incremental updates.

    Without this, removing or recomputing a source can silently strand
    stale evidence inside merged entity/relation profiles. See
    project recommendation §22.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: StrictStr
    source_hash: StrictStr = ""
    chunk_ids: list[StrictStr] = Field(default_factory=list)
    entity_ids: list[StrictStr] = Field(default_factory=list)
    relation_ids: list[StrictStr] = Field(default_factory=list)
    status: Literal["fresh", "stale", "missing"] = "fresh"
    requires_review: bool = False


class LightGraphBuildManifest(BaseModel):
    """Persisted freshness metadata for a built LightGraph index."""

    model_config = ConfigDict(extra="forbid")

    built_at: StrictStr
    source_hashes: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    chunking: dict[StrictStr, int] = Field(default_factory=dict)
    extraction_prompt_hash: StrictStr = ""
    provider_identity: StrictStr | None = None
    embedding_provider: StrictStr = "bm25"
    embedding_model: StrictStr = "bm25-fallback"
    embedding_dimension: int = 0
    extractor: StrictStr = "deterministic"
    index_schema_version: int = 1


class LightGraphIndex(BaseModel):
    """In-memory representation of the LightRAG-style index."""

    model_config = ConfigDict(extra="forbid")

    built_at: StrictStr = ""
    chunks: list[LightChunk] = Field(default_factory=list)
    entities: list[EntityProfile] = Field(default_factory=list)
    relations: list[RelationProfile] = Field(default_factory=list)
    contributions: list[SourceContribution] = Field(default_factory=list)
    manifest: LightGraphBuildManifest = Field(
        default_factory=lambda: LightGraphBuildManifest(built_at="")
    )

    @property
    def chunk_count(self) -> int:
        """Return the number of stored chunks."""
        return len(self.chunks)

    @property
    def entity_count(self) -> int:
        """Return the number of canonical entities."""
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        """Return the number of canonical relations."""
        return len(self.relations)


class LightRetrievedContext(BaseModel):
    """A single retrieved item with the source-anchor citation it carries."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["entity", "relation", "chunk"]
    id: StrictStr
    title: StrictStr
    score: float
    text: StrictStr = ""
    path: StrictStr | None = None
    chunk_index: int | None = None
    source_ids: list[StrictStr] = Field(default_factory=list)
    trace: list[StrictStr] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def citation_ref(self) -> str:
        """Return a printable citation ref for the retrieval evidence.

        Chunk contexts include a ``#chunk-N`` anchor; entity / relation
        contexts return the canonical id so callers can distinguish
        scaffolding rows from grounding rows.
        """
        if self.kind == "chunk" and self.path and self.chunk_index is not None:
            return f"{self.path}#chunk-{self.chunk_index}"
        if self.path:
            return self.path
        return self.id


class LightRetrievedBundle(BaseModel):
    """The structured retrieval bundle assembled before answer synthesis."""

    model_config = ConfigDict(extra="forbid")

    question: StrictStr
    method: LightQueryMethod
    low_level_keywords: list[StrictStr] = Field(default_factory=list)
    high_level_keywords: list[StrictStr] = Field(default_factory=list)
    entities: list[EntityProfile] = Field(default_factory=list)
    relations: list[RelationProfile] = Field(default_factory=list)
    chunks: list[LightChunk] = Field(default_factory=list)
    contexts: list[LightRetrievedContext] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[StrictStr] = Field(default_factory=list)


class LightGraphBuildReport(BaseModel):
    """Report returned from :func:`build_lightgraph_index`."""

    model_config = ConfigDict(extra="forbid")

    built_at: StrictStr
    chunk_count: int
    entity_count: int
    relation_count: int
    source_count: int
    missing_source_count: int = 0
    extracted_entity_count: int = 0
    extracted_relation_count: int = 0
    entity_dedupe_rate: float = 0.0
    relation_dedupe_rate: float = 0.0
    extractor: StrictStr = "deterministic"
    embedding_provider: StrictStr = "bm25"
    embedding_model: StrictStr = "bm25-fallback"
    incremental: bool = False
    artifacts: list[StrictStr] = Field(default_factory=list)
    warnings: list[StrictStr] = Field(default_factory=list)
