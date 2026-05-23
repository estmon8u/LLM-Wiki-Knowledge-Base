"""Pydantic models for the LightRAG-style WikiGraphRAG backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LightQueryMethod = Literal[
    "local",
    "global",
    "hybrid",
    "basic",
    "drift-lite",
    "auto",
]


class LightChunk(BaseModel):
    """Token-aware source chunk used as primary retrieval evidence."""

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


class ExtractedEntity(BaseModel):
    """Entity extracted from a single chunk."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    chunk_ids: list[str]
    source_ids: list[str]
    confidence: float | None = None


class ExtractedRelation(BaseModel):
    """Relation extracted from a single chunk."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    relation_type: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    chunk_ids: list[str]
    source_ids: list[str]
    weight: float = 1.0
    confidence: float | None = None


class EntityProfile(BaseModel):
    """Canonical entity profile used for vector retrieval."""

    model_config = ConfigDict(extra="forbid")

    id: str
    canonical_name: str
    type: str
    aliases: list[str]
    description: str
    profile_text: str
    keywords: list[str]
    chunk_ids: list[str]
    source_ids: list[str]
    relation_ids: list[str]
    embedding_text: str
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationProfile(BaseModel):
    """Canonical relation profile used for vector retrieval."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    description: str
    profile_text: str
    keywords: list[str]
    chunk_ids: list[str]
    source_ids: list[str]
    embedding_text: str
    weight: float = 1.0
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LightGraphIndex(BaseModel):
    """In-memory LightRAG-style graph index."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    chunks: list[LightChunk]
    entities: list[EntityProfile]
    relations: list[RelationProfile]
    source_hashes: dict[str, str]
    extraction_prompt_hash: str
    embedding_model: str
    embedding_dimension: int
    provider_identity: str | None = None
    chunk_count: int
    entity_count: int
    relation_count: int


class LightGraphBuildReport(BaseModel):
    """Summary returned from a LightGraph index build."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    mode: Literal["lightrag"] = "lightrag"
    chunk_count: int
    entity_count: int
    relation_count: int
    source_count: int
    incremental: bool = False
    vector_rewrite: bool = False
    fallback_mode: str = ""
    artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QueryKeywords(BaseModel):
    """Low/high-level keywords extracted from a user question."""

    model_config = ConfigDict(extra="forbid")

    low_level_keywords: list[str] = Field(default_factory=list)
    high_level_keywords: list[str] = Field(default_factory=list)


class LightRetrievedBundle(BaseModel):
    """Structured retrieval bundle for LightRAG answer synthesis."""

    model_config = ConfigDict(extra="forbid")

    question: str
    method: LightQueryMethod
    low_level_keywords: list[str] = Field(default_factory=list)
    high_level_keywords: list[str] = Field(default_factory=list)
    entities: list[EntityProfile] = Field(default_factory=list)
    relations: list[RelationProfile] = Field(default_factory=list)
    chunks: list[LightChunk] = Field(default_factory=list)
    contexts: list[Any] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_backend: str = "lightrag"


class LightGraphFindResult(BaseModel):
    """Provider-free find result for the LightRAG backend."""

    model_config = ConfigDict(extra="forbid")

    engine: Literal["wikigraph"] = "wikigraph"
    mode: Literal["lightrag"] = "lightrag"
    query: str
    method: LightQueryMethod
    low_level_keywords: list[str] = Field(default_factory=list)
    high_level_keywords: list[str] = Field(default_factory=list)
    entities: list[EntityProfile] = Field(default_factory=list)
    relations: list[RelationProfile] = Field(default_factory=list)
    contexts: list[Any] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    retrieval_backend: str = "lightrag"
