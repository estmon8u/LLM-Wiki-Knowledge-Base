"""Strict Pydantic data models for the WikiGraphRAG backend.

These models intentionally mirror the high-level concepts that Microsoft
GraphRAG exposes (entities, claims, relationships, communities, chunks) so
that retrieval and answer artifacts produced by either backend can be
compared with the same evaluator.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

NodeKind = Literal[
    "source_page",
    "concept_page",
    "analysis_page",
    "graph_page",
    "chunk",
    "entity",
    "claim",
    "community",
]

EdgeKind = Literal[
    "links_to",
    "mentions",
    "contains",
    "supports",
    "derived_from",
    "related_to",
    "member_of",
    "co_mentions",
    "cites",
]

QueryMethod = Literal["basic", "local", "global", "drift-lite", "auto"]


class WikiGraphNode(BaseModel):
    """A node in the wiki graph."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: NodeKind
    title: str
    path: str | None = None
    text: str = ""
    aliases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WikiGraphEdge(BaseModel):
    """A directed (logically undirected for ranking) edge in the wiki graph."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    kind: EdgeKind
    weight: float = 1.0
    evidence: list[str] = Field(default_factory=list)


class WikiGraphCommunity(BaseModel):
    """A community in the wiki graph along with a summary card."""

    model_config = ConfigDict(extra="forbid")

    id: str
    level: int = 0
    title: str = ""
    members: list[str] = Field(default_factory=list)
    summary: str = ""
    top_entities: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class WikiGraphRetrievedContext(BaseModel):
    """A single retrieved evidence unit produced by WikiGraphRAG."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    node_kind: NodeKind
    title: str
    path: str | None = None
    text: str
    score: float
    source_ids: list[str] = Field(default_factory=list)
    section: str = ""
    chunk_index: int | None = None
    trace: list[str] = Field(default_factory=list)

    @property
    def citation_ref(self) -> str:
        """A printable citation reference, mirroring legacy ``SearchResult``."""
        if self.chunk_index is None or self.chunk_index < 0 or not self.path:
            return self.path or self.node_id
        return f"{self.path}#chunk-{self.chunk_index}"


class WikiGraphFindResult(BaseModel):
    """Result of a provider-free retrieval-only ``kb wikigraph find`` call."""

    model_config = ConfigDict(extra="forbid")

    query: str
    method: QueryMethod
    contexts: list[WikiGraphRetrievedContext] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    communities: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


class WikiGraphAnswer(BaseModel):
    """The end-to-end answer produced by ``kb wikigraph ask``."""

    model_config = ConfigDict(extra="forbid")

    engine: Literal["wikigraph"] = "wikigraph"
    method: QueryMethod
    question: str
    answer: str
    contexts: list[WikiGraphRetrievedContext] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False
    provider_status: dict[str, Any] = Field(default_factory=dict)
    saved_path: str | None = None


class WikiGraphIndex(BaseModel):
    """In-memory representation of a built wiki graph index."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[WikiGraphNode] = Field(default_factory=list)
    edges: list[WikiGraphEdge] = Field(default_factory=list)
    communities: list[WikiGraphCommunity] = Field(default_factory=list)
    built_at: str = ""
    include_graphrag_export_pages: bool = False
    source_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0


class WikiGraphBuildReport(BaseModel):
    """Summary returned from ``kb wikigraph build``."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    node_count: int
    edge_count: int
    chunk_count: int
    entity_count: int
    community_count: int
    source_count: int
    include_graphrag_export_pages: bool
    artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
