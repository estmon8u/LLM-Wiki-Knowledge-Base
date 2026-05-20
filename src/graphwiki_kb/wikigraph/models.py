"""Pydantic models for WikiGraphRAG nodes, edges, and answers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WikiGraphNode(BaseModel):
    """A node in the custom wiki-artifact graph."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal[
        "source_page",
        "concept_page",
        "analysis_page",
        "chunk",
        "entity",
        "claim",
        "community",
    ]
    title: str
    path: str | None = None
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WikiGraphEdge(BaseModel):
    """A directed edge in the custom wiki-artifact graph."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    kind: Literal[
        "links_to",
        "mentions",
        "contains",
        "supports",
        "derived_from",
        "related_to",
        "member_of",
        "cites",
    ]
    weight: float = 1.0
    evidence: list[str] = Field(default_factory=list)


class WikiGraphRetrievedContext(BaseModel):
    """One retrieved context unit with provenance trace."""

    node_id: str
    node_kind: str
    title: str
    path: str | None
    text: str
    score: float
    source_ids: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)


class WikiGraphAnswer(BaseModel):
    """Structured WikiGraphRAG answer comparable to GraphRAG metadata."""

    engine: Literal["wikigraph"] = "wikigraph"
    method: Literal["basic", "local", "global", "drift-lite", "auto"]
    question: str
    answer: str
    contexts: list[WikiGraphRetrievedContext]
    citations: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)


class WikiGraphIndexSnapshot(BaseModel):
    """Serialized index metadata written to graph/wikigraph/."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    node_count: int
    edge_count: int
    chunk_count: int
    community_count: int
    include_graphrag_export_pages: bool
    lexical_backend: str
    community_algorithm: str
    source_dirs: list[str]
