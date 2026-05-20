"""Pydantic models for kb agent runtime, tool I/O, research, and recommendations.

This module belongs to `graphwiki_kb.agents.models` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Tool input / output models
# ---------------------------------------------------------------------------

ClaimSupportLevel = Literal[
    "cited-graph-answer",
    "graph-index-answer",
    "insufficient-evidence",
    "stale-index",
    "no-answer",
    "unverified",
]

RelevanceLevel = Literal["low", "medium", "high"]
NoveltyLevel = Literal["low", "medium", "high"]
ConfidenceLevel = Literal["low", "medium", "high"]
SearchContextSize = Literal["low", "medium", "high"]
SourceType = Literal[
    "paper",
    "docs",
    "article",
    "github",
    "blog",
    "unknown",
]


AskKbEngine = Literal["graphrag", "wikigraph"]
AskKbMethod = Literal["auto", "basic", "local", "global", "drift", "drift-lite"]


class AskKbInput(BaseModel):
    """Inputs accepted by the ask_kb agent tool."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., description="Natural-language KB question.")
    engine: AskKbEngine = Field(
        "wikigraph",
        description=(
            "Retrieval backend. `wikigraph` (default) runs the custom "
            "WikiGraphRAG backend built from the maintained wiki -- fast, "
            "cheap, fully grounded, and the recommended default. `graphrag` "
            "runs the Microsoft GraphRAG controller; choose it when the "
            "user explicitly asks for GraphRAG or for synthesis-heavy "
            "whole-corpus answers."
        ),
    )
    method: AskKbMethod = Field(
        "auto",
        description=(
            "Retrieval method. GraphRAG supports auto/basic/local/global/drift; "
            "WikiGraphRAG supports auto/basic/local/global/drift-lite."
        ),
    )
    save: bool = Field(
        False,
        description="Save the answer as an analysis page in wiki/analysis/.",
    )
    show_source_trace: bool = Field(
        False,
        description="Include the graph source trace and routing metadata.",
    )


class AskKbOutput(BaseModel):
    """A small, agent-friendly projection of GraphRAGQueryAnswer."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    method: str
    planner: str | None = None
    route_reason: str | None = None
    route_confidence: str | None = None
    claim_support: ClaimSupportLevel | None = None
    staleness_warnings: list[str] = Field(default_factory=list)
    source_trace: dict[str, str | None] = Field(default_factory=dict)
    saved_path: str | None = None
    index_run_id: str | None = None


FindKbEngine = Literal["auto", "graphrag", "wiki", "wikigraph", "all"]


class FindKbInput(BaseModel):
    """Inputs accepted by the find_kb agent tool."""

    model_config = ConfigDict(extra="forbid")

    query: str
    limit: int = Field(5, ge=1, le=50)
    engine: FindKbEngine = Field(
        "auto",
        description=(
            "Retrieval backend. `auto`/`all` fuse GraphRAG, wiki, and "
            "WikiGraphRAG via reciprocal rank fusion."
        ),
    )


class FindKbResult(BaseModel):
    """A single search hit returned by find_kb."""

    model_config = ConfigDict(extra="forbid")

    title: str
    path: str
    score: float
    snippet: str
    retriever: Literal["graph", "wiki", "wikigraph"]


class FindKbOutput(BaseModel):
    """Output of the find_kb agent tool."""

    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[FindKbResult] = Field(default_factory=list)
    graph_diagnostics: list[str] = Field(default_factory=list)


class WikiGraphStatusBlock(BaseModel):
    """Compact snapshot of the WikiGraphRAG index for the agent."""

    model_config = ConfigDict(extra="forbid")

    initialized: bool
    built_at: str | None = None
    node_count: int = 0
    edge_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0
    community_count: int = 0
    source_count: int = 0
    include_graphrag_export_pages: bool = False
    readable: bool = True
    message: str = ""


class StatusOutput(BaseModel):
    """Status snapshot consumable by the agent."""

    model_config = ConfigDict(extra="forbid")

    project_initialized: bool
    source_count: int
    compiled_source_count: int
    concept_count: int
    analysis_count: int
    graph_state: str
    graph_freshness: str
    next_action: str
    staleness_reasons: list[str] = Field(default_factory=list)
    wikigraph: WikiGraphStatusBlock | None = None


class LintOutput(BaseModel):
    """Lint findings projection for the agent."""

    model_config = ConfigDict(extra="forbid")

    error_count: int
    warning_count: int
    suggestion_count: int
    issues: list[dict[str, Any]] = Field(default_factory=list)


class ReviewOutput(BaseModel):
    """Review findings projection for the agent."""

    model_config = ConfigDict(extra="forbid")

    total_issues: int
    issues: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Research models
# ---------------------------------------------------------------------------


class ResearchInput(BaseModel):
    """Inputs accepted by the research agent tool."""

    model_config = ConfigDict(extra="forbid")

    question: str
    use_web: bool = True
    recommend_sources: bool = True
    search_context_size: SearchContextSize = "medium"
    max_recommendations: int = Field(5, ge=1, le=25)


class WebFinding(BaseModel):
    """A single web finding produced by the research pipeline."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    summary: str
    relevance: RelevanceLevel = "medium"
    supports_recommendation: bool = False


class SourceRecommendation(BaseModel):
    """A durable recommendation that can be ingested into the KB."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(..., ge=1)
    title: str
    url: str
    source_type: SourceType = "unknown"
    publisher: str | None = None
    published_at: str | None = None
    retrieved_at: str
    why_add: str
    knowledge_gap: str = ""
    novelty: NoveltyLevel = "medium"
    confidence: ConfidenceLevel = "medium"
    ingestable: bool = True
    suggested_tags: list[str] = Field(default_factory=list)
    citation_urls: list[str] = Field(default_factory=list)


class ResearchOutput(BaseModel):
    """Final research tool output for one run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    question: str
    created_at: str
    local_answer: AskKbOutput
    kb_gaps: list[str] = Field(default_factory=list)
    web_findings: list[WebFinding] = Field(default_factory=list)
    recommendations: list[SourceRecommendation] = Field(default_factory=list)
    saved_report_path: str | None = None
    web_used: bool = True


class WebResearchResult(BaseModel):
    """Raw output of the WebResearchService."""

    model_config = ConfigDict(extra="forbid")

    findings: list[WebFinding]
    recommendations: list[SourceRecommendation]
    sources: list[str] = Field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Ingest recommendation models
# ---------------------------------------------------------------------------


class ListRecommendationsInput(BaseModel):
    """Inputs accepted by the list_recommendations tool."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(
        "latest",
        description=(
            "Research run identifier or 'latest'. With 'latest' the tool "
            "returns the newest run that has at least one recommendation."
        ),
    )


class ListRecommendationsOutput(BaseModel):
    """Output of the list_recommendations tool."""

    model_config = ConfigDict(extra="forbid")

    run_id: str | None
    question: str | None
    created_at: str | None = None
    recommendations: list[SourceRecommendation] = Field(default_factory=list)
    message: str = ""


class IngestRecommendationInput(BaseModel):
    """Inputs accepted by the ingest_recommendation tool."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field("latest", description="Research run identifier or 'latest'.")
    ids: list[int] = Field(default_factory=list)


class IngestRecommendationItemResult(BaseModel):
    """Result of ingesting a single recommendation."""

    model_config = ConfigDict(extra="forbid")

    id: int
    title: str
    url: str
    created: bool
    message: str
    staged_path: str | None = None
    source_id: str | None = None


class IngestRecommendationOutput(BaseModel):
    """Output of the ingest_recommendation tool."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    results: list[IngestRecommendationItemResult] = Field(default_factory=list)
    next_command: str | None = None


# ---------------------------------------------------------------------------
# Update tool
# ---------------------------------------------------------------------------


GraphMethod = Literal["auto", "standard", "fast", "standard-update", "fast-update"]


class UpdateInput(BaseModel):
    """Inputs accepted by the update agent tool."""

    model_config = ConfigDict(extra="forbid")

    force: bool = False
    dry_run: bool = False
    graph_method: GraphMethod = "auto"
    no_graph: bool = False
    graph_only: bool = False
    wikigraph: bool | None = Field(
        None,
        description=(
            "Refresh the WikiGraphRAG index after compile. When unset, "
            "`wikigraph.enabled` from project config drives the behavior "
            "(defaults to true)."
        ),
    )
    wikigraph_include_graphrag_export_pages: bool = Field(
        False,
        description=(
            "Include wiki/graph (GraphRAG-exported) pages in the WikiGraphRAG "
            "build for the optional ablation."
        ),
    )
    export_wikigraph_artifacts: bool | None = Field(
        None,
        description=(
            "After the WikiGraphRAG build, write generated entity, community, "
            "and chunk cards under wiki/wikigraph/. When unset, "
            "`wikigraph.export_generated_artifacts` from project config "
            "drives the behavior (defaults to false)."
        ),
    )


class UpdateOutput(BaseModel):
    """Output of the update agent tool."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    summary: str
    next_action: str = ""
    method: str | None = None
    diagnostics: list[str] = Field(default_factory=list)
    graph_freshness: str | None = None
    staleness_warnings: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Run record models
# ---------------------------------------------------------------------------


class AgentToolResult(BaseModel):
    """Trace record for a single tool call inside an agent run."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    ok: bool
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PendingApproval(BaseModel):
    """Represents a write action that requires user approval."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentRunRecord(BaseModel):
    """Durable record of one agent run for trace/replay purposes."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    prompt: str
    created_at: str
    tool_results: list[AgentToolResult] = Field(default_factory=list)
    final_output: str = ""
    pending_approvals: list[PendingApproval] = Field(default_factory=list)
    session_id: str | None = None


class ResearchRunRecord(BaseModel):
    """Persisted research run with local answer, gaps, findings, and recs."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    question: str
    created_at: str
    local_answer: dict[str, Any]
    kb_gaps: list[str] = Field(default_factory=list)
    web_findings: list[WebFinding] = Field(default_factory=list)
    recommendations: list[SourceRecommendation] = Field(default_factory=list)
    web_used: bool = True


__all__ = [
    "AgentRunRecord",
    "AgentToolResult",
    "AskKbEngine",
    "AskKbInput",
    "AskKbMethod",
    "AskKbOutput",
    "ClaimSupportLevel",
    "ConfidenceLevel",
    "FindKbEngine",
    "FindKbInput",
    "FindKbOutput",
    "FindKbResult",
    "GraphMethod",
    "IngestRecommendationInput",
    "IngestRecommendationItemResult",
    "IngestRecommendationOutput",
    "LintOutput",
    "ListRecommendationsInput",
    "ListRecommendationsOutput",
    "NoveltyLevel",
    "PendingApproval",
    "RelevanceLevel",
    "ResearchInput",
    "ResearchOutput",
    "ResearchRunRecord",
    "ReviewOutput",
    "SearchContextSize",
    "SourceRecommendation",
    "SourceType",
    "StatusOutput",
    "UpdateInput",
    "UpdateOutput",
    "WebFinding",
    "WebResearchResult",
    "WikiGraphStatusBlock",
]
