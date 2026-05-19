"""Pydantic models for KB agent runs, research, and tool I/O."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentToolResult(BaseModel):
    """Summary of one tool invocation during an agent run."""

    tool_name: str
    ok: bool
    summary: str
    data: dict[str, object] = Field(default_factory=dict)
    error: str | None = None


class AgentRunRecord(BaseModel):
    """Persisted trace of a completed agent run."""

    run_id: str
    prompt: str
    created_at: str
    tool_results: list[AgentToolResult] = Field(default_factory=list)
    final_output: str
    pending_approvals: list[dict[str, object]] = Field(default_factory=list)


class AgentRunResult(BaseModel):
    """Result returned to the CLI from one agent invocation."""

    run_id: str
    final_output: str
    tool_results: list[AgentToolResult] = Field(default_factory=list)
    pending_approvals: list[dict[str, object]] = Field(default_factory=list)
    planned_tools: list[str] = Field(default_factory=list)


class AskKbInput(BaseModel):
    """Input for the ask_kb agent tool."""

    question: str
    method: Literal["auto", "basic", "local", "global", "drift"] = "auto"
    save: bool = False
    show_source_trace: bool = False


class AskKbOutput(BaseModel):
    """Projection of GraphRAGQueryAnswer for agent tools."""

    answer: str
    method: str
    planner: str | None = None
    route_reason: str | None = None
    claim_support: str | None = None
    staleness_warnings: list[str] = Field(default_factory=list)
    source_trace: dict[str, object] = Field(default_factory=dict)
    saved_path: str | None = None


class FindKbInput(BaseModel):
    """Input for the find_kb agent tool."""

    query: str
    limit: int = 5


class FindKbResultItem(BaseModel):
    """One merged graph or wiki search hit."""

    title: str
    path: str
    score: float
    kind: str
    snippet: str = ""


class FindKbOutput(BaseModel):
    """Output from find_kb."""

    query: str
    diagnostics: list[str] = Field(default_factory=list)
    results: list[FindKbResultItem] = Field(default_factory=list)


class StatusKbOutput(BaseModel):
    """Combined project and GraphRAG health snapshot."""

    initialized: bool
    summary: str
    graph_freshness: str | None = None
    graph_stale_reasons: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    details: dict[str, object] = Field(default_factory=dict)


class LintKbOutput(BaseModel):
    """Lint report summary for the agent."""

    ok: bool
    error_count: int
    warning_count: int
    issues: list[dict[str, object]] = Field(default_factory=list)


class ReviewKbOutput(BaseModel):
    """Review report summary for the agent."""

    issue_count: int
    issues: list[dict[str, object]] = Field(default_factory=list)


class WebFinding(BaseModel):
    """One web research finding distinct from local KB evidence."""

    title: str
    url: str
    summary: str
    relevance: Literal["low", "medium", "high"] = "medium"
    supports_recommendation: bool = False


class SourceRecommendation(BaseModel):
    """A numbered source candidate from a research run."""

    id: int
    title: str
    url: str
    source_type: Literal["paper", "docs", "article", "github", "blog", "unknown"] = (
        "unknown"
    )
    publisher: str | None = None
    published_at: str | None = None
    retrieved_at: str
    why_add: str
    knowledge_gap: str
    novelty: Literal["low", "medium", "high"] = "medium"
    confidence: Literal["low", "medium", "high"] = "medium"
    ingestable: bool
    suggested_tags: list[str] = Field(default_factory=list)
    citation_urls: list[str] = Field(default_factory=list)


class ResearchInput(BaseModel):
    """Input for the research agent tool."""

    question: str
    use_web: bool = True
    recommend_sources: bool = True
    search_context_size: Literal["low", "medium", "high"] = "medium"
    max_recommendations: int = 5


class ResearchOutput(BaseModel):
    """Research tool output separating local KB from web findings."""

    run_id: str
    question: str
    local_answer: AskKbOutput
    kb_gaps: list[str] = Field(default_factory=list)
    web_findings: list[WebFinding] = Field(default_factory=list)
    recommendations: list[SourceRecommendation] = Field(default_factory=list)
    saved_report_path: str | None = None


class ResearchRunRecord(BaseModel):
    """Persisted research run on disk."""

    run_id: str
    question: str
    created_at: str
    local_answer: dict[str, object]
    kb_gaps: list[str] = Field(default_factory=list)
    web_findings: list[WebFinding] = Field(default_factory=list)
    recommendations: list[SourceRecommendation] = Field(default_factory=list)
    saved_report_path: str | None = None


class IngestRecommendationInput(BaseModel):
    """Input for ingesting numbered recommendations."""

    recommendation_ids: list[int] = Field(
        description="Recommendation IDs from the latest or specified research run."
    )
    run_id: str | None = Field(
        default=None,
        description="Research run id; defaults to latest persisted run.",
    )


class IngestRecommendationOutput(BaseModel):
    """Result of ingesting one or more recommendations."""

    ingested: list[dict[str, object]] = Field(default_factory=list)
    skipped: list[dict[str, object]] = Field(default_factory=list)
    next_command: str = "kb update"


class UpdateKbInput(BaseModel):
    """Input for the update_kb agent tool."""

    graph_method: Literal[
        "auto", "standard", "fast", "standard-update", "fast-update"
    ] = "auto"
    no_graph: bool = False
    graph_only: bool = False


class UpdateKbOutput(BaseModel):
    """Summary of an update_kb tool run."""

    ok: bool
    summary: str
    graph_freshness: str | None = None
    staleness_warnings: list[str] = Field(default_factory=list)
    details: dict[str, object] = Field(default_factory=dict)
