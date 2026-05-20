"""Combine local KB answers with optional web research and recommendations.

This module belongs to `graphwiki_kb.services.research_service` and keeps
related behavior close to the command, service, model, provider, storage,
script, or test surface that uses it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from graphwiki_kb.agents.models import (
    AskKbOutput,
    ResearchInput,
    ResearchOutput,
    ResearchRunRecord,
    SourceRecommendation,
    WebFinding,
)
from graphwiki_kb.services.graph_ask_controller_service import (
    GraphAskControllerError,
    GraphAskControllerService,
)
from graphwiki_kb.services.graphrag_query_service import (
    GraphRAGQueryAnswer,
    GraphRAGQueryError,
)
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
    utc_now_iso,
)
from graphwiki_kb.services.source_recommendation_store import (
    SourceRecommendationStore,
)
from graphwiki_kb.services.web_research_service import (
    WebResearchError,
    WebResearchService,
)

logger = logging.getLogger(__name__)


class ResearchService:
    """Orchestrates local-KB lookup, web research, and recommendation ranking."""

    def __init__(
        self,
        paths: ProjectPaths,
        controller: GraphAskControllerService,
        recommendation_store: SourceRecommendationStore,
        *,
        web_service: WebResearchService | None = None,
    ) -> None:
        self.paths = paths
        self.controller = controller
        self.store = recommendation_store
        self.web_service = web_service

    # ------------------------------------------------------------------
    def research(self, request: ResearchInput) -> ResearchOutput:
        """Run the research pipeline for one question."""
        created_at = utc_now_iso()
        local_output, local_answer_obj = self._ask_local_kb(request.question)
        kb_gaps = derive_kb_gaps(local_output, local_answer_obj)

        web_findings: list[WebFinding] = []
        recommendations: list[SourceRecommendation] = []
        web_used = False
        if request.use_web and self.web_service is not None:
            try:
                web_result = self.web_service.research(
                    question=request.question,
                    local_answer=local_output.answer,
                    kb_gaps=kb_gaps,
                    search_context_size=request.search_context_size,
                    max_recommendations=request.max_recommendations,
                )
            except WebResearchError as exc:
                logger.warning("Web research skipped: %s", exc)
            else:
                web_used = True
                web_findings = list(web_result.findings)
                if request.recommend_sources:
                    recommendations = renumber_recommendations(
                        web_result.recommendations
                    )[: request.max_recommendations]
        run_id = self.store.generate_run_id(request.question, created_at=created_at)
        record = ResearchRunRecord(
            run_id=run_id,
            question=request.question,
            created_at=created_at,
            local_answer=local_output.model_dump(),
            kb_gaps=kb_gaps,
            web_findings=web_findings,
            recommendations=recommendations,
            web_used=web_used,
        )
        self.store.save(record)
        saved_report_path = self._save_markdown_report(record)
        return ResearchOutput(
            run_id=run_id,
            question=request.question,
            created_at=created_at,
            local_answer=local_output,
            kb_gaps=kb_gaps,
            web_findings=web_findings,
            recommendations=recommendations,
            saved_report_path=saved_report_path,
            web_used=web_used,
        )

    # ------------------------------------------------------------------
    def _ask_local_kb(
        self,
        question: str,
    ) -> tuple[AskKbOutput, GraphRAGQueryAnswer | None]:
        try:
            answer = self.controller.ask(question, method="auto")
        except (GraphAskControllerError, GraphRAGQueryError) as exc:
            placeholder = AskKbOutput(
                answer="",
                method="auto",
                staleness_warnings=[str(exc)],
                claim_support="no-answer",
            )
            return placeholder, None
        return project_ask_kb_output(answer), answer

    # ------------------------------------------------------------------
    def _save_markdown_report(self, record: ResearchRunRecord) -> str | None:
        self.paths.wiki_analysis_dir.mkdir(parents=True, exist_ok=True)
        slug = slugify(record.question)[:60] or "agent-research"
        filename = f"agent-research-{slug}.md"
        path = self.paths.wiki_analysis_dir / filename
        if path.exists():
            path = self.paths.wiki_analysis_dir / (
                f"agent-research-{slug}-{record.run_id[-8:]}.md"
            )
        atomic_write_text(path, _render_report(record))
        try:
            return path.relative_to(self.paths.root).as_posix()
        except ValueError:
            return path.as_posix()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def project_ask_kb_output(answer: GraphRAGQueryAnswer) -> AskKbOutput:
    """Project a GraphRAGQueryAnswer onto AskKbOutput."""
    return AskKbOutput(
        answer=answer.answer or "",
        method=answer.method,
        planner=answer.planner,
        route_reason=answer.route_reason,
        route_confidence=answer.route_confidence,
        claim_support=_coerce_claim_support(answer.claim_support),
        staleness_warnings=list(answer.staleness_warnings or []),
        source_trace=_stringify_source_trace(answer.source_trace),
        saved_path=answer.saved_path,
        index_run_id=answer.index_run_id,
    )


def _coerce_claim_support(value: str | None) -> str | None:
    if value is None:
        return None
    allowed = {
        "cited-graph-answer",
        "graph-index-answer",
        "insufficient-evidence",
        "stale-index",
        "no-answer",
        "unverified",
    }
    return value if value in allowed else "unverified"


def _stringify_source_trace(trace: dict[str, Any] | None) -> dict[str, str | None]:
    if not trace:
        return {}
    result: dict[str, str | None] = {}
    for key, value in trace.items():
        if value is None:
            result[str(key)] = None
        else:
            result[str(key)] = str(value)
    return result


_GAP_STALENESS_HINT = re.compile(r"stale", re.IGNORECASE)


def derive_kb_gaps(
    local_output: AskKbOutput,
    answer_obj: GraphRAGQueryAnswer | None,
) -> list[str]:
    """Build a small list of gap statements about the local KB answer."""
    gaps: list[str] = []
    answer_text = (local_output.answer or "").strip()
    if not answer_text:
        gaps.append("The local KB returned no answer for this question.")
    if local_output.claim_support in {"no-answer", "unverified"}:
        gaps.append(
            "Local KB answer is not citation-grounded; recent sources may be missing."
        )
    if local_output.claim_support == "stale-index":
        gaps.append("Local graph index is stale; run `kb update` to refresh.")
    for warning in local_output.staleness_warnings:
        if _GAP_STALENESS_HINT.search(warning):
            gaps.append(warning)
    if answer_obj is not None and not answer_obj.graph_data_references:
        gaps.append(
            "Local KB answer cites no graph data references "
            "(no [Data: ...] markers)."
        )
    seen: set[str] = set()
    deduped: list[str] = []
    for gap in gaps:
        key = gap.strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(gap.strip())
    return deduped


def renumber_recommendations(
    recommendations: list[SourceRecommendation],
) -> list[SourceRecommendation]:
    """Reassign monotonically increasing 1-based IDs."""
    renumbered: list[SourceRecommendation] = []
    for index, rec in enumerate(recommendations, start=1):
        renumbered.append(rec.model_copy(update={"id": index}))
    return renumbered


def _render_report(record: ResearchRunRecord) -> str:
    local = record.local_answer or {}
    claim_support = local.get("claim_support") or "unverified"
    method = local.get("method") or "auto"
    lines: list[str] = []
    lines.append("---")
    lines.append("type: agent_research")
    lines.append("agent: graphwiki-agent")
    lines.append(f"run_id: {record.run_id}")
    lines.append(f"question: {_yaml_safe(record.question)}")
    lines.append(f"created_at: {record.created_at}")
    lines.append(f"local_method: {method}")
    lines.append(f"local_claim_support: {claim_support}")
    lines.append(f"recommendation_count: {len(record.recommendations)}")
    lines.append(f"web_used: {str(record.web_used).lower()}")
    lines.append("---")
    lines.append("")
    lines.append(f"# Research Report: {record.question}")
    lines.append("")
    lines.append("## Local KB Answer")
    lines.append("")
    answer_text = local.get("answer") or "(no local KB answer was returned)"
    lines.append(answer_text.rstrip())
    lines.append("")
    lines.append("## KB Gaps")
    lines.append("")
    if record.kb_gaps:
        for gap in record.kb_gaps:
            lines.append(f"- {gap}")
    else:
        lines.append("- No gaps reported.")
    lines.append("")
    lines.append("## Web Findings")
    lines.append("")
    if record.web_findings:
        for finding in record.web_findings:
            lines.append(f"- [{finding.title}]({finding.url}) — {finding.summary}")
    else:
        lines.append("- No web findings recorded.")
    lines.append("")
    lines.append("## Recommended Sources to Add")
    lines.append("")
    if record.recommendations:
        for rec in record.recommendations:
            lines.append(
                f"{rec.id}. [{rec.title}]({rec.url}) — {rec.why_add or '(no rationale)'}"
            )
    else:
        lines.append("- No recommendations produced.")
    lines.append("")
    lines.append("## What Was Not Added")
    lines.append("")
    lines.append(
        'No sources were added to the KB. Use `kb agent "add recommendation N"`.'
    )
    lines.append("")
    lines.append("## Suggested Next Commands")
    lines.append("")
    if record.recommendations:
        lines.append('- `kb agent "add recommendation 1"`')
        lines.append("- `kb update`")
    else:
        lines.append("- `kb update`")
    lines.append("")
    return "\n".join(lines)


def _yaml_safe(value: str) -> str:
    if not value:
        return '""'
    if any(ch in value for ch in (":", "#", "\n", '"', "'")):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        return f'"{escaped}"'
    return value
