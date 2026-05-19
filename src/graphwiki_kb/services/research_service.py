"""Local KB answer, gap extraction, web research, and recommendations."""

from __future__ import annotations

from typing import Any

from graphwiki_kb.agents.models import (
    AskKbOutput,
    ResearchOutput,
    ResearchRunRecord,
    SourceRecommendation,
)
from graphwiki_kb.services.graph_ask_controller_service import (
    GraphAskControllerError,
    GraphAskControllerService,
)
from graphwiki_kb.services.project_service import ProjectPaths, slugify, utc_now_iso
from graphwiki_kb.services.source_recommendation_store import SourceRecommendationStore
from graphwiki_kb.services.web_research_service import (
    WebResearchService,
    build_recommendations_from_urls,
)


def project_ask_output(answer: Any) -> AskKbOutput:
    """Project GraphRAGQueryAnswer into AskKbOutput."""
    return AskKbOutput(
        answer=answer.answer or "",
        method=answer.method,
        planner=answer.planner,
        route_reason=answer.route_reason,
        claim_support=answer.claim_support,
        staleness_warnings=list(answer.staleness_warnings),
        source_trace=dict(answer.source_trace),
        saved_path=answer.saved_path,
    )


def extract_kb_gaps(local: AskKbOutput, question: str) -> list[str]:
    """Heuristic KB gap list from a local answer."""
    gaps: list[str] = []
    text = (local.answer or "").strip().lower()
    if not text or text in {"no answer text returned.", "no answer"}:
        gaps.append("Local KB returned no substantive answer for this question.")
    if local.staleness_warnings:
        gaps.extend(local.staleness_warnings)
    if local.claim_support in {"unverified", "no-answer", "stale-index"}:
        gaps.append(
            f"Local answer claim support is {local.claim_support or 'unknown'}; "
            "additional sources may be needed."
        )
    if not gaps:
        gaps.append(f"Local KB may not fully cover recent developments for: {question}")
    return gaps


class ResearchService:
    """Coordinates local ask, web research, and recommendation persistence."""

    def __init__(
        self,
        paths: ProjectPaths,
        config: dict[str, Any],
        *,
        ask_controller: GraphAskControllerService,
        store: SourceRecommendationStore | None = None,
        web_research: WebResearchService | None = None,
    ) -> None:
        self.paths = paths
        self.config = config
        research_cfg = dict(config.get("research", {}) or {})
        self._store = store or SourceRecommendationStore(paths)
        self._ask = ask_controller
        self._web = web_research or WebResearchService(
            model=str(research_cfg.get("web_model", "gpt-5.4-nano")),
            blocked_domains=list(research_cfg.get("default_domains_blocklist", [])),
        )
        self._research_cfg = research_cfg

    def run(
        self,
        *,
        question: str,
        use_web: bool = True,
        recommend_sources: bool = True,
        search_context_size: str | None = None,
        max_recommendations: int | None = None,
        method: str = "auto",
    ) -> ResearchOutput:
        """Run the full research pipeline."""
        try:
            answer = self._ask.ask(question, method=method)
        except GraphAskControllerError as exc:
            local = AskKbOutput(
                answer="",
                method=method,
                claim_support="no-answer",
                staleness_warnings=[str(exc)],
            )
        else:
            local = project_ask_output(answer)

        kb_gaps = extract_kb_gaps(local, question)
        web_findings = []
        recommendations: list[SourceRecommendation] = []
        web_enabled = bool(self._research_cfg.get("web_enabled", True))
        context_size = search_context_size or str(
            self._research_cfg.get("search_context_size", "medium")
        )
        max_recs = max_recommendations or int(
            self._research_cfg.get("max_recommendations", 5)
        )

        if use_web and web_enabled:
            web_result = self._web.research(
                question=question,
                local_answer=local.answer,
                kb_gaps=kb_gaps,
                search_context_size=context_size,
            )
            web_findings = web_result.web_findings
            if recommend_sources:
                recommendations = build_recommendations_from_urls(
                    web_result.source_urls,
                    question=question,
                    kb_gaps=kb_gaps,
                    max_recommendations=max_recs,
                )

        run_id = self._store.make_run_id(question)
        saved_report_path = self._save_markdown_report(
            run_id=run_id,
            question=question,
            local=local,
            kb_gaps=kb_gaps,
            web_findings=web_findings,
            recommendations=recommendations,
        )
        record = ResearchRunRecord(
            run_id=run_id,
            question=question,
            created_at=utc_now_iso(),
            local_answer=local.model_dump(),
            kb_gaps=kb_gaps,
            web_findings=web_findings,
            recommendations=recommendations,
            saved_report_path=saved_report_path,
        )
        self._store.save_run(record)

        return ResearchOutput(
            run_id=run_id,
            question=question,
            local_answer=local,
            kb_gaps=kb_gaps,
            web_findings=web_findings,
            recommendations=recommendations,
            saved_report_path=saved_report_path,
        )

    def _save_markdown_report(
        self,
        *,
        run_id: str,
        question: str,
        local: AskKbOutput,
        kb_gaps: list[str],
        web_findings: list,
        recommendations: list[SourceRecommendation],
    ) -> str | None:
        analysis_dir = self.paths.wiki_analysis_dir
        analysis_dir.mkdir(parents=True, exist_ok=True)
        slug = slugify(question)[:60] or "research"
        path = analysis_dir / f"agent-research-{slug}.md"
        lines = [
            "---",
            "type: agent_research",
            "agent: graphwiki-agent",
            f"run_id: {run_id}",
            f"question: {question}",
            f"created_at: {utc_now_iso()}",
            f"local_claim_support: {local.claim_support or 'unknown'}",
            f"recommendation_count: {len(recommendations)}",
            "---",
            "",
            "# Research Report",
            "",
            "## Local KB Answer",
            "",
            local.answer or "(no local answer)",
            "",
            "## KB Gaps",
            "",
        ]
        lines.extend(f"- {gap}" for gap in kb_gaps) or lines.append("- (none)")
        lines.extend(["", "## Web Findings", ""])
        if web_findings:
            for finding in web_findings:
                lines.append(f"- [{finding.title}]({finding.url}): {finding.summary}")
        else:
            lines.append("- (none)")
        lines.extend(["", "## Recommended Sources to Add", ""])
        if recommendations:
            for rec in recommendations:
                lines.append(
                    f"{rec.id}. [{rec.title}]({rec.url}) — {rec.why_add} "
                    f"(ingestable={rec.ingestable})"
                )
        else:
            lines.append("- (none)")
        lines.extend(
            [
                "",
                "## What Was Not Added",
                "",
                "No sources were ingested automatically.",
                "",
                "## Suggested Next Commands",
                "",
                '- `kb agent "add recommendation 1"`',
                "- `kb update`",
            ]
        )
        from graphwiki_kb.services.project_service import atomic_write_text

        atomic_write_text(path, "\n".join(lines) + "\n")
        try:
            return path.relative_to(self.paths.root).as_posix()
        except ValueError:
            return path.as_posix()
