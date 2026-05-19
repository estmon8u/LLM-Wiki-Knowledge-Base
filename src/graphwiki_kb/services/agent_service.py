"""Public agent service used by the kb agent command.

This module belongs to `graphwiki_kb.services.agent_service` and keeps
related behavior close to the command, service, model, provider, storage,
script, or test surface that uses it.

The service wires the OpenAI Agents SDK runtime to the existing
:class:`ServiceContainer`, the recommendation store, and the research
pipeline. The Click command stays small and thin; everything else lives here.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AgentRunRecord,
    PendingApproval,
)
from graphwiki_kb.agents.runtime import (
    AgentRuntimeError,
    build_kb_agent,
    build_session,
    is_agents_sdk_available,
    run_agent,
)
from graphwiki_kb.services.project_service import (
    atomic_write_text,
    slugify,
    utc_now_iso,
)
from graphwiki_kb.services.research_service import ResearchService
from graphwiki_kb.services.source_recommendation_store import (
    SourceRecommendationStore,
)
from graphwiki_kb.services.web_research_service import WebResearchService
from graphwiki_kb.services.web_source_acquisition_service import (
    WebSourceAcquisitionService,
)

if TYPE_CHECKING:
    from graphwiki_kb.models.command_models import CommandContext
    from graphwiki_kb.services.container import ServiceContainer


@dataclass
class AgentRunResult:
    """Final result of a single agent run."""

    record: AgentRunRecord
    pending_approvals: list[PendingApproval] = field(default_factory=list)
    saved_run_path: str | None = None


class AgentService:
    """Coordinates an OpenAI Agents SDK run for the kb agent command."""

    def __init__(
        self,
        config: dict[str, Any],
        services: ServiceContainer,
        *,
        recommendation_store: SourceRecommendationStore | None = None,
        research_service: ResearchService | None = None,
        web_research_service: WebResearchService | None = None,
        web_source_acquisition_service: WebSourceAcquisitionService | None = None,
    ) -> None:
        self.config = config
        self.services = services
        self.agent_config = (config or {}).get("agent", {}) or {}
        self.research_config = (config or {}).get("research", {}) or {}
        paths = services.project.paths
        self.recommendation_store = recommendation_store or SourceRecommendationStore(
            paths
        )
        self.web_source_acquisition = (
            web_source_acquisition_service or WebSourceAcquisitionService(paths)
        )
        self.web_research_service = web_research_service or _maybe_build_web_research(
            self.research_config
        )
        self.research_service = research_service or ResearchService(
            paths,
            services.graph_ask_controller,
            self.recommendation_store,
            web_service=self.web_research_service,
        )

    # ------------------------------------------------------------------
    # Public entrypoints
    # ------------------------------------------------------------------
    def run_once(
        self,
        prompt: str,
        *,
        command_context: CommandContext,
        auto_approve: bool = False,
        show_plan: bool = False,
        session_id: str | None = None,
    ) -> AgentRunResult:
        """Synchronously run the agent for one prompt."""
        return asyncio.run(
            self.run_once_async(
                prompt,
                command_context=command_context,
                auto_approve=auto_approve,
                show_plan=show_plan,
                session_id=session_id,
            )
        )

    async def run_once_async(
        self,
        prompt: str,
        *,
        command_context: CommandContext,
        auto_approve: bool = False,
        show_plan: bool = False,
        session_id: str | None = None,
    ) -> AgentRunResult:
        """Run the agent once and persist a durable run record."""
        runtime = self.build_runtime(
            command_context=command_context,
            auto_approve=auto_approve,
            show_plan=show_plan,
            session_id=session_id,
        )
        if not is_agents_sdk_available():
            raise AgentRuntimeError(
                "openai-agents is not installed. "
                "Install the 'agent' extra: poetry install -E agent"
            )
        agent = build_kb_agent(
            model=str(self.agent_config.get("model") or "gpt-5.5"),
            allow_writes=True,
        )
        session = build_session(
            session_id=session_id,
            storage_path=self._session_storage_path(),
        )
        max_turns = int(self.agent_config.get("max_turns", 8) or 8)
        result = await run_agent(
            agent=agent,
            prompt=prompt,
            runtime=runtime,
            session=session,
            max_turns=max_turns,
        )
        final_output = _extract_final_output(result)
        record = self._build_record(
            prompt=prompt,
            runtime=runtime,
            final_output=final_output,
            session_id=session_id,
        )
        saved_path = self._maybe_save_record(record)
        return AgentRunResult(
            record=record,
            pending_approvals=list(runtime.pending_approvals),
            saved_run_path=saved_path,
        )

    # ------------------------------------------------------------------
    # Runtime construction
    # ------------------------------------------------------------------
    def build_runtime(
        self,
        *,
        command_context: CommandContext,
        auto_approve: bool = False,
        show_plan: bool = False,
        session_id: str | None = None,
    ) -> AgentRuntimeContext:
        """Construct an AgentRuntimeContext wired to research and ingest."""
        runtime = AgentRuntimeContext(
            command_context=command_context,
            services=self.services,
            recommendation_store=self.recommendation_store,
            auto_approve=auto_approve,
            show_plan=show_plan,
            session_id=session_id,
        )
        runtime.metadata["research_service"] = self.research_service
        runtime.metadata["web_source_acquisition"] = self.web_source_acquisition
        return runtime

    def build_run_record(
        self,
        *,
        prompt: str,
        runtime: AgentRuntimeContext,
        final_output: str,
        session_id: str | None = None,
    ) -> AgentRunRecord:
        """Public helper to build a run record (used in tests)."""
        return self._build_record(
            prompt=prompt,
            runtime=runtime,
            final_output=final_output,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _session_storage_path(self) -> str | None:
        if self.agent_config.get("session_backend", "sqlite") != "sqlite":
            return None
        paths = self.services.project.paths
        directory = paths.graph_dir / "runs" / "agent"
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory / "sessions.sqlite")

    def _build_record(
        self,
        *,
        prompt: str,
        runtime: AgentRuntimeContext,
        final_output: str,
        session_id: str | None,
    ) -> AgentRunRecord:
        created_at = utc_now_iso()
        run_id = _build_run_id(prompt, created_at=created_at)
        return AgentRunRecord(
            run_id=run_id,
            prompt=prompt,
            created_at=created_at,
            tool_results=list(runtime.tool_results),
            final_output=final_output,
            pending_approvals=list(runtime.pending_approvals),
            session_id=session_id,
        )

    def _maybe_save_record(self, record: AgentRunRecord) -> str | None:
        if not bool(self.agent_config.get("save_runs", True)):
            return None
        paths = self.services.project.paths
        directory = paths.graph_dir / "runs" / "agent"
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"agent-run-{record.run_id}.json"
        target = directory / filename
        payload = json.loads(record.model_dump_json())
        atomic_write_text(target, json.dumps(payload, indent=2, sort_keys=False))
        try:
            return target.relative_to(paths.root).as_posix()
        except ValueError:
            return str(target)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _maybe_build_web_research(
    research_config: dict[str, Any],
) -> WebResearchService | None:
    if not research_config:
        return WebResearchService()
    if not bool(research_config.get("web_enabled", True)):
        return None
    return WebResearchService(
        model=str(research_config.get("web_model") or "gpt-5.5"),
        blocked_domains=list(research_config.get("default_domains_blocklist") or []),
        allowed_domains=list(research_config.get("default_domains_allowlist") or []),
    )


def _build_run_id(prompt: str, *, created_at: str) -> str:
    timestamp = created_at.replace("-", "").replace(":", "")
    slug = slugify(prompt)[:32] or "run"
    return f"{timestamp}_{slug}"


def _extract_final_output(result: Any) -> str:
    if result is None:
        return ""
    final = getattr(result, "final_output", None)
    if isinstance(final, str):
        return final
    if final is not None:
        try:
            return json.dumps(final, default=str)
        except (TypeError, ValueError):
            return str(final)
    output_text = getattr(result, "output_text", None)
    return str(output_text or "")
