"""Agent tool: research a topic against local KB plus optional web search.

This module belongs to `graphwiki_kb.agents.tools.research` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.
"""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AgentToolResult,
    ResearchInput,
    ResearchOutput,
)
from graphwiki_kb.services.research_service import ResearchService

TOOL_NAME = "research"
TOOL_DESCRIPTION = (
    "Research a topic. Combines the local KB answer with optional OpenAI "
    "web_search-backed findings and produces durable, numbered source "
    "recommendations. Never ingests sources by itself."
)


def run_research(
    runtime: AgentRuntimeContext,
    payload: ResearchInput,
) -> ResearchOutput:
    """Build a ResearchService and execute it for one question."""
    service: ResearchService | None = runtime.metadata.get("research_service")
    if service is None:
        raise RuntimeError(
            "research tool invoked but research_service is not wired. "
            "Make sure the agent runtime was built with build_kb_agent_runtime()."
        )
    result = service.research(payload)
    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=True,
            summary=(
                f"research produced {len(result.recommendations)} recommendation(s) "
                f"and {len(result.web_findings)} web finding(s)"
            ),
            data={
                "run_id": result.run_id,
                "web_used": result.web_used,
                "recommendation_count": len(result.recommendations),
                "saved_report_path": result.saved_report_path,
            },
        )
    )
    return result
