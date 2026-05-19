"""research agent tool."""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import ResearchInput
from graphwiki_kb.agents.tools._helpers import record_tool, tool_json
from graphwiki_kb.services.research_service import ResearchService


def run_research(runtime: AgentRuntimeContext, params: ResearchInput) -> str:
    """Run local KB answer plus optional web research and recommendations."""
    service = ResearchService(
        runtime.services.graph_ask_controller.paths,
        dict(runtime.config),
        ask_controller=runtime.services.graph_ask_controller,
    )
    output = service.run(
        question=params.question,
        use_web=params.use_web,
        recommend_sources=params.recommend_sources,
        search_context_size=params.search_context_size,
        max_recommendations=params.max_recommendations,
    )
    record_tool(
        runtime,
        tool_name="research",
        ok=True,
        summary=(
            f"Research run {output.run_id} with "
            f"{len(output.recommendations)} recommendation(s). "
            "No sources were added."
        ),
        data=output.model_dump(),
    )
    return tool_json(output)
