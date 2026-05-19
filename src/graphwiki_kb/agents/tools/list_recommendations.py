"""list_recommendations agent tool."""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import ListRecommendationsOutput
from graphwiki_kb.agents.tools._helpers import record_tool, tool_json
from graphwiki_kb.services.source_recommendation_store import SourceRecommendationStore


def run_list_recommendations(
    runtime: AgentRuntimeContext,
    *,
    run_id: str | None = None,
) -> str:
    """List numbered recommendations from a persisted research run (disk only)."""
    store = SourceRecommendationStore(runtime.services.project.paths)
    if run_id:
        record = store.load_run(run_id)
    else:
        record = store.latest_run_with_recommendations()
    if record is None:
        output = ListRecommendationsOutput(
            run_id=None,
            question=None,
            recommendations=[],
            message="No research run with recommendations found. Run research first.",
        )
        record_tool(
            runtime,
            tool_name="list_recommendations",
            ok=True,
            summary=output.message,
            data=output.model_dump(),
        )
        return tool_json(output)

    output = ListRecommendationsOutput(
        run_id=record.run_id,
        question=record.question,
        recommendations=record.recommendations,
        message=(
            f"Found {len(record.recommendations)} recommendation(s) "
            f"from run {record.run_id}. No sources were added."
        ),
    )
    record_tool(
        runtime,
        tool_name="list_recommendations",
        ok=True,
        summary=output.message,
        data=output.model_dump(),
    )
    return tool_json(output)
