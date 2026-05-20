"""Agent tool: list persisted research recommendations from disk.

This module belongs to ``graphwiki_kb.agents.tools.list_recommendations`` and
keeps related behavior close to the command, service, model, provider,
storage, script, or test surface that uses it.

The tool is a read-only complement to ``research``: it does not call the web
or the local KB, it simply returns the numbered recommendations saved by a
previous research run. The agent should call this tool whenever the user asks
to see prior recommendations or before invoking ``ingest_recommendation`` so
that recommendation IDs are resolved against persisted data.
"""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AgentToolResult,
    ListRecommendationsInput,
    ListRecommendationsOutput,
)
from graphwiki_kb.services.source_recommendation_store import (
    SourceRecommendationStoreError,
)

TOOL_NAME = "list_recommendations"
TOOL_DESCRIPTION = (
    "List numbered source recommendations from a persisted research run "
    "(reads disk only, no web access). Use this for prompts like "
    "'show previous recommendations' or before ingest_recommendation. "
    "Defaults to the most recent run that has at least one recommendation."
)


def run_list_recommendations(
    runtime: AgentRuntimeContext,
    payload: ListRecommendationsInput,
) -> ListRecommendationsOutput:
    """Return persisted recommendations for the requested research run."""
    store = runtime.recommendation_store
    run_id = payload.run_id or "latest"
    record = None
    error_message: str | None = None
    try:
        if run_id == "latest":
            record = store.latest_with_recommendations() or store.latest()
        else:
            record = store.load(run_id)
    except SourceRecommendationStoreError as exc:
        error_message = str(exc)

    if record is None or not record.recommendations:
        message = (
            error_message
            or "No research run with recommendations was found. "
            "Run `research` first."
        )
        output = ListRecommendationsOutput(
            run_id=record.run_id if record is not None else None,
            question=record.question if record is not None else None,
            created_at=record.created_at if record is not None else None,
            recommendations=[],
            message=message,
        )
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=record is not None,
                summary=message,
                data=output.model_dump(),
            )
        )
        return output

    message = (
        f"Found {len(record.recommendations)} recommendation(s) from run "
        f"{record.run_id}. No sources were added by this call."
    )
    output = ListRecommendationsOutput(
        run_id=record.run_id,
        question=record.question,
        created_at=record.created_at,
        recommendations=list(record.recommendations),
        message=message,
    )
    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=True,
            summary=message,
            data={
                "run_id": record.run_id,
                "recommendation_count": len(record.recommendations),
            },
        )
    )
    return output
