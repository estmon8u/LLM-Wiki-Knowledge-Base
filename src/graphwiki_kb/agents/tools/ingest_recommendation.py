"""Agent tool: ingest one or more research recommendations into the KB.

This module belongs to `graphwiki_kb.agents.tools.ingest_recommendation` and
keeps related behavior close to the command, service, model, provider,
storage, script, or test surface that uses it.
"""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AgentToolResult,
    IngestRecommendationInput,
    IngestRecommendationItemResult,
    IngestRecommendationOutput,
    PendingApproval,
)
from graphwiki_kb.services.source_recommendation_store import (
    SourceRecommendationStoreError,
)
from graphwiki_kb.services.web_source_acquisition_service import (
    WebSourceAcquisitionError,
    WebSourceAcquisitionService,
)

TOOL_NAME = "ingest_recommendation"
TOOL_DESCRIPTION = (
    "Stage one or more source recommendations from a research run and ingest "
    "them through the KB ingest pipeline. Mutates the KB and requires approval "
    "unless the agent runtime was launched with auto-approve."
)


def run_ingest_recommendation(
    runtime: AgentRuntimeContext,
    payload: IngestRecommendationInput,
) -> IngestRecommendationOutput:
    """Resolve recommendations and ingest them with approval gating."""
    try:
        record, recommendations = runtime.recommendation_store.resolve_recommendations(
            payload.ids,
            run_id=payload.run_id,
        )
    except SourceRecommendationStoreError as exc:
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=False,
                summary=str(exc),
                error=str(exc),
            )
        )
        return IngestRecommendationOutput(run_id=payload.run_id, results=[])

    if not runtime.auto_approve:
        approval = PendingApproval(
            tool_name=TOOL_NAME,
            summary=(
                f"Ingest {len(recommendations)} recommendation(s) from run "
                f"{record.run_id} into the KB."
            ),
            payload={
                "run_id": record.run_id,
                "ids": [rec.id for rec in recommendations],
                "titles": [rec.title for rec in recommendations],
                "urls": [rec.url for rec in recommendations],
            },
        )
        runtime.add_pending_approval(approval)
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=True,
                summary="awaiting approval before ingest",
                data={"run_id": record.run_id, "pending": len(recommendations)},
            )
        )
        return IngestRecommendationOutput(
            run_id=record.run_id,
            results=[],
            next_command=None,
        )

    acquisition: WebSourceAcquisitionService | None = runtime.metadata.get(
        "web_source_acquisition"
    )
    if acquisition is None:
        raise RuntimeError(
            "ingest_recommendation invoked but web_source_acquisition is not wired."
        )
    ingest_service = runtime.services.ingest
    results: list[IngestRecommendationItemResult] = []
    any_created = False
    for rec in recommendations:
        try:
            staged = acquisition.stage_recommendation(rec, run_id=record.run_id)
        except WebSourceAcquisitionError as exc:
            results.append(
                IngestRecommendationItemResult(
                    id=rec.id,
                    title=rec.title,
                    url=rec.url,
                    created=False,
                    message=f"Failed to stage source: {exc}",
                )
            )
            continue
        try:
            ingest_result = ingest_service.ingest_path(staged.staged_path)
        except (FileNotFoundError, ValueError) as exc:
            results.append(
                IngestRecommendationItemResult(
                    id=rec.id,
                    title=rec.title,
                    url=rec.url,
                    created=False,
                    message=f"Ingest pipeline error: {exc}",
                    staged_path=str(staged.staged_path),
                )
            )
            continue
        source = ingest_result.source or ingest_result.duplicate_of
        results.append(
            IngestRecommendationItemResult(
                id=rec.id,
                title=rec.title,
                url=rec.url,
                created=ingest_result.created,
                message=ingest_result.message,
                staged_path=str(staged.staged_path),
                source_id=source.source_id if source is not None else None,
            )
        )
        any_created = any_created or ingest_result.created
    next_command = "kb update" if any_created else None
    output = IngestRecommendationOutput(
        run_id=record.run_id,
        results=results,
        next_command=next_command,
    )
    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=True,
            summary=(
                f"ingest_recommendation processed {len(results)} item(s), "
                f"{sum(1 for r in results if r.created)} created"
            ),
            data={
                "run_id": record.run_id,
                "next_command": next_command,
            },
        )
    )
    return output
