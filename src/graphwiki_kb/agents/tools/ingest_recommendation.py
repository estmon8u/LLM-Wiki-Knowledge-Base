"""ingest_recommendation agent tool."""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    IngestRecommendationInput,
    IngestRecommendationOutput,
)
from graphwiki_kb.agents.tools._helpers import record_tool, tool_json
from graphwiki_kb.services.source_recommendation_store import SourceRecommendationStore
from graphwiki_kb.services.web_source_acquisition_service import (
    WebSourceAcquisitionError,
    WebSourceAcquisitionService,
)


def run_ingest_recommendation(
    runtime: AgentRuntimeContext,
    params: IngestRecommendationInput,
) -> str:
    """Stage and ingest selected research recommendations."""
    store = SourceRecommendationStore(runtime.services.project.paths)
    acquisition = WebSourceAcquisitionService(runtime.services.ingest.paths)
    record, recommendations = store.resolve_recommendations(
        params.recommendation_ids,
        run_id=params.run_id,
    )
    ingested: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for recommendation in recommendations:
        try:
            staged = acquisition.stage(recommendation, run_id=record.run_id)
            result = runtime.services.ingest.ingest_path(staged.staged_path)
            ingested.append(
                {
                    "recommendation_id": recommendation.id,
                    "title": recommendation.title,
                    "url": recommendation.url,
                    "destination": staged.destination,
                    "created": result.created,
                    "message": result.message,
                }
            )
        except (WebSourceAcquisitionError, ValueError) as exc:
            skipped.append(
                {
                    "recommendation_id": recommendation.id,
                    "title": recommendation.title,
                    "reason": str(exc),
                }
            )
    output = IngestRecommendationOutput(ingested=ingested, skipped=skipped)
    record_tool(
        runtime,
        tool_name="ingest_recommendation",
        ok=bool(ingested),
        summary=f"Ingested {len(ingested)} source(s); skipped {len(skipped)}.",
        data=output.model_dump(),
    )
    return tool_json(output)
