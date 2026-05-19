"""Agent tool: ask the local KB through the GraphRAG ask controller.

This module belongs to `graphwiki_kb.agents.tools.ask_kb` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.
"""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AgentToolResult,
    AskKbInput,
    AskKbOutput,
)
from graphwiki_kb.services.graph_ask_controller_service import GraphAskControllerError
from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryError
from graphwiki_kb.services.research_service import project_ask_kb_output

TOOL_NAME = "ask_kb"
TOOL_DESCRIPTION = (
    "Ask the local GraphWiki KB a question. Routes through the GraphRAG-aware "
    "answer controller. Use this for any question about the user's KB contents."
)


def run_ask_kb(
    runtime: AgentRuntimeContext,
    payload: AskKbInput,
) -> AskKbOutput:
    """Execute ask_kb against the GraphAskControllerService.

    Returns an ``AskKbOutput`` even for unexpected failures so the agent
    runtime always sees a structured result and records a trace entry. This
    prevents low-level errors (subprocess fallback failures, missing
    GraphRAG binary on PATH, etc.) from surfacing to the LLM as raw
    ``Error: ...`` strings and from being omitted from the run trace.
    """
    controller = runtime.services.graph_ask_controller
    try:
        answer = controller.ask(
            payload.question,
            method=payload.method,
            save=payload.save,
        )
    except (GraphAskControllerError, GraphRAGQueryError) as exc:
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=False,
                summary="ask_kb failed",
                data={"question": payload.question, "method": payload.method},
                error=str(exc),
            )
        )
        return AskKbOutput(
            answer="",
            method=payload.method,
            staleness_warnings=[str(exc)],
            claim_support="no-answer",
        )
    except Exception as exc:
        message = f"{exc.__class__.__name__}: {exc}"
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=False,
                summary="ask_kb failed (unexpected error)",
                data={"question": payload.question, "method": payload.method},
                error=message,
            )
        )
        return AskKbOutput(
            answer="",
            method=payload.method,
            staleness_warnings=[
                "ask_kb failed unexpectedly; "
                "tell the user the KB answer service is unavailable.",
                message,
            ],
            claim_support="no-answer",
        )

    projection = project_ask_kb_output(answer)
    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=True,
            summary=f"Answered via {projection.method} ({projection.claim_support})",
            data={
                "method": projection.method,
                "claim_support": projection.claim_support,
                "saved_path": projection.saved_path,
                "staleness_warnings": projection.staleness_warnings,
            },
        )
    )
    return projection
