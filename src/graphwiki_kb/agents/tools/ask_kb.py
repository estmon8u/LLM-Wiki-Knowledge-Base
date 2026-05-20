"""Agent tool: ask the local KB through GraphRAG or WikiGraphRAG.

This module belongs to `graphwiki_kb.agents.tools.ask_kb` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.
"""

from __future__ import annotations

import click

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AgentToolResult,
    AskKbInput,
    AskKbOutput,
)
from graphwiki_kb.commands.retrieval_engines import (
    graphrag_ask_errors,
    normalize_ask_engine,
    normalize_wikigraph_method,
    run_wikigraph_ask,
    validate_ask_method_for_engine,
)
from graphwiki_kb.services.research_service import (
    project_ask_kb_output,
    project_wikigraph_ask_kb_output,
)

TOOL_NAME = "ask_kb"
TOOL_DESCRIPTION = (
    "Ask the local GraphWiki KB a question. Default engine is GraphRAG "
    "(graphrag); set engine=wikigraph for the custom WikiGraphRAG index built "
    "by kb update. Use this for any question about the user's KB contents."
)


def _ask_failure_output(
    payload: AskKbInput,
    *,
    warnings: list[str],
) -> AskKbOutput:
    return AskKbOutput(
        answer="",
        method=payload.method,
        staleness_warnings=warnings,
        claim_support="no-answer",
    )


def _record_ask_failure(
    runtime: AgentRuntimeContext,
    payload: AskKbInput,
    *,
    summary: str,
    error: str,
    warnings: list[str],
) -> AskKbOutput:
    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=False,
            summary=summary,
            data={
                "question": payload.question,
                "method": payload.method,
                "engine": payload.engine,
            },
            error=error,
        )
    )
    return _ask_failure_output(payload, warnings=warnings)


def _run_graphrag_ask(
    runtime: AgentRuntimeContext,
    payload: AskKbInput,
) -> AskKbOutput:
    controller = runtime.services.graph_ask_controller
    graph_method = payload.method if payload.method != "drift-lite" else "drift"
    try:
        answer = controller.ask(
            payload.question,
            method=graph_method,
            save=payload.save,
        )
    except graphrag_ask_errors() as exc:
        return _record_ask_failure(
            runtime,
            payload,
            summary="ask_kb failed",
            error=str(exc),
            warnings=[str(exc)],
        )
    except Exception as exc:
        message = f"{exc.__class__.__name__}: {exc}"
        return _record_ask_failure(
            runtime,
            payload,
            summary="ask_kb failed (unexpected error)",
            error=message,
            warnings=[
                "ask_kb failed unexpectedly; "
                "tell the user the KB answer service is unavailable.",
                message,
            ],
        )

    projection = project_ask_kb_output(answer)
    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=True,
            summary=f"Answered via {projection.method} ({projection.claim_support})",
            data={
                "engine": "graphrag",
                "method": projection.method,
                "claim_support": projection.claim_support,
                "saved_path": projection.saved_path,
                "staleness_warnings": projection.staleness_warnings,
            },
        )
    )
    return projection


def _run_wikigraph_ask(
    runtime: AgentRuntimeContext,
    payload: AskKbInput,
) -> AskKbOutput:
    ctx = runtime.command_context
    try:
        answer = run_wikigraph_ask(
            ctx,
            payload.question,
            method=normalize_wikigraph_method(payload.method),
            save_answer=payload.save,
        )
    except click.ClickException as exc:
        return _record_ask_failure(
            runtime,
            payload,
            summary="ask_kb failed (wikigraph)",
            error=str(exc),
            warnings=[str(exc)],
        )
    except Exception as exc:
        message = f"{exc.__class__.__name__}: {exc}"
        return _record_ask_failure(
            runtime,
            payload,
            summary="ask_kb failed (wikigraph, unexpected error)",
            error=message,
            warnings=[
                "WikiGraphRAG ask failed unexpectedly; run `kb update` to rebuild.",
                message,
            ],
        )

    projection = project_wikigraph_ask_kb_output(answer)
    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=True,
            summary=(
                f"Answered via wikigraph/{projection.method} "
                f"({projection.claim_support})"
            ),
            data={
                "engine": "wikigraph",
                "method": projection.method,
                "claim_support": projection.claim_support,
                "staleness_warnings": projection.staleness_warnings,
            },
        )
    )
    return projection


def run_ask_kb(
    runtime: AgentRuntimeContext,
    payload: AskKbInput,
) -> AskKbOutput:
    """Execute ask_kb against GraphRAG or WikiGraphRAG.

    Returns an ``AskKbOutput`` even for unexpected failures so the agent
    runtime always sees a structured result and records a trace entry.
    """
    try:
        engine = normalize_ask_engine(payload.engine)
        validate_ask_method_for_engine(engine, payload.method)
    except click.ClickException as exc:
        return _record_ask_failure(
            runtime,
            payload,
            summary="ask_kb rejected invalid engine/method",
            error=str(exc),
            warnings=[str(exc)],
        )

    if engine == "wikigraph":
        return _run_wikigraph_ask(runtime, payload)
    return _run_graphrag_ask(runtime, payload)
