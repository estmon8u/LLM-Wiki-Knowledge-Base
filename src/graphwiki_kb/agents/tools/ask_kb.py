"""Agent tool: ask the local KB through either backend.

This module belongs to `graphwiki_kb.agents.tools.ask_kb` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.

The tool routes to one of two backends:

* ``engine="graphrag"`` (default) -- Microsoft GraphRAG via the existing
  :class:`GraphAskControllerService`.
* ``engine="wikigraph"`` -- the custom WikiGraphRAG backend via
  :class:`WikiGraphQueryService`.

Method validation matches ``kb ask --engine ... --method ...`` so the agent
sees the same friendly errors as a human operator on the CLI.
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
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryError
from graphwiki_kb.wikigraph.models import QueryMethod, WikiGraphAnswer

TOOL_NAME = "ask_kb"
TOOL_DESCRIPTION = (
    "Ask the local GraphWiki KB a question. Routes through the GraphRAG-aware "
    "answer controller by default, or the custom WikiGraphRAG backend when "
    "engine='wikigraph'. Use this for any question about the user's KB contents."
)

_GRAPHRAG_METHODS: frozenset[str] = frozenset(
    {"auto", "basic", "local", "global", "drift"}
)
_WIKIGRAPH_METHODS: frozenset[str] = frozenset(
    {"auto", "basic", "local", "global", "drift-lite"}
)


def run_ask_kb(
    runtime: AgentRuntimeContext,
    payload: AskKbInput,
) -> AskKbOutput:
    """Execute ask_kb against the configured backend.

    Returns an ``AskKbOutput`` even for unexpected failures so the agent
    runtime always sees a structured result and records a trace entry.
    """
    if payload.engine == "wikigraph":
        return _ask_wikigraph(runtime, payload)
    return _ask_graphrag(runtime, payload)


def _ask_graphrag(runtime: AgentRuntimeContext, payload: AskKbInput) -> AskKbOutput:
    if payload.method not in _GRAPHRAG_METHODS:
        message = (
            f"--method={payload.method!r} is not valid for engine='graphrag'. "
            f"Choose one of: {', '.join(sorted(_GRAPHRAG_METHODS))}."
        )
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=False,
                summary="ask_kb refused: invalid method for engine=graphrag",
                data={
                    "engine": payload.engine,
                    "method": payload.method,
                },
                error=message,
            )
        )
        return AskKbOutput(
            answer="",
            method=payload.method,
            staleness_warnings=[message],
            claim_support="no-answer",
        )

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
                data={
                    "engine": payload.engine,
                    "question": payload.question,
                    "method": payload.method,
                },
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
                data={
                    "engine": payload.engine,
                    "question": payload.question,
                    "method": payload.method,
                },
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
            summary=(
                f"Answered via graphrag/{projection.method} "
                f"({projection.claim_support})"
            ),
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


def _ask_wikigraph(runtime: AgentRuntimeContext, payload: AskKbInput) -> AskKbOutput:
    if payload.method not in _WIKIGRAPH_METHODS:
        message = (
            f"--method={payload.method!r} is not valid for engine='wikigraph'. "
            f"Choose one of: {', '.join(sorted(_WIKIGRAPH_METHODS))}."
        )
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=False,
                summary="ask_kb refused: invalid method for engine=wikigraph",
                data={
                    "engine": payload.engine,
                    "method": payload.method,
                },
                error=message,
            )
        )
        return AskKbOutput(
            answer="",
            method=payload.method,
            staleness_warnings=[message],
            claim_support="no-answer",
        )

    service = runtime.services.wikigraph_query
    method: QueryMethod = payload.method  # type: ignore[assignment]
    try:
        answer: WikiGraphAnswer = service.ask(
            payload.question,
            method=method,
            save=payload.save,
        )
    except WikiGraphQueryError as exc:
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=False,
                summary="ask_kb failed (wikigraph)",
                data={
                    "engine": payload.engine,
                    "question": payload.question,
                    "method": payload.method,
                },
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
                summary="ask_kb failed (wikigraph unexpected error)",
                data={
                    "engine": payload.engine,
                    "question": payload.question,
                    "method": payload.method,
                },
                error=message,
            )
        )
        return AskKbOutput(
            answer="",
            method=payload.method,
            staleness_warnings=[
                "ask_kb failed unexpectedly via wikigraph; "
                "tell the user the WikiGraphRAG answer service is unavailable.",
                message,
            ],
            claim_support="no-answer",
        )

    projection = _project_wikigraph_answer(answer)
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
                "saved_path": projection.saved_path,
                "staleness_warnings": projection.staleness_warnings,
            },
        )
    )
    return projection


def _project_wikigraph_answer(answer: WikiGraphAnswer) -> AskKbOutput:
    """Project a :class:`WikiGraphAnswer` onto the agent ``AskKbOutput`` shape.

    The agent contract is GraphRAG-shaped (planner/route_reason/source_trace);
    WikiGraphRAG does not produce all of those fields, so the projection
    fills the closest equivalents:

    * ``claim_support``: ``cited-graph-answer`` when at least one citation is
      attached and the synthesis is not flagged insufficient; ``no-answer``
      when ``insufficient_evidence`` is true and no answer text was produced;
      ``unverified`` otherwise.
    * ``source_trace`` carries the wikigraph provider mode plus a comma-joined
      list of context ``citation_ref`` values for the cited contexts.
    """
    has_answer = bool(answer.answer.strip())
    claim_support = "unverified"
    if not has_answer:
        claim_support = "no-answer"
    elif answer.insufficient_evidence:
        # WikiGraphRAG's "insufficient_evidence" does not mean the index is
        # stale -- it means there were not enough contexts (empty index, bad
        # query, or missing topic). Use the dedicated label instead of
        # overloading ``stale-index``.
        claim_support = "insufficient-evidence"
    elif answer.citations:
        claim_support = "cited-graph-answer"

    provider = answer.provider_status or {}
    citation_refs = ",".join(
        ctx.citation_ref for ctx in answer.contexts if ctx.citation_ref
    )
    source_trace: dict[str, str | None] = {
        "engine": "wikigraph",
        "method": answer.method,
        "provider_mode": str(provider.get("mode", "")) or None,
        "provider": str(provider.get("provider", "")) or None,
        "model": str(provider.get("model", "")) or None,
        "citation_refs": citation_refs or None,
    }
    return AskKbOutput(
        answer=answer.answer or "",
        method=answer.method,
        planner="wikigraph",
        route_reason=None,
        route_confidence=None,
        claim_support=claim_support,
        staleness_warnings=list(answer.warnings),
        source_trace=source_trace,
        saved_path=answer.saved_path,
        index_run_id=None,
    )
