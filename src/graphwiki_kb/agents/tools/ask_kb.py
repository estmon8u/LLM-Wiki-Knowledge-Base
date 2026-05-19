"""ask_kb agent tool."""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import AskKbInput
from graphwiki_kb.agents.tools._helpers import record_tool, tool_json
from graphwiki_kb.services.graph_ask_controller_service import GraphAskControllerError
from graphwiki_kb.services.research_service import project_ask_output


def run_ask_kb(runtime: AgentRuntimeContext, params: AskKbInput) -> str:
    """Answer a question using the GraphRAG ask controller."""
    controller = runtime.services.graph_ask_controller
    try:
        answer = controller.ask(
            params.question,
            method=params.method,
            save=params.save,
        )
    except GraphAskControllerError as exc:
        record_tool(
            runtime,
            tool_name="ask_kb",
            ok=False,
            summary=str(exc),
            error=str(exc),
        )
        raise
    output = project_ask_output(answer)
    record_tool(
        runtime,
        tool_name="ask_kb",
        ok=True,
        summary=f"Answered via GraphRAG method {output.method}.",
        data=output.model_dump(),
    )
    return tool_json(output)
