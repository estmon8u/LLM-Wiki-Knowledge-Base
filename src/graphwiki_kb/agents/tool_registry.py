"""Build OpenAI Agents SDK function tools from KB service wrappers.

This module belongs to `graphwiki_kb.agents.tool_registry` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.

The functions exposed here intentionally accept a Pydantic model (or no
arguments) and return a Pydantic model so the SDK can build strict JSON
schemas. Each wrapper records a trace entry on the runtime context before
returning so a durable run record can be produced even when the SDK is
mocked in tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

try:  # The SDK is optional; only required to build the FunctionTool list.
    from agents import RunContextWrapper
except ImportError:  # pragma: no cover - optional extra
    RunContextWrapper = None

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AskKbInput,
    AskKbOutput,
    FindKbInput,
    FindKbOutput,
    IngestRecommendationInput,
    IngestRecommendationOutput,
    LintOutput,
    ResearchInput,
    ResearchOutput,
    ReviewOutput,
    StatusOutput,
    UpdateInput,
    UpdateOutput,
)
from graphwiki_kb.agents.tools.ask_kb import TOOL_DESCRIPTION as ASK_KB_DESC
from graphwiki_kb.agents.tools.ask_kb import run_ask_kb
from graphwiki_kb.agents.tools.find_kb import TOOL_DESCRIPTION as FIND_KB_DESC
from graphwiki_kb.agents.tools.find_kb import run_find_kb
from graphwiki_kb.agents.tools.ingest_recommendation import (
    TOOL_DESCRIPTION as INGEST_DESC,
)
from graphwiki_kb.agents.tools.ingest_recommendation import (
    run_ingest_recommendation,
)
from graphwiki_kb.agents.tools.lint import TOOL_DESCRIPTION as LINT_DESC
from graphwiki_kb.agents.tools.lint import run_lint
from graphwiki_kb.agents.tools.research import TOOL_DESCRIPTION as RESEARCH_DESC
from graphwiki_kb.agents.tools.research import run_research
from graphwiki_kb.agents.tools.review import TOOL_DESCRIPTION as REVIEW_DESC
from graphwiki_kb.agents.tools.review import run_review
from graphwiki_kb.agents.tools.status import TOOL_DESCRIPTION as STATUS_DESC
from graphwiki_kb.agents.tools.status import run_status
from graphwiki_kb.agents.tools.update import TOOL_DESCRIPTION as UPDATE_DESC
from graphwiki_kb.agents.tools.update import run_update_kb

if TYPE_CHECKING:
    from agents import FunctionTool

# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------

READ_ONLY_TOOL_NAMES: tuple[str, ...] = (
    "ask_kb",
    "find_kb",
    "status",
    "lint",
    "review",
    "research",
)
WRITE_TOOL_NAMES: tuple[str, ...] = (
    "ingest_recommendation",
    "update_kb",
)
ALL_TOOL_NAMES: tuple[str, ...] = READ_ONLY_TOOL_NAMES + WRITE_TOOL_NAMES


# ---------------------------------------------------------------------------
# Direct callables — useful for tests and when the SDK is unavailable.
# ---------------------------------------------------------------------------


def ask_kb_callable(
    runtime: AgentRuntimeContext,
    payload: AskKbInput | dict[str, Any],
) -> AskKbOutput:
    """Direct callable wrapper for ask_kb."""
    return run_ask_kb(runtime, _coerce(payload, AskKbInput))


def find_kb_callable(
    runtime: AgentRuntimeContext,
    payload: FindKbInput | dict[str, Any],
) -> FindKbOutput:
    """Direct callable wrapper for find_kb."""
    return run_find_kb(runtime, _coerce(payload, FindKbInput))


def status_callable(runtime: AgentRuntimeContext) -> StatusOutput:
    """Direct callable wrapper for status."""
    return run_status(runtime)


def lint_callable(runtime: AgentRuntimeContext) -> LintOutput:
    """Direct callable wrapper for lint."""
    return run_lint(runtime)


def review_callable(runtime: AgentRuntimeContext) -> ReviewOutput:
    """Direct callable wrapper for review."""
    return run_review(runtime)


def research_callable(
    runtime: AgentRuntimeContext,
    payload: ResearchInput | dict[str, Any],
) -> ResearchOutput:
    """Direct callable wrapper for research."""
    return run_research(runtime, _coerce(payload, ResearchInput))


def ingest_recommendation_callable(
    runtime: AgentRuntimeContext,
    payload: IngestRecommendationInput | dict[str, Any],
) -> IngestRecommendationOutput:
    """Direct callable wrapper for ingest_recommendation."""
    return run_ingest_recommendation(
        runtime, _coerce(payload, IngestRecommendationInput)
    )


def update_kb_callable(
    runtime: AgentRuntimeContext,
    payload: UpdateInput | dict[str, Any],
) -> UpdateOutput:
    """Direct callable wrapper for update_kb."""
    return run_update_kb(runtime, _coerce(payload, UpdateInput))


CALLABLE_REGISTRY: dict[str, Any] = {
    "ask_kb": ask_kb_callable,
    "find_kb": find_kb_callable,
    "status": status_callable,
    "lint": lint_callable,
    "review": review_callable,
    "research": research_callable,
    "ingest_recommendation": ingest_recommendation_callable,
    "update_kb": update_kb_callable,
}


# ---------------------------------------------------------------------------
# Agents SDK function-tool builder
# ---------------------------------------------------------------------------


def build_agent_tools(
    *,
    allow_writes: bool = True,
) -> list[FunctionTool]:
    """Build the list of OpenAI Agents SDK FunctionTools.

    Each tool's callable accepts a ``RunContextWrapper`` whose
    ``.context`` is an :class:`AgentRuntimeContext`. The first parameter must
    be annotated as ``RunContextWrapper`` so the SDK strips it from the
    generated JSON schema.

    The wrappers are intentionally **synchronous**: GraphRAG (via
    ``graphrag_llm``) calls ``nest_asyncio2.apply()`` at import time, which
    makes nested ``asyncio.run`` calls safe on the agent's main thread. The
    indexing entrypoint also registers POSIX signal handlers, which require
    the main thread; offloading to ``asyncio.to_thread`` would break
    ``update_kb`` with ``signal only works in main thread of the main
    interpreter``. The OpenAI Agents SDK happily invokes sync function tools
    inside its async loop.
    """
    from agents import function_tool

    @function_tool(
        name_override="ask_kb",
        description_override=ASK_KB_DESC,
    )
    def ask_kb_tool(
        ctx: RunContextWrapper,
        payload: AskKbInput,
    ) -> AskKbOutput:
        return run_ask_kb(_runtime(ctx), payload)

    @function_tool(
        name_override="find_kb",
        description_override=FIND_KB_DESC,
    )
    def find_kb_tool(
        ctx: RunContextWrapper,
        payload: FindKbInput,
    ) -> FindKbOutput:
        return run_find_kb(_runtime(ctx), payload)

    @function_tool(
        name_override="status",
        description_override=STATUS_DESC,
    )
    def status_tool(ctx: RunContextWrapper) -> StatusOutput:
        return run_status(_runtime(ctx))

    @function_tool(
        name_override="lint",
        description_override=LINT_DESC,
    )
    def lint_tool(ctx: RunContextWrapper) -> LintOutput:
        return run_lint(_runtime(ctx))

    @function_tool(
        name_override="review",
        description_override=REVIEW_DESC,
    )
    def review_tool(ctx: RunContextWrapper) -> ReviewOutput:
        return run_review(_runtime(ctx))

    @function_tool(
        name_override="research",
        description_override=RESEARCH_DESC,
    )
    def research_tool(
        ctx: RunContextWrapper,
        payload: ResearchInput,
    ) -> ResearchOutput:
        return run_research(_runtime(ctx), payload)

    tools = [
        ask_kb_tool,
        find_kb_tool,
        status_tool,
        lint_tool,
        review_tool,
        research_tool,
    ]
    if allow_writes:

        @function_tool(
            name_override="ingest_recommendation",
            description_override=INGEST_DESC,
        )
        def ingest_tool(
            ctx: RunContextWrapper,
            payload: IngestRecommendationInput,
        ) -> IngestRecommendationOutput:
            return run_ingest_recommendation(_runtime(ctx), payload)

        @function_tool(
            name_override="update_kb",
            description_override=UPDATE_DESC,
        )
        def update_tool(
            ctx: RunContextWrapper,
            payload: UpdateInput,
        ) -> UpdateOutput:
            return run_update_kb(_runtime(ctx), payload)

        tools.extend([ingest_tool, update_tool])
    return tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runtime(ctx: Any) -> AgentRuntimeContext:
    runtime = getattr(ctx, "context", None) if ctx is not None else None
    if not isinstance(runtime, AgentRuntimeContext):
        raise RuntimeError(
            "Tool was invoked without an AgentRuntimeContext. "
            "Pass context=AgentRuntimeContext(...) to Runner.run()."
        )
    return runtime


def _coerce(payload: Any, model_cls: type[BaseModel]) -> BaseModel:
    if isinstance(payload, model_cls):
        return payload
    if isinstance(payload, dict):
        return model_cls.model_validate(payload)
    if payload is None:
        return model_cls()
    return model_cls.model_validate(payload)
