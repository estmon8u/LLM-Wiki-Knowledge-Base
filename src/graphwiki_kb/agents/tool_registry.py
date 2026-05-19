"""Build OpenAI Agents SDK function tools from KB services."""

from __future__ import annotations

from typing import Any

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AskKbInput,
    FindKbInput,
    IngestRecommendationInput,
    ResearchInput,
    UpdateKbInput,
)
from graphwiki_kb.agents.tools.ask_kb import run_ask_kb
from graphwiki_kb.agents.tools.find_kb import run_find_kb
from graphwiki_kb.agents.tools.ingest_recommendation import run_ingest_recommendation
from graphwiki_kb.agents.tools.lint import run_lint_kb
from graphwiki_kb.agents.tools.research import run_research
from graphwiki_kb.agents.tools.review import run_review_kb
from graphwiki_kb.agents.tools.status import run_status_kb
from graphwiki_kb.agents.tools.update import run_update_kb

_WRITE_TOOLS = frozenset({"ingest_recommendation", "update_kb"})


def agents_sdk_available() -> bool:
    """Return whether the optional openai-agents package is installed."""
    try:
        import agents  # noqa: F401

        return True
    except ImportError:
        return False


def _needs_write_approval(runtime: AgentRuntimeContext, tool_name: str) -> bool:
    if tool_name not in _WRITE_TOOLS:
        return False
    if runtime.auto_approve:
        return False
    agent_cfg = dict(runtime.config.get("agent", {}) or {})
    return bool(agent_cfg.get("require_approval_for_writes", True))


def build_tools(runtime: AgentRuntimeContext) -> list[Any]:
    """Create function tools bound to the current runtime context."""
    if not agents_sdk_available():
        raise RuntimeError(
            "The agent extra is not installed. Run: poetry install --extras agent"
        )
    from agents import function_tool

    @function_tool
    def ask_kb(
        question: str,
        method: str = "auto",
        save: bool = False,
        show_source_trace: bool = False,
    ) -> str:
        """Answer a question from the local GraphRAG knowledge base."""
        params = AskKbInput(
            question=question,
            method=method,  # type: ignore[arg-type]
            save=save,
            show_source_trace=show_source_trace,
        )
        return run_ask_kb(runtime, params)

    @function_tool
    def find_kb(query: str, limit: int = 5) -> str:
        """Find wiki pages and graph entities related to a query."""
        return run_find_kb(runtime, FindKbInput(query=query, limit=limit))

    @function_tool
    def status_kb() -> str:
        """Report project and GraphRAG index health."""
        return run_status_kb(runtime)

    @function_tool
    def lint_kb() -> str:
        """Run deterministic wiki lint checks."""
        return run_lint_kb(runtime)

    @function_tool
    def review_kb() -> str:
        """Run semantic KB quality review (requires provider)."""
        return run_review_kb(runtime)

    @function_tool
    def research(
        question: str,
        use_web: bool = True,
        recommend_sources: bool = True,
        search_context_size: str = "medium",
        max_recommendations: int = 5,
    ) -> str:
        """Research a topic using local KB plus optional web search."""
        params = ResearchInput(
            question=question,
            use_web=use_web,
            recommend_sources=recommend_sources,
            search_context_size=search_context_size,  # type: ignore[arg-type]
            max_recommendations=max_recommendations,
        )
        return run_research(runtime, params)

    ingest_needs = _needs_write_approval(runtime, "ingest_recommendation")
    update_needs = _needs_write_approval(runtime, "update_kb")

    if ingest_needs:

        @function_tool(name_override="ingest_recommendation", needs_approval=True)
        def ingest_recommendation(
            recommendation_ids: list[int],
            run_id: str | None = None,
        ) -> str:
            """Ingest numbered sources from a prior research run."""
            params = IngestRecommendationInput(
                recommendation_ids=recommendation_ids,
                run_id=run_id,
            )
            return run_ingest_recommendation(runtime, params)

    else:

        @function_tool(name_override="ingest_recommendation")
        def ingest_recommendation(
            recommendation_ids: list[int],
            run_id: str | None = None,
        ) -> str:
            """Ingest numbered sources from a prior research run."""
            params = IngestRecommendationInput(
                recommendation_ids=recommendation_ids,
                run_id=run_id,
            )
            return run_ingest_recommendation(runtime, params)

    if update_needs:

        @function_tool(name_override="update_kb", needs_approval=True)
        def update_kb(
            graph_method: str = "auto",
            no_graph: bool = False,
            graph_only: bool = False,
        ) -> str:
            """Update the wiki and GraphRAG index."""
            params = UpdateKbInput(
                graph_method=graph_method,  # type: ignore[arg-type]
                no_graph=no_graph,
                graph_only=graph_only,
            )
            return run_update_kb(runtime, params)

    else:

        @function_tool(name_override="update_kb")
        def update_kb(
            graph_method: str = "auto",
            no_graph: bool = False,
            graph_only: bool = False,
        ) -> str:
            """Update the wiki and GraphRAG index."""
            params = UpdateKbInput(
                graph_method=graph_method,  # type: ignore[arg-type]
                no_graph=no_graph,
                graph_only=graph_only,
            )
            return run_update_kb(runtime, params)

    return [
        ask_kb,
        find_kb,
        status_kb,
        lint_kb,
        review_kb,
        research,
        ingest_recommendation,
        update_kb,
    ]


def tool_names(runtime: AgentRuntimeContext) -> list[str]:
    """Return tool names that would be registered (for --show-plan)."""
    return [
        "ask_kb",
        "find_kb",
        "status_kb",
        "lint_kb",
        "review_kb",
        "research",
        "ingest_recommendation",
        "update_kb",
    ]
