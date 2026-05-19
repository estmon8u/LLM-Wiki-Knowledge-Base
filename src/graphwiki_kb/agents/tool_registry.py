"""Build OpenAI Agents SDK function tools from KB services."""

from __future__ import annotations

from typing import Any

from graphwiki_kb.agents.config_helpers import config_section
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
from graphwiki_kb.agents.tools.list_recommendations import run_list_recommendations
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
    agent_cfg = config_section(runtime.config, "agent")
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
        params = AskKbInput.model_validate(
            {
                "question": question,
                "method": method,
                "save": save,
                "show_source_trace": show_source_trace,
            }
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
        """Run NEW research (local KB + optional web). Not for listing prior recommendations."""
        params = ResearchInput.model_validate(
            {
                "question": question,
                "use_web": use_web,
                "recommend_sources": recommend_sources,
                "search_context_size": search_context_size,
                "max_recommendations": max_recommendations,
            }
        )
        return run_research(runtime, params)

    @function_tool(name_override="list_recommendations")
    def list_recommendations(run_id: str | None = None) -> str:
        """List saved numbered recommendations from disk. Use for 'previous recommendations' or before ingest."""
        return run_list_recommendations(runtime, run_id=run_id)

    ingest_needs = _needs_write_approval(runtime, "ingest_recommendation")
    update_needs = _needs_write_approval(runtime, "update_kb")

    if ingest_needs:

        @function_tool(name_override="ingest_recommendation", needs_approval=True)
        def ingest_recommendation(
            recommendation_ids: list[int],
            run_id: str | None = None,
        ) -> str:
            """Ingest numbered sources from a saved research run. Use list_recommendations if ids are unclear."""
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
            """Ingest numbered sources from a saved research run. Use list_recommendations if ids are unclear."""
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
            params = UpdateKbInput.model_validate(
                {
                    "graph_method": graph_method,
                    "no_graph": no_graph,
                    "graph_only": graph_only,
                }
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
            params = UpdateKbInput.model_validate(
                {
                    "graph_method": graph_method,
                    "no_graph": no_graph,
                    "graph_only": graph_only,
                }
            )
            return run_update_kb(runtime, params)

    return [
        ask_kb,
        find_kb,
        status_kb,
        lint_kb,
        review_kb,
        research,
        list_recommendations,
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
        "list_recommendations",
        "ingest_recommendation",
        "update_kb",
    ]
