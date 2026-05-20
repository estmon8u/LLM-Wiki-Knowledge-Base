"""Shared retrieval engine dispatch for kb ask and kb find."""

from __future__ import annotations

import click

from graphwiki_kb.commands.common import console, echo_section, emit_json, make_table
from graphwiki_kb.models.command_models import CommandContext
from graphwiki_kb.services.graph_ask_controller_service import GraphAskControllerError
from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryError
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryFacade
from graphwiki_kb.wikigraph.deps import wikigraph_extra_hint
from graphwiki_kb.wikigraph.models import WikiGraphAnswer

ASK_ENGINES = ("graphrag", "wikigraph")
FIND_ENGINES = ("graph", "wikigraph")
WIKIGRAPH_ASK_METHODS = ("auto", "basic", "local", "global", "drift-lite")


def normalize_ask_engine(engine: str) -> str:
    """Normalize the ask engine name."""
    normalized = engine.strip().lower()
    if normalized not in ASK_ENGINES:
        supported = ", ".join(ASK_ENGINES)
        raise click.ClickException(
            f"Unsupported ask engine '{engine}'. Use one of: {supported}."
        )
    return normalized


def normalize_find_engine(engine: str) -> str:
    """Normalize the find engine name."""
    normalized = engine.strip().lower()
    if normalized not in FIND_ENGINES:
        supported = ", ".join(FIND_ENGINES)
        raise click.ClickException(
            f"Unsupported find engine '{engine}'. Use one of: {supported}."
        )
    return normalized


def normalize_wikigraph_method(method: str) -> str:
    """Map GraphRAG method aliases onto WikiGraphRAG methods."""
    normalized = method.strip().lower()
    if normalized == "drift":
        return "drift-lite"
    if normalized not in WIKIGRAPH_ASK_METHODS:
        supported = ", ".join(WIKIGRAPH_ASK_METHODS)
        raise click.ClickException(
            f"Unsupported WikiGraphRAG method '{method}'. Use one of: {supported}."
        )
    return normalized


def run_wikigraph_find(
    command_context: CommandContext,
    query: str,
    *,
    method: str,
    limit: int,
    as_json: bool,
) -> None:
    """Run WikiGraphRAG retrieval through kb find."""
    facade = WikiGraphQueryFacade(
        command_context.services.project.paths,
        command_context.config,
    )
    try:
        payload = facade.find(
            query,
            method=normalize_wikigraph_method(method),  # type: ignore[arg-type]
            limit=limit,
        )
    except ImportError as exc:
        raise click.ClickException(
            f"{exc}. Install extras with: {wikigraph_extra_hint()}"
        ) from exc
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"{exc} Run `kb update` to rebuild the WikiGraphRAG index."
        ) from exc
    if as_json:
        emit_json(payload)
        return
    echo_section(f"WikiGraphRAG find ({payload['method']})")
    if payload["matched_entities"]:
        console.print("Matched entities:")
        for entity in payload["matched_entities"]:
            console.print(f"  - {entity['title']} ({entity['score']:.2f})")
    rows = [
        (
            context["title"][:50],
            context["node_kind"],
            f"{context['score']:.2f}",
            (context.get("path") or "")[:40],
        )
        for context in payload["contexts"]
    ]
    if not rows:
        console.print("No WikiGraphRAG contexts matched that query.")
        return
    console.print(
        make_table(
            columns=[
                ("Title", {"style": "bold"}),
                ("Kind", {}),
                ("Score", {"justify": "right"}),
                ("Path", {"style": "dim"}),
            ],
            rows=rows,
            title="WikiGraphRAG Contexts",
        )
    )


def run_wikigraph_ask(
    command_context: CommandContext,
    question: str,
    *,
    method: str,
    save_answer: bool,
) -> WikiGraphAnswer:
    """Run WikiGraphRAG answer generation through kb ask."""
    provider = None
    try:
        provider = command_context.services.query.provider
    except Exception:
        provider = None
    facade = WikiGraphQueryFacade(
        command_context.services.project.paths,
        command_context.config,
        provider=provider,
    )
    try:
        return facade.ask(
            question,
            method=normalize_wikigraph_method(method),  # type: ignore[arg-type]
            save=save_answer,
        )
    except ImportError as exc:
        raise click.ClickException(
            f"{exc}. Install extras with: {wikigraph_extra_hint()}"
        ) from exc
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"{exc} Run `kb update` to rebuild the WikiGraphRAG index."
        ) from exc


def validate_ask_method_for_engine(engine: str, method: str) -> None:
    """Reject method/engine pairings that are not supported."""
    normalized_engine = normalize_ask_engine(engine)
    normalized_method = method.strip().lower()
    if normalized_engine == "wikigraph":
        normalize_wikigraph_method(normalized_method)
        return
    if normalized_method == "drift-lite":
        raise click.ClickException(
            "--method drift-lite is only supported with --engine wikigraph. "
            "Use --method drift for GraphRAG or switch engines."
        )


def graphrag_ask_errors() -> tuple[type[Exception], ...]:
    """Exception types raised by GraphRAG ask."""
    return (GraphAskControllerError, GraphRAGQueryError)
