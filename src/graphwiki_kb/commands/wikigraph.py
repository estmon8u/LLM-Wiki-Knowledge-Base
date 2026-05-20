"""Click command implementation for the kb wikigraph command group.

This module belongs to :mod:`graphwiki_kb.commands.wikigraph` and exposes
the WikiGraphRAG backend through ``kb wikigraph build``,
``kb wikigraph status``, ``kb wikigraph find``, and ``kb wikigraph ask``.
"""

from __future__ import annotations

from typing import Any

import click
from rich.markdown import Markdown as RichMarkdown

from graphwiki_kb.commands.common import (
    console,
    echo_bullet,
    echo_kv,
    echo_section,
    emit_json,
    make_table,
    require_initialized,
)
from graphwiki_kb.models.command_models import CommandContext, CommandSpec
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryError
from graphwiki_kb.wikigraph.models import WikiGraphAnswer, WikiGraphFindResult

SUMMARY = "WikiGraphRAG backend: build, inspect, and query a custom wiki graph index."

_METHOD_CHOICES = ["auto", "basic", "local", "global", "drift-lite"]


def build_spec(_: CommandContext | None = None) -> CommandSpec:
    """Builds the command registry specification for this module."""
    return CommandSpec(name="wikigraph", summary=SUMMARY)


def create_command() -> click.Command:
    """Creates the Click command exposed by this module."""

    @click.group(
        name="wikigraph",
        help=SUMMARY,
        short_help="Custom WikiGraphRAG backend.",
    )
    def wikigraph_group() -> None:
        """Group entry point."""

    @wikigraph_group.command(
        name="build",
        help="Build the WikiGraphRAG index from wiki/sources, wiki/concepts, wiki/analysis.",
        short_help="Build wiki graph index.",
    )
    @click.option(
        "--include-graphrag-export-pages",
        is_flag=True,
        help="Include wiki/graph (GraphRAG-exported) pages for an ablation.",
    )
    @click.option(
        "--chunk-char-limit",
        type=click.IntRange(min=200, max=5000),
        default=1200,
        show_default=True,
        help="Maximum characters per section-level chunk.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def build_cmd(
        command_context: CommandContext,
        include_graphrag_export_pages: bool,
        chunk_char_limit: int,
        as_json: bool,
    ) -> None:
        """Build command implementation."""
        require_initialized(command_context)
        service = command_context.services.wikigraph_index
        report = service.build(
            include_graphrag_export_pages=include_graphrag_export_pages,
            chunk_char_limit=chunk_char_limit,
        )
        if as_json:
            emit_json(report.model_dump())
            return
        echo_section("WikiGraphRAG Build")
        echo_kv("Built at", report.built_at)
        echo_kv("Nodes", report.node_count)
        echo_kv("Edges", report.edge_count)
        echo_kv("Chunks", report.chunk_count)
        echo_kv("Entities", report.entity_count)
        echo_kv("Communities", report.community_count)
        echo_kv("Source pages", report.source_count)
        echo_kv(
            "GraphRAG export pages included",
            "yes" if report.include_graphrag_export_pages else "no",
        )
        if report.warnings:
            echo_section("Warnings")
            for warning in report.warnings:
                echo_bullet(warning)
        echo_section("Artifacts")
        for artifact in report.artifacts:
            echo_bullet(artifact)

    @wikigraph_group.command(
        name="status",
        help="Show WikiGraphRAG index status and counts.",
        short_help="Wiki graph index status.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def status_cmd(command_context: CommandContext, as_json: bool) -> None:
        """Status command implementation."""
        require_initialized(command_context)
        service = command_context.services.wikigraph_index
        snapshot = service.status()
        if as_json:
            emit_json(snapshot)
            return
        echo_section("WikiGraphRAG Status")
        for key, value in snapshot.items():
            echo_kv(key, str(value))

    @wikigraph_group.command(
        name="find",
        help="Retrieve evidence from the WikiGraphRAG index without provider calls.",
        short_help="Wiki graph search.",
    )
    @click.argument("query_terms", nargs=-1)
    @click.option(
        "--method",
        type=click.Choice(_METHOD_CHOICES),
        default="auto",
        show_default=True,
    )
    @click.option("--limit", default=8, type=click.IntRange(min=1, max=50))
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def find_cmd(
        command_context: CommandContext,
        query_terms: tuple[str, ...],
        method: str,
        limit: int,
        as_json: bool,
    ) -> None:
        """Find command implementation."""
        require_initialized(command_context)
        if not query_terms:
            raise click.ClickException("Provide at least one search term.")
        query_service = command_context.services.wikigraph_query
        question = " ".join(query_terms).strip()
        try:
            result = query_service.find(question, method=method)  # type: ignore[arg-type]
        except WikiGraphQueryError as exc:
            raise click.ClickException(str(exc)) from exc

        if as_json:
            emit_json(_find_payload(result, limit=limit))
            return

        echo_section(f"WikiGraphRAG Find ({result.method})")
        if result.entities:
            console.print("Matched entities: " + ", ".join(result.entities))
        if result.communities:
            console.print("Selected communities: " + ", ".join(result.communities))
        if not result.contexts:
            console.print("No wiki contexts retrieved.")
            return
        rows = [
            (
                ctx.title,
                ctx.citation_ref,
                f"{ctx.score:.3f}",
                (ctx.text[:120] + "...") if len(ctx.text) > 120 else ctx.text,
            )
            for ctx in result.contexts[:limit]
        ]
        table = make_table(
            columns=[
                ("Title", {"style": "bold"}),
                ("Citation", {}),
                ("Score", {"justify": "right"}),
                ("Snippet", {"style": "dim"}),
            ],
            rows=rows,
            title="WikiGraphRAG Contexts",
        )
        console.print(table)

    @wikigraph_group.command(
        name="ask",
        help="Answer a question using the WikiGraphRAG backend.",
        short_help="Ask the WikiGraphRAG backend.",
    )
    @click.argument("question_terms", nargs=-1)
    @click.option(
        "--method",
        type=click.Choice(_METHOD_CHOICES),
        default="auto",
        show_default=True,
    )
    @click.option(
        "--require-provider",
        is_flag=True,
        help="Fail if no provider is configured (skip provider-free fallback).",
    )
    @click.option(
        "--save",
        is_flag=True,
        help="Save the WikiGraphRAG answer to wiki/analysis.",
    )
    @click.option(
        "--save-as",
        type=str,
        default=None,
        help="Save the WikiGraphRAG answer with a custom slug.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def ask_cmd(
        command_context: CommandContext,
        question_terms: tuple[str, ...],
        method: str,
        require_provider: bool,
        save: bool,
        save_as: str | None,
        as_json: bool,
    ) -> None:
        """Ask command implementation."""
        require_initialized(command_context)
        if not question_terms:
            raise click.ClickException("Provide a question to answer.")
        question = " ".join(question_terms).strip()
        query_service = command_context.services.wikigraph_query
        try:
            answer = query_service.ask(
                question,
                method=method,  # type: ignore[arg-type]
                require_provider=require_provider,
                save=save,
                save_as=save_as,
            )
        except WikiGraphQueryError as exc:
            raise click.ClickException(str(exc)) from exc

        if as_json:
            emit_json(_answer_payload(answer))
            return
        _render_answer(answer)

    return wikigraph_group


def _find_payload(result: WikiGraphFindResult, *, limit: int) -> dict[str, Any]:
    return {
        "engine": "wikigraph",
        "query": result.query,
        "method": result.method,
        "entities": result.entities,
        "communities": result.communities,
        "diagnostics": result.diagnostics,
        "contexts": [ctx.model_dump() for ctx in result.contexts[:limit]],
        "trace": result.trace,
    }


def _answer_payload(answer: WikiGraphAnswer) -> dict[str, Any]:
    payload = answer.model_dump()
    payload["citation_count"] = len(answer.citations)
    return payload


def _render_answer(answer: WikiGraphAnswer) -> None:
    console.print(
        f"[dim]\\[engine: wikigraph, method: {answer.method}, "
        f"provider: {answer.provider_status.get('mode', 'provider-free')}][/dim]"
    )
    if answer.warnings:
        console.print("[yellow]Warnings: " + ", ".join(answer.warnings) + "[/yellow]")
    console.print("")
    console.print(RichMarkdown(answer.answer))
    if answer.contexts:
        console.print("")
        echo_section("Contexts")
        for ctx in answer.contexts:
            echo_bullet(f"{ctx.title} [{ctx.citation_ref}] (score={ctx.score:.3f})")
    if answer.saved_path:
        console.print(f"\nSaved analysis page: {answer.saved_path}")
