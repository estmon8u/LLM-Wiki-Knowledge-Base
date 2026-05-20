"""Click command implementation for the kb ask command.

This module belongs to `graphwiki_kb.commands.ask` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import click
from rich.markdown import Markdown as RichMarkdown

from graphwiki_kb.commands.common import (
    console,
    echo_bullet,
    echo_section,
    emit_json,
    live_status,
    require_initialized,
)
from graphwiki_kb.models.command_models import CommandContext, CommandSpec
from graphwiki_kb.services.graph_ask_controller_service import GraphAskControllerError
from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryError
from graphwiki_kb.services.query_router_service import GRAPH_ASK_METHODS
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryError
from graphwiki_kb.wikigraph.models import QueryMethod, WikiGraphAnswer

SUMMARY = (
    "Ask with the GraphRAG-aware answer controller or the custom WikiGraphRAG "
    "backend. Legacy FTS-backed ask lives under `kb legacy ask`."
)

ENGINE_CHOICES = ("graphrag", "wikigraph")
WIKIGRAPH_METHODS = ("auto", "basic", "local", "global", "drift-lite")
ALL_METHOD_CHOICES = tuple(sorted(set(GRAPH_ASK_METHODS).union(WIKIGRAPH_METHODS)))


def build_spec(_: CommandContext | None = None) -> CommandSpec:
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(name="ask", summary=SUMMARY)


def create_command() -> click.Command:
    """Creates the Click command exposed by this module.

    Returns:
        click.Command produced by the operation.
    """

    @click.command(name="ask", help=SUMMARY, short_help="Ask with GraphRAG.")
    @click.argument("question_terms", nargs=-1)
    @click.option(
        "--engine",
        type=click.Choice(ENGINE_CHOICES),
        default="graphrag",
        show_default=True,
        help=(
            "Backend to use. `graphrag` runs Microsoft GraphRAG. `wikigraph` "
            "runs the custom WikiGraphRAG backend over the maintained wiki."
        ),
    )
    @click.option(
        "--method",
        type=click.Choice(ALL_METHOD_CHOICES),
        default="auto",
        show_default=True,
        help=(
            "Retrieval method. GraphRAG supports auto/basic/local/global/drift; "
            "WikiGraphRAG supports auto/basic/local/global/drift-lite."
        ),
    )
    @click.option(
        "--community-level",
        type=int,
        help="Forward GraphRAG's community level option.",
    )
    @click.option(
        "--dynamic-community-selection/--no-dynamic-selection",
        default=None,
        help="Forward GraphRAG dynamic community selection behavior.",
    )
    @click.option(
        "--response-type",
        help="Forward GraphRAG's response type option.",
    )
    @click.option("--limit", type=int, help="Deprecated; use GraphRAG routing instead.")
    @click.option(
        "--save",
        "save_answer",
        is_flag=True,
        help="Save the answer as an analysis page in the wiki.",
    )
    @click.option(
        "--save-as",
        "save_as_name",
        type=str,
        default=None,
        help="Save the answer as an analysis page with a custom slug.",
    )
    @click.option(
        "--show-source-trace",
        "show_source_trace",
        is_flag=True,
        help="Print the graph source trace and routing metadata before the answer.",
    )
    @click.option(
        "--show-evidence",
        is_flag=True,
        hidden=True,
    )
    @click.option("--verbose", is_flag=True, help="Forward GraphRAG's verbose flag.")
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def command(
        command_context: CommandContext,
        question_terms: tuple[str, ...],
        engine: str,
        method: str,
        community_level: int | None,
        dynamic_community_selection: bool | None,
        response_type: str | None,
        limit: int | None,
        save_answer: bool,
        save_as_name: str | None,
        show_source_trace: bool,
        show_evidence: bool,
        verbose: bool,
        as_json: bool,
    ) -> None:
        """Command.

        Args:
            command_context: Command context value used by the operation.
            question_terms: Question terms value used by the operation.
            method: Method value used by the operation.
            community_level: Community level value used by the operation.
            dynamic_community_selection: Dynamic community selection value used by the operation.
            response_type: Response type value used by the operation.
            limit: Maximum number of results to return or process.
            save_answer: Save answer value used by the operation.
            save_as_name: Save as name value used by the operation.
            show_source_trace: Show source trace value used by the operation.
            show_evidence: Deprecated show source trace alias.
            verbose: Whether to emit verbose command output.
            as_json: As json value used by the operation.
        """
        require_initialized(command_context)
        if not question_terms:
            raise click.ClickException(
                'Provide a question to answer, for example: kb ask "What changed?"'
            )
        if limit is not None:
            raise click.ClickException(
                "--limit is not supported by kb ask because GraphRAG controls "
                "retrieval internally."
            )
        if show_evidence and not as_json:
            console.print(
                "[yellow]--show-evidence is deprecated; use "
                "--show-source-trace.[/yellow]"
            )
        question = " ".join(question_terms).strip()

        if engine == "wikigraph":
            _validate_method_for_engine(engine, method)
            _run_wikigraph_ask(
                command_context,
                question,
                method=method,
                save_answer=save_answer,
                save_as_name=save_as_name,
                as_json=as_json,
                show_source_trace=show_source_trace,
            )
            return

        _validate_method_for_engine(engine, method)
        controller = command_context.services.graph_ask_controller

        try:
            with live_status("Querying GraphRAG"):
                answer = controller.ask(
                    question,
                    method=method,
                    community_level=community_level,
                    dynamic_community_selection=dynamic_community_selection,
                    response_type=response_type,
                    streaming=None,
                    verbose=verbose,
                    save=save_answer,
                    save_as=save_as_name,
                )
        except (GraphAskControllerError, GraphRAGQueryError) as exc:
            raise click.ClickException(str(exc)) from exc

        if as_json:
            emit_json(answer.to_dict())
            return

        console.print(
            f"[dim]\\[retriever: {answer.retriever}, method: {answer.method}, "
            f"planner: {answer.planner or 'none'}, "
            f"index_run_id: {answer.index_run_id or 'unknown'}][/dim]"
        )
        if answer.route_reason:
            route_detail = answer.route_reason
            if answer.route_confidence:
                route_detail += f" ({answer.route_confidence})"
            console.print(f"[dim]Route: {route_detail}[/dim]")
        if show_source_trace or show_evidence:
            console.print("")
            console.print("Source Trace")
            console.print(f"  GraphRAG input: {answer.source_trace.get('input_path')}")
            console.print(f"  GraphRAG output: {answer.source_trace.get('output_dir')}")
            console.print(f"  Route reason: {answer.route_reason or 'unknown'}")
            console.print(f"  Route confidence: {answer.route_confidence or 'unknown'}")
            if answer.route_matched_terms:
                console.print(
                    "  Route matched terms: " + ", ".join(answer.route_matched_terms)
                )
            console.print(
                "  Parsed graph references: " f"{len(answer.graph_data_references)}"
            )
            console.print(f"  Support level: {answer.claim_support or 'unverified'}")
        if answer.staleness_warnings:
            console.print("")
            for warning in answer.staleness_warnings:
                console.print(f"[yellow]{warning}[/yellow]")
        console.print("")
        console.print(RichMarkdown(answer.answer or "No answer text returned."))
        if answer.saved_path:
            console.print(f"\nSaved analysis page: {answer.saved_path}")

    return command


def _validate_method_for_engine(engine: str, method: str) -> None:
    """Raise a friendly error when `--method` is not valid for the engine."""
    if engine == "wikigraph":
        if method not in WIKIGRAPH_METHODS:
            raise click.ClickException(
                f"--method {method!r} is not valid for engine=wikigraph. "
                f"Choose one of: {', '.join(WIKIGRAPH_METHODS)}."
            )
        return
    if method not in GRAPH_ASK_METHODS:
        raise click.ClickException(
            f"--method {method!r} is not valid for engine=graphrag. "
            f"Choose one of: {', '.join(GRAPH_ASK_METHODS)}."
        )


def _run_wikigraph_ask(
    command_context: CommandContext,
    question: str,
    *,
    method: str,
    save_answer: bool,
    save_as_name: str | None,
    as_json: bool,
    show_source_trace: bool = False,
) -> None:
    """Route ``kb ask --engine wikigraph`` through WikiGraphQueryService."""
    query_service = command_context.services.wikigraph_query
    try:
        with live_status("Querying WikiGraphRAG"):
            wikigraph_method: QueryMethod = method  # type: ignore[assignment]
            answer: WikiGraphAnswer = query_service.ask(
                question,
                method=wikigraph_method,
                save=save_answer,
                save_as=save_as_name,
            )
    except WikiGraphQueryError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        payload = answer.model_dump()
        payload["citation_count"] = len(answer.citations)
        emit_json(payload)
        return

    provider_mode = answer.provider_status.get("mode", "provider-free")
    console.print(
        f"[dim]\\[engine: wikigraph, method: {answer.method}, "
        f"provider: {provider_mode}][/dim]"
    )
    if answer.warnings:
        console.print("[yellow]Warnings: " + ", ".join(answer.warnings) + "[/yellow]")
    if show_source_trace:
        console.print("")
        echo_section("WikiGraphRAG Source Trace")
        seed_entities: list[str] = []
        communities: list[str] = []
        sub_questions: list[str] = []
        for step in answer.trace:
            if not isinstance(step, dict):
                continue
            seed_entities.extend(step.get("seed_entities") or [])
            communities.extend(step.get("communities") or [])
            sub_questions.extend(step.get("sub_questions") or [])
        if seed_entities:
            console.print("  Seed entities: " + ", ".join(dict.fromkeys(seed_entities)))
        if communities:
            console.print("  Communities: " + ", ".join(dict.fromkeys(communities)))
        if sub_questions:
            console.print("  Sub-questions:")
            for question_text in dict.fromkeys(sub_questions):
                echo_bullet(question_text)
        if answer.contexts:
            console.print(f"  Retrieved contexts: {len(answer.contexts)}")
            for ctx in answer.contexts:
                trace_label = ",".join(ctx.trace) if ctx.trace else "(none)"
                echo_bullet(f"{ctx.title} [{ctx.citation_ref}] trace={trace_label}")
        if answer.provider_status:
            console.print(
                "  Provider: "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in answer.provider_status.items()
                    if value not in (None, "")
                )
            )
    console.print("")
    console.print(RichMarkdown(answer.answer or "No answer text returned."))
    if answer.contexts and not show_source_trace:
        console.print("")
        echo_section("Contexts")
        for ctx in answer.contexts:
            echo_bullet(f"{ctx.title} [{ctx.citation_ref}] (score={ctx.score:.3f})")
    if answer.saved_path:
        console.print(f"\nSaved analysis page: {answer.saved_path}")
