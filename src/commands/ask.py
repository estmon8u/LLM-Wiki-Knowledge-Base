"""Click command implementation for the kb ask command.

This module belongs to `src.commands.ask` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from typing import Optional

import click
from rich.markdown import Markdown as RichMarkdown

from src.commands.common import console, emit_json, live_status, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from src.services.graph_ask_controller_service import GraphAskControllerError
from src.services.graphrag_query_service import GraphRAGQueryError
from src.services.query_router_service import GRAPH_ASK_METHODS


SUMMARY = (
    "Ask with the GraphRAG-aware answer controller. Legacy FTS-backed ask lives "
    "under kb legacy ask."
)


def build_spec(_: CommandContext = None) -> CommandSpec:
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
        "--method",
        type=click.Choice(GRAPH_ASK_METHODS),
        default="auto",
        show_default=True,
        help="GraphRAG method or deterministic auto-routing.",
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
    @click.option(
        "--streaming/--no-streaming",
        default=None,
        hidden=True,
    )
    @click.option("--limit", type=int, help="Deprecated; ignored.")
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
        method: str,
        community_level: int | None,
        dynamic_community_selection: bool | None,
        response_type: str | None,
        streaming: bool | None,
        limit: int | None,
        save_answer: bool,
        save_as_name: Optional[str],
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
            streaming: Streaming value used by the operation.
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
            raise click.ClickException("Provide a question to answer.")
        if streaming is not None:
            raise click.ClickException(
                "--streaming is not supported by kb ask yet; GraphRAG query output "
                "is captured before rendering."
            )
        if limit is not None and not as_json:
            console.print("[yellow]--limit is ignored for GraphRAG queries.[/yellow]")
        if show_evidence and not as_json:
            console.print(
                "[yellow]--show-evidence is deprecated; use "
                "--show-source-trace.[/yellow]"
            )
        question = " ".join(question_terms).strip()
        controller = command_context.services.graph_ask_controller

        try:
            with live_status("Querying GraphRAG"):
                answer = controller.ask(
                    question,
                    method=method,
                    community_level=community_level,
                    dynamic_community_selection=dynamic_community_selection,
                    response_type=response_type,
                    streaming=streaming,
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
        if show_source_trace or show_evidence:
            console.print("")
            console.print("Source Trace")
            console.print(f"  GraphRAG input: {answer.source_trace.get('input_path')}")
            console.print(f"  GraphRAG output: {answer.source_trace.get('output_dir')}")
            console.print(f"  Route reason: {answer.route_reason or 'unknown'}")
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
