from __future__ import annotations

from typing import Optional

import click
from rich.markdown import Markdown as RichMarkdown

from src.commands.common import console, emit_json, err_console, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from src.services.graph_ask_controller_service import GraphAskControllerError
from src.services.graphrag_query_service import GraphRAGQueryError
from src.services.query_router_service import GRAPH_ASK_METHODS


SUMMARY = (
    "Ask with the GraphRAG-aware answer controller. Legacy FTS-backed ask lives "
    "under kb legacy ask."
)


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="ask", summary=SUMMARY)


def create_command() -> click.Command:
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
    @click.option("--limit", default=3, type=int, help="Deprecated; ignored.")
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
        "--show-evidence",
        is_flag=True,
        help="Print the retrieved evidence snippets before the answer.",
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
        limit: int,
        save_answer: bool,
        save_as_name: Optional[str],
        show_evidence: bool,
        verbose: bool,
        as_json: bool,
    ) -> None:
        require_initialized(command_context)
        if not question_terms:
            raise click.ClickException("Provide a question to answer.")
        _ = limit
        question = " ".join(question_terms).strip()
        controller = command_context.services["graph_ask_controller"]

        try:
            from rich.status import Status

            spinner = Status("Querying GraphRAG…", console=err_console, spinner="dots")
            spinner.start()
            try:
                answer = controller.ask(
                    question,
                    method=method,
                    community_level=community_level,
                    dynamic_community_selection=dynamic_community_selection,
                    response_type=response_type,
                    verbose=verbose,
                    save=save_answer,
                    save_as=save_as_name,
                )
            finally:
                spinner.stop()
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
        if show_evidence:
            console.print("")
            console.print("Source Trace")
            console.print(f"  GraphRAG input: {answer.source_trace.get('input_path')}")
            console.print(f"  GraphRAG output: {answer.source_trace.get('output_dir')}")
            console.print(f"  Route reason: {answer.route_reason or 'unknown'}")
            console.print(f"  Claim support: {answer.claim_support or 'unverified'}")
        console.print("")
        console.print(RichMarkdown(answer.answer or "No answer text returned."))
        if answer.saved_path:
            console.print(f"\nSaved analysis page: {answer.saved_path}")

    return command
