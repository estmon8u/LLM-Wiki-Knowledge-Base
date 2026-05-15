"""Click command implementation for the kb legacy command.

This module belongs to `src.commands.legacy` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from typing import Optional

import click
from rich.markdown import Markdown as RichMarkdown

from src.commands.common import (
    console,
    echo_bullet,
    echo_section,
    emit_json,
    make_table,
    require_initialized,
)
from src.models.command_models import CommandContext, CommandSpec
from src.providers import ProviderError


SUMMARY = "Deprecated SQLite FTS5 commands for comparison and exact lookup."
LEGACY_WARNING = (
    "Deprecated: SQLite FTS5 retrieval is legacy-only. "
    "GraphRAG is the default target retrieval path."
)


def build_spec(_: CommandContext = None) -> CommandSpec:
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(name="legacy", summary=SUMMARY)


def create_command() -> click.Command:
    """Creates the Click command exposed by this module.

    Returns:
        click.Command produced by the operation.
    """

    @click.group(
        name="legacy", help=SUMMARY, short_help="Run deprecated FTS5 commands."
    )
    def legacy_group() -> None:
        """Deprecated legacy retrieval commands."""

    @legacy_group.command(
        name="find",
        help="Search the deprecated SQLite FTS5 wiki index.",
        short_help="Deprecated FTS5 search.",
    )
    @click.argument("query_terms", nargs=-1)
    @click.option("--limit", default=5, show_default=True, type=int)
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def legacy_find(
        command_context: CommandContext,
        query_terms: tuple[str, ...],
        limit: int,
        as_json: bool,
    ) -> None:
        """Legacy find.

        Args:
            command_context: Command context value used by the operation.
            query_terms: Query terms value used by the operation.
            limit: Maximum number of results to return or process.
            as_json: As json value used by the operation.
        """
        require_initialized(command_context)
        if not query_terms:
            raise click.ClickException("Provide at least one search term.")
        if not as_json:
            _warn_legacy()

        search_service = command_context.services["search"]
        query = " ".join(query_terms)
        results = search_service.search(
            query,
            limit=limit,
            include_concepts=False,
            include_analysis=False,
            page_types={"source"},
        )

        if as_json:
            emit_json(
                {
                    "retriever": "legacy-fts",
                    "deprecated": True,
                    "warning": LEGACY_WARNING,
                    "query": query,
                    "results": [_search_result_payload(result) for result in results],
                }
            )
            return

        if not results:
            console.print("No wiki pages matched that query.")
            return

        rows = [
            (result.title, result.path, f"{result.score:.2f}", result.snippet)
            for result in results
        ]
        table = make_table(
            columns=[
                ("Title", {"style": "bold"}),
                ("Path", {}),
                ("Score", {"justify": "right"}),
                ("Snippet", {"style": "dim"}),
            ],
            rows=rows,
            title="Legacy FTS Search Results",
        )
        console.print(table)

    @legacy_group.command(
        name="ask",
        help="Answer with the deprecated SQLite FTS5 source-page retrieval path.",
        short_help="Deprecated FTS5-backed ask.",
    )
    @click.argument("question_terms", nargs=-1)
    @click.option("--limit", default=3, type=int, help="Evidence page limit.")
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
    @click.pass_obj
    def legacy_ask(
        command_context: CommandContext,
        question_terms: tuple[str, ...],
        limit: int,
        save_answer: bool,
        save_as_name: Optional[str],
        show_evidence: bool,
    ) -> None:
        """Legacy ask.

        Args:
            command_context: Command context value used by the operation.
            question_terms: Question terms value used by the operation.
            limit: Maximum number of results to return or process.
            save_answer: Save answer value used by the operation.
            save_as_name: Save as name value used by the operation.
            show_evidence: Show evidence value used by the operation.
        """
        require_initialized(command_context)
        if not question_terms:
            raise click.ClickException("Provide a question to answer.")
        _warn_legacy()

        query_service = command_context.services["query"]
        question = " ".join(question_terms)
        try:
            answer = query_service.answer_question(
                question,
                limit=limit,
            )
        except ProviderError as exc:
            raise click.ClickException(str(exc)) from exc

        if show_evidence and answer.citations:
            echo_section("Evidence")
            for citation in answer.citations:
                echo_bullet(f"{citation.title} [{citation.citation_ref}]")
            console.print("")

        echo_section("Answer")
        console.print(f"[dim]\\[retriever: legacy-fts, mode: {answer.mode}][/dim]")
        console.print("")
        console.print(RichMarkdown(answer.answer))

        if answer.citations:
            console.print("")
            echo_section("Citations")
            for citation in answer.citations:
                line = f"{citation.title} [{citation.citation_ref}]"
                if citation.section and citation.section != citation.title:
                    line += f" - {citation.section}"
                echo_bullet(line)

        should_save = save_answer or save_as_name is not None
        if should_save and answer.mode != "no-matches":
            saved_path = query_service.save_answer(question, answer, slug=save_as_name)
            console.print(f"\nSaved analysis page: {saved_path}")

    return legacy_group


def _warn_legacy() -> None:
    click.echo(LEGACY_WARNING, err=True)


def _search_result_payload(result: object) -> dict[str, object]:
    return {
        "retriever": "legacy-fts",
        "deprecated": True,
        "title": result.title,
        "path": result.path,
        "score": result.score,
        "snippet": result.snippet,
        "section": result.section,
        "chunk_index": result.chunk_index,
    }
