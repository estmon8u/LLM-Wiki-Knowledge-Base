from __future__ import annotations

from typing import Optional

import click
from rich.markdown import Markdown as RichMarkdown

from src.commands.common import console, echo_bullet, echo_section, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from src.providers import ProviderError


SUMMARY = (
    "Answer a question from compiled wiki evidence with provider-backed citations."
)


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="ask", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(
        name="ask", help=SUMMARY, short_help="Answer from compiled wiki evidence."
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
    def command(
        command_context: CommandContext,
        question_terms: tuple[str, ...],
        limit: int,
        save_answer: bool,
        save_as_name: Optional[str],
        show_evidence: bool,
    ) -> None:
        require_initialized(command_context)
        if not question_terms:
            raise click.ClickException("Provide a question to answer.")

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
        console.print(f"[dim]\\[mode: {answer.mode}][/dim]")
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
        if should_save and answer.citations:
            saved_path = query_service.save_answer(question, answer, slug=save_as_name)
            console.print(f"\nSaved analysis page: {saved_path}")

    return command
