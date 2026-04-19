from __future__ import annotations

import click

from src.commands.common import echo_bullet, echo_section, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from src.providers import ProviderError


SUMMARY = (
    "Answer a question from compiled wiki evidence with provider-backed citations."
)


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="query ask", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(
        name="ask", help=SUMMARY, short_help="Answer from compiled wiki evidence."
    )
    @click.argument("question_terms", nargs=-1)
    @click.option("--limit", default=3, show_default=True, type=int)
    @click.option(
        "--self-consistency",
        default=1,
        show_default=True,
        type=click.IntRange(1),
        help="Sample N independent provider answers from the same evidence and merge them deterministically.",
    )
    @click.pass_obj
    def command(
        command_context: CommandContext,
        question_terms: tuple[str, ...],
        limit: int,
        self_consistency: int,
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
                self_consistency=self_consistency,
            )
        except ProviderError as exc:
            raise click.ClickException(str(exc)) from exc
        echo_section("Answer")
        click.echo(f"[mode: {answer.mode}]")
        click.echo("")
        click.echo(answer.answer)
        if answer.citations:
            click.echo("")
            echo_section("Citations")
            for citation in answer.citations:
                echo_bullet(f"{citation.title} [{citation.path}]")
        if answer.citations:
            try:
                if click.confirm("\nSave this answer as an analysis page?"):
                    saved_path = query_service.save_answer(question, answer)
                    click.echo(f"Saved analysis page: {saved_path}")
            except (click.Abort, EOFError):
                pass

    return command
