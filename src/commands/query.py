from __future__ import annotations

import click

from src.commands.common import require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Answer a question from compiled wiki evidence with simple citations."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="query", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(
        name="query", help=SUMMARY, short_help="Answer from compiled wiki evidence."
    )
    @click.argument("question_terms", nargs=-1)
    @click.option("--limit", default=3, show_default=True, type=int)
    @click.pass_obj
    def command(
        command_context: CommandContext,
        question_terms: tuple[str, ...],
        limit: int,
    ) -> None:
        require_initialized(command_context)
        if not question_terms:
            raise click.ClickException("Provide a question to answer.")
        query_service = command_context.services["query"]
        question = " ".join(question_terms)
        answer = query_service.answer_question(question, limit=limit)
        click.echo(f"[mode: {answer.mode}]\n")
        click.echo(answer.answer)
        if answer.citations:
            click.echo("")
            click.echo("Citations:")
            for citation in answer.citations:
                click.echo(f"- {citation.title} [{citation.path}]")
        if answer.citations:
            try:
                if click.confirm("\nSave this answer as an analysis page?"):
                    saved_path = query_service.save_answer(question, answer)
                    click.echo(f"Saved analysis page: {saved_path}")
            except (click.Abort, EOFError):
                pass

    return command
