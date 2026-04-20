from __future__ import annotations

from typing import Optional

import click

from src.commands.common import echo_bullet, echo_section, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from src.providers import ProviderError, build_provider
from src.services.model_registry_service import ModelRegistryService


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
    @click.option("--limit", default=None, type=int, help="Evidence page limit.")
    @click.option(
        "--self-consistency",
        default=None,
        type=click.IntRange(1),
        hidden=True,
        help="(Advanced) Sample N independent provider answers.",
    )
    @click.option(
        "--quality",
        type=click.Choice(["fast", "normal", "deep"], case_sensitive=False),
        default=None,
        help="Preset quality level: fast (1 call, 2 pages), normal (1 call, 3 pages), deep (self-consistency, 5 pages).",
    )
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
        limit: Optional[int],
        self_consistency: Optional[int],
        quality: Optional[str],
        save_answer: bool,
        save_as_name: Optional[str],
        show_evidence: bool,
    ) -> None:
        require_initialized(command_context)
        if not question_terms:
            raise click.ClickException("Provide a question to answer.")

        # Resolve --quality preset into limit + self_consistency
        quality_presets = {
            "fast": (2, 1),
            "normal": (3, 1),
            "deep": (5, 3),
        }
        if quality:
            preset_limit, preset_sc = quality_presets[quality]
            if limit is None:
                limit = preset_limit
            if self_consistency is None:
                self_consistency = preset_sc

            # --quality also implies a model tier when the user did not
            # supply an explicit --tier or --model on the top-level CLI.
            _quality_tier_map = {"fast": "fast", "normal": "balanced", "deep": "deep"}
            runtime = command_context.config.get("_runtime") or {}
            provider_name = (command_context.config.get("provider") or {}).get(
                "name", ""
            )
            if provider_name and "tier" not in runtime and "model" not in runtime:
                implied_tier = _quality_tier_map[quality]
                registry = ModelRegistryService()
                try:
                    resolved = registry.resolve(
                        config=command_context.config,
                        tier=implied_tier,
                        task="ask",
                    )
                    new_provider = build_provider(
                        command_context.config, resolved=resolved
                    )
                    if new_provider is not None:
                        command_context.services["query"].provider = new_provider
                except ValueError:
                    pass  # unknown provider — keep existing
        if limit is None:
            limit = 3
        if self_consistency is None:
            self_consistency = 1
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

        if show_evidence and answer.citations:
            echo_section("Evidence")
            for citation in answer.citations:
                echo_bullet(f"{citation.title} [{citation.citation_ref}]")
            click.echo("")

        echo_section("Answer")
        click.echo(f"[mode: {answer.mode}]")
        click.echo("")
        click.echo(answer.answer)

        if answer.citations:
            click.echo("")
            echo_section("Citations")
            for citation in answer.citations:
                line = f"{citation.title} [{citation.citation_ref}]"
                if citation.section and citation.section != citation.title:
                    line += f" - {citation.section}"
                echo_bullet(line)

        should_save = save_answer or save_as_name is not None
        if should_save and answer.citations:
            saved_path = query_service.save_answer(question, answer, slug=save_as_name)
            click.echo(f"\nSaved analysis page: {saved_path}")

    return command
