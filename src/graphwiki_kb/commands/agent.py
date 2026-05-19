"""Click command implementation for the kb agent command."""

from __future__ import annotations

import click

from graphwiki_kb.commands.common import console, emit_json, require_initialized
from graphwiki_kb.models.command_models import CommandContext, CommandSpec
from graphwiki_kb.services.agent_service import AgentService

SUMMARY = "Interact with the KB using natural language."


def build_spec(_: CommandContext | None = None) -> CommandSpec:
    """Build the command registry specification."""
    return CommandSpec(name="agent", summary=SUMMARY)


def create_command() -> click.Command:
    """Create the kb agent Click command."""

    @click.command(
        name="agent",
        help=SUMMARY,
        short_help="Natural-language KB assistant.",
    )
    @click.argument("prompt_terms", nargs=-1)
    @click.option(
        "--yes",
        is_flag=True,
        help="Approve safe write actions automatically.",
    )
    @click.option(
        "--show-plan",
        is_flag=True,
        help="Show registered tools before execution.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def command(
        command_context: CommandContext,
        prompt_terms: tuple[str, ...],
        yes: bool,
        show_plan: bool,
        as_json: bool,
    ) -> None:
        """Run one-shot or interactive natural-language KB control."""
        require_initialized(command_context)
        agent_service = AgentService(command_context.config, command_context.services)
        try:
            agent_service.ensure_available()
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc

        if prompt_terms:
            _run_prompt(
                agent_service,
                command_context,
                " ".join(prompt_terms).strip(),
                auto_approve=yes,
                show_plan=show_plan,
                as_json=as_json,
            )
            return

        console.print(
            "[dim]Interactive kb agent. Type 'exit' or 'quit' to leave.[/dim]"
        )
        session_id = "repl"
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not line:
                continue
            if line.lower() in {"exit", "quit"}:
                break
            _run_prompt(
                agent_service,
                command_context,
                line,
                auto_approve=yes,
                show_plan=show_plan,
                as_json=as_json,
                session_id=session_id,
            )

    return command


def _run_prompt(
    agent_service: AgentService,
    command_context: CommandContext,
    prompt: str,
    *,
    auto_approve: bool,
    show_plan: bool,
    as_json: bool,
    session_id: str | None = None,
) -> None:
    try:
        result = agent_service.run_once(
            prompt,
            command_context=command_context,
            auto_approve=auto_approve,
            show_plan=show_plan,
            session_id=session_id,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    if show_plan and result.planned_tools:
        console.print("[dim]Tools:[/dim] " + ", ".join(result.planned_tools))

    if as_json:
        emit_json(result.model_dump())
        return

    if result.pending_approvals:
        for item in result.pending_approvals:
            console.print(f"[dim]Approval: {item}[/dim]")
    console.print(result.final_output or "(no output)")
