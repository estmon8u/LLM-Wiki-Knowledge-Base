"""Click command implementation for the kb agent command.

This module belongs to `graphwiki_kb.commands.agent` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import json
from typing import Any

import click
from rich.markdown import Markdown as RichMarkdown

from graphwiki_kb.agents.runtime import AgentRuntimeError, is_agents_sdk_available
from graphwiki_kb.commands.common import (
    console,
    emit_json,
    require_initialized,
)
from graphwiki_kb.models.command_models import CommandContext, CommandSpec
from graphwiki_kb.services.agent_service import AgentRunResult, AgentService

SUMMARY = (
    "Interact with the KB using natural language. Uses the OpenAI Agents SDK "
    "to route requests to existing KB services."
)

_INTERACTIVE_EXIT = {"exit", "quit", ":q", ":quit", ":exit"}


def build_spec(_: CommandContext | None = None) -> CommandSpec:
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(name="agent", summary=SUMMARY)


def create_command() -> click.Command:
    """Creates the Click command exposed by this module.

    Returns:
        click.Command produced by the operation.
    """

    @click.command(
        name="agent",
        help=SUMMARY,
        short_help="Natural-language KB assistant.",
    )
    @click.argument("prompt_terms", nargs=-1)
    @click.option(
        "--yes",
        is_flag=True,
        help="Auto-approve safe write actions (ingest, kb update).",
    )
    @click.option(
        "--show-plan",
        is_flag=True,
        help="Print the planned tool calls before execution.",
    )
    @click.option(
        "--json",
        "as_json",
        is_flag=True,
        help="Emit a JSON record of the run instead of pretty output.",
    )
    @click.option(
        "--session",
        "session_id",
        type=str,
        default="default",
        show_default=True,
        help="SQLiteSession id to use for cross-call memory.",
    )
    @click.pass_obj
    def command(
        command_context: CommandContext,
        prompt_terms: tuple[str, ...],
        yes: bool,
        show_plan: bool,
        as_json: bool,
        session_id: str,
    ) -> None:
        """Command.

        Args:
            command_context: Command context value used by the operation.
            prompt_terms: Free-form natural-language prompt parts.
            yes: Auto-approve safe write actions.
            show_plan: Print the planned tool calls before execution.
            as_json: Emit JSON instead of pretty output.
            session_id: SQLiteSession id used for cross-call memory.
        """
        require_initialized(command_context)
        prompt = " ".join(prompt_terms).strip()
        agent_service = AgentService(command_context.config, command_context.services)
        if prompt:
            _run_one_shot(
                agent_service=agent_service,
                command_context=command_context,
                prompt=prompt,
                yes=yes,
                show_plan=show_plan,
                as_json=as_json,
                session_id=session_id,
            )
            return
        if as_json:
            raise click.ClickException(
                "kb agent --json requires a prompt argument. Use one-shot mode."
            )
        _run_interactive(
            agent_service=agent_service,
            command_context=command_context,
            yes=yes,
            show_plan=show_plan,
            session_id=session_id,
        )

    return command


# ---------------------------------------------------------------------------
# One-shot mode
# ---------------------------------------------------------------------------


def _run_one_shot(
    *,
    agent_service: AgentService,
    command_context: CommandContext,
    prompt: str,
    yes: bool,
    show_plan: bool,
    as_json: bool,
    session_id: str,
) -> None:
    if not is_agents_sdk_available():
        message = (
            "openai-agents is not installed. Install with: "
            "poetry install -E agent (or pip install openai-agents)."
        )
        if as_json:
            emit_json({"ok": False, "error": message})
            return
        raise click.ClickException(message)
    try:
        result = agent_service.run_once(
            prompt,
            command_context=command_context,
            auto_approve=yes,
            show_plan=show_plan,
            session_id=session_id,
        )
    except AgentRuntimeError as exc:
        if as_json:
            emit_json({"ok": False, "error": str(exc)})
            return
        raise click.ClickException(str(exc)) from exc
    if as_json:
        emit_json(_run_payload(result))
        return
    _render_result(result, show_plan=show_plan)


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


def _run_interactive(
    *,
    agent_service: AgentService,
    command_context: CommandContext,
    yes: bool,
    show_plan: bool,
    session_id: str,
) -> None:
    if not is_agents_sdk_available():
        console.print(
            "[red]openai-agents is not installed.[/red] "
            "Install with: poetry install -E agent"
        )
        return
    console.print(
        "[bold]GraphWiki KB agent[/bold] - " "type your request, or 'exit' to quit."
    )
    while True:
        try:
            prompt = click.prompt("> ", prompt_suffix="").strip()
        except (click.Abort, EOFError):
            console.print("")
            return
        if not prompt:
            continue
        if prompt.casefold() in _INTERACTIVE_EXIT:
            return
        try:
            result = agent_service.run_once(
                prompt,
                command_context=command_context,
                auto_approve=yes,
                show_plan=show_plan,
                session_id=session_id,
            )
        except AgentRuntimeError as exc:
            console.print(f"[red]Agent error: {exc}[/red]")
            continue
        _render_result(result, show_plan=show_plan)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_result(result: AgentRunResult, *, show_plan: bool) -> None:
    if show_plan and result.record.tool_results:
        console.print("[bold]Tool plan:[/bold]")
        for tool_result in result.record.tool_results:
            status = "ok" if tool_result.ok else "fail"
            console.print(
                f"  - {tool_result.tool_name} \\[{status}]: {tool_result.summary}"
            )
        console.print("")
    if result.record.final_output.strip():
        console.print(RichMarkdown(result.record.final_output))
    else:
        console.print("[dim]Agent returned no final output.[/dim]")
    for approval in result.pending_approvals:
        console.print("")
        console.print(
            f"[yellow]Approval required[/yellow] for {approval.tool_name}: "
            f"{approval.summary}"
        )
        if approval.payload:
            console.print(_format_payload(approval.payload))
        console.print("Re-run with [bold]--yes[/bold] (after reviewing) to proceed.")
    if result.saved_run_path:
        console.print(f"\n[dim]Run record saved to: {result.saved_run_path}[/dim]")


def _format_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=False, default=str)


def _run_payload(result: AgentRunResult) -> dict[str, Any]:
    return {
        "ok": True,
        "run_id": result.record.run_id,
        "final_output": result.record.final_output,
        "tool_results": [tr.model_dump() for tr in result.record.tool_results],
        "pending_approvals": [pa.model_dump() for pa in result.pending_approvals],
        "saved_run_path": result.saved_run_path,
    }
