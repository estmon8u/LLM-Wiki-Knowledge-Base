"""Click command implementation for the kb config command.

This module belongs to `graphwiki_kb.commands.config_cmd` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import click
import yaml

from graphwiki_kb.commands.common import console, echo_section, require_initialized
from graphwiki_kb.models.command_models import CommandContext, CommandSpec
from graphwiki_kb.providers import validate_provider_name

SUMMARY = "View or edit project configuration."


def build_spec(_: CommandContext = None) -> CommandSpec:
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(name="config", summary=SUMMARY)


def create_command() -> click.BaseCommand:
    """Creates the Click command exposed by this module.

    Returns:
        click.BaseCommand produced by the operation.
    """

    @click.group(
        name="config",
        help=SUMMARY,
        short_help="View or edit configuration.",
        invoke_without_command=True,
    )
    @click.pass_context
    def config_group(ctx: click.Context) -> None:
        """Config group.

        Args:
            ctx: Click context carrying command invocation state.
        """
        if ctx.invoked_subcommand is None:
            ctx.invoke(show_cmd)

    @config_group.command(name="show", help="Display the current configuration.")
    @click.pass_obj
    def show_cmd(command_context: CommandContext) -> None:
        """Show cmd.

        Args:
            command_context: Command context value used by the operation.
        """
        require_initialized(command_context)
        config_service = command_context.services.config
        persisted = config_service.load()
        visible = {k: v for k, v in persisted.items() if not k.startswith("_")}
        echo_section("Configuration")
        console.print(yaml.dump(visible, default_flow_style=False).rstrip())

    # -- provider subgroup --------------------------------------------------

    @config_group.group(name="provider", help="Manage the LLM provider setting.")
    def provider_group() -> None:
        """Provider group."""
        pass

    @provider_group.command(name="set", help="Set the LLM provider.")
    @click.argument("name", type=str)
    @click.option("--model", default=None, help="Override the default model.")
    @click.pass_obj
    def provider_set(
        command_context: CommandContext,
        name: str,
        model: str | None,
    ) -> None:
        """Provider set.

        Args:
            command_context: Command context value used by the operation.
            name: Name value used for lookup or display.
            model: Model value used by the operation.
        """
        require_initialized(command_context)
        config_service = command_context.services.config
        try:
            config = config_service.load()
            validated_name = validate_provider_name(
                name,
                provider_catalog=config.get("providers"),
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        provider_section = config.setdefault("provider", {})
        for stale_key in (
            "model",
            "tier",
            "api_key_env",
            "reasoning_effort",
            "thinking_budget",
            "thinking_effort",
        ):
            provider_section.pop(stale_key, None)
        provider_section["name"] = validated_name

        if model:
            config["providers"][validated_name]["model"] = model

        config_service.save(config)
        msg = f"Provider set to {validated_name}"
        if model:
            msg += f" (model={model})"
        console.print(msg)

    @provider_group.command(name="clear", help="Remove the LLM provider setting.")
    @click.pass_obj
    def provider_clear(command_context: CommandContext) -> None:
        """Provider clear.

        Args:
            command_context: Command context value used by the operation.
        """
        require_initialized(command_context)
        config_service = command_context.services.config
        config = config_service.load()
        config["provider"] = {}
        config_service.save(config)
        console.print("Provider cleared.")

    return config_group
