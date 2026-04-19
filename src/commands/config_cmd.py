from __future__ import annotations

import click
import yaml

from src.commands.common import echo_section, require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "View or edit project configuration."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="config", summary=SUMMARY)


def create_command() -> click.BaseCommand:
    @click.group(
        name="config",
        help=SUMMARY,
        short_help="View or edit configuration.",
        invoke_without_command=True,
    )
    @click.pass_context
    def config_group(ctx: click.Context) -> None:
        if ctx.invoked_subcommand is None:
            ctx.invoke(show_cmd)

    @config_group.command(name="show", help="Display the current configuration.")
    @click.pass_obj
    def show_cmd(command_context: CommandContext) -> None:
        require_initialized(command_context)
        echo_section("Configuration")
        click.echo(yaml.dump(command_context.config, default_flow_style=False).rstrip())

    # -- provider subgroup --------------------------------------------------

    @config_group.group(name="provider", help="Manage the LLM provider setting.")
    def provider_group() -> None:
        pass

    @provider_group.command(name="set", help="Set the LLM provider.")
    @click.argument("name")
    @click.option("--model", default=None, help="Override the default model.")
    @click.pass_obj
    def provider_set(
        command_context: CommandContext, name: str, model: str | None
    ) -> None:
        require_initialized(command_context)
        config_service = command_context.services.get("config")
        if config_service is None:
            raise click.ClickException("Config service unavailable.")
        config = config_service.load()
        provider_section = config.setdefault("provider", {})
        provider_section["name"] = name
        if model:
            provider_section["model"] = model
        config_service.save(config)
        msg = f"Provider set to {name}"
        if model:
            msg += f" (model={model})"
        click.echo(msg)

    @provider_group.command(name="clear", help="Remove the LLM provider setting.")
    @click.pass_obj
    def provider_clear(command_context: CommandContext) -> None:
        require_initialized(command_context)
        config_service = command_context.services.get("config")
        if config_service is None:
            raise click.ClickException("Config service unavailable.")
        config = config_service.load()
        config["provider"] = {}
        config_service.save(config)
        click.echo("Provider cleared.")

    return config_group
