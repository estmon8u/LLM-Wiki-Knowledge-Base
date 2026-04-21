from __future__ import annotations

import click
import yaml

from src.commands.common import console, echo_section, require_initialized
from src.models.command_models import CommandContext, CommandSpec


_SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini")

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
        config_service = command_context.services.get("config")
        if config_service is None:
            raise click.ClickException("Config service unavailable.")
        persisted = config_service.load()
        visible = {k: v for k, v in persisted.items() if not k.startswith("_")}
        echo_section("Configuration")
        console.print(yaml.dump(visible, default_flow_style=False).rstrip())

    # -- provider subgroup --------------------------------------------------

    @config_group.group(name="provider", help="Manage the LLM provider setting.")
    def provider_group() -> None:
        pass

    @provider_group.command(name="set", help="Set the LLM provider.")
    @click.argument(
        "name", type=click.Choice(list(_SUPPORTED_PROVIDERS), case_sensitive=False)
    )
    @click.option("--model", default=None, help="Override the default model.")
    @click.pass_obj
    def provider_set(
        command_context: CommandContext,
        name: str,
        model: str | None,
    ) -> None:
        require_initialized(command_context)
        config_service = command_context.services.get("config")
        if config_service is None:
            raise click.ClickException("Config service unavailable.")
        config = config_service.load()
        provider_section = config.setdefault("provider", {})
        old_name = provider_section.get("name", "")
        provider_section["name"] = name

        if model:
            provider_section["model"] = model
            provider_section.pop("tier", None)
        elif old_name and old_name != name:
            # Clear a stale model that belongs to a different provider.
            provider_section.pop("model", None)
            provider_section.pop("tier", None)

        config_service.save(config)
        msg = f"Provider set to {name}"
        if model:
            msg += f" (model={model})"
        console.print(msg)

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
        console.print("Provider cleared.")

    return config_group
