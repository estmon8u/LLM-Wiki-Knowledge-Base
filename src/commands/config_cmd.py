from __future__ import annotations

import click
import yaml

from src.commands.common import console, echo_section, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from src.providers import validate_provider_name

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
    @click.argument("name", type=str)
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
        require_initialized(command_context)
        config_service = command_context.services.get("config")
        if config_service is None:
            raise click.ClickException("Config service unavailable.")
        config = config_service.load()
        config["provider"] = {}
        config_service.save(config)
        console.print("Provider cleared.")

    return config_group
