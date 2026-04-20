from __future__ import annotations

import click
import yaml

from src.commands.common import echo_section, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from src.services.model_registry_service import (
    PROVIDERS,
    TIERS,
    ModelRegistryService,
)


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
    @click.option(
        "--tier",
        default=None,
        type=click.Choice(list(TIERS), case_sensitive=False),
        help="Persist a default model tier (fast, balanced, deep).",
    )
    @click.pass_obj
    def provider_set(
        command_context: CommandContext,
        name: str,
        model: str | None,
        tier: str | None,
    ) -> None:
        require_initialized(command_context)
        config_service = command_context.services.get("config")
        if config_service is None:
            raise click.ClickException("Config service unavailable.")
        config = config_service.load()
        provider_section = config.setdefault("provider", {})
        old_name = provider_section.get("name", "")
        provider_section["name"] = name

        if model and tier:
            raise click.ClickException("--model and --tier are mutually exclusive.")

        if tier:
            registry = ModelRegistryService()
            profile = registry.list_profiles(name)
            tier_map = {p.tier: p for p in profile}
            if tier not in tier_map:
                raise click.ClickException(f"No {tier!r} tier for provider {name!r}.")
            provider_section["model"] = tier_map[tier].model
            provider_section["tier"] = tier
        elif model:
            provider_section["model"] = model
            provider_section.pop("tier", None)
        elif old_name and old_name != name:
            # Clear a stale model that belongs to a different provider.
            provider_section.pop("model", None)
            provider_section.pop("tier", None)

        config_service.save(config)
        msg = f"Provider set to {name}"
        if tier:
            msg += f" (tier={tier}, model={tier_map[tier].model})"
        elif model:
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

    # -- model inspection commands ------------------------------------------

    @config_group.command(name="providers", help="List supported providers.")
    def providers_cmd() -> None:
        for name in PROVIDERS:
            click.echo(name)

    @config_group.command(
        name="models", help="Show available model tiers for a provider."
    )
    @click.argument("provider_name", default="", required=False)
    @click.pass_obj
    def models_cmd(command_context: CommandContext, provider_name: str) -> None:
        registry = ModelRegistryService()
        name = provider_name or (
            command_context.config.get("provider", {}).get("name", "")
            if command_context
            else ""
        )
        if not name:
            raise click.ClickException(
                "Specify a provider name or configure one in kb.config.yaml."
            )
        profiles = registry.list_profiles(name)
        if not profiles:
            raise click.ClickException(f"Unknown provider: {name}")
        echo_section(f"Model tiers for {name}")
        for p in profiles:
            click.echo(f"  {p.tier:<10s} {p.model}")

    return config_group
