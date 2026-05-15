"""Command-line entry point for the knowledge-base CLI.

This module belongs to `src.cli` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from src.engine.command_registry import (
    get_click_command,
    list_command_names,
)
from src.models.command_models import CommandContext
from src.providers import validate_provider_name
from src.services import build_services
from src.services.config_service import ConfigService
from src.services.project_service import build_project_paths, discover_project_root


def _extract_project_root(ctx: click.Context) -> Path:
    """Return the discovered project root for a Click invocation context."""
    project_root = None
    if isinstance(ctx.params, dict):
        project_root = ctx.params.get("project_root")
    return discover_project_root(project_root or Path.cwd())


def build_runtime_context(
    project_root: Path,
    *,
    verbose: bool,
    provider_override: Optional[str] = None,
) -> CommandContext:
    """Builds runtime context.

    Args:
        project_root: Project root used to resolve knowledge-base paths.
        verbose: Whether to emit verbose command output.
        provider_override: Optional provider name overriding the configured provider.

    Returns:
        CommandContext produced by the operation.
    """
    paths = build_project_paths(project_root)
    config_service = ConfigService(paths)
    try:
        config = config_service.load()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if provider_override:
        try:
            config.setdefault("provider", {})["name"] = validate_provider_name(
                provider_override,
                provider_catalog=config.get("providers"),
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        # Clear stale fields that belong to a different provider.
        config["provider"].pop("model", None)
        config["provider"].pop("tier", None)
        config["provider"].pop("api_key_env", None)
        config["provider"].pop("reasoning_effort", None)
        config["provider"].pop("thinking_budget", None)
        config["provider"].pop("thinking_effort", None)
    schema_text = config_service.load_schema()

    services = build_services(paths, config)

    return CommandContext(
        project_root=paths.root,
        cwd=Path.cwd().resolve(),
        config=config,
        schema_text=schema_text,
        services=services,
        verbose=verbose,
    )


class KBGroup(click.Group):
    """Lazy-loading group that discovers commands from the registry."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List commands.

        Args:
            ctx: Click context carrying command invocation state.

        Returns:
            list[str] produced by the operation.
        """
        return list_command_names()

    def get_command(self, ctx: click.Context, cmd_name: str) -> Optional[click.Command]:
        """Get command.

        Args:
            ctx: Click context carrying command invocation state.
            cmd_name: Cmd name value used by the operation.

        Returns:
            Optional[click.Command] produced by the operation.
        """
        return get_click_command(cmd_name)


@click.command(
    cls=KBGroup,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--project-root",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True, resolve_path=True),
    default=None,
    help="Run the CLI against a specific project root.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose CLI output.")
@click.option(
    "--provider",
    "provider_override",
    type=str,
    default=None,
    help="Override the configured provider for this invocation.",
)
@click.pass_context
def main(
    ctx: click.Context,
    project_root: Optional[Path],
    verbose: bool,
    provider_override: Optional[str],
) -> None:
    """Runs the command-line entry point.

    Args:
        ctx: Click context carrying command invocation state.
        project_root: Project root used to resolve knowledge-base paths.
        verbose: Whether to emit verbose command output.
        provider_override: Optional provider name overriding the configured provider.
    """
    root = _extract_project_root(ctx)
    ctx.obj = build_runtime_context(
        root,
        verbose=verbose,
        provider_override=provider_override,
    )
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


if __name__ == "__main__":  # pragma: no cover
    main()
