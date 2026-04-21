from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from src.engine.command_registry import (
    build_command_specs,
    get_click_command,
    list_command_names,
)
from src.models.command_models import CommandContext
from src.services import build_services
from src.services.config_service import ConfigService
from src.services.project_service import build_project_paths, discover_project_root


def build_runtime_context(
    project_root: Path,
    *,
    verbose: bool,
    provider_override: Optional[str] = None,
) -> CommandContext:
    paths = build_project_paths(project_root)
    config_service = ConfigService(paths)
    config = config_service.load()
    if provider_override:
        config.setdefault("provider", {})["name"] = provider_override
        # Clear stale fields that belong to a different provider.
        config["provider"].pop("model", None)
        config["provider"].pop("tier", None)
        config["provider"].pop("api_key_env", None)
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
        return list_command_names()

    def get_command(self, ctx: click.Context, cmd_name: str) -> Optional[click.Command]:
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
    type=click.Choice(["openai", "anthropic", "gemini"], case_sensitive=False),
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
    root = discover_project_root(project_root or Path.cwd())
    ctx.obj = build_runtime_context(
        root,
        verbose=verbose,
        provider_override=provider_override,
    )
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def _extract_project_root(ctx: click.Context) -> Path:
    project_root = ctx.params.get("project_root")
    if isinstance(project_root, Path):
        return discover_project_root(project_root)
    return discover_project_root(Path.cwd())


if __name__ == "__main__":  # pragma: no cover
    main()
