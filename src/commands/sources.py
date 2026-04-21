from __future__ import annotations

import click

from src.commands.common import (
    console,
    echo_kv,
    echo_section,
    emit_json,
    make_table,
    require_initialized,
)
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Manage source inventory."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="sources", summary=SUMMARY)


def create_command() -> click.Command:
    @click.group(
        name="sources",
        help=SUMMARY,
        short_help="Manage source inventory.",
        invoke_without_command=True,
    )
    @click.pass_context
    def sources_group(ctx: click.Context) -> None:
        if ctx.invoked_subcommand is None:
            ctx.invoke(sources_list)

    @sources_group.command(name="list")
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def sources_list(command_context: CommandContext, as_json: bool) -> None:
        """List all ingested sources."""
        require_initialized(command_context)
        manifest_service = command_context.services["manifest"]
        sources = manifest_service.list_sources()
        if not sources:
            if as_json:
                emit_json([])
            else:
                console.print("No sources ingested yet.")
            return

        if as_json:
            emit_json(
                [
                    {
                        "slug": s.slug,
                        "raw_path": s.raw_path,
                        "status": "compiled" if s.compiled_from_hash else "pending",
                    }
                    for s in sources
                ]
            )
            return

        rows = []
        for source in sources:
            status = "compiled" if source.compiled_from_hash else "pending"
            rows.append((source.slug, source.raw_path, status))

        table = make_table(
            columns=[
                ("Slug", {"style": "bold"}),
                ("Path", {}),
                ("Status", {}),
            ],
            rows=rows,
            title="Sources",
        )
        console.print(table)

        console.print("")
        echo_kv("total", len(sources))

    @sources_group.command(name="show")
    @click.argument("slug")
    @click.pass_obj
    def sources_show(command_context: CommandContext, slug: str) -> None:
        """Show details for a single source."""
        require_initialized(command_context)
        manifest_service = command_context.services["manifest"]
        sources = manifest_service.list_sources()
        match = [s for s in sources if s.slug == slug]
        if not match:
            raise click.ClickException(f"Source not found: {slug}")
        source = match[0]
        echo_section(f"Source: {source.slug}")
        echo_kv("source_id", source.source_id)
        echo_kv("raw_path", source.raw_path)
        echo_kv("normalized_path", source.normalized_path)
        echo_kv("title", source.title)
        echo_kv("content_hash", source.content_hash)
        echo_kv("source_type", source.source_type)
        echo_kv("ingested_at", source.ingested_at)
        echo_kv("compiled_from_hash", source.compiled_from_hash)
        echo_kv("compiled_at", source.compiled_at)

    return sources_group
