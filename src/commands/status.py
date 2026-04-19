from __future__ import annotations

import click

from src.commands.common import (
    echo_kv,
    echo_section,
    echo_status_line,
    require_initialized,
)
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Show what exists, what changed, what is stale, and what to do next."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="status", summary=SUMMARY)


def create_command(
    *,
    name: str = "status",
    help_text: str = SUMMARY,
    short_help: str = "Show project status.",
) -> click.Command:
    @click.command(name=name, help=help_text, short_help=short_help)
    @click.option(
        "--changed",
        is_flag=True,
        help="Show a pre-compile preview of new, changed, and up-to-date sources.",
    )
    @click.pass_obj
    def command(command_context: CommandContext, changed: bool) -> None:
        project_service = command_context.services["project"]
        status_service = command_context.services["status"]
        snapshot = status_service.snapshot(initialized=project_service.is_initialized())

        if changed:
            require_initialized(command_context)
            diff_service = command_context.services["diff"]
            report = diff_service.diff()

            echo_section("Source Diff")
            for entry in report.entries:
                label = {"new": "NEW", "changed": "CHANGED", "up_to_date": "OK"}[
                    entry.status
                ]
                suffix = f"  ({entry.details})" if entry.details else ""
                echo_status_line(label, f"{entry.title} — {entry.raw_path}{suffix}")

            click.echo("")
            echo_section("Summary")
            echo_kv("new", report.new_count)
            echo_kv("changed", report.changed_count)
            echo_kv("up_to_date", report.up_to_date_count)
            return

        echo_section("Knowledge Base")
        click.echo(f"  {command_context.project_root}")
        click.echo("")

        echo_section("Sources")
        click.echo(f"  {snapshot.source_count} total")
        stale = snapshot.source_count - snapshot.compiled_source_count
        if stale > 0:
            click.echo(f"  {stale} need compiling")
        click.echo(f"  {snapshot.compiled_source_count} compiled")
        click.echo("")

        echo_section("Wiki")
        click.echo(f"  {snapshot.compiled_source_count} source page(s)")
        click.echo(f"  {snapshot.concept_page_count} concept page(s)")
        click.echo(f"  {snapshot.analysis_page_count} analysis page(s)")
        click.echo("")

        if snapshot.last_compile_at:
            echo_kv("last_compile_at", snapshot.last_compile_at)
            click.echo("")

        echo_section("Provider")
        click.echo(f"  {snapshot.provider_summary}")
        click.echo("")

        echo_section("Index")
        click.echo(f"  {snapshot.index_status}")
        click.echo("")

        echo_section("Export")
        click.echo(f"  {snapshot.export_status}")
        click.echo("")

        # Suggest what to do next
        if not snapshot.initialized:
            click.echo("Next\n  Run: kb init")
        elif snapshot.source_count == 0:
            click.echo("Next\n  Run: kb add <file|folder>")
        elif stale > 0:
            click.echo("Next\n  Run: kb update")
        else:
            click.echo("Next\n  Knowledge base is current.")

    return command
