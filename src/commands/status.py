"""Click command implementation for the kb status command.

This module belongs to `src.commands.status` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import click

from src.commands.common import (
    console,
    echo_kv,
    echo_section,
    echo_status_line,
    emit_json,
    require_initialized,
)
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Show what exists, what changed, what is stale, and what to do next."


def build_spec(_: CommandContext = None) -> CommandSpec:
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(name="status", summary=SUMMARY)


def create_command(
    *,
    name: str = "status",
    help_text: str = SUMMARY,
    short_help: str = "Show project status.",
) -> click.Command:
    """Creates the Click command exposed by this module.

    Args:
        name: Name value used for lookup or display.
        help_text: Help text value used by the operation.
        short_help: Short help value used by the operation.

    Returns:
        click.Command produced by the operation.
    """

    @click.command(name=name, help=help_text, short_help=short_help)
    @click.option(
        "--changed",
        is_flag=True,
        help=(
            "Show a pre-compile preview of new, changed, missing, and up-to-date "
            "sources."
        ),
    )
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def command(command_context: CommandContext, changed: bool, as_json: bool) -> None:
        """Command.

        Args:
            command_context: Command context value used by the operation.
            changed: Changed value used by the operation.
            as_json: As json value used by the operation.
        """
        project_service = command_context.services["project"]
        status_service = command_context.services["status"]
        snapshot = status_service.snapshot(initialized=project_service.is_initialized())

        if changed:
            require_initialized(command_context)
            diff_service = command_context.services["diff"]
            report = diff_service.diff()

            if as_json:
                emit_json(
                    {
                        "new": report.new_count,
                        "changed": report.changed_count,
                        "missing": report.missing_count,
                        "up_to_date": report.up_to_date_count,
                        "entries": [
                            {
                                "title": e.title,
                                "raw_path": e.raw_path,
                                "status": e.status,
                                "details": e.details,
                            }
                            for e in report.entries
                        ],
                    }
                )
                return

            echo_section("Source Diff")
            for entry in report.entries:
                label = {
                    "new": "NEW",
                    "changed": "CHANGED",
                    "missing": "MISSING",
                    "up_to_date": "OK",
                }[entry.status]
                suffix = f"  ({entry.details})" if entry.details else ""
                echo_status_line(label, f"{entry.title} — {entry.raw_path}{suffix}")

            click.echo("")
            echo_section("Summary")
            echo_kv("new", report.new_count)
            echo_kv("changed", report.changed_count)
            echo_kv("missing", report.missing_count)
            echo_kv("up_to_date", report.up_to_date_count)
            return

        if as_json:
            emit_json(
                {
                    "project_root": str(command_context.project_root),
                    "initialized": snapshot.initialized,
                    "source_count": snapshot.source_count,
                    "compiled_source_count": snapshot.compiled_source_count,
                    "concept_page_count": snapshot.concept_page_count,
                    "analysis_page_count": snapshot.analysis_page_count,
                    "last_compile_at": snapshot.last_compile_at,
                    "provider_summary": snapshot.provider_summary,
                    "index_status": snapshot.index_status,
                    "export_status": snapshot.export_status,
                    "graph_status": snapshot.graph_status,
                    "graph": snapshot.graph_status,
                }
            )
            return

        echo_section("Knowledge Base")
        console.print(f"  {command_context.project_root}")
        console.print("")

        echo_section("Sources")
        console.print(f"  {snapshot.source_count} total")
        stale = snapshot.source_count - snapshot.compiled_source_count
        if stale > 0:
            console.print(f"  [yellow]{stale} need compiling[/yellow]")
        console.print(f"  {snapshot.compiled_source_count} compiled")
        console.print("")

        echo_section("Wiki")
        console.print(f"  {snapshot.compiled_source_count} source page(s)")
        console.print(f"  {snapshot.concept_page_count} concept page(s)")
        console.print(f"  {snapshot.analysis_page_count} analysis page(s)")
        console.print("")

        if snapshot.last_compile_at:
            echo_kv("last_compile_at", snapshot.last_compile_at)
            console.print("")

        echo_section("Provider")
        console.print(f"  {snapshot.provider_summary}")
        console.print("")

        echo_section("Index")
        console.print(f"  {snapshot.index_status}")
        console.print("")

        echo_section("Export")
        console.print(f"  {snapshot.export_status}")
        console.print("")

        if snapshot.graph_status:
            graph = snapshot.graph_status
            echo_section("GraphRAG")
            workspace = (
                "initialized" if graph.get("workspace_initialized") else "missing"
            )
            output = graph.get("state") or (
                "complete"
                if graph.get("output_complete")
                else "partial" if graph.get("output_present") else "missing"
            )
            console.print(f"  Workspace: {workspace}")
            console.print(f"  Input documents: {graph.get('input_document_count', 0)}")
            console.print(f"  Index output: {output}")
            if graph.get("last_index_run_at"):
                console.print(f"  Last index: {graph.get('last_index_run_at')}")
                console.print(f"  Index method: {graph.get('last_index_method')}")
            for label, key in (
                ("Entities", "entity_count"),
                ("Relationships", "relationship_count"),
                ("Communities", "community_count"),
                ("Community reports", "community_report_count"),
            ):
                value = graph.get(key)
                console.print(f"  {label}: {value if value is not None else 'unknown'}")
            console.print(f"  Next action: {graph.get('next_action')}")
            console.print("")

        # Suggest what to do next
        if not snapshot.initialized:
            console.print("Next\n  Run: kb init")
        elif snapshot.source_count == 0:
            console.print("Next\n  Run: kb add <file|folder>")
        elif stale > 0:
            console.print("Next\n  Run: kb update")
        else:
            console.print("Next\n  Knowledge base is current.")

    return command
