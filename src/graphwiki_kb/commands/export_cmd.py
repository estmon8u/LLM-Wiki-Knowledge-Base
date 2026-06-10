"""Click command implementation for the kb export command.

This module belongs to `graphwiki_kb.commands.export_cmd` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from copy import deepcopy

import click

from graphwiki_kb.commands.common import (
    console,
    echo_bullet,
    echo_section,
    require_initialized,
)
from graphwiki_kb.models.command_models import CommandContext, CommandSpec
from graphwiki_kb.services.graphrag_wiki_export_service import GraphRAGWikiExportError
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService

SUMMARY = (
    "Export the compiled wiki to the configured target (defaults to Obsidian vault)."
)
WIKIGRAPH_EXPORT_MODES = ("none", "current", "classic", "lightrag", "all")


def build_spec(_: CommandContext | None = None) -> CommandSpec:
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(name="export", summary=SUMMARY)


def _wikigraph_modes(value: str) -> tuple[str, ...]:
    if value == "none":
        return ()
    if value == "all":
        return ("classic", "lightrag")
    if value == "current":
        return ("current",)
    return (value,)


def _wikigraph_mode_label(command_context: CommandContext, mode: str) -> str:
    if mode != "current":
        return mode
    configured = command_context.config.get("wikigraph", {})
    if isinstance(configured, dict):
        raw = str(configured.get("mode", "classic")).strip().lower()
        if raw in {"classic", "lightrag"}:
            return raw
    return "classic"


def _export_wikigraph_artifacts(
    command_context: CommandContext,
    *,
    modes: tuple[str, ...],
) -> list[tuple[str, str, list[str]]]:
    exports: list[tuple[str, str, list[str]]] = []
    for requested_mode in modes:
        mode = _wikigraph_mode_label(command_context, requested_mode)
        config = deepcopy(command_context.config)
        config.setdefault("wikigraph", {})["mode"] = mode
        service = WikiGraphIndexService(
            paths=command_context.services.wikigraph_index.paths,
            config=config,
            manifest_service=command_context.services.manifest,
        )
        base_subdir = (
            "wikigraph" if requested_mode == "current" else f"wikigraph/{mode}"
        )
        try:
            paths = service.export_artifacts(base_subdir=base_subdir)
        except FileNotFoundError as exc:
            console.print(f"[yellow]WikiGraphRAG {mode} export skipped: {exc}[/yellow]")
            continue
        except ValueError as exc:
            console.print(f"[yellow]WikiGraphRAG {mode} export skipped: {exc}[/yellow]")
            continue
        exports.append((mode, base_subdir, paths))
    return exports


def create_command() -> click.Command:
    """Creates the Click command exposed by this module.

    Returns:
        click.Command produced by the operation.
    """

    @click.command(name="export", help=SUMMARY, short_help="Export the wiki.")
    @click.option(
        "--clean",
        is_flag=True,
        help="Remove stale vault files that no longer exist in the wiki.",
    )
    @click.option(
        "--wikigraph-modes",
        type=click.Choice(WIKIGRAPH_EXPORT_MODES),
        default="none",
        show_default=True,
        help=(
            "Refresh generated WikiGraphRAG inspection cards before vault export. "
            "`current` writes wiki/wikigraph/, while `classic`, `lightrag`, "
            "and `all` write mode-separated cards under wiki/wikigraph/<mode>/."
        ),
    )
    @click.pass_obj
    def command(
        command_context: CommandContext,
        clean: bool,
        wikigraph_modes: str,
    ) -> None:
        """Command.

        Args:
            command_context: Command context value used by the operation.
            clean: Clean value used by the operation.
            wikigraph_modes: WikiGraph artifact export selector.
        """
        require_initialized(command_context)
        export_service = command_context.services.export

        graph_result = None
        graph_status = command_context.services.graphrag_status.status()
        if graph_status.workspace_initialized and graph_status.output_complete:
            try:
                graph_result = (
                    command_context.services.graphrag_wiki_export.export_wiki()
                )
            except GraphRAGWikiExportError as exc:
                console.print(f"[yellow]Graph wiki export skipped: {exc}[/yellow]")

        wikigraph_exports = _export_wikigraph_artifacts(
            command_context,
            modes=_wikigraph_modes(wikigraph_modes),
        )

        result = export_service.export_vault(clean=clean)
        echo_section("Vault Export")
        console.print(f"Exported {len(result.exported_paths)} file(s) to the vault")
        for path in result.exported_paths:
            echo_bullet(path)
        if result.removed_paths:
            console.print("")
            echo_section("Removed Stale Files")
            console.print(
                f"Removed {len(result.removed_paths)} stale file(s) from the vault"
            )
            for path in result.removed_paths:
                echo_bullet(path)

        if graph_result is None:
            graph_result_text = ""
        else:
            graph_result_text = (
                f"Exported {graph_result.exported_count} GraphRAG wiki page(s) "
                "to wiki/graph"
            )
        if graph_result_text:
            console.print("")
            echo_section("Graph Wiki Export")
            console.print(graph_result_text)
        if wikigraph_exports:
            console.print("")
            echo_section("WikiGraphRAG Export")
            for mode, base_subdir, paths in wikigraph_exports:
                console.print(
                    f"Exported {len(paths)} {mode} WikiGraphRAG card(s) "
                    f"under wiki/{base_subdir}"
                )

    return command
