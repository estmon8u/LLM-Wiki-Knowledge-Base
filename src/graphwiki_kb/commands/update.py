"""Click command implementation for the kb update command.

This module belongs to `graphwiki_kb.commands.update` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from pathlib import Path

import click

from graphwiki_kb.commands.common import (
    console,
    echo_bullet,
    echo_section,
    echo_status_line,
    lazy_live_status,
    progress_report,
    require_initialized,
)
from graphwiki_kb.models.command_models import CommandContext, CommandSpec
from graphwiki_kb.services.update_service import (
    GRAPH_INDEX_METHODS,
    UpdateOptions,
    UpdatePreflightError,
    UpdateService,
)

SUMMARY = (
    "Bring the knowledge base current. Optionally add new sources first, "
    "then compile, sync GraphRAG, and refresh indexes."
)


def build_spec(_: CommandContext | None = None) -> CommandSpec:
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(
        name="update",
        summary=SUMMARY,
    )


def _get_update_service(command_context: CommandContext) -> UpdateService:
    return UpdateService(
        ingest_service=command_context.services.ingest,
        compile_service=command_context.services.compile,
        concept_service=command_context.services.concepts,
        search_service=command_context.services.search,
        config=command_context.config,
        graphrag_workspace_service=command_context.services.graphrag_workspace,
        graphrag_sync_service=command_context.services.graphrag_sync,
        graphrag_wiki_export_service=command_context.services.graphrag_wiki_export,
        wikigraph_index_service=command_context.services.wikigraph_index,
    )


def _mode_label(options: UpdateOptions) -> str:
    if options.graph_only:
        return "graph-only"
    if options.no_graph:
        return "wiki-only"
    return "full"


def _status_label(options: UpdateOptions) -> str:
    if options.graph_only:
        return "Updating GraphRAG"
    if options.no_graph:
        return "Updating wiki"
    return "Updating knowledge base"


def create_command() -> click.Command:
    """Creates the Click command exposed by this module.

    Returns:
        click.Command produced by the operation.
    """

    @click.command(
        name="update",
        help=SUMMARY,
        short_help="Update the knowledge base.",
    )
    @click.argument(
        "source_paths",
        nargs=-1,
        type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    )
    @click.option(
        "--force",
        is_flag=True,
        help="Rebuild every source page even if nothing changed.",
    )
    @click.option(
        "--resume",
        is_flag=True,
        help="Resume the most recent interrupted or failed update run.",
    )
    @click.option(
        "--no-graph",
        is_flag=True,
        help="Skip GraphRAG sync, indexing, and graph wiki export.",
    )
    @click.option(
        "--graph-only",
        is_flag=True,
        help="Sync/index/export GraphRAG without legacy compile or search refresh.",
    )
    @click.option(
        "--graph-method",
        type=click.Choice(GRAPH_INDEX_METHODS),
        default="auto",
        show_default=True,
        help="GraphRAG indexing method to use.",
    )
    @click.option(
        "--allow-partial",
        is_flag=True,
        help="Treat GraphRAG sync/index/export failures as warnings.",
    )
    @click.option(
        "--concepts/--no-concepts",
        default=None,
        help="Opt in or out of legacy concept page generation for this update.",
    )
    @click.option(
        "--wikigraph/--no-wikigraph",
        default=None,
        show_default=False,
        help=(
            "Build the WikiGraphRAG index after wiki compile. When neither "
            "flag is passed, the `wikigraph.enabled` config value drives the "
            "behavior (defaults to true)."
        ),
    )
    @click.option(
        "--wikigraph-include-graphrag-export-pages",
        is_flag=True,
        help=(
            "Ablation: also feed wiki/graph (GraphRAG export) pages to the "
            "WikiGraphRAG build."
        ),
    )
    @click.option(
        "--export-wikigraph-artifacts/--no-export-wikigraph-artifacts",
        default=None,
        show_default=False,
        help=(
            "After building the WikiGraphRAG index, write generated entity, "
            "community, and chunk cards under wiki/wikigraph/. When neither "
            "flag is passed, the `wikigraph.export_generated_artifacts` "
            "config value drives the behavior (defaults to false)."
        ),
    )
    @click.option(
        "--artifact-types",
        "artifact_types",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of wikigraph artifact types to export "
            "(entities,communities,chunks). Defaults to all three when "
            "--export-wikigraph-artifacts is on."
        ),
    )
    @click.pass_obj
    def command(
        command_context: CommandContext,
        source_paths: tuple[Path, ...],
        force: bool,
        resume: bool,
        no_graph: bool,
        graph_only: bool,
        graph_method: str,
        allow_partial: bool,
        concepts: bool | None,
        wikigraph: bool | None,
        wikigraph_include_graphrag_export_pages: bool,
        export_wikigraph_artifacts: bool | None,
        artifact_types: str | None,
    ) -> None:
        """Command.

        Args:
            command_context: Command context value used by the operation.
            source_paths: Source paths value used by the operation.
            force: Force value used by the operation.
            resume: Resume value used by the operation.
            no_graph: No graph value used by the operation.
            graph_only: Graph only value used by the operation.
            graph_method: GraphRAG indexing method to request.
            allow_partial: Allow partial value used by the operation.
            concepts: Concepts value used by the operation.
        """
        require_initialized(command_context)
        service = _get_update_service(command_context)

        options = UpdateOptions(
            source_paths=source_paths,
            force=force,
            resume=resume,
            no_graph=no_graph,
            graph_only=graph_only,
            graph_method=graph_method,
            allow_partial=allow_partial,
            concepts=concepts,
            wikigraph=wikigraph,
            wikigraph_include_graphrag_export_pages=(
                wikigraph_include_graphrag_export_pages
            ),
            export_wikigraph_artifacts=export_wikigraph_artifacts,
            wikigraph_artifact_types=(
                tuple(
                    item.strip() for item in artifact_types.split(",") if item.strip()
                )
                if artifact_types
                else None
            ),
        )

        console.print(f"Mode: {_mode_label(options)}")

        try:

            def _progress_factory(pending_count):
                return progress_report(
                    label="Compiling",
                    length=pending_count,
                    item_label="source",
                )

            with lazy_live_status(_status_label(options)) as graph_status_update:
                result = service.run(
                    options,
                    compile_progress_factory=_progress_factory,
                    graph_status_callback=graph_status_update,
                )
        except (UpdatePreflightError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

        # Render ingest phase
        for summary in result.ingest_summaries:
            if summary.is_dir:
                console.print(
                    f"Added {summary.created_count} source(s) from {summary.path}"
                )
            elif summary.message:
                console.print(summary.message)
            else:
                console.print(f"Added {summary.path.name}")
        if result.ingest_summaries:
            console.print("")

        # Render build phase
        cr = result.compile_result
        if cr is not None and cr.resumed_from_run_id:
            echo_status_line(
                "resume", f"resumed interrupted update run {cr.resumed_from_run_id}"
            )
            console.print("")

        if cr is not None:
            echo_section("Update Summary")
            console.print(f"Compiled {cr.compiled_count} source page(s)")
            console.print(f"Skipped {cr.skipped_count} source page(s)")
            for path in cr.compiled_paths:
                echo_bullet(f"updated {path}")

        # Render concept phase
        if cr is not None:
            console.print("")
            echo_section("Concept Summary")
            concept_result = result.concept_result
            if result.concepts_skipped:
                console.print(f"Skipped: {result.concepts_skip_reason}")
            elif concept_result is not None:
                console.print(
                    f"Generated {len(concept_result.concept_paths)} concept page(s)"
                )
                console.print(
                    "Updated "
                    f"{len(concept_result.updated_source_paths)} source page backlink "
                    "section(s)"
                )
                if concept_result.removed_paths:
                    console.print(
                        f"Removed {len(concept_result.removed_paths)} stale concept page(s)"
                    )
                for path in concept_result.concept_paths:
                    echo_bullet(path)
            if result.search_warning:
                console.print("")
                echo_section("Search Summary")
                console.print(f"[yellow]{result.search_warning}[/yellow]")

        graph_result = result.graph_result
        if graph_result is not None:
            console.print("")
            echo_section("GraphRAG Summary")
            if graph_result.initialized:
                echo_bullet("initialized graph workspace")
            if graph_result.preflight_result is not None:
                decision = graph_result.preflight_result.decision
                if decision.action == "index":
                    console.print(
                        f"Graph index action: {decision.method} ({decision.reason})"
                    )
                    if decision.cost_warning:
                        console.print(decision.cost_warning)
                else:
                    console.print(f"Graph index action: {decision.reason}")
            if graph_result.sync_result and graph_result.sync_result.index_run:
                run = graph_result.sync_result.index_run
                console.print(f"Graph index run: {run.run_id} ({run.method})")
            output_path = graph_result.active_output_dir
            if (
                not output_path
                and graph_result.sync_result
                and graph_result.sync_result.index_run
            ):
                output_path = graph_result.sync_result.index_run.active_output_dir
            if output_path:
                console.print(f"Graph output: {output_path}")
            if graph_result.export_result:
                console.print(
                    "Graph wiki export: "
                    f"{graph_result.export_result.exported_count} page(s)"
                )
            if graph_result.warning:
                console.print(f"[yellow]{graph_result.warning}[/yellow]")
            elif graph_result.skipped and graph_result.skip_reason:
                console.print(f"Graph skipped: {graph_result.skip_reason}")

        # Render WikiGraphRAG phase
        console.print("")
        echo_section("WikiGraphRAG Summary")
        if result.wikigraph_skipped:
            console.print(
                f"WikiGraphRAG skipped: {result.wikigraph_skip_reason or 'unknown'}"
            )
        elif result.wikigraph_result is not None:
            report = result.wikigraph_result
            console.print(
                f"Built {report.node_count} node(s), {report.edge_count} edge(s), "
                f"{report.community_count} community(ies) "
                f"from {report.source_count} source page(s)"
            )
            for warning in report.warnings:
                console.print(f"[yellow]{warning}[/yellow]")
            if result.wikigraph_artifact_paths:
                per_type: dict[str, int] = {}
                for rel_path in result.wikigraph_artifact_paths:
                    parts = rel_path.split("/")
                    bucket = parts[2] if len(parts) > 2 else "(other)"
                    per_type[bucket] = per_type.get(bucket, 0) + 1
                console.print(
                    f"Exported {len(result.wikigraph_artifact_paths)} generated "
                    "card(s) under wiki/wikigraph/:"
                )
                for bucket in sorted(per_type):
                    echo_bullet(f"{per_type[bucket]} {bucket} card(s)")
        else:
            console.print("WikiGraphRAG build did not run.")

    return command
