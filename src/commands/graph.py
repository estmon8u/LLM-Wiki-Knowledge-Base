from __future__ import annotations

import click
from rich.markdown import Markdown as RichMarkdown

from src.commands.common import console, emit_json, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from src.services.graphrag_command_service import GraphRAGCommandError
from src.services.graphrag_input_sync_service import GraphRAGInputSyncError
from src.services.graphrag_query_service import GRAPH_QUERY_METHODS, GraphRAGQueryError
from src.services.graphrag_status_service import GraphRAGStatus
from src.services.graphrag_wiki_export_service import GraphRAGWikiExportError


SUMMARY = "GraphRAG workspace commands."
DEFAULT_GRAPH_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_METHODS = ("standard", "fast", "standard-update", "fast-update")


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="graph", summary=SUMMARY)


def create_command() -> click.Command:
    @click.group(name="graph", help=SUMMARY, short_help="Run GraphRAG commands.")
    def graph_group() -> None:
        """GraphRAG workspace commands."""

    @graph_group.command(
        name="init",
        help="Initialize the local GraphRAG workspace.",
        short_help="Initialize GraphRAG.",
    )
    @click.option(
        "--model",
        default=DEFAULT_GRAPH_MODEL,
        show_default=True,
        help="Chat model GraphRAG should write into settings.yaml.",
    )
    @click.option(
        "--embedding",
        default=DEFAULT_EMBEDDING_MODEL,
        show_default=True,
        help="Embedding model GraphRAG should write into settings.yaml.",
    )
    @click.option(
        "--force/--no-force",
        default=True,
        show_default=True,
        help="Allow GraphRAG to overwrite an existing workspace config.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def init(
        command_context: CommandContext,
        model: str,
        embedding: str,
        force: bool,
        as_json: bool,
    ) -> None:
        require_initialized(command_context)
        workspace_service = command_context.services["graphrag_workspace"]

        try:
            result = workspace_service.init_workspace(
                model=model,
                embedding=embedding,
                force=force,
            )
        except GraphRAGCommandError as exc:
            raise click.ClickException(str(exc)) from exc

        workspace_path = result.workspace_dir.relative_to(command_context.project_root)
        settings_path = result.settings_path.relative_to(command_context.project_root)

        if as_json:
            emit_json(
                {
                    "workspace_dir": workspace_path.as_posix(),
                    "settings_path": settings_path.as_posix(),
                    "command": list(result.result.command),
                    "returncode": result.result.returncode,
                    "stdout": result.result.stdout,
                    "stderr": result.result.stderr,
                }
            )
            return

        console.print(f"Initialized GraphRAG workspace at {workspace_path.as_posix()}")
        console.print(f"Settings: {settings_path.as_posix()}")

    @graph_group.command(
        name="sync",
        help="Sync normalized source artifacts into GraphRAG JSON input.",
        short_help="Sync normalized sources.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def sync(command_context: CommandContext, as_json: bool) -> None:
        require_initialized(command_context)
        sync_service = command_context.services["graphrag_input_sync"]

        try:
            result = sync_service.sync()
        except GraphRAGInputSyncError as exc:
            raise click.ClickException(str(exc)) from exc

        output_path = result.output_path.relative_to(command_context.project_root)
        settings_path = result.settings_path.relative_to(command_context.project_root)

        if as_json:
            emit_json(
                {
                    "source_count": result.source_count,
                    "output_path": output_path.as_posix(),
                    "settings_path": settings_path.as_posix(),
                    "metadata_fields": list(result.metadata_fields),
                    "settings_updated": result.settings_updated,
                }
            )
            return

        console.print(
            f"Synced {result.source_count} normalized source(s) to "
            f"{output_path.as_posix()}"
        )
        console.print(f"Configured GraphRAG JSON input in {settings_path.as_posix()}")

    @graph_group.command(
        name="index",
        help="Run GraphRAG indexing for the synced JSON input.",
        short_help="Index the graph workspace.",
    )
    @click.option(
        "--method",
        type=click.Choice(INDEX_METHODS),
        default="fast",
        show_default=True,
        help="GraphRAG indexing method.",
    )
    @click.option("--dry-run", is_flag=True, help="Validate without running indexing.")
    @click.option(
        "--cache/--no-cache",
        default=True,
        show_default=True,
        help="Enable GraphRAG cache use.",
    )
    @click.option(
        "--skip-validation",
        is_flag=True,
        help="Forward GraphRAG's --skip-validation flag.",
    )
    @click.option("--verbose", is_flag=True, help="Forward GraphRAG's verbose flag.")
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def index(
        command_context: CommandContext,
        method: str,
        dry_run: bool,
        cache: bool,
        skip_validation: bool,
        verbose: bool,
        as_json: bool,
    ) -> None:
        require_initialized(command_context)
        status_service = command_context.services["graphrag_status"]
        command_service = command_context.services["graphrag_command"]
        status = status_service.status()
        _require_graph_index_ready(status)

        try:
            command_result = command_service.index(
                method=method,
                dry_run=dry_run,
                cache=cache,
                skip_validation=skip_validation,
                verbose=verbose,
            )
        except GraphRAGCommandError as exc:
            if exc.result is not None:
                status_service.record_index_run(
                    method=method,
                    dry_run=dry_run,
                    result=exc.result,
                )
            raise click.ClickException(str(exc)) from exc

        run = status_service.record_index_run(
            method=method,
            dry_run=dry_run,
            result=command_result,
        )

        if as_json:
            emit_json(
                {
                    "run": run.to_dict(),
                    "command": list(command_result.command),
                    "stdout": command_result.stdout,
                    "stderr": command_result.stderr,
                }
            )
            return

        mode = "dry run" if dry_run else "index"
        console.print(f"GraphRAG {mode} completed with method {method}.")
        console.print(f"Run ID: {run.run_id}")

    @graph_group.command(
        name="ask",
        help="Ask a question through an explicit GraphRAG query mode.",
        short_help="Ask with GraphRAG.",
    )
    @click.argument("question")
    @click.option(
        "--method",
        type=click.Choice(GRAPH_QUERY_METHODS),
        required=True,
        help="GraphRAG query method.",
    )
    @click.option(
        "--community-level",
        type=int,
        help="Forward GraphRAG's community level option.",
    )
    @click.option(
        "--dynamic-community-selection/--no-dynamic-selection",
        default=None,
        help="Forward GraphRAG dynamic community selection behavior.",
    )
    @click.option(
        "--response-type",
        help="Forward GraphRAG's response type option.",
    )
    @click.option("--verbose", is_flag=True, help="Forward GraphRAG's verbose flag.")
    @click.option("--save", is_flag=True, help="Save the answer as an analysis page.")
    @click.option("--save-as", help="Save the answer using a custom analysis slug.")
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def ask(
        command_context: CommandContext,
        question: str,
        method: str,
        community_level: int | None,
        dynamic_community_selection: bool | None,
        response_type: str | None,
        verbose: bool,
        save: bool,
        save_as: str | None,
        as_json: bool,
    ) -> None:
        require_initialized(command_context)
        query_service = command_context.services["graphrag_query"]

        try:
            answer = query_service.ask(
                question,
                method=method,
                community_level=community_level,
                dynamic_community_selection=dynamic_community_selection,
                response_type=response_type,
                verbose=verbose,
            )
            if save or save_as:
                query_service.save_answer(answer, slug=save_as)
        except GraphRAGQueryError as exc:
            raise click.ClickException(str(exc)) from exc

        if as_json:
            emit_json(answer.to_dict())
            return

        console.print(
            f"[dim]\\[retriever: graphrag, method: {answer.method}, "
            f"index_run_id: {answer.index_run_id or 'unknown'}][/dim]"
        )
        console.print("")
        console.print(RichMarkdown(answer.answer or "No answer text returned."))
        if answer.saved_path:
            console.print(f"\nSaved analysis page: {answer.saved_path}")

    @graph_group.command(
        name="export-wiki",
        help="Export GraphRAG output tables into wiki/graph markdown pages.",
        short_help="Export graph wiki pages.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def export_wiki(command_context: CommandContext, as_json: bool) -> None:
        require_initialized(command_context)
        export_service = command_context.services["graphrag_wiki_export"]

        try:
            result = export_service.export_wiki()
        except GraphRAGWikiExportError as exc:
            raise click.ClickException(str(exc)) from exc

        if as_json:
            emit_json(result.to_dict())
            return

        console.print(
            f"Exported {result.exported_count} GraphRAG wiki page(s) to wiki/graph"
        )
        if result.missing_tables:
            console.print(
                "Missing GraphRAG table(s): " + ", ".join(result.missing_tables)
            )

    @graph_group.command(
        name="status",
        help="Show GraphRAG workspace and index status.",
        short_help="Show graph status.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def status(command_context: CommandContext, as_json: bool) -> None:
        require_initialized(command_context)
        status_service = command_context.services["graphrag_status"]
        snapshot = status_service.status()

        if as_json:
            emit_json(snapshot.to_dict(command_context.project_root))
            return

        _print_status(snapshot, command_context.project_root)

    return graph_group


def _require_graph_index_ready(status: GraphRAGStatus) -> None:
    if not status.workspace_initialized:
        raise click.ClickException(
            "GraphRAG workspace is not initialized. Run `kb graph init` first."
        )
    if not status.input_exists:
        raise click.ClickException(
            "GraphRAG input not found. Run `kb graph sync` first."
        )
    if status.input_document_count == 0:
        raise click.ClickException(
            "GraphRAG input has no documents. Add and compile sources, then run "
            "`kb graph sync`."
        )


def _print_status(status: GraphRAGStatus, project_root) -> None:
    payload = status.to_dict(project_root)
    initialized = "yes" if status.workspace_initialized else "no"
    input_state = "present" if status.input_exists else "missing"
    output_state = "present" if status.output_present else "missing"
    console.print(f"Workspace initialized: {initialized}")
    console.print(f"Workspace path: {payload['workspace_dir']}")
    console.print(f"Input: {input_state} ({status.input_document_count} document(s))")
    console.print(f"Index output: {output_state}")
    console.print(f"Documents table: {_present(status.documents_present)}")
    console.print(f"Text units table: {_present(status.text_units_present)}")
    console.print(f"Entities table: {_present(status.entities_present)}")
    console.print(f"Relationships table: {_present(status.relationships_present)}")
    console.print(f"Communities table: {_present(status.communities_present)}")
    console.print(
        f"Community reports table: {_present(status.community_reports_present)}"
    )
    if status.last_index_run_at:
        success = "yes" if status.last_index_success else "no"
        console.print(
            "Last index run: "
            f"{status.last_index_run_at} "
            f"({status.last_index_method}, success: {success})"
        )
    else:
        console.print("Last index run: none")
    console.print(f"Next action: {status.next_action}")


def _present(value: bool) -> str:
    return "present" if value else "missing"
