from __future__ import annotations

import click

from src.commands.common import echo_kv, echo_section, require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Show prior ask, review, compile, and update runs."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="history", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(
        name="history",
        help=SUMMARY,
        short_help="Show run history.",
    )
    @click.option(
        "--command",
        "filter_command",
        type=str,
        default=None,
        help="Filter runs by command name (e.g. ask, review, compile).",
    )
    @click.option("--limit", default=20, show_default=True, type=int)
    @click.pass_obj
    def command(
        command_context: CommandContext,
        filter_command: str | None,
        limit: int,
    ) -> None:
        require_initialized(command_context)
        run_store = command_context.services.get("run_store")
        compile_run_store = command_context.services.get("compile_run_store")

        # Collect provider-backed runs (ask, review, etc.)
        provider_runs: list[dict] = []
        if run_store is not None and filter_command != "compile":
            for run in run_store.list_runs(command=filter_command, limit=limit):
                ts = run.timestamp[:19] if len(run.timestamp) > 19 else run.timestamp
                provider_runs.append(
                    {
                        "timestamp": ts,
                        "kind": run.command,
                        "detail": run.model_id,
                        "extra": f"citations={run_store.citation_count(run.run_id)}",
                        "preview": run.final_text,
                    }
                )

        # Collect compile runs
        compile_runs: list[dict] = []
        if compile_run_store is not None and filter_command in (None, "compile"):
            for rec in compile_run_store.load_history():
                ts = rec.started_at[:19] if len(rec.started_at) > 19 else rec.started_at
                n_done = len(rec.completed_source_slugs)
                n_planned = len(rec.planned_source_slugs)
                detail = f"{rec.status}  {n_done}/{n_planned} sources"
                if rec.force:
                    detail += "  --force"
                compile_runs.append(
                    {
                        "timestamp": ts,
                        "kind": "compile",
                        "detail": detail,
                        "extra": "",
                        "preview": rec.error if rec.error else "",
                    }
                )

        # Merge and sort by timestamp descending, then limit
        all_runs = provider_runs + compile_runs
        all_runs.sort(key=lambda r: r["timestamp"], reverse=True)
        all_runs = all_runs[:limit]

        if not all_runs:
            click.echo("No runs recorded yet.")
            return

        echo_section("Run History")
        for run in all_runs:
            line = f"  {run['timestamp']}  {run['kind']:<10s}  {run['detail']}"
            if run["extra"]:
                line += f"  {run['extra']}"
            click.echo(line)
            if run["preview"]:
                preview = run["preview"][:80].replace("\n", " ")
                click.echo(f"    {preview}...")

        click.echo("")
        echo_kv("total", len(all_runs))

    return command
