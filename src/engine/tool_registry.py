from __future__ import annotations

from src.models.tool_models import ToolContext, ToolResult, ToolSpec


def build_tool_specs() -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            name="SearchWiki",
            summary="Search the compiled wiki for relevant pages.",
            access_level="read",
            is_concurrency_safe=True,
            run=_search_wiki,
        ),
        ToolSpec(
            name="ReadManifest",
            summary="Read the raw-source manifest.",
            access_level="read",
            is_concurrency_safe=True,
            run=_read_manifest,
        ),
        ToolSpec(
            name="IngestSource",
            summary="Ingest a new source into the raw corpus.",
            access_level="write",
            is_concurrency_safe=False,
            run=_unsupported,
        ),
        ToolSpec(
            name="LintWiki",
            summary="Run structural wiki health checks.",
            access_level="read",
            is_concurrency_safe=True,
            run=_lint_wiki,
        ),
        ToolSpec(
            name="ReviewWiki",
            summary="Run semantic review checks for contradictions and terminology.",
            access_level="read",
            is_concurrency_safe=True,
            run=_review_wiki,
        ),
        ToolSpec(
            name="ExportVault",
            summary="Export wiki pages into the vault.",
            access_level="export",
            is_concurrency_safe=False,
            run=_unsupported,
        ),
    )


def _search_wiki(arguments: dict[str, object], tool_context: ToolContext) -> ToolResult:
    query = str(arguments.get("query", "")).strip()
    if not query:
        return ToolResult(ok=False, content="Missing search query.")
    results = tool_context.services["search"].search(
        query, limit=int(arguments.get("limit", 5))
    )
    return ToolResult(
        ok=True,
        content="\n".join(f"- {item.title} [{item.path}]" for item in results),
        data={"results": [item.__dict__ for item in results]},
    )


def _read_manifest(_: dict[str, object], tool_context: ToolContext) -> ToolResult:
    sources = tool_context.services["manifest"].list_sources()
    return ToolResult(
        ok=True,
        content=f"Loaded {len(sources)} source record(s).",
        data={"sources": [item.to_dict() for item in sources]},
    )


def _lint_wiki(_: dict[str, object], tool_context: ToolContext) -> ToolResult:
    report = tool_context.services["lint"].lint()
    return ToolResult(
        ok=True,
        content=f"Found {len(report.issues)} lint issue(s).",
        data={
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "path": issue.path,
                    "message": issue.message,
                }
                for issue in report.issues
            ]
        },
    )


def _review_wiki(_: dict[str, object], tool_context: ToolContext) -> ToolResult:
    report = tool_context.services["review"].review()
    return ToolResult(
        ok=True,
        content=f"Found {report.issue_count} review issue(s) ({report.mode} mode).",
        data={
            "mode": report.mode,
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "pages": issue.pages,
                    "message": issue.message,
                }
                for issue in report.issues
            ],
        },
    )


def _unsupported(_: dict[str, object], __: ToolContext) -> ToolResult:
    return ToolResult(
        ok=False,
        content="This tool contract is defined, but direct tool execution is not wired into the CLI yet.",
    )
