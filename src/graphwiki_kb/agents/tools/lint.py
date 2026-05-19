"""lint_kb agent tool."""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import LintKbOutput
from graphwiki_kb.agents.tools._helpers import record_tool, tool_json


def run_lint_kb(runtime: AgentRuntimeContext) -> str:
    """Run deterministic wiki lint checks."""
    report = runtime.services.lint.lint()
    issues = [
        {
            "severity": issue.severity,
            "code": issue.code,
            "message": issue.message,
            "path": issue.path,
        }
        for issue in report.issues[:25]
    ]
    output = LintKbOutput(
        ok=report.error_count == 0,
        error_count=report.error_count,
        warning_count=report.warning_count,
        issues=issues,
    )
    record_tool(
        runtime,
        tool_name="lint_kb",
        ok=output.ok,
        summary=(
            f"Lint: {report.error_count} error(s), {report.warning_count} warning(s)."
        ),
        data=output.model_dump(),
    )
    return tool_json(output)
