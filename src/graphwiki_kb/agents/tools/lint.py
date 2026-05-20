"""Agent tool: lint the KB.

This module belongs to `graphwiki_kb.agents.tools.lint` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.
"""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import AgentToolResult, LintOutput

TOOL_NAME = "lint"
TOOL_DESCRIPTION = (
    "Lint the KB for stale, missing, or broken artifacts and report a small "
    "summary of issues by severity. Read-only."
)

_MAX_ISSUES = 20


def run_lint(runtime: AgentRuntimeContext) -> LintOutput:
    """Run the lint service and project its result."""
    report = runtime.services.lint.lint()
    issues = [
        {
            "severity": issue.severity,
            "code": issue.code,
            "path": issue.path,
            "message": issue.message,
        }
        for issue in report.issues[:_MAX_ISSUES]
    ]
    output = LintOutput(
        error_count=report.error_count,
        warning_count=report.warning_count,
        suggestion_count=report.suggestion_count,
        issues=issues,
    )
    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=True,
            summary=(
                f"lint errors={output.error_count} "
                f"warnings={output.warning_count} "
                f"suggestions={output.suggestion_count}"
            ),
            data={
                "error_count": output.error_count,
                "warning_count": output.warning_count,
                "suggestion_count": output.suggestion_count,
            },
        )
    )
    return output
