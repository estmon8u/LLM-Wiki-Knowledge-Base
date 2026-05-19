"""Agent tool: KB quality review.

This module belongs to `graphwiki_kb.agents.tools.review` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.
"""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import AgentToolResult, ReviewOutput
from graphwiki_kb.providers import (
    ProviderConfigurationError,
    ProviderExecutionError,
)

TOOL_NAME = "review"
TOOL_DESCRIPTION = (
    "Run a KB quality review and return summarized issues such as duplicate "
    "topics, contradictions, and terminology drift. Read-only."
)

_MAX_ISSUES = 20


def run_review(runtime: AgentRuntimeContext) -> ReviewOutput:
    """Run the review service and project its result."""
    try:
        report = runtime.services.review.review()
    except (
        ProviderConfigurationError,
        ProviderExecutionError,
    ) as exc:
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=False,
                summary="review failed (provider unavailable)",
                error=str(exc),
            )
        )
        return ReviewOutput(
            total_issues=0,
            issues=[],
            summary=f"Review requires a configured provider: {exc}",
        )
    issues = [
        {
            "severity": issue.severity,
            "code": issue.code,
            "pages": list(issue.pages),
            "message": issue.message,
        }
        for issue in report.issues[:_MAX_ISSUES]
    ]
    summary = (
        f"{len(report.issues)} issue(s) detected" if report.issues else "No issues."
    )
    output = ReviewOutput(
        total_issues=len(report.issues),
        issues=issues,
        summary=summary,
    )
    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=True,
            summary=summary,
            data={"total_issues": output.total_issues},
        )
    )
    return output
