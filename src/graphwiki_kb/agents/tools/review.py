"""review_kb agent tool."""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import ReviewKbOutput
from graphwiki_kb.agents.tools._helpers import record_tool, tool_json
from graphwiki_kb.providers import ProviderError


def run_review_kb(runtime: AgentRuntimeContext) -> str:
    """Run semantic KB quality review (requires provider)."""
    try:
        report = runtime.services.review.review()
    except ProviderError as exc:
        record_tool(
            runtime,
            tool_name="review_kb",
            ok=False,
            summary=str(exc),
            error=str(exc),
        )
        raise
    issues = [
        {
            "severity": issue.severity,
            "code": issue.code,
            "message": issue.message,
            "pages": list(issue.pages),
        }
        for issue in report.issues[:25]
    ]
    output = ReviewKbOutput(issue_count=len(report.issues), issues=issues)
    record_tool(
        runtime,
        tool_name="review_kb",
        ok=True,
        summary=f"Review found {len(report.issues)} issue(s).",
        data=output.model_dump(),
    )
    return tool_json(output)
