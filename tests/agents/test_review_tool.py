"""Tests for review_kb tool."""

from __future__ import annotations

import json

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.tools.review import run_review_kb
from graphwiki_kb.models.wiki_models import ReviewIssue, ReviewReport


def test_review_kb_tool(test_project) -> None:
    class _Review:
        def review(self) -> ReviewReport:
            return ReviewReport(
                issues=[
                    ReviewIssue(
                        severity="warning",
                        code="drift",
                        message="Terminology drift detected.",
                        pages=["wiki/sources/a.md"],
                    )
                ]
            )

    test_project.services.review = _Review()  # type: ignore[assignment]
    runtime = AgentRuntimeContext(
        command_context=test_project.command_context,
        services=test_project.services,
    )
    payload = json.loads(run_review_kb(runtime))
    assert payload["issue_count"] == 1
