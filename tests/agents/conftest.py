"""Shared fixtures for kb agent tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.services.source_recommendation_store import (
    SourceRecommendationStore,
)
from tests.conftest import TestProject


@dataclass
class FakeAgentRuntime(AgentRuntimeContext):
    """An AgentRuntimeContext pre-populated with a recommendation store."""


@pytest.fixture
def runtime(test_project: TestProject) -> AgentRuntimeContext:
    """Build a real AgentRuntimeContext backed by the test_project fixture."""
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    runtime = AgentRuntimeContext(
        command_context=test_project.command_context,
        services=test_project.services,
        recommendation_store=store,
        auto_approve=False,
        show_plan=False,
    )
    return runtime
