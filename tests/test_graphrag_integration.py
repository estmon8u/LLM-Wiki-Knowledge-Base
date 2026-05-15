"""Opt-in provider-backed GraphRAG integration smoke tests."""

from __future__ import annotations

import os

from click.testing import CliRunner
import pytest

from graphwiki_kb.cli import main


pytestmark = [
    pytest.mark.graphrag_integration,
    pytest.mark.skipif(
        os.environ.get("RUN_PROVIDER_TESTS") != "1"
        or not os.environ.get("OPENAI_API_KEY"),
        reason=(
            "set RUN_PROVIDER_TESTS=1 and OPENAI_API_KEY to run live GraphRAG "
            "integration smoke tests"
        ),
    ),
]


def test_tiny_live_graphrag_smoke() -> None:
    """Indexes and queries a tiny corpus only when live provider calls are enabled."""
    runner = CliRunner(mix_stderr=False)
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        with open("seed.md", "w", encoding="utf-8") as handle:
            handle.write(
                "# Retrieval Notes\n\n"
                "GraphRAG builds graph summaries from source text for cross-document "
                "questions.\n"
            )

        update_result = runner.invoke(
            main, ["update", "seed.md", "--graph-method", "fast"]
        )
        assert update_result.exit_code == 0, update_result.output

        status_result = runner.invoke(main, ["status", "--strict"])
        assert status_result.exit_code == 0, status_result.output

        ask_result = runner.invoke(
            main,
            ["ask", "--method", "basic", "What does GraphRAG build?"],
        )
        assert ask_result.exit_code == 0, ask_result.output
        assert "graph" in ask_result.output.casefold()
