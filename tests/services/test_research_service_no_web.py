"""Research service without web."""

from __future__ import annotations

from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryAnswer
from graphwiki_kb.services.research_service import ResearchService


def test_research_without_web(test_project) -> None:
    class _Ask:
        def ask(self, question: str, **kwargs: object) -> GraphRAGQueryAnswer:
            return GraphRAGQueryAnswer(
                question=question,
                answer="Only local.",
                raw_output="",
                method="local",
                created_at="2026-05-19T00:00:00+00:00",
                index_run_id=None,
                command=(),
                stdout="",
                stderr="",
                graph_input_hash="",
            )

    service = ResearchService(
        test_project.paths,
        test_project.config,
        ask_controller=_Ask(),  # type: ignore[arg-type]
        web_research=None,
    )
    output = service.run(question="local only", use_web=False, recommend_sources=False)
    assert output.local_answer.answer == "Only local."
    assert not output.recommendations
