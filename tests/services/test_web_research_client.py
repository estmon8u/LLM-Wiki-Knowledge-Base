"""Web research service client invocation tests."""

from __future__ import annotations

from graphwiki_kb.services.web_research_service import WebResearchService


class _FakeResponses:
    def create(self, **kwargs: object) -> dict[str, object]:
        return {
            "output_text": "Found https://example.com/paper",
            "output": [
                {
                    "type": "web_search_call",
                    "action": {"sources": [{"url": "https://example.com/paper"}]},
                }
            ],
        }


class _FakeClient:
    responses = _FakeResponses()


def test_web_research_service_calls_client() -> None:
    service = WebResearchService(client=_FakeClient(), model="test-model")
    result = service.research(
        question="RAG",
        local_answer="local",
        kb_gaps=["gap"],
        search_context_size="low",
    )
    assert result.source_urls
    assert "example.com" in result.source_urls[0]
