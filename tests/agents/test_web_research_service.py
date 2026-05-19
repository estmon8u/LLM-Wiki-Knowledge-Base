"""Tests for the WebResearchService and Responses parsing."""

from __future__ import annotations

import json
from typing import Any

import pytest

from graphwiki_kb.services.web_research_service import (
    WebResearchError,
    WebResearchService,
    parse_web_research_response,
)


class _FakeResponses:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


class _FakeClient:
    def __init__(self, responses: _FakeResponses) -> None:
        self.responses = responses


def _response_object(text: str, sources: list[str] | None = None) -> Any:
    output = []
    output.append(
        {
            "type": "web_search_call",
            "action": {
                "sources": [{"url": s} for s in (sources or [])],
            },
        }
    )
    output.append(
        {
            "type": "message",
            "content": [{"type": "output_text", "text": text}],
        }
    )
    return type("Resp", (), {"output": output, "output_text": text})()


def test_parse_web_research_response_extracts_json_fenced_body() -> None:
    text = (
        "```json\n"
        + json.dumps(
            {
                "findings": [
                    {
                        "title": "A",
                        "url": "https://a",
                        "summary": "s",
                        "relevance": "high",
                    }
                ],
                "recommendations": [
                    {
                        "title": "Paper",
                        "url": "https://b/paper.pdf",
                        "source_type": "paper",
                        "why_add": "fills gap",
                        "knowledge_gap": "g",
                        "novelty": "high",
                        "confidence": "high",
                        "ingestable": True,
                        "suggested_tags": ["ml"],
                    }
                ],
            }
        )
        + "\n```"
    )
    result = parse_web_research_response(_response_object(text, sources=["https://b"]))
    assert len(result.findings) == 1
    assert result.findings[0].relevance == "high"
    assert len(result.recommendations) == 1
    assert result.recommendations[0].id == 1
    assert result.recommendations[0].source_type == "paper"
    assert result.sources == ["https://b"]


def test_parse_web_research_response_recovers_from_malformed_json() -> None:
    text = (
        "Here is the data: "
        + json.dumps({"findings": [], "recommendations": []})
        + " trailing stuff."
    )
    result = parse_web_research_response(_response_object(text))
    assert result.findings == []
    assert result.recommendations == []


def test_parse_web_research_response_caps_recommendations() -> None:
    items = [
        {
            "title": f"R{i}",
            "url": f"https://example.com/{i}",
            "source_type": "paper",
            "why_add": "r",
            "knowledge_gap": "g",
        }
        for i in range(10)
    ]
    text = json.dumps({"findings": [], "recommendations": items})
    result = parse_web_research_response(_response_object(text), max_recommendations=3)
    assert [r.id for r in result.recommendations] == [1, 2, 3]


def test_web_research_service_passes_filters_and_tool_choice() -> None:
    fake_resp = _response_object('{"findings": [], "recommendations": []}')
    fake_responses = _FakeResponses(fake_resp)
    service = WebResearchService(
        client=_FakeClient(fake_responses),
        model="test-model",
        blocked_domains=["reddit.com"],
        allowed_domains=["arxiv.org"],
    )
    service.research(
        question="q",
        local_answer="a",
        kb_gaps=["g"],
        max_recommendations=5,
    )
    call = fake_responses.calls[0]
    assert call["model"] == "test-model"
    assert call["tool_choice"] == "required"
    tool = call["tools"][0]
    assert tool["type"] == "web_search"
    assert tool["search_context_size"] == "medium"
    assert tool["filters"]["blocked_domains"] == ["reddit.com"]
    assert tool["filters"]["allowed_domains"] == ["arxiv.org"]
    assert call["include"] == ["web_search_call.action.sources"]


def test_web_research_service_wraps_openai_errors() -> None:
    class _Broken:
        def create(self, **kwargs):
            raise RuntimeError("boom")

    class _Client:
        responses = _Broken()

    service = WebResearchService(client=_Client(), model="m")
    with pytest.raises(WebResearchError):
        service.research(question="q", local_answer="", kb_gaps=[])


def test_web_research_service_skips_filters_when_unset() -> None:
    fake_resp = _response_object('{"findings": [], "recommendations": []}')
    fake_responses = _FakeResponses(fake_resp)
    service = WebResearchService(client=_FakeClient(fake_responses), model="m")
    service.research(question="q", local_answer="a", kb_gaps=[])
    tool = fake_responses.calls[0]["tools"][0]
    assert "filters" not in tool


def test_parse_response_handles_object_style_output() -> None:
    """The parser should accept objects with .output / .output_text attributes."""

    class _Action:
        sources = [{"url": "https://example.com"}]

    class _Item:
        type = "web_search_call"
        action = _Action()

    class _Message:
        type = "message"

        class _Chunk:
            type = "output_text"
            text = '{"findings": [], "recommendations": []}'

        content = [_Chunk()]

    class _Resp:
        output = [_Item(), _Message()]
        output_text = None

    result = parse_web_research_response(_Resp())
    assert result.sources == ["https://example.com"]


def test_parse_response_handles_empty_text() -> None:
    class _Resp:
        output = []
        output_text = None

    result = parse_web_research_response(_Resp())
    assert result.findings == []
    assert result.recommendations == []
    assert result.raw_text == ""


def test_normalize_recommendation_clamps_invalid_enums() -> None:
    text = json.dumps(
        {
            "findings": [],
            "recommendations": [
                {
                    "title": "x",
                    "url": "https://x",
                    "source_type": "magazine",  # not a known type
                    "why_add": "fills gap",
                    "knowledge_gap": "g",
                    "novelty": "extreme",
                    "confidence": "perfect",
                    "ingestable": True,
                    "suggested_tags": "tag",  # not a list
                }
            ],
        }
    )
    result = parse_web_research_response(_response_object(text))
    assert result.recommendations[0].source_type == "unknown"
    assert result.recommendations[0].novelty == "medium"
    assert result.recommendations[0].confidence == "medium"
    assert result.recommendations[0].suggested_tags == []
