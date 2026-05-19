"""Tests for web research parsing."""

from __future__ import annotations

from graphwiki_kb.services.web_research_service import (
    build_recommendations_from_urls,
    parse_web_research_response,
)


def test_parse_web_research_response_extracts_urls() -> None:
    payload = {
        "output_text": "See https://example.com/paper and https://docs.example.com/guide",
        "output": [
            {
                "type": "web_search_call",
                "action": {
                    "sources": [
                        {"url": "https://example.com/paper"},
                        "https://docs.example.com/guide",
                    ]
                },
            }
        ],
    }
    result = parse_web_research_response(payload)
    assert "example.com" in result.summary_text or result.source_urls
    assert len(result.source_urls) >= 1


def test_build_recommendations_numbered() -> None:
    recs = build_recommendations_from_urls(
        ["https://arxiv.org/abs/1234", "https://github.com/org/repo"],
        question="agents",
        kb_gaps=["gap"],
        max_recommendations=5,
    )
    assert recs[0].id == 1
    assert recs[1].id == 2
    assert recs[0].ingestable
