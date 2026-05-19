"""Direct OpenAI Responses API web_search integration for agent research."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from urllib.parse import urlparse

from graphwiki_kb.agents.models import SourceRecommendation, WebFinding
from graphwiki_kb.agents.prompts import WEB_RESEARCH_SYNTHESIS_PROMPT
from graphwiki_kb.services.project_service import utc_now_iso

_URL_PATTERN = re.compile(r"https?://[^\s\])<>\"']+")


@dataclass
class WebResearchResult:
    """Parsed output from a Responses API web search call."""

    summary_text: str
    web_findings: list[WebFinding] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    raw_response: dict[str, object] = field(default_factory=dict)


def build_web_research_prompt(
    question: str,
    local_answer: str,
    kb_gaps: list[str],
) -> str:
    """Build the user prompt for web research synthesis."""
    gaps = "\n".join(f"- {gap}" for gap in kb_gaps) or "- (none identified)"
    return (
        f"{WEB_RESEARCH_SYNTHESIS_PROMPT}\n\n"
        f"## Question\n{question}\n\n"
        f"## Local KB answer\n{local_answer or '(empty)'}\n\n"
        f"## KB gaps\n{gaps}\n"
    )


def _response_to_dict(response: Any) -> dict[str, object]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if isinstance(response, dict):
        return response
    return {"output_text": getattr(response, "output_text", "") or str(response)}


def _collect_output_text(payload: dict[str, object]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    chunks: list[str] = []
    raw_output = payload.get("output")
    output_items = raw_output if isinstance(raw_output, list) else []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for content in item.get("content", []) or []:
                if isinstance(content, dict) and content.get("type") in {
                    "output_text",
                    "text",
                }:
                    text = content.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
    return "\n".join(chunks).strip()


def _collect_web_sources(payload: dict[str, object]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    def add(url: str) -> None:
        normalized = url.rstrip(").,;")
        if normalized and normalized not in seen:
            seen.add(normalized)
            urls.append(normalized)

    raw_output = payload.get("output")
    output_items = raw_output if isinstance(raw_output, list) else []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "web_search_call":
            continue
        action = item.get("action")
        if isinstance(action, dict):
            for source in action.get("sources", []) or []:
                if isinstance(source, dict):
                    url = source.get("url")
                    if isinstance(url, str):
                        add(url)
                elif isinstance(source, str):
                    add(source)
    if not urls:
        text = _collect_output_text(payload)
        for match in _URL_PATTERN.findall(text):
            add(match)
    return urls


def _guess_source_type(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    path = (urlparse(url).path or "").lower()
    if host.endswith("github.com"):
        return "github"
    if any(part in path for part in (".pdf", "/pdf")):
        return "paper"
    if any(token in host for token in ("arxiv.org", "doi.org", "acm.org", "ieee.org")):
        return "paper"
    if any(token in host for token in ("docs.", "readthedocs", "developer.")):
        return "docs"
    if any(token in host for token in ("blog.", "medium.com", "substack.com")):
        return "blog"
    return "article"


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = (
        parsed.path.strip("/").split("/")[-1]
        if parsed.path.strip("/")
        else parsed.netloc
    )
    return path.replace("-", " ").replace("_", " ")[:120] or url


def build_recommendations_from_urls(
    urls: list[str],
    *,
    question: str,
    kb_gaps: list[str],
    max_recommendations: int,
) -> list[SourceRecommendation]:
    """Turn consulted URLs into numbered source recommendations."""
    gap_text = kb_gaps[0] if kb_gaps else question
    recommendations: list[SourceRecommendation] = []
    for index, url in enumerate(urls[:max_recommendations], start=1):
        guessed_type = _guess_source_type(url)
        source_type = cast(
            Literal["paper", "docs", "article", "github", "blog", "unknown"],
            guessed_type,
        )
        ingestable = source_type in {
            "paper",
            "docs",
            "article",
            "blog",
        } or url.endswith(".pdf")
        recommendations.append(
            SourceRecommendation(
                id=index,
                title=_title_from_url(url),
                url=url,
                source_type=source_type,
                retrieved_at=utc_now_iso(),
                why_add=f"Web search surfaced this source while researching: {question}",
                knowledge_gap=gap_text,
                novelty="medium",
                confidence="medium",
                ingestable=ingestable,
                suggested_tags=[],
                citation_urls=[url],
            )
        )
    return recommendations


def parse_web_research_response(response: Any) -> WebResearchResult:
    """Parse a Responses API payload into structured web research output."""
    payload = _response_to_dict(response)
    summary_text = _collect_output_text(payload)
    source_urls = _collect_web_sources(payload)
    findings: list[WebFinding] = []
    for url in source_urls:
        findings.append(
            WebFinding(
                title=_title_from_url(url),
                url=url,
                summary=summary_text[:400] if summary_text else url,
                relevance="medium",
                supports_recommendation=True,
            )
        )
    return WebResearchResult(
        summary_text=summary_text,
        web_findings=findings,
        source_urls=source_urls,
        raw_response=payload,
    )


class WebResearchService:
    """Calls OpenAI Responses web_search for agent research."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.4-nano",
        client: Any | None = None,
        blocked_domains: list[str] | None = None,
    ) -> None:
        self.model = model
        self._client = client
        self._blocked_domains = list(blocked_domains or [])

    def _client_or_create(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import OpenAI

        return OpenAI()

    def research(
        self,
        *,
        question: str,
        local_answer: str,
        kb_gaps: list[str],
        search_context_size: str = "medium",
    ) -> WebResearchResult:
        """Run a required web_search Responses call."""
        tool: dict[str, object] = {
            "type": "web_search",
            "search_context_size": search_context_size,
        }
        if self._blocked_domains:
            tool["filters"] = {"blocked_domains": self._blocked_domains}
        client = self._client_or_create()
        response = client.responses.create(
            model=self.model,
            tools=[tool],
            tool_choice="required",
            include=["web_search_call.action.sources"],
            input=build_web_research_prompt(question, local_answer, kb_gaps),
        )
        return parse_web_research_response(response)

    def research_from_text(
        self,
        *,
        question: str,
        local_answer: str,
        kb_gaps: list[str],
        summary_text: str,
        source_urls: list[str],
    ) -> WebResearchResult:
        """Build a WebResearchResult without calling the API (tests)."""
        findings = [
            WebFinding(
                title=_title_from_url(url),
                url=url,
                summary=summary_text[:400] or url,
                relevance="medium",
                supports_recommendation=True,
            )
            for url in source_urls
        ]
        return WebResearchResult(
            summary_text=summary_text,
            web_findings=findings,
            source_urls=source_urls,
            raw_response={"output_text": summary_text, "source_urls": source_urls},
        )
