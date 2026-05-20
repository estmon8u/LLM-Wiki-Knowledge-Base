"""OpenAI Responses ``web_search`` integration for kb agent research.

This module belongs to `graphwiki_kb.services.web_research_service` and keeps
related behavior close to the command, service, model, provider, storage,
script, or test surface that uses it.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from graphwiki_kb.agents.models import (
    SearchContextSize,
    SourceRecommendation,
    WebFinding,
    WebResearchResult,
)
from graphwiki_kb.agents.prompts import (
    WEB_RESEARCH_SYSTEM_PROMPT,
    build_web_research_prompt,
)

logger = logging.getLogger(__name__)


class WebResearchError(RuntimeError):
    """Raised when the Responses API web_search call fails or is unparseable."""


class WebResearchService:
    """Calls the OpenAI Responses API with the ``web_search`` tool."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = "gpt-5.5",
        blocked_domains: list[str] | None = None,
        allowed_domains: list[str] | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self.blocked_domains = list(blocked_domains or [])
        self.allowed_domains = list(allowed_domains or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def research(
        self,
        *,
        question: str,
        local_answer: str,
        kb_gaps: list[str],
        search_context_size: SearchContextSize = "medium",
        max_recommendations: int = 5,
    ) -> WebResearchResult:
        """Run a web research call and return parsed findings + recommendations."""
        client = self._resolve_client()
        prompt = build_web_research_prompt(
            question=question,
            local_answer=local_answer,
            kb_gaps=kb_gaps,
            max_recommendations=max_recommendations,
        )
        tool: dict[str, Any] = {
            "type": "web_search",
            "search_context_size": search_context_size,
        }
        filters: dict[str, Any] = {}
        if self.allowed_domains:
            filters["allowed_domains"] = list(self.allowed_domains[:100])
        if self.blocked_domains:
            filters["blocked_domains"] = list(self.blocked_domains[:100])
        if filters:
            tool["filters"] = filters

        try:
            response = client.responses.create(
                model=self.model,
                tools=[tool],
                tool_choice="required",
                include=["web_search_call.action.sources"],
                input=[
                    {"role": "system", "content": WEB_RESEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            raise WebResearchError(f"OpenAI web_search call failed: {exc}") from exc

        return parse_web_research_response(
            response, max_recommendations=max_recommendations
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional extra
            raise WebResearchError(
                "OpenAI client is not available. Install the 'openai' extra."
            ) from exc
        self._client = OpenAI()
        return self._client


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_web_research_response(
    response: Any,
    *,
    max_recommendations: int = 5,
) -> WebResearchResult:
    """Translate a Responses API result into a WebResearchResult."""
    text = _extract_output_text(response)
    payload = _extract_json_payload(text)
    findings = _parse_findings(payload.get("findings", []))
    recommendations = _parse_recommendations(
        payload.get("recommendations", []),
        max_recommendations=max_recommendations,
    )
    sources = _extract_sources(response)
    return WebResearchResult(
        findings=findings,
        recommendations=recommendations,
        sources=sources,
        raw_text=text,
    )


def _extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if not output:
        return ""
    parts: list[str] = []
    for item in output:
        item_type = _attr(item, "type")
        if item_type != "message":
            continue
        content = _attr(item, "content") or []
        for chunk in content:
            chunk_type = _attr(chunk, "type")
            chunk_text = _attr(chunk, "text")
            if chunk_type in {"output_text", "text"} and isinstance(chunk_text, str):
                parts.append(chunk_text)
    return "\n".join(parts).strip()


_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json_payload(text: str) -> dict[str, Any]:
    if not text:
        return {}
    fence_match = _JSON_FENCE_PATTERN.search(text)
    candidate = fence_match.group(1) if fence_match else text
    candidate = candidate.strip()
    if not candidate:
        return {}
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.debug("Web research output did not contain JSON: %s", text[:200])
            return {}
        try:
            data = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            logger.debug("Web research JSON unparseable: %s", candidate[:200])
            return {}
    return data if isinstance(data, dict) else {}


def _parse_findings(items: Any) -> list[WebFinding]:
    if not isinstance(items, list):
        return []
    findings: list[WebFinding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            findings.append(WebFinding.model_validate(_normalize_finding(item)))
        except ValidationError as exc:
            logger.debug("Skipped malformed web finding: %s", exc)
    return findings


def _parse_recommendations(
    items: Any,
    *,
    max_recommendations: int,
) -> list[SourceRecommendation]:
    if not isinstance(items, list):
        return []
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    recommendations: list[SourceRecommendation] = []
    for index, item in enumerate(items[:max_recommendations], start=1):
        if not isinstance(item, dict):
            continue
        try:
            normalized = _normalize_recommendation(item, index=index, now=now)
            recommendations.append(SourceRecommendation.model_validate(normalized))
        except ValidationError as exc:
            logger.debug("Skipped malformed recommendation: %s", exc)
    return recommendations


def _normalize_finding(item: dict[str, Any]) -> dict[str, Any]:
    relevance = item.get("relevance")
    if relevance not in {"low", "medium", "high"}:
        relevance = "medium"
    return {
        "title": str(item.get("title") or item.get("url") or "Untitled").strip(),
        "url": str(item.get("url") or "").strip(),
        "summary": str(item.get("summary") or "").strip(),
        "relevance": relevance,
        "supports_recommendation": bool(item.get("supports_recommendation")),
    }


def _normalize_recommendation(
    item: dict[str, Any],
    *,
    index: int,
    now: str,
) -> dict[str, Any]:
    source_type = str(item.get("source_type") or "unknown").lower()
    if source_type not in {"paper", "docs", "article", "github", "blog", "unknown"}:
        source_type = "unknown"
    novelty = str(item.get("novelty") or "medium").lower()
    if novelty not in {"low", "medium", "high"}:
        novelty = "medium"
    confidence = str(item.get("confidence") or "medium").lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    raw_tags = item.get("suggested_tags") or []
    if not isinstance(raw_tags, list):
        raw_tags = []
    raw_citations = item.get("citation_urls") or []
    if not isinstance(raw_citations, list):
        raw_citations = []
    return {
        "id": int(item.get("id") or index),
        "title": str(item.get("title") or "Untitled").strip(),
        "url": str(item.get("url") or "").strip(),
        "source_type": source_type,
        "publisher": (item.get("publisher") or None) and str(item["publisher"]).strip(),
        "published_at": (item.get("published_at") or None)
        and str(item["published_at"]).strip(),
        "retrieved_at": str(item.get("retrieved_at") or now),
        "why_add": str(item.get("why_add") or "").strip(),
        "knowledge_gap": str(item.get("knowledge_gap") or "").strip(),
        "novelty": novelty,
        "confidence": confidence,
        "ingestable": bool(item.get("ingestable", True)),
        "suggested_tags": [str(tag).strip() for tag in raw_tags if str(tag).strip()],
        "citation_urls": [
            str(url).strip() for url in raw_citations if str(url).strip()
        ],
    }


def _extract_sources(response: Any) -> list[str]:
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    if not output:
        return []
    sources: list[str] = []
    for item in output:
        if _attr(item, "type") != "web_search_call":
            continue
        action = _attr(item, "action") or {}
        action_sources = _attr(action, "sources") or []
        for entry in action_sources:
            url = _attr(entry, "url") or _attr(entry, "src")
            if isinstance(url, str) and url.strip():
                sources.append(url.strip())
    return sources


def _attr(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
