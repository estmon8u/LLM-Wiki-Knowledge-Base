"""Query keyword extraction for LightRAG dual-level retrieval.

LightRAG splits a query into *low-level* keywords (specific entities, methods,
datasets, metrics, papers, tools, names) used to match entity vectors, and
*high-level* keywords (broad themes, relation topics, tradeoffs) used to match
relation vectors. A provider is used when available; otherwise a deterministic
fallback extracts capitalized phrases / acronyms (low-level) and broad-theme
hint words (high-level).
"""

from __future__ import annotations

import re

from graphwiki_kb.providers.base import ProviderRequest, TextProvider
from graphwiki_kb.providers.structured import (
    StructuredOutputError,
    parse_model_payload,
)
from graphwiki_kb.services.stopwords import STOPWORDS
from graphwiki_kb.wikigraph.light_models import QueryKeywords

_CAPITALIZED_PHRASE = re.compile(
    r"\b([A-Z][A-Za-z0-9][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9][A-Za-z0-9\-]*){0,3})\b"
)
_ACRONYM = re.compile(r"\b([A-Z]{2,6})\b")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")

_HIGH_LEVEL_HINTS: frozenset[str] = frozenset(
    {
        "tradeoff",
        "tradeoffs",
        "theme",
        "themes",
        "compare",
        "comparison",
        "impact",
        "limitation",
        "limitations",
        "architecture",
        "overview",
        "main",
        "ideas",
        "across",
        "landscape",
        "differences",
        "difference",
        "relationship",
        "relationships",
        "versus",
        "approaches",
        "trends",
        "challenges",
        "advantages",
        "benefits",
    }
)

_KEYWORD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "low_level_keywords": {"type": "array", "items": {"type": "string"}},
        "high_level_keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["low_level_keywords", "high_level_keywords"],
}

_KEYWORD_PROMPT = (
    "Extract query keywords for graph retrieval.\n\n"
    "Low-level keywords are specific entities, methods, datasets, metrics, "
    "papers, tools, or names.\nHigh-level keywords are broad themes, relation "
    "topics, goals, tradeoffs, or abstract concepts.\n\n"
    "Return JSON with 'low_level_keywords' and 'high_level_keywords'.\n\n"
    "Question: {question}"
)


def extract_query_keywords(
    question: str,
    *,
    provider: TextProvider | None = None,
    known_aliases: set[str] | None = None,
) -> QueryKeywords:
    """Return low/high-level keywords for ``question``.

    Uses ``provider`` structured output when available, otherwise a
    deterministic fallback. The fallback always runs first so its results can
    backstop an empty provider response.
    """
    fallback = _fallback_keywords(question, known_aliases or set())
    if provider is None:
        return fallback
    ensure = getattr(provider, "ensure_available", None)
    if callable(ensure):
        try:
            ensure()
        except Exception:
            return fallback
    try:
        response = provider.generate(
            ProviderRequest(
                prompt=_KEYWORD_PROMPT.format(question=question),
                system_prompt="Extract retrieval keywords as JSON.",
                max_tokens=512,
                response_schema=_KEYWORD_SCHEMA,
                response_schema_name="lightrag_query_keywords",
                reasoning_effort="low",
            )
        )
        keywords = parse_model_payload(
            response.text, QueryKeywords, label="LightRAG query keywords"
        )
    except (StructuredOutputError, Exception):
        return fallback

    low = _dedupe(keywords.low_level_keywords) or fallback.low_level_keywords
    high = _dedupe(keywords.high_level_keywords) or fallback.high_level_keywords
    return QueryKeywords(low_level_keywords=low, high_level_keywords=high)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def _fallback_keywords(question: str, known_aliases: set[str]) -> QueryKeywords:
    low: list[str] = []
    alias_lookup = {alias.casefold(): alias for alias in known_aliases}
    lowered = question.casefold()
    for alias_key, alias in alias_lookup.items():
        if alias_key and alias_key in lowered:
            low.append(alias)
    for match in _CAPITALIZED_PHRASE.finditer(question):
        phrase = " ".join(match.group(1).split())
        if len(phrase) >= 2:
            low.append(phrase)
    for match in _ACRONYM.finditer(question):
        low.append(match.group(1))

    high: list[str] = []
    for word in _WORD.findall(lowered):
        if word in _HIGH_LEVEL_HINTS:
            high.append(word)
    # If no explicit theme hints, fall back to salient non-stopword tokens.
    if not high:
        for word in _WORD.findall(lowered):
            if (
                word not in STOPWORDS
                and len(word) > 3
                and word not in {kw.casefold() for kw in low}
            ):
                high.append(word)
    return QueryKeywords(
        low_level_keywords=_dedupe(low),
        high_level_keywords=_dedupe(high),
    )
