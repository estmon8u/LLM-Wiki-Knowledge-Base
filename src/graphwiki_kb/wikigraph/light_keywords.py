"""Query keyword extraction for LightRAG dual-level retrieval."""

from __future__ import annotations

import json
import re

from graphwiki_kb.providers.base import ProviderRequest, TextProvider
from graphwiki_kb.wikigraph.light_models import EntityProfile, QueryKeywords

_HIGH_LEVEL_TERMS = frozenset(
    {
        "tradeoff",
        "trade-off",
        "themes",
        "theme",
        "compare",
        "impact",
        "limitations",
        "architecture",
        "overall",
        "across",
        "landscape",
        "patterns",
    }
)


def extract_query_keywords(
    question: str,
    *,
    provider: TextProvider | None = None,
    entity_catalog: list[EntityProfile] | None = None,
) -> QueryKeywords:
    """Extract low/high-level keywords for retrieval routing."""
    if provider is not None:
        try:
            return _extract_with_provider(question, provider=provider)
        except Exception:
            pass
    return _extract_fallback(question, entity_catalog=entity_catalog)


def _extract_with_provider(question: str, *, provider: TextProvider) -> QueryKeywords:
    prompt = (
        "Extract query keywords for graph retrieval.\n\n"
        "Low-level keywords are specific entities, methods, datasets, metrics, "
        "papers, tools, or names.\n"
        "High-level keywords are broad themes, relation topics, goals, tradeoffs, "
        "or abstract concepts.\n\n"
        "Return JSON only:\n"
        '{"low_level_keywords": [...], "high_level_keywords": [...]}\n\n'
        f"Question: {question}\n"
    )
    response = provider.generate(
        ProviderRequest(
            prompt=prompt,
            system_prompt="Return strict JSON only.",
            max_tokens=512,
            response_schema_name="light_keywords",
        )
    )
    payload = json.loads(
        response.text[response.text.find("{") : response.text.rfind("}") + 1]
    )
    return QueryKeywords(
        low_level_keywords=_clean_list(payload.get("low_level_keywords")),
        high_level_keywords=_clean_list(payload.get("high_level_keywords")),
    )


def _extract_fallback(
    question: str,
    *,
    entity_catalog: list[EntityProfile] | None,
) -> QueryKeywords:
    low: list[str] = []
    high: list[str] = []
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+./-]*", question)
    for token in tokens:
        if token.casefold() in _HIGH_LEVEL_TERMS:
            high.append(token)
        elif token[0].isupper() or token.isupper():
            low.append(token)
    if entity_catalog:
        lowered = question.casefold()
        for profile in entity_catalog:
            names = [profile.canonical_name, *profile.aliases]
            for name in names:
                if name and name.casefold() in lowered and name not in low:
                    low.append(name)
    if not high:
        for token in tokens:
            if len(token) >= 5 and token.casefold() in _HIGH_LEVEL_TERMS:
                high.append(token)
    if not low and tokens:
        low = tokens[:6]
    if not high:
        high = [token for token in tokens if len(token) >= 6][:4]
    return QueryKeywords(
        low_level_keywords=_clean_list(low),
        high_level_keywords=_clean_list(high),
    )


def _clean_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return list(dict.fromkeys(cleaned))
