"""Query keyword extraction for LightRAG-style dual-level retrieval.

Provider-backed keyword extraction is the LightRAG paper's default; the
project keeps that path open through the :class:`LightKeywordProvider`
protocol below. The offline / fallback path here is rule-based: it
matches known entity aliases, lifts capitalized phrases & acronyms as
low-level keywords, and treats a short list of abstract terms as
high-level keywords.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from graphwiki_kb.services.stopwords import STOPWORDS

_CAPITALIZED = re.compile(
    r"\b([A-Z][A-Za-z0-9][A-Za-z0-9\-]*"
    r"(?:\s+[A-Z][A-Za-z0-9][A-Za-z0-9\-]*){0,2})\b"
)
_ACRONYM = re.compile(r"\b([A-Z]{2,6})\b")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")

# Abstract / theme words that indicate a high-level question intent.
_HIGH_LEVEL_TERMS: tuple[str, ...] = (
    "trade-off",
    "tradeoff",
    "tradeoffs",
    "themes",
    "main themes",
    "main ideas",
    "compare",
    "comparison",
    "impact",
    "limitations",
    "architecture",
    "patterns",
    "landscape",
    "overall",
    "across",
    "ecosystem",
    "approach",
    "approaches",
    "evolution",
    "challenges",
    "trends",
    "differ",
    "difference",
    "differences",
    "relationship",
    "relationships",
    "relate",
    "related",
)


class QueryKeywords(BaseModel):
    """Structured low/high-level keyword bundle for one question."""

    low_level_keywords: list[str] = Field(default_factory=list)
    high_level_keywords: list[str] = Field(default_factory=list)


@runtime_checkable
class LightKeywordProvider(Protocol):
    """Any object that can extract retrieval keywords for a question."""

    name: str

    def extract(self, question: str) -> QueryKeywords:
        """Return :class:`QueryKeywords` for ``question``."""
        ...


@dataclass
class RuleBasedKeywordProvider:
    """Provider-free keyword extractor used as the offline fallback."""

    known_aliases: tuple[str, ...] = ()
    name: str = "rule-based"

    def __post_init__(self) -> None:
        self._alias_lookup: dict[str, str] = {}
        for alias in self.known_aliases:
            key = alias.casefold()
            if key not in self._alias_lookup:
                self._alias_lookup[key] = alias

    def extract(self, question: str) -> QueryKeywords:
        """Return low-level and high-level keywords for ``question``."""
        low_level = list(_extract_low_level(question, self._alias_lookup))
        high_level = list(_extract_high_level(question))
        return QueryKeywords(
            low_level_keywords=low_level,
            high_level_keywords=high_level,
        )


def _extract_low_level(question: str, alias_lookup: dict[str, str]) -> Iterable[str]:
    seen: set[str] = set()
    text = question
    lower = question.casefold()

    for alias_key, alias in alias_lookup.items():
        if not alias_key:
            continue
        if _word_in(lower, alias_key):
            key = alias.casefold()
            if key not in seen:
                seen.add(key)
                yield alias

    for match in _CAPITALIZED.finditer(text):
        phrase = " ".join(match.group(1).split()).strip(" .,:;")
        key = phrase.casefold()
        if not phrase or key in STOPWORDS or key in seen:
            continue
        seen.add(key)
        yield phrase

    for match in _ACRONYM.finditer(text):
        phrase = match.group(1)
        key = phrase.casefold()
        if key in seen or key in STOPWORDS:
            continue
        seen.add(key)
        yield phrase


def _word_in(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])"
    return re.search(pattern, haystack) is not None


def _extract_high_level(question: str) -> Iterable[str]:
    lower = question.casefold()
    seen: set[str] = set()
    for term in _HIGH_LEVEL_TERMS:
        if term in lower and term not in seen:
            seen.add(term)
            yield term
    # Fallback: significant nouns when no theme keywords matched.
    if not seen:
        for match in _WORD.finditer(question):
            word = match.group(0)
            key = word.casefold()
            if (
                len(word) >= 5
                and key not in STOPWORDS
                and key not in seen
                and not word[0].isupper()
            ):
                seen.add(key)
                yield word
                if len(seen) >= 5:
                    break
