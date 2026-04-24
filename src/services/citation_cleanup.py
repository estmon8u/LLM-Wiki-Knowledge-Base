"""Strip raw citation-ref markers that providers embed in answer prose."""

from __future__ import annotations

import re

# Matches patterns like:
#   [wiki/sources/page.md#chunk-0]
#   [wiki/sources/page.md#chunk-0, wiki/sources/other.md#chunk-1]
#   [`wiki/sources/page.md#chunk-0`]
#   (wiki/sources/page.md#chunk-0)
# With optional leading whitespace and surrounding punctuation.
_CITATION_REF_PATTERN = re.compile(
    r"\s*"
    r"[\[\(]"
    r"[`\s]*"
    r"(?:wiki/[^\]\)]+#chunk-\d+)"
    r"(?:\s*,\s*(?:wiki/[^\]\)]+#chunk-\d+))*"
    r"[`\s]*"
    r"[\]\)]",
)


def clean_citation_refs(text: str) -> str:
    """Remove raw wiki citation-ref markers from provider answer text."""
    return _CITATION_REF_PATTERN.sub("", text)
