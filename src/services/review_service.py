from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Literal, Optional

from nltk.stem import SnowballStemmer
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from src.models.wiki_models import ReviewIssue, ReviewReport
from src.providers import (
    ProviderConfigurationError,
    ProviderExecutionError,
    UnavailableProvider,
)
from src.providers.base import ProviderRequest, TextProvider
from src.providers.structured import StructuredOutputError, parse_json_payload
from src.services.markdown_document import (
    parse_frontmatter as markdown_parse_frontmatter,
    sections as markdown_sections,
)
from src.services.project_service import ProjectPaths
from src.services.stopwords import STOPWORDS

logger = logging.getLogger(__name__)


_STEMMER = SnowballStemmer("english")
_WORD_PATTERN = re.compile(r"[a-z]+(?:-[a-z]+)*")

_OVERLAP_THRESHOLD = 0.55
_VARIANT_SIMILARITY_THRESHOLD = 85
_VARIANT_MIN_TERM_LENGTH = 6
_VARIANT_MAX_TERMS = 500
_VARIANT_MAX_ISSUES = 25
_REVIEWABLE_PAGE_TYPES = frozenset({"source", "concept"})
_NEGATING_PREFIXES = ("anti", "dis", "in", "non", "un")
_PROVIDER_UNVERIFIABLE_CODES = frozenset(
    {
        "truncated_content",
        "truncated_summary",
        "truncated-page",
        "truncated-summary",
    }
)

_REVIEW_SYSTEM_PROMPT = (
    "You are a knowledge-base quality reviewer. Analyze the wiki pages below "
    "and report concrete knowledge-base quality issues only: broken assumptions, "
    "duplicate or conflicting pages, missing prerequisites, terminology drift, "
    "stale content, and cross-page inconsistencies. Return only JSON matching "
    "the provided schema. Use an empty issues array when no issues are found. "
    "The page excerpts are intentionally partial for prompt budget; do not report "
    "truncation, missing sections, or incomplete files solely because an excerpt "
    "ends abruptly."
)

_REVIEW_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["error", "warning", "suggestion"],
                    },
                    "code": {"type": "string"},
                    "pages": {"type": "array", "items": {"type": "string"}},
                    "message": {"type": "string"},
                },
                "required": ["severity", "code", "pages", "message"],
            },
        }
    },
    "required": ["issues"],
}


class _ProviderReviewIssue(BaseModel):
    severity: Literal["error", "warning", "suggestion"] = "suggestion"
    code: str = Field(min_length=1)
    pages: list[str] = Field(default_factory=list)
    message: str = Field(min_length=1)


class _ProviderReviewReport(BaseModel):
    issues: list[_ProviderReviewIssue] = Field(default_factory=list)


class ReviewService:
    """Semantic review checks for the maintained wiki."""

    def __init__(
        self,
        paths: ProjectPaths,
        *,
        provider: Optional[TextProvider] = None,
    ) -> None:
        self.paths = paths
        self.provider = provider

    def review(self) -> ReviewReport:
        self._require_provider("kb review")
        issues: list[ReviewIssue] = []
        page_tokens = self._load_page_tokens()

        issues.extend(self._check_summary_overlap(page_tokens))
        issues.extend(self._check_terminology_variants(page_tokens))

        provider_issues, mode = self._provider_review()
        issues.extend(provider_issues)

        return ReviewReport(issues=issues, mode=mode)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_provider(self, feature_name: str) -> TextProvider:
        if self.provider is None:
            raise ProviderConfigurationError(
                f"{feature_name} requires a configured provider. Add a provider "
                "section to kb.config.yaml and set the matching API key environment variable."
            )
        if isinstance(self.provider, UnavailableProvider):
            self.provider.ensure_available()
        return self.provider

    def _load_page_tokens(self) -> dict[str, list[str]]:
        """Return ``{relative_path: [lowered tokens]}`` for every wiki page."""
        page_tokens: dict[str, list[str]] = {}
        for md_file in sorted(self.paths.wiki_dir.rglob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            relative = md_file.relative_to(self.paths.root).as_posix()
            if not _is_reviewable_page(relative, text):
                continue
            tokens = [
                word
                for word in _WORD_PATTERN.findall(text.lower())
                if word not in STOPWORDS and len(word) >= 3
            ]
            page_tokens[relative] = tokens
        return page_tokens

    def _check_summary_overlap(
        self, page_tokens: dict[str, list[str]]
    ) -> list[ReviewIssue]:
        """Flag pairs of source pages whose significant terms overlap heavily."""
        issues: list[ReviewIssue] = []
        source_pages = {
            path: tokens
            for path, tokens in page_tokens.items()
            if path.startswith("wiki/sources/")
        }
        for (path_a, tokens_a), (path_b, tokens_b) in combinations(
            source_pages.items(), 2
        ):
            set_a = set(tokens_a)
            set_b = set(tokens_b)
            union = set_a | set_b
            if not union:
                continue
            jaccard = len(set_a & set_b) / len(union)
            if jaccard >= _OVERLAP_THRESHOLD:
                issues.append(
                    ReviewIssue(
                        severity="suggestion",
                        code="overlapping-topics",
                        pages=[path_a, path_b],
                        message=(
                            f"Source pages share {jaccard:.0%} term overlap "
                            "and may benefit from a shared concept page."
                        ),
                    )
                )
        return issues

    def _check_terminology_variants(
        self, page_tokens: dict[str, list[str]]
    ) -> list[ReviewIssue]:
        """Flag terms that appear in variant forms across different pages."""
        issues: list[ReviewIssue] = []
        per_page_terms: dict[str, set[str]] = {}
        for path, tokens in page_tokens.items():
            per_page_terms[path] = set(tokens)

        term_pages: dict[str, set[str]] = {}
        for path, terms in per_page_terms.items():
            for term in terms:
                if len(term) < _VARIANT_MIN_TERM_LENGTH:
                    continue
                term_pages.setdefault(term, set()).add(path)

        all_terms = sorted(
            term_pages,
            key=lambda term: (-len(term_pages[term]), term),
        )[:_VARIANT_MAX_TERMS]

        checked: set[tuple[str, str]] = set()
        for i, term in enumerate(all_terms):
            for other in all_terms[i + 1 :]:
                pair = (term, other)
                if pair in checked:
                    continue
                if term == other:
                    continue
                if abs(len(term) - len(other)) > max(3, len(term) // 2):
                    continue
                if _variant_stem(term) == _variant_stem(other):
                    continue
                if _collapse_term(term) == _collapse_term(other):
                    continue
                if not _looks_like_terminology_variant(term, other):
                    continue
                score = fuzz.ratio(term, other)
                if score >= _VARIANT_SIMILARITY_THRESHOLD and score < 100:
                    pages_with_term = sorted(term_pages.get(term, set()))
                    pages_with_other = sorted(term_pages.get(other, set()))
                    if pages_with_term and pages_with_other:
                        affected = sorted(set(pages_with_term) | set(pages_with_other))
                        issues.append(
                            ReviewIssue(
                                severity="suggestion",
                                code="terminology-variant",
                                pages=affected,
                                message=(
                                    f'Term appears as both "{term}" and '
                                    f'"{other}" across pages '
                                    f"(similarity {score}%)."
                                ),
                            )
                        )
                        if len(issues) >= _VARIANT_MAX_ISSUES:
                            return issues
                    checked.add(pair)

        return issues

    def _provider_review(self) -> tuple[list[ReviewIssue], str]:
        """Run a model-backed review pass over source pages."""
        provider = self._require_provider("Provider-backed review")
        source_dir = self.paths.wiki_dir / "sources"
        if not source_dir.exists():
            return [], "no-sources"

        page_texts: list[str] = []
        for md_file in sorted(source_dir.rglob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            rel = md_file.relative_to(self.paths.root).as_posix()
            page_texts.append(f"### {rel}\n{_review_page_excerpt(text)}")

        if not page_texts:
            return [], "no-sources"

        prompt = "## Wiki Pages\n\n" + "\n\n".join(page_texts)
        try:
            response = provider.generate(
                ProviderRequest(
                    prompt=prompt,
                    system_prompt=_REVIEW_SYSTEM_PROMPT,
                    max_tokens=4096,
                    response_schema=_REVIEW_RESPONSE_SCHEMA,
                    response_schema_name="kb_review_report",
                    reasoning_effort="low",
                )
            )
            issues = _filter_provider_issues(self._parse_provider_issues(response.text))
            return issues, f"provider:{response.model_name}"
        except Exception as exc:
            raise ProviderExecutionError(f"Provider review failed: {exc}") from exc

    @staticmethod
    def _parse_provider_issues(raw: str) -> list[ReviewIssue]:
        """Parse provider JSON output."""
        stripped = raw.strip()
        if stripped == "NO_ISSUES":
            return []

        try:
            payload = parse_json_payload(stripped, label="Provider review response")
            if isinstance(payload, list):
                payload = {"issues": payload}
            report = _ProviderReviewReport.model_validate(payload)
            return [
                ReviewIssue(
                    severity=issue.severity,
                    code=issue.code.strip() or "provider-issue",
                    pages=[page.strip() for page in issue.pages if page.strip()],
                    message=issue.message.strip(),
                )
                for issue in report.issues
                if issue.message.strip()
            ]
        except (StructuredOutputError, TypeError, ValueError) as exc:
            raise ValueError(
                "Provider review response did not match the structured JSON schema."
            ) from exc


def _variant_stem(term: str) -> str:
    return "-".join(_STEMMER.stem(part) for part in term.split("-"))


def _is_reviewable_page(relative_path: str, text: str) -> bool:
    if not (
        relative_path.startswith("wiki/sources/")
        or relative_path.startswith("wiki/concepts/")
    ):
        return False
    page_type = markdown_parse_frontmatter(text).get("type")
    return not isinstance(page_type, str) or page_type in _REVIEWABLE_PAGE_TYPES


def _looks_like_terminology_variant(term: str, other: str) -> bool:
    collapsed_a = _collapse_term(term)
    collapsed_b = _collapse_term(other)
    if collapsed_a == collapsed_b:
        return True
    if _is_specificity_pair(term, other):
        return False
    if _has_negating_prefix_pair(collapsed_a, collapsed_b):
        return False
    if _has_same_suffix_different_prefix(collapsed_a, collapsed_b):
        return False
    if collapsed_a[:4] == collapsed_b[:4] or _first_part(term) == _first_part(other):
        return fuzz.ratio(term, other) >= 92
    return False


def _collapse_term(term: str) -> str:
    return term.replace("-", "")


def _term_parts(term: str) -> set[str]:
    return {part for part in term.split("-") if part}


def _first_part(term: str) -> str:
    return next((part for part in term.split("-") if part), term)


def _is_specificity_pair(term: str, other: str) -> bool:
    parts = _term_parts(term)
    other_parts = _term_parts(other)
    if not parts or not other_parts or parts == other_parts:
        return False
    return parts.issubset(other_parts) or other_parts.issubset(parts)


def _has_negating_prefix_pair(term: str, other: str) -> bool:
    return any(
        term == f"{prefix}{other}" or other == f"{prefix}{term}"
        for prefix in _NEGATING_PREFIXES
    )


def _has_same_suffix_different_prefix(term: str, other: str) -> bool:
    if len(term) != len(other) or len(term) < 6:
        return False
    return term[1:] == other[1:] and term[0] != other[0]


def _review_page_excerpt(text: str) -> str:
    excerpts: list[str] = []
    for section in markdown_sections(text, default_title="content"):
        title = section.title.strip()
        if title not in {"Summary", "Key Excerpt"}:
            continue
        body = " ".join(" ".join(p.split()) for p in section.paragraphs).strip()
        if body:
            excerpts.append(f"## {title}\n{body[:1400]}")
    if excerpts:
        return "\n\n".join(excerpts)
    return " ".join(text.split())[:1800]


def _filter_provider_issues(issues: list[ReviewIssue]) -> list[ReviewIssue]:
    filtered: list[ReviewIssue] = []
    for issue in issues:
        code = issue.code.strip().casefold()
        message = issue.message.casefold()
        if code in _PROVIDER_UNVERIFIABLE_CODES or "truncat" in message:
            continue
        filtered.append(issue)
    return filtered
