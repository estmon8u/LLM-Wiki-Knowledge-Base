from __future__ import annotations

import json
import logging
import re
from itertools import combinations
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError
from rapidfuzz import fuzz

from src.models.wiki_models import ReviewIssue, ReviewReport
from src.providers import (
    ProviderConfigurationError,
    ProviderExecutionError,
    UnavailableProvider,
)
from src.providers.base import ProviderRequest, TextProvider
from src.services.project_service import ProjectPaths

logger = logging.getLogger(__name__)


_WORD_PATTERN = re.compile(r"[a-z]+(?:-[a-z]+)*")
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "not",
        "but",
        "can",
        "its",
        "also",
        "may",
        "more",
        "into",
        "each",
        "than",
        "which",
        "when",
        "how",
        "where",
        "what",
        "use",
        "used",
        "using",
        "such",
        "will",
        "been",
        "does",
        "should",
        "would",
        "could",
        "about",
        "other",
        "some",
        "them",
        "they",
        "their",
        "then",
        "only",
        "over",
        "most",
        "just",
    }
)

_OVERLAP_THRESHOLD = 0.55
_VARIANT_SIMILARITY_THRESHOLD = 85
_VARIANT_MIN_TERM_LENGTH = 6
_VARIANT_MAX_TERMS = 500
_VARIANT_MAX_ISSUES = 25

_REVIEW_SYSTEM_PROMPT = (
    "You are a knowledge-base quality reviewer. Analyze the wiki pages below "
    "and report concrete knowledge-base quality issues only: broken assumptions, "
    "duplicate or conflicting pages, missing prerequisites, terminology drift, "
    "stale content, and cross-page inconsistencies. Return only JSON matching "
    "the provided schema. Use an empty issues array when no issues are found."
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
            tokens = [
                word
                for word in _WORD_PATTERN.findall(text.lower())
                if word not in _STOPWORDS and len(word) >= 3
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
            page_texts.append(f"### {rel}\n{text[:2000]}")

        if not page_texts:
            return [], "no-sources"

        prompt = "## Wiki Pages\n\n" + "\n\n".join(page_texts)
        try:
            response = provider.generate(
                ProviderRequest(
                    prompt=prompt,
                    system_prompt=_REVIEW_SYSTEM_PROMPT,
                    max_tokens=1024,
                    response_schema=_REVIEW_RESPONSE_SCHEMA,
                    response_schema_name="kb_review_report",
                )
            )
            issues = self._parse_provider_issues(response.text)
            return issues, f"provider:{response.model_name}"
        except Exception as exc:
            raise ProviderExecutionError(f"Provider review failed: {exc}") from exc

    @staticmethod
    def _parse_provider_issues(raw: str) -> list[ReviewIssue]:
        """Parse provider JSON output with a legacy pipe-format fallback."""
        stripped = raw.strip()
        if not stripped or stripped == "NO_ISSUES":
            return []

        try:
            payload = json.loads(stripped)
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
        except (json.JSONDecodeError, TypeError, ValidationError):
            pass

        issues: list[ReviewIssue] = []
        for line in stripped.splitlines():
            line = line.strip()
            if line == "NO_ISSUES" or not line.startswith("ISSUE|"):
                continue
            parts = line.split("|", 4)
            if len(parts) < 5:
                continue
            _, severity, code, pages_str, message = parts
            severity = severity.strip()
            if severity not in ("error", "warning", "suggestion"):
                severity = "suggestion"
            pages = [p.strip() for p in pages_str.split(",") if p.strip()]
            issues.append(
                ReviewIssue(
                    severity=severity,
                    code=code.strip(),
                    pages=pages,
                    message=message.strip(),
                )
            )
        return issues
