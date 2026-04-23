from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Optional

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

_REVIEW_SYSTEM_PROMPT = (
    "You are a knowledge-base quality reviewer. Analyze the wiki pages below "
    "and report issues. For each issue, output exactly one line in the format:\n"
    "ISSUE|severity|code|pages|message\n"
    "Where severity is one of: error, warning, suggestion.\n"
    "code is a short kebab-case label like 'contradiction', 'stale-claim', "
    "'terminology-drift', 'redundant-content'.\n"
    "pages is a comma-separated list of page paths.\n"
    "message is a concise explanation.\n"
    "If there are no issues, output exactly: NO_ISSUES\n"
    "Do not output anything else."
)


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

        all_terms = sorted(
            {term for terms in per_page_terms.values() for term in terms}
        )

        checked: set[tuple[str, str]] = set()
        for i, term in enumerate(all_terms):
            for other in all_terms[i + 1 :]:
                pair = (term, other)
                if pair in checked:
                    continue
                if term == other:
                    continue
                # Skip very short terms or identical base forms
                if len(term) < 4 or len(other) < 4:
                    continue
                score = fuzz.ratio(term, other)
                if score >= _VARIANT_SIMILARITY_THRESHOLD and score < 100:
                    pages_with_term = [
                        p for p, ts in per_page_terms.items() if term in ts
                    ]
                    pages_with_other = [
                        p for p, ts in per_page_terms.items() if other in ts
                    ]
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
                )
            )
            issues = self._parse_provider_issues(response.text)
            return issues, f"provider:{response.model_name}"
        except Exception as exc:
            raise ProviderExecutionError(f"Provider review failed: {exc}") from exc

    @staticmethod
    def _parse_provider_issues(raw: str) -> list[ReviewIssue]:
        """Parse structured ``ISSUE|…`` lines from provider output."""
        issues: list[ReviewIssue] = []
        for line in raw.strip().splitlines():
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
