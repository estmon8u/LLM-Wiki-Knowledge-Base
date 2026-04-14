from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

from src.models.wiki_models import ReviewIssue, ReviewReport
from src.services.project_service import ProjectPaths


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


class ReviewService:
    """Semantic review checks for the maintained wiki.

    Currently uses deterministic heuristics.  When a provider is wired,
    the ``review`` method can delegate to a model-backed pass instead.
    """

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def review(self) -> ReviewReport:
        issues: list[ReviewIssue] = []
        page_tokens = self._load_page_tokens()

        issues.extend(self._check_summary_overlap(page_tokens))
        issues.extend(self._check_terminology_variants(page_tokens))

        return ReviewReport(issues=issues, mode="heuristic")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

        all_terms: set[str] = set()
        for terms in per_page_terms.values():
            all_terms.update(terms)

        checked: set[tuple[str, str]] = set()
        for term in sorted(all_terms):
            hyphenated = term.replace("-", "")
            for other in sorted(all_terms):
                if other == term:
                    continue
                pair = (min(term, other), max(term, other))
                if pair in checked:
                    continue
                other_flat = other.replace("-", "")
                if hyphenated == other_flat and hyphenated != term:
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
                                    f'"{other}" across pages.'
                                ),
                            )
                        )
                    checked.add(pair)

        return issues
