from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

from src.models.wiki_models import ReviewIssue, ReviewReport
from src.providers import (
    ProviderConfigurationError,
    ProviderExecutionError,
    UnavailableProvider,
)
from src.providers.base import ProviderRequest, TextProvider
from src.schemas.claims import EvidenceBundle, EvidenceItem
from src.schemas.review import ReviewFinding, Verdict
from src.schemas.runs import RunRecord
from src.storage.run_store import RunStore
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
_PAIR_TERM_THRESHOLD = 4
_PAIR_HEADING_THRESHOLD = 1

_TITLE_PATTERN = re.compile(r"(?m)^#\s+(.+?)\s*$")
_HEADING_PATTERN = re.compile(r"(?m)^#{2,6}\s+(.+?)\s*$")
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

_ADVERSARIAL_PROMPT_VERSION = "review-adversarial-v1"

_EXTRACTOR_SYSTEM_PROMPT = (
    "You are a claim extractor for a markdown knowledge base. "
    "Given two wiki source pages, extract only the overlapping claims or terminology statements that are worth adversarial review. "
    "Output exactly one line per claim in the format:\n"
    "CLAIM|claim text|citations\n"
    "Where citations is a comma-separated list of page paths.\n"
    "If there are no candidate claims, output exactly: NO_CLAIMS\n"
    "Do not output anything else."
)

_SKEPTIC_SYSTEM_PROMPT = (
    "You are a skeptical reviewer for a markdown knowledge base. "
    "Challenge the extracted claims using only the two pages provided. "
    "Output exactly one line per concern in the format:\n"
    "CRITIQUE|issue_type|claim text|evidence_against|citations\n"
    "issue_type must be one of contradiction, term-drift, or needs-review.\n"
    "If there are no critiques, output exactly: NO_CRITIQUES\n"
    "Do not output anything else."
)

_ARBITER_SYSTEM_PROMPT = (
    "You are the final arbiter for adversarial knowledge-base review. "
    "Given two wiki pages, extracted claims, and skeptic critiques, emit final typed findings. "
    "Output exactly one line per finding in the format:\n"
    "FINDING|issue_type|verdict|confidence|claim|evidence_for|evidence_against|citations\n"
    "verdict must be one of consistent, contradictory, term_drift, needs_review.\n"
    "confidence must be a float between 0 and 1.\n"
    "citations must be a comma-separated list of page paths.\n"
    "If there are no findings, output exactly: NO_FINDINGS\n"
    "Do not output anything else."
)


@dataclass(frozen=True)
class PageSnapshot:
    path: str
    title: str
    text: str
    tokens: set[str]
    headings: set[str]
    years: set[str]


@dataclass(frozen=True)
class ReviewPair:
    left: PageSnapshot
    right: PageSnapshot
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ExtractedClaim:
    text: str
    citations: tuple[str, ...]


@dataclass(frozen=True)
class PairReviewResult:
    pair: ReviewPair
    findings: list[ReviewFinding]
    model_name: str = ""
    error: str | None = None


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
        run_store: Optional[RunStore] = None,
        workflow_backend: str = "python",
    ) -> None:
        self.paths = paths
        self.provider = provider
        self.run_store = run_store
        self.workflow_backend = workflow_backend

    def review(self, *, adversarial: bool = False) -> ReviewReport:
        self._require_provider("kb review")
        issues: list[ReviewIssue] = []
        page_tokens = self._load_page_tokens()

        issues.extend(self._check_summary_overlap(page_tokens))
        issues.extend(self._check_terminology_variants(page_tokens))

        if adversarial:
            if self.workflow_backend == "langgraph":
                from src.workflows.review_graph import run_review_graph

                findings, mode, run_id = run_review_graph(self)
            else:
                findings, mode, run_id = self._adversarial_review()
            issues.extend(self._findings_to_issues(findings))
            return ReviewReport(
                issues=issues,
                mode=mode,
                findings=findings,
                run_id=run_id,
            )

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

    def _adversarial_review(self) -> tuple[list[ReviewFinding], str, str | None]:
        provider = self._require_provider("kb review --adversarial")
        snapshots = self._load_source_page_snapshots()
        if not snapshots:
            return [], f"adversarial:{getattr(provider, 'name', '')}", None

        pairs = self._build_candidate_pairs(snapshots)
        evidence_bundle = self._build_review_evidence_bundle(pairs, snapshots)

        if not pairs:
            run_id = self._persist_review_run(
                evidence_bundle=evidence_bundle,
                findings=[],
                model_id=getattr(provider, "name", ""),
                wall_time_ms=0,
                unresolved_disagreement=False,
            )
            model_name = getattr(provider, "name", "")
            return [], f"adversarial:{model_name}", run_id

        try:
            started_at = time.perf_counter()
            results = asyncio.run(self._run_adversarial_pairs(pairs))
            wall_time_ms = int((time.perf_counter() - started_at) * 1000)
        except Exception as exc:
            raise ProviderExecutionError(f"Adversarial review failed: {exc}") from exc

        findings = [finding for result in results for finding in result.findings]
        errors = [result.error for result in results if result.error]
        if results and len(errors) == len(results):
            raise ProviderExecutionError(
                "Adversarial review failed: " + "; ".join(errors)
            )
        model_name = self._review_model_name(results)
        unresolved = any(
            finding.verdict == Verdict.NEEDS_REVIEW for finding in findings
        ) or bool(errors)
        run_id = self._persist_review_run(
            evidence_bundle=evidence_bundle,
            findings=findings,
            model_id=model_name,
            wall_time_ms=wall_time_ms,
            unresolved_disagreement=unresolved,
        )
        return findings, f"adversarial:{model_name}", run_id

    async def _run_adversarial_pairs(
        self, pairs: list[ReviewPair]
    ) -> list[PairReviewResult]:
        results: list[PairReviewResult | None] = [None] * len(pairs)

        async def review_pair(pair_index: int, pair: ReviewPair) -> None:
            try:
                extracted_claims, extractor_model = await asyncio.to_thread(
                    self._extract_pair_claims,
                    pair,
                )
                skeptic_text, skeptic_model = await asyncio.to_thread(
                    self._skeptic_review,
                    pair,
                    extracted_claims,
                )
                findings, arbiter_model = await asyncio.to_thread(
                    self._arbiter_review,
                    pair,
                    extracted_claims,
                    skeptic_text,
                )
                results[pair_index] = PairReviewResult(
                    pair=pair,
                    findings=findings,
                    model_name=arbiter_model or skeptic_model or extractor_model,
                )
            except Exception as exc:
                results[pair_index] = PairReviewResult(
                    pair=pair,
                    findings=[],
                    error=str(exc),
                )

        async with asyncio.TaskGroup() as task_group:
            for pair_index, pair in enumerate(pairs):
                task_group.create_task(review_pair(pair_index, pair))

        return [result for result in results if result is not None]

    def _load_source_page_snapshots(self) -> list[PageSnapshot]:
        source_dir = self.paths.wiki_dir / "sources"
        if not source_dir.exists():
            return []

        snapshots: list[PageSnapshot] = []
        for md_file in sorted(source_dir.rglob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            relative = md_file.relative_to(self.paths.root).as_posix()
            tokens = {
                word
                for word in _WORD_PATTERN.findall(text.lower())
                if word not in _STOPWORDS and len(word) >= 3
            }
            headings = {
                heading.strip().lower()
                for heading in _HEADING_PATTERN.findall(text)
                if heading.strip()
            }
            years = set(_YEAR_PATTERN.findall(text))
            snapshots.append(
                PageSnapshot(
                    path=relative,
                    title=self._extract_page_title(md_file.stem, text),
                    text=text,
                    tokens=tokens,
                    headings=headings,
                    years=years,
                )
            )
        return snapshots

    def _extract_page_title(self, fallback: str, text: str) -> str:
        match = _TITLE_PATTERN.search(text)
        if match is not None:
            return match.group(1).strip()
        return fallback.replace("-", " ").strip().title()

    def _build_candidate_pairs(self, snapshots: list[PageSnapshot]) -> list[ReviewPair]:
        pairs: list[ReviewPair] = []
        for left, right in combinations(snapshots, 2):
            shared_terms = sorted(left.tokens & right.tokens)
            shared_headings = sorted(left.headings & right.headings)
            shared_years = sorted(left.years & right.years)
            variants = self._shared_variant_terms(left.tokens, right.tokens)
            reasons: list[str] = []

            if len(shared_terms) >= _PAIR_TERM_THRESHOLD:
                reasons.append("shared terms: " + ", ".join(shared_terms[:4]))
            if len(shared_headings) >= _PAIR_HEADING_THRESHOLD:
                reasons.append("shared headings: " + ", ".join(shared_headings[:2]))
            if shared_years:
                reasons.append("shared years: " + ", ".join(shared_years[:2]))
            if variants:
                reasons.append("variant terms: " + ", ".join(variants[:2]))

            if reasons:
                pairs.append(ReviewPair(left=left, right=right, reasons=tuple(reasons)))

        return pairs

    def _shared_variant_terms(
        self, left_terms: set[str], right_terms: set[str]
    ) -> list[str]:
        variants: set[str] = set()
        for left in left_terms:
            left_flat = left.replace("-", "")
            for right in right_terms:
                if left == right:
                    continue
                if left_flat == right.replace("-", ""):
                    variants.add(left)
                    variants.add(right)
        return sorted(variants)

    def _extract_pair_claims(
        self, pair: ReviewPair
    ) -> tuple[list[ExtractedClaim], str]:
        prompt = (
            f"## Candidate Pair Reasons\n\n- "
            + "\n- ".join(pair.reasons)
            + "\n\n"
            + self._pair_pages_block(pair)
        )
        response = self.provider.generate(
            ProviderRequest(
                prompt=prompt,
                system_prompt=_EXTRACTOR_SYSTEM_PROMPT,
                max_tokens=1024,
            )
        )
        return self._parse_extracted_claims(response.text), response.model_name

    def _skeptic_review(
        self, pair: ReviewPair, extracted_claims: list[ExtractedClaim]
    ) -> tuple[str, str]:
        claims_block = self._claims_block(extracted_claims)
        prompt = (
            self._pair_pages_block(pair) + "\n\n## Extracted Claims\n\n" + claims_block
        )
        response = self.provider.generate(
            ProviderRequest(
                prompt=prompt,
                system_prompt=_SKEPTIC_SYSTEM_PROMPT,
                max_tokens=1024,
            )
        )
        return response.text, response.model_name

    def _arbiter_review(
        self,
        pair: ReviewPair,
        extracted_claims: list[ExtractedClaim],
        skeptic_text: str,
    ) -> tuple[list[ReviewFinding], str]:
        claims_block = self._claims_block(extracted_claims)
        prompt = (
            self._pair_pages_block(pair)
            + "\n\n## Extracted Claims\n\n"
            + claims_block
            + "\n\n## Skeptic Review\n\n"
            + skeptic_text
        )
        response = self.provider.generate(
            ProviderRequest(
                prompt=prompt,
                system_prompt=_ARBITER_SYSTEM_PROMPT,
                max_tokens=1024,
            )
        )
        return (
            self._parse_adversarial_findings(response.text, pair),
            response.model_name,
        )

    def _pair_pages_block(self, pair: ReviewPair) -> str:
        return (
            f"## Page A\n\n### {pair.left.path}\n# {pair.left.title}\n\n{pair.left.text[:2500]}"
            f"\n\n## Page B\n\n### {pair.right.path}\n# {pair.right.title}\n\n{pair.right.text[:2500]}"
        )

    def _claims_block(self, extracted_claims: list[ExtractedClaim]) -> str:
        if not extracted_claims:
            return "NO_CLAIMS"
        return "\n".join(
            f"- {claim.text} ({', '.join(claim.citations)})"
            for claim in extracted_claims
        )

    @staticmethod
    def _parse_extracted_claims(raw: str) -> list[ExtractedClaim]:
        claims: list[ExtractedClaim] = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if line == "NO_CLAIMS" or not line.startswith("CLAIM|"):
                continue
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            _, claim_text, citations_str = parts
            citations = tuple(
                citation.strip()
                for citation in citations_str.split(",")
                if citation.strip()
            )
            if claim_text.strip():
                claims.append(
                    ExtractedClaim(text=claim_text.strip(), citations=citations)
                )
        return claims

    def _parse_adversarial_findings(
        self, raw: str, pair: ReviewPair
    ) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        default_pages = [pair.left.path, pair.right.path]
        for line in raw.strip().splitlines():
            line = line.strip()
            if line == "NO_FINDINGS" or not line.startswith("FINDING|"):
                continue
            parts = line.split("|", 7)
            if len(parts) < 8:
                continue
            (
                _,
                issue_type,
                verdict_text,
                confidence_text,
                claim,
                evidence_for,
                evidence_against,
                citations_str,
            ) = parts
            try:
                verdict = Verdict(verdict_text.strip())
            except ValueError:
                verdict = Verdict.NEEDS_REVIEW
            try:
                confidence = float(confidence_text.strip())
            except ValueError:
                confidence = 0.5
            confidence = min(max(confidence, 0.0), 1.0)
            citations = [
                citation.strip()
                for citation in citations_str.split(",")
                if citation.strip()
            ] or default_pages
            findings.append(
                ReviewFinding(
                    issue_type=issue_type.strip() or verdict.value.replace("_", "-"),
                    affected_pages=default_pages,
                    claim=claim.strip(),
                    evidence_for=evidence_for.strip(),
                    evidence_against=evidence_against.strip(),
                    verdict=verdict,
                    confidence=confidence,
                    citations=citations,
                )
            )
        return findings

    def _findings_to_issues(self, findings: list[ReviewFinding]) -> list[ReviewIssue]:
        issues: list[ReviewIssue] = []
        for finding in findings:
            if finding.verdict == Verdict.CONSISTENT:
                continue
            severity = {
                Verdict.CONTRADICTORY: "error",
                Verdict.TERM_DRIFT: "suggestion",
                Verdict.NEEDS_REVIEW: "warning",
            }.get(finding.verdict, "warning")
            message = self._finding_message(finding)
            issues.append(
                ReviewIssue(
                    severity=severity,
                    code=finding.issue_type.replace("_", "-")
                    or finding.verdict.value.replace("_", "-"),
                    pages=finding.affected_pages,
                    message=message,
                )
            )
        return issues

    def _finding_message(self, finding: ReviewFinding) -> str:
        if finding.claim:
            return finding.claim
        if finding.evidence_against:
            return finding.evidence_against
        if finding.evidence_for:
            return finding.evidence_for
        return finding.verdict.value.replace("_", " ").title()

    def _build_review_evidence_bundle(
        self, pairs: list[ReviewPair], snapshots: list[PageSnapshot]
    ) -> EvidenceBundle:
        selected_paths = {
            page.path for pair in pairs for page in (pair.left, pair.right)
        }
        items: list[EvidenceItem] = []
        for snapshot in snapshots:
            if selected_paths and snapshot.path not in selected_paths:
                continue
            items.append(
                EvidenceItem(
                    page_path=snapshot.path,
                    title=snapshot.title,
                    snippet=re.sub(r"\s+", " ", snapshot.text).strip()[:300],
                    score=0,
                )
            )
        return EvidenceBundle(question="adversarial review", items=items)

    def _persist_review_run(
        self,
        *,
        evidence_bundle: EvidenceBundle,
        findings: list[ReviewFinding],
        model_id: str,
        wall_time_ms: int,
        unresolved_disagreement: bool,
    ) -> str | None:
        if self.run_store is None:
            return None
        record = RunRecord(
            command="review",
            model_id=model_id,
            prompt_version=_ADVERSARIAL_PROMPT_VERSION,
            evidence_bundle=evidence_bundle,
            context_hash=evidence_bundle.context_hash,
            review_findings=findings,
            final_text=self._review_summary(findings),
            token_cost=0,
            wall_time_ms=wall_time_ms,
            unresolved_disagreement=unresolved_disagreement,
        )
        return self.run_store.save_run(record)

    def _review_model_name(self, results: list[PairReviewResult]) -> str:
        for result in results:
            if result.model_name:
                return result.model_name
        return getattr(self.provider, "name", "")

    def _review_summary(self, findings: list[ReviewFinding]) -> str:
        if not findings:
            return "No adversarial review findings."
        counts = Counter(finding.verdict.value for finding in findings)
        ordered = ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
        return f"Adversarial review findings: {ordered}"

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
