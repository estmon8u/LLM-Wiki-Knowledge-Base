from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import yaml

from src.models.wiki_models import SearchResult
from src.providers.base import ProviderRequest, TextProvider
from src.schemas.claims import (
    CandidateAnswer,
    Claim,
    EvidenceBundle,
    EvidenceItem,
    MergedAnswer,
)
from src.schemas.runs import RunRecord
from src.storage.run_store import RunStore
from src.services.project_service import ProjectPaths, slugify, utc_now_iso
from src.services.search_service import SearchService

logger = logging.getLogger(__name__)

_QUERY_SYSTEM_PROMPT = (
    "You are a research assistant for a curated markdown knowledge base. "
    "Answer the user's question using ONLY the evidence provided below. "
    "Cite each claim by referencing the source title in square brackets, "
    "e.g. [Source Title]. If the evidence is insufficient, say so."
)

_QUERY_PROMPT_VERSION = "query-self-consistency-v1"
_MIN_CONSENSUS_SUPPORT = 2
_STRONG_EVIDENCE_CONFIDENCE = 0.85
_NO_CONSENSUS_ANSWER = (
    "The sampled answers did not converge on a sufficiently grounded response "
    "from the available evidence."
)
_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_LIST_PREFIX_PATTERN = re.compile(r"^(?:[-*]|\d+\.)\s+")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?!\[)")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class QueryAnswer:
    answer: str
    citations: list[SearchResult]
    saved_path: str | None = None
    mode: str = "heuristic"
    run_id: str | None = None


class QueryService:
    def __init__(
        self,
        paths: ProjectPaths,
        search_service: SearchService,
        *,
        provider: Optional[TextProvider] = None,
        run_store: Optional[RunStore] = None,
    ) -> None:
        self.paths = paths
        self.search_service = search_service
        self.provider = provider
        self.run_store = run_store

    def answer_question(
        self, question: str, *, limit: int = 3, self_consistency: int = 1
    ) -> QueryAnswer:
        matches = self.search_service.search(question, limit=limit)
        if not matches:
            return QueryAnswer(
                answer="No compiled wiki pages matched that question yet. Ingest more sources or re-run compile.",
                citations=[],
            )

        if self_consistency > 1:
            if self.provider is not None:
                return self._self_consistent_answer(
                    question, matches, sample_count=self_consistency
                )
            return self._heuristic_answer(matches, mode="heuristic:no-provider")

        if self.provider is not None:
            return self._provider_answer(question, matches)

        return self._heuristic_answer(matches)

    def _heuristic_answer(
        self, matches: list[SearchResult], *, mode: str = "heuristic"
    ) -> QueryAnswer:
        evidence_lines = [f"{match.title}: {match.snippet}" for match in matches]
        answer = " ".join(evidence_lines)
        return QueryAnswer(answer=answer, citations=matches, mode=mode)

    def _provider_answer(
        self, question: str, matches: list[SearchResult]
    ) -> QueryAnswer:
        prompt = self._build_prompt(question, matches)
        try:
            response = self.provider.generate(
                ProviderRequest(
                    prompt=prompt,
                    system_prompt=_QUERY_SYSTEM_PROMPT,
                    max_tokens=1024,
                )
            )
            return QueryAnswer(
                answer=response.text,
                citations=matches,
                mode=f"provider:{response.model_name}",
            )
        except Exception as exc:
            logger.warning(
                "Provider query failed (%s); falling back to heuristic.", exc
            )
            evidence_lines = [f"{m.title}: {m.snippet}" for m in matches]
            return QueryAnswer(
                answer=" ".join(evidence_lines),
                citations=matches,
                mode="heuristic-fallback",
            )

    def _self_consistent_answer(
        self, question: str, matches: list[SearchResult], *, sample_count: int
    ) -> QueryAnswer:
        evidence_bundle = self._build_evidence_bundle(question, matches)
        started_at = time.perf_counter()

        try:
            candidates = asyncio.run(
                self._sample_candidates(
                    question, matches, evidence_bundle, sample_count
                )
            )
        except Exception as exc:
            logger.warning(
                "Self-consistency query failed (%s); falling back to heuristic.", exc
            )
            return self._heuristic_answer(matches, mode="heuristic-fallback")

        successful_candidates = [
            candidate for candidate in candidates if not candidate.error
        ]
        if not successful_candidates:
            return self._heuristic_answer(matches, mode="heuristic-fallback")

        merged = self._merge_candidates(successful_candidates, evidence_bundle)
        wall_time_ms = int((time.perf_counter() - started_at) * 1000)
        model_id = next(
            (
                candidate.model_name
                for candidate in successful_candidates
                if candidate.model_name
            ),
            getattr(self.provider, "name", ""),
        )
        unresolved_disagreement = bool(
            merged.dropped_claims or any(candidate.error for candidate in candidates)
        )

        run_id: str | None = None
        if self.run_store is not None:
            record = RunRecord(
                command="query",
                model_id=model_id,
                prompt_version=_QUERY_PROMPT_VERSION,
                evidence_bundle=evidence_bundle,
                context_hash=evidence_bundle.context_hash,
                candidates=candidates,
                merged_answer=merged,
                final_text=merged.text,
                token_cost=0,
                wall_time_ms=wall_time_ms,
                unresolved_disagreement=unresolved_disagreement,
            )
            run_id = self.run_store.save_run(record)

        return QueryAnswer(
            answer=merged.text,
            citations=matches,
            mode=f"self-consistency:{model_id}:{sample_count}",
            run_id=run_id,
        )

    def _build_evidence_bundle(
        self, question: str, matches: list[SearchResult]
    ) -> EvidenceBundle:
        return EvidenceBundle(
            question=question,
            items=[
                EvidenceItem(
                    page_path=match.path,
                    title=match.title,
                    snippet=match.snippet,
                    score=match.score,
                )
                for match in matches
            ],
        )

    def _build_prompt(
        self,
        question: str,
        matches: list[SearchResult],
        *,
        sample_index: int | None = None,
    ) -> str:
        evidence_block = "\n\n".join(
            f"### {match.title} ({match.path})\n{match.snippet}" for match in matches
        )
        sample_instruction = ""
        if sample_index is not None:
            sample_instruction = (
                "\n\n## Sampling\n\n"
                f"This is independent sample {sample_index + 1}. Work independently from other possible samples."
            )
        return (
            f"## Evidence\n\n{evidence_block}"
            f"{sample_instruction}\n\n"
            "## Output Rules\n\n"
            "Use only the evidence above. Keep claims concise and cite factual sentences with [Source Title]. "
            "If the evidence is insufficient, say so explicitly.\n\n"
            f"## Question\n\n{question}"
        )

    async def _sample_candidates(
        self,
        question: str,
        matches: list[SearchResult],
        evidence_bundle: EvidenceBundle,
        sample_count: int,
    ) -> list[CandidateAnswer]:
        candidates: list[CandidateAnswer | None] = [None] * sample_count

        async def run_sample(sample_index: int) -> None:
            started_at = time.perf_counter()
            prompt = self._build_prompt(question, matches, sample_index=sample_index)
            try:
                response = await asyncio.to_thread(
                    self.provider.generate,
                    ProviderRequest(
                        prompt=prompt,
                        system_prompt=_QUERY_SYSTEM_PROMPT,
                        max_tokens=1024,
                    ),
                )
                claims = self._normalize_claims(response.text, evidence_bundle)
                candidates[sample_index] = CandidateAnswer(
                    raw_text=response.text,
                    claims=claims,
                    model_name=response.model_name,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                )
            except Exception as exc:
                candidates[sample_index] = CandidateAnswer(
                    raw_text="",
                    claims=[],
                    model_name=getattr(self.provider, "name", ""),
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    error=str(exc),
                )

        async with asyncio.TaskGroup() as task_group:
            for sample_index in range(sample_count):
                task_group.create_task(run_sample(sample_index))

        return [candidate for candidate in candidates if candidate is not None]

    def _normalize_claims(
        self, raw_text: str, evidence_bundle: EvidenceBundle
    ) -> list[Claim]:
        normalized_titles = {
            self._normalize_text(item.title): item for item in evidence_bundle.items
        }
        claims: list[Claim] = []

        for segment in self._split_claim_segments(raw_text):
            cited_titles = [
                title.strip()
                for title in _CITATION_PATTERN.findall(segment)
                if title.strip()
            ]
            claim_text = _CITATION_PATTERN.sub("", segment)
            claim_text = re.sub(r"\s+", " ", claim_text).strip(" -:;,.\n\t")
            if not claim_text:
                continue

            matched_item: EvidenceItem | None = None
            confidence = 0.2
            grounded = False
            for cited_title in cited_titles:
                matched_item = normalized_titles.get(self._normalize_text(cited_title))
                if matched_item is not None:
                    confidence = 1.0
                    grounded = True
                    break

            if matched_item is None:
                matched_item, support_ratio, overlap_count = self._best_evidence_match(
                    claim_text, evidence_bundle
                )
                if (
                    matched_item is not None
                    and overlap_count >= 2
                    and support_ratio >= 0.4
                ):
                    grounded = True
                    confidence = 0.6 if support_ratio < 0.6 else 0.85

            claims.append(
                Claim(
                    text=claim_text,
                    source_page=matched_item.page_path
                    if matched_item is not None
                    else "",
                    section=matched_item.title if matched_item is not None else "",
                    confidence=confidence,
                    grounded=grounded,
                )
            )

        return claims

    def _best_evidence_match(
        self, claim_text: str, evidence_bundle: EvidenceBundle
    ) -> tuple[EvidenceItem | None, float, int]:
        claim_tokens = set(self._tokenize_text(claim_text))
        if not claim_tokens:
            return None, 0.0, 0

        best_item: EvidenceItem | None = None
        best_ratio = 0.0
        best_overlap_count = 0
        for item in evidence_bundle.items:
            evidence_tokens = set(self._tokenize_text(f"{item.title} {item.snippet}"))
            overlap_count = len(claim_tokens & evidence_tokens)
            if overlap_count == 0:
                continue
            ratio = overlap_count / len(claim_tokens)
            if (ratio, overlap_count, item.score, item.page_path) > (
                best_ratio,
                best_overlap_count,
                best_item.score if best_item is not None else -1,
                best_item.page_path if best_item is not None else "",
            ):
                best_item = item
                best_ratio = ratio
                best_overlap_count = overlap_count

        return best_item, best_ratio, best_overlap_count

    def _merge_candidates(
        self, candidates: list[CandidateAnswer], evidence_bundle: EvidenceBundle
    ) -> MergedAnswer:
        clusters: list[dict[str, object]] = []

        for candidate_index, candidate in enumerate(candidates):
            for claim in candidate.claims:
                if not claim.text.strip():
                    continue
                for cluster in clusters:
                    representative = cluster["representative"]
                    if self._claims_match(claim, representative):
                        cluster["claims"].append(claim)
                        cluster["candidate_indexes"].add(candidate_index)
                        if self._claim_rank(claim) > self._claim_rank(representative):
                            cluster["representative"] = claim
                        break
                else:
                    clusters.append(
                        {
                            "representative": claim,
                            "claims": [claim],
                            "candidate_indexes": {candidate_index},
                        }
                    )

        accepted_claims: list[Claim] = []
        dropped_claims: list[Claim] = []
        for cluster in clusters:
            representative = cluster["representative"]
            cluster_claims = cluster["claims"]
            support_count = len(cluster["candidate_indexes"])
            strongest_confidence = max(claim.confidence for claim in cluster_claims)
            grounded = any(claim.grounded for claim in cluster_claims)
            if grounded and (
                support_count >= _MIN_CONSENSUS_SUPPORT
                or strongest_confidence >= _STRONG_EVIDENCE_CONFIDENCE
            ):
                accepted_claims.append(representative)
            else:
                dropped_claims.append(representative)

        title_by_path = {item.page_path: item.title for item in evidence_bundle.items}
        if accepted_claims:
            text = " ".join(
                self._render_claim(claim, title_by_path) for claim in accepted_claims
            )
        else:
            text = _NO_CONSENSUS_ANSWER

        return MergedAnswer(
            text=text,
            accepted_claims=accepted_claims,
            dropped_claims=dropped_claims,
            candidate_count=len(candidates),
        )

    def _render_claim(self, claim: Claim, title_by_path: dict[str, str]) -> str:
        sentence = claim.text.strip()
        if sentence and sentence[-1] not in ".!?":
            sentence = f"{sentence}."
        title = title_by_path.get(claim.source_page)
        if title:
            return f"{sentence} [{title}]"
        return sentence

    def _claims_match(self, left: Claim, right: Claim) -> bool:
        if self._normalize_text(left.text) == self._normalize_text(right.text):
            return True
        left_tokens = set(self._tokenize_text(left.text))
        right_tokens = set(self._tokenize_text(right.text))
        if not left_tokens or not right_tokens:
            return False
        overlap = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        similarity = overlap / union if union else 0.0
        containment = max(
            overlap / len(left_tokens),
            overlap / len(right_tokens),
        )
        return similarity >= 0.6 or containment >= 0.75

    def _claim_rank(self, claim: Claim) -> tuple[int, float, int, str, str]:
        return (
            int(claim.grounded),
            claim.confidence,
            -len(claim.text),
            claim.text,
            claim.source_page,
        )

    def _split_claim_segments(self, text: str) -> list[str]:
        segments: list[str] = []
        for line in text.splitlines():
            stripped = _LIST_PREFIX_PATTERN.sub("", line.strip())
            if not stripped:
                continue
            segments.extend(
                segment.strip()
                for segment in _SENTENCE_SPLIT_PATTERN.split(stripped)
                if segment.strip()
            )
        return segments

    def _tokenize_text(self, text: str) -> list[str]:
        return [
            self._stem_token(token) for token in _TOKEN_PATTERN.findall(text.lower())
        ]

    def _normalize_text(self, text: str) -> str:
        return " ".join(self._tokenize_text(text))

    def _stem_token(self, token: str) -> str:
        for suffix in ("ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[: -len(suffix)]
        return token

    def save_answer(self, question: str, answer: QueryAnswer) -> str:
        slug = slugify(question)
        if not slug or slug == "untitled":
            slug = "analysis-" + slugify(answer.answer[:40])
        timestamp = utc_now_iso()
        frontmatter = {
            "title": question,
            "type": "analysis",
            "question": question,
            "saved_at": timestamp,
            "citations": [c.path for c in answer.citations],
        }
        yaml_block = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        citation_lines = ""
        if answer.citations:
            citation_lines = "\n".join(
                f"- [[{c.title}]] (`{c.path}`)" for c in answer.citations
            )
        page_text = (
            f"---\n{yaml_block}\n---\n\n"
            f"# {question}\n\n"
            "## Answer\n\n"
            f"{answer.answer}\n\n"
            "## Citations\n\n"
            f"{citation_lines or 'No citations.'}\n"
        )
        dest = self.paths.wiki_concepts_dir / f"{slug}.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page_text, encoding="utf-8")
        return dest.relative_to(self.paths.root).as_posix()
