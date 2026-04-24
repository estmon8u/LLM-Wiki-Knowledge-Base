from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field, ValidationError
import yaml

from src.models.wiki_models import SearchResult
from src.providers import (
    ProviderConfigurationError,
    ProviderExecutionError,
    UnavailableProvider,
)
from src.providers.base import ProviderRequest, TextProvider
from src.services.config_service import schema_excerpt
from src.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
    utc_now_iso,
)
from src.services.search_service import SearchService

logger = logging.getLogger(__name__)

_QUERY_SYSTEM_PROMPT = (
    "You are a research assistant for a curated markdown knowledge base. "
    "Answer the user's question using ONLY the evidence provided below. "
    "Return only JSON matching the provided schema. Cite each claim with the "
    "exact citation_ref values from the evidence bundle. If the evidence is "
    "insufficient, set insufficient_evidence to true and explain the gap."
)

_QUERY_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer_markdown": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "citation_refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "citation_refs"],
            },
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ref": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["ref", "title"],
            },
        },
        "insufficient_evidence": {"type": "boolean"},
    },
    "required": [
        "answer_markdown",
        "claims",
        "citations",
        "insufficient_evidence",
    ],
}


class _ProviderQueryClaim(BaseModel):
    text: str = Field(min_length=1)
    citation_refs: list[str] = Field(default_factory=list)


class _ProviderQueryCitation(BaseModel):
    ref: str = Field(min_length=1)
    title: str = Field(default="")


class _ProviderQueryAnswer(BaseModel):
    answer_markdown: str = ""
    claims: list[_ProviderQueryClaim] = Field(default_factory=list)
    citations: list[_ProviderQueryCitation] = Field(default_factory=list)
    insufficient_evidence: bool = False


@dataclass
class QueryClaim:
    text: str
    citation_refs: list[str] = field(default_factory=list)


@dataclass
class QueryCitation:
    ref: str
    title: str = ""


@dataclass
class QueryAnswer:
    answer: str
    citations: list[SearchResult]
    saved_path: str | None = None
    mode: str = ""
    claims: list[QueryClaim] = field(default_factory=list)
    declared_citations: list[QueryCitation] = field(default_factory=list)
    insufficient_evidence: bool = False


class QueryService:
    def __init__(
        self,
        paths: ProjectPaths,
        search_service: SearchService,
        *,
        provider: Optional[TextProvider] = None,
        refresh_index: Optional["Callable[[], None]"] = None,
        schema_text: str = "",
    ) -> None:
        self.paths = paths
        self.search_service = search_service
        self.provider = provider
        self._refresh_index = refresh_index
        self.schema_text = schema_text

    def answer_question(self, question: str, *, limit: int = 3) -> QueryAnswer:
        provider = self._require_provider()
        matches = self.search_service.search(question, limit=limit)
        if not matches:
            return QueryAnswer(
                answer="No compiled wiki pages matched that question yet. Ingest more sources or re-run compile.",
                citations=[],
                mode="no-matches",
            )

        return self._provider_answer(question, matches, provider=provider)

    def _require_provider(self) -> TextProvider:
        if self.provider is None:
            raise ProviderConfigurationError(
                "kb ask requires a configured provider. Add a provider section "
                "to kb.config.yaml and set the matching API key environment variable."
            )
        if isinstance(self.provider, UnavailableProvider):
            self.provider.ensure_available()
        return self.provider

    def _provider_answer(
        self, question: str, matches: list[SearchResult], *, provider: TextProvider
    ) -> QueryAnswer:
        prompt = self._build_prompt(question, matches)
        system_prompt = _QUERY_SYSTEM_PROMPT
        if self.schema_text:
            excerpt = schema_excerpt(self.schema_text, ["Query Behavior"])
            if excerpt:
                system_prompt = f"{system_prompt}\n\n{excerpt}"
        try:
            response = provider.generate(
                ProviderRequest(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=1024,
                    response_schema=_QUERY_RESPONSE_SCHEMA,
                    response_schema_name="kb_query_answer",
                )
            )
            structured_answer = _parse_provider_query_answer(response.text)
            if structured_answer is not None:
                return _query_answer_from_structured_response(
                    structured_answer,
                    matches,
                    mode=f"provider:{response.model_name}",
                )
            return QueryAnswer(
                answer=response.text,
                citations=matches,
                mode=f"provider:{response.model_name}",
            )
        except Exception as exc:
            raise ProviderExecutionError(f"Provider query failed: {exc}") from exc

    def _build_prompt(self, question: str, matches: list[SearchResult]) -> str:
        evidence_block = "\n\n".join(
            self._format_prompt_match(match) for match in matches
        )
        return (
            f"## Evidence\n\n{evidence_block}\n\n"
            "## Output Rules\n\n"
            "Use only the evidence above. Keep claims concise. For every factual "
            "claim, include at least one exact citation_ref from the evidence. "
            "Do not invent citation refs. If the evidence is insufficient, set "
            "insufficient_evidence to true and explain what is missing.\n\n"
            f"## Question\n\n{question}"
        )

    def _format_prompt_match(self, match: SearchResult) -> str:
        lines = [f"### {match.title}", f"citation_ref: {match.citation_ref}"]
        if match.section and match.section != match.title:
            lines.append(f"Section: {match.section}")
        lines.append(match.snippet)
        return "\n".join(lines)

    def save_answer(
        self, question: str, answer: QueryAnswer, *, slug: str | None = None
    ) -> str:
        if slug:
            safe_slug = slugify(slug)
        else:
            safe_slug = slugify(question)
        if not safe_slug or safe_slug == "untitled":
            safe_slug = "analysis-" + slugify(answer.answer[:40])
        timestamp = utc_now_iso()
        summary = answer.answer.replace("\n", " ").strip()[:280].rstrip()
        if not summary:
            summary = "Analysis page for: " + question[:250]
        frontmatter = {
            "title": question,
            "summary": summary,
            "type": "analysis",
            "question": question,
            "saved_at": timestamp,
            "citations": [c.citation_ref for c in answer.citations],
            "insufficient_evidence": answer.insufficient_evidence,
            "claim_count": len(answer.claims),
            "citation_count": len(answer.citations),
        }
        if answer.claims:
            frontmatter["claims"] = [
                {
                    "text": claim.text,
                    "citation_refs": claim.citation_refs,
                }
                for claim in answer.claims
            ]
        if answer.declared_citations:
            frontmatter["provider_citations"] = [
                {
                    "ref": citation.ref,
                    "title": citation.title,
                }
                for citation in answer.declared_citations
            ]
        yaml_block = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        citation_lines = ""
        if answer.citations:
            citation_lines = "\n".join(
                self._format_saved_citation(c) for c in answer.citations
            )
        claim_lines = ""
        if answer.claims:
            claim_lines = "\n".join(self._format_saved_claim(c) for c in answer.claims)
        page_text = (
            f"---\n{yaml_block}\n---\n\n"
            f"# {question}\n\n"
            "## Answer\n\n"
            f"{answer.answer}\n\n"
            "## Claims\n\n"
            f"{claim_lines or 'No structured claims.'}\n\n"
            "## Citations\n\n"
            f"{citation_lines or 'No citations.'}\n"
        )
        dest = self.paths.wiki_analysis_dir / f"{safe_slug}.md"
        atomic_write_text(dest, page_text)
        self.search_service.refresh_file(dest)
        self._append_log(question, dest)
        if self._refresh_index is not None:
            self._refresh_index()
        return dest.relative_to(self.paths.root).as_posix()

    def _append_log(self, question: str, dest: "Path") -> None:
        """Append a saved-analysis entry to wiki/log.md."""
        timestamp = utc_now_iso()[:10]
        current = "# Activity Log\n"
        if self.paths.wiki_log_file.exists():
            current = self.paths.wiki_log_file.read_text(encoding="utf-8")
        if not current.endswith("\n"):
            current += "\n"
        rel = dest.relative_to(self.paths.root).as_posix()
        question_summary = _log_safe_text(question)
        current += f"\n## [{timestamp}] ask --save | {question_summary} -> {rel}\n"
        atomic_write_text(self.paths.wiki_log_file, current)

    def _format_saved_citation(self, citation: SearchResult) -> str:
        line = f"- [[{citation.title}]] (`{citation.citation_ref}`)"
        if citation.section and citation.section != citation.title:
            line += f" - Section: {citation.section}"
        return line

    def _format_saved_claim(self, claim: QueryClaim) -> str:
        refs = ", ".join(f"`{ref}`" for ref in claim.citation_refs)
        if not refs:
            refs = "No citation refs"
        return f"- {claim.text} ({refs})"


def _parse_provider_query_answer(raw: str) -> _ProviderQueryAnswer | None:
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
        return _ProviderQueryAnswer.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValidationError):
        return None


def _query_answer_from_structured_response(
    structured: _ProviderQueryAnswer,
    matches: list[SearchResult],
    *,
    mode: str,
) -> QueryAnswer:
    known_by_ref: dict[str, SearchResult] = {}
    for match in matches:
        known_by_ref[match.citation_ref] = match
        known_by_ref[match.path] = match

    claims: list[QueryClaim] = []
    used_refs: set[str] = set()
    for claim in structured.claims:
        refs: list[str] = []
        for raw_ref in claim.citation_refs:
            ref = raw_ref.strip()
            if ref not in known_by_ref or ref in refs:
                continue
            refs.append(ref)
            used_refs.add(ref)
        claims.append(QueryClaim(text=claim.text.strip(), citation_refs=refs))

    declared_citations: list[QueryCitation] = []
    for citation in structured.citations:
        ref = citation.ref.strip()
        if ref not in known_by_ref:
            continue
        used_refs.add(ref)
        declared_citations.append(
            QueryCitation(
                ref=known_by_ref[ref].citation_ref,
                title=citation.title.strip(),
            )
        )

    citations = [
        match
        for match in matches
        if match.citation_ref in used_refs or match.path in used_refs
    ]

    return QueryAnswer(
        answer=structured.answer_markdown.strip(),
        citations=citations,
        mode=mode,
        claims=claims,
        declared_citations=declared_citations,
        insufficient_evidence=structured.insufficient_evidence,
    )


def _log_safe_text(text: str, *, max_length: int = 160) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > max_length:
        collapsed = collapsed[: max_length - 3].rstrip() + "..."
    return json.dumps(collapsed)
