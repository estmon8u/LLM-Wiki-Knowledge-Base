from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field
import yaml

from src.models.wiki_models import SearchResult
from src.providers import (
    ProviderConfigurationError,
    ProviderExecutionError,
    UnavailableProvider,
)
from src.providers.base import ProviderRequest, TextProvider
from src.providers.structured import StructuredOutputError, parse_model_payload
from src.services.citation_cleanup import clean_citation_refs
from src.services.config_service import schema_excerpt
from src.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
    unique_markdown_heading,
    utc_now_iso,
)
from src.services.search_service import SearchService

logger = logging.getLogger(__name__)

_QUERY_SYSTEM_PROMPT = (
    "You are a research assistant for a curated markdown knowledge base. "
    "Answer the user's question using ONLY the evidence provided below. "
    "Return only JSON matching the provided schema. Cite each claim with the "
    "exact citation_ref values from the evidence bundle. Never return a claim "
    "object with empty citation_refs; omit unsupported claims instead. If the "
    "evidence is insufficient, set insufficient_evidence to true and explain "
    "the gap."
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
    provider_status: dict[str, object] = field(default_factory=dict)


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
        matches = self.search_service.search(
            question, limit=limit, include_analysis=False
        )
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
                    max_tokens=4096,
                    response_schema=_QUERY_RESPONSE_SCHEMA,
                    response_schema_name="kb_query_answer",
                    reasoning_effort="low",
                )
            )
            try:
                structured_answer = _parse_provider_query_answer(response.text)
                _validate_provider_query_answer(structured_answer, matches)
            except ValueError as exc:
                raise ProviderExecutionError(
                    _format_provider_response_error(response, str(exc))
                ) from exc
            return _query_answer_from_structured_response(
                structured_answer,
                matches,
                mode=f"provider:{response.model_name}",
                provider_status=_provider_status_from_response(
                    response,
                    provider=provider,
                ),
            )
        except ProviderExecutionError:
            raise
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
            "Do not invent citation refs. Do not include claim objects that have "
            "no citation_refs; describe unsupported gaps in answer_markdown and "
            "set insufficient_evidence to true instead. If the evidence is "
            "insufficient, explain what is missing.\n\n"
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
        _validate_saved_answer(answer)
        if slug:
            safe_slug = slugify(slug)
        else:
            safe_slug = slugify(question)
        if not safe_slug or safe_slug == "untitled":
            safe_slug = "analysis-" + slugify(answer.answer[:40])
        timestamp = utc_now_iso()
        summary = answer.answer.replace("\n", " ").strip()[:280].rstrip()
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
        if answer.provider_status:
            frontmatter["provider_status"] = answer.provider_status
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
        timestamp = utc_now_iso()
        current = "# Activity Log\n"
        if self.paths.wiki_log_file.exists():
            current = self.paths.wiki_log_file.read_text(encoding="utf-8")
        if not current.endswith("\n"):
            current += "\n"
        rel = dest.relative_to(self.paths.root).as_posix()
        question_summary = _log_safe_text(question)
        heading = unique_markdown_heading(
            current,
            f"## [{timestamp}] ask --save | {question_summary} -> {rel}",
        )
        current += f"\n{heading}\n"
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


def _parse_provider_query_answer(raw: str) -> _ProviderQueryAnswer:
    try:
        return parse_model_payload(
            raw,
            _ProviderQueryAnswer,
            label="Provider query response",
        )
    except StructuredOutputError as exc:
        raise ValueError(str(exc)) from exc


def _validate_provider_query_answer(
    structured: _ProviderQueryAnswer,
    matches: list[SearchResult],
) -> None:
    answer_text = structured.answer_markdown.strip()
    if not answer_text:
        raise ValueError("Provider returned empty answer_markdown.")

    known_refs = _known_evidence_refs(matches)
    claim_refs: list[str] = []
    for claim in structured.claims:
        refs = [ref.strip() for ref in claim.citation_refs if ref.strip()]
        if not refs:
            raise ValueError("Provider returned a claim without citation_refs.")
        unknown = [ref for ref in refs if ref not in known_refs]
        if unknown:
            raise ValueError(
                "Provider returned citation_refs outside retrieved evidence: "
                + ", ".join(sorted(set(unknown)))
            )
        claim_refs.extend(refs)

    declared_refs = [citation.ref.strip() for citation in structured.citations]
    unknown_declared = [ref for ref in declared_refs if ref and ref not in known_refs]
    if unknown_declared:
        raise ValueError(
            "Provider returned citations outside retrieved evidence: "
            + ", ".join(sorted(set(unknown_declared)))
        )

    if not structured.insufficient_evidence:
        if not structured.claims:
            raise ValueError(
                "Provider returned insufficient_evidence=false but no claims."
            )
        if not claim_refs:
            raise ValueError(
                "Provider returned insufficient_evidence=false but no grounded citation_refs."
            )


def _known_evidence_refs(matches: list[SearchResult]) -> set[str]:
    refs: set[str] = set()
    for match in matches:
        refs.add(match.citation_ref)
        refs.add(match.path)
    return refs


def _validate_saved_answer(answer: QueryAnswer) -> None:
    if answer.answer.strip():
        return
    raise ValueError("Refusing to save an empty analysis answer.")


def _format_provider_response_error(response: object, message: str) -> str:
    details = [message]
    finish_reason = getattr(response, "finish_reason", None)
    if finish_reason:
        details.append(f"finish_reason={finish_reason}")
    output_tokens = getattr(response, "output_tokens", None)
    if output_tokens is not None:
        details.append(f"output_tokens={output_tokens}")
    return "Provider query failed: " + "; ".join(details)


def _provider_status_from_response(
    response: object,
    *,
    provider: TextProvider,
) -> dict[str, object]:
    status: dict[str, object] = {
        "parsed": True,
        "semantically_valid": True,
    }
    provider_name = getattr(response, "provider", "") or getattr(provider, "name", "")
    if provider_name:
        status["provider"] = provider_name
    model_name = getattr(response, "model_name", "")
    if model_name:
        status["model"] = model_name
    finish_reason = getattr(response, "finish_reason", None)
    if finish_reason:
        status["finish_reason"] = finish_reason
    input_tokens = getattr(response, "input_tokens", None)
    if input_tokens is not None:
        status["input_tokens"] = input_tokens
    output_tokens = getattr(response, "output_tokens", None)
    if output_tokens is not None:
        status["output_tokens"] = output_tokens
    return status


def _query_answer_from_structured_response(
    structured: _ProviderQueryAnswer,
    matches: list[SearchResult],
    *,
    mode: str,
    provider_status: dict[str, object],
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
        answer=clean_citation_refs(structured.answer_markdown.strip()),
        citations=citations,
        mode=mode,
        claims=claims,
        declared_citations=declared_citations,
        insufficient_evidence=structured.insufficient_evidence,
        provider_status=provider_status,
    )


def _log_safe_text(text: str, *, max_length: int = 160) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > max_length:
        collapsed = collapsed[: max_length - 3].rstrip() + "..."
    return json.dumps(collapsed)
