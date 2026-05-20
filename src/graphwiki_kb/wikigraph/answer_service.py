"""Answer synthesis for the WikiGraphRAG backend.

The answer service has two modes:

* **Provider-free** (default when no provider is available): produces a
  deterministic, evidence-only summary by concatenating the top retrieved
  contexts with clear citation markers. This keeps the WikiGraphRAG backend
  fully usable -- and the evaluator fully reproducible -- without any API
  keys.
* **Provider-backed**: when a :class:`graphwiki_kb.providers.base.TextProvider`
  is supplied, the service runs the same structured-output prompt used by
  the legacy ``kb ask`` path so that claims are grounded in WikiGraphRAG
  contexts and validated against retrieved citation refs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from graphwiki_kb.providers import ProviderConfigurationError, UnavailableProvider
from graphwiki_kb.providers.base import ProviderRequest, TextProvider
from graphwiki_kb.providers.structured import (
    StructuredOutputError,
    parse_model_payload,
)
from graphwiki_kb.services.citation_cleanup import clean_citation_refs
from graphwiki_kb.wikigraph.models import (
    QueryMethod,
    WikiGraphAnswer,
    WikiGraphRetrievedContext,
)
from graphwiki_kb.wikigraph.query_service import (
    WikiGraphFindResult,
    WikiGraphQueryEngine,
)

_WIKIGRAPH_SYSTEM_PROMPT = (
    "You are a research assistant for a curated markdown knowledge base served "
    "by the WikiGraphRAG backend. Answer the user's question using ONLY the "
    "evidence below. Return JSON matching the supplied schema. Cite every "
    "claim using the exact citation_ref values from the evidence bundle. Never "
    "return a claim with empty citation_refs; omit unsupported claims and set "
    "insufficient_evidence to true instead."
)

_WIKIGRAPH_RESPONSE_SCHEMA = {
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
    "required": ["answer_markdown", "claims", "citations", "insufficient_evidence"],
}


class _ProviderWikiGraphClaim(BaseModel):
    text: str
    citation_refs: list[str] = Field(default_factory=list)


class _ProviderWikiGraphCitation(BaseModel):
    ref: str
    title: str = ""


class _ProviderWikiGraphAnswer(BaseModel):
    answer_markdown: str = ""
    claims: list[_ProviderWikiGraphClaim] = Field(default_factory=list)
    citations: list[_ProviderWikiGraphCitation] = Field(default_factory=list)
    insufficient_evidence: bool = False


@dataclass
class WikiGraphAnswerService:
    """Thin synthesis layer over :class:`WikiGraphQueryEngine`."""

    engine: WikiGraphQueryEngine
    provider: TextProvider | None = None

    def ask(
        self,
        question: str,
        *,
        method: QueryMethod = "auto",
        require_provider: bool = False,
    ) -> WikiGraphAnswer:
        """Answer ``question`` using ``method`` and the configured provider."""
        find = self.engine.find(question, method=method)
        if not find.contexts:
            return WikiGraphAnswer(
                method=find.method,
                question=question,
                answer=(
                    "WikiGraphRAG did not match any wiki evidence for this "
                    "question. Build the index with `kb wikigraph build` and "
                    "ingest more sources, then try again."
                ),
                contexts=[],
                citations=[],
                trace=[*find.trace, {"step": "answer", "mode": "no-context"}],
                warnings=["no_context"],
                insufficient_evidence=True,
            )

        provider = self._maybe_provider()
        if provider is None:
            if require_provider:
                raise ProviderConfigurationError(
                    "WikiGraphRAG ask --require-provider is set but no provider "
                    "is configured."
                )
            return _provider_free_answer(question, find)

        return _provider_backed_answer(question, find, provider=provider)

    def _maybe_provider(self) -> TextProvider | None:
        if self.provider is None:
            return None
        if isinstance(self.provider, UnavailableProvider):
            try:
                self.provider.ensure_available()
            except ProviderConfigurationError:
                return None
        return self.provider


# --------------------------------------------------------------------------- #
# Provider-free synthesis                                                     #
# --------------------------------------------------------------------------- #


def _provider_free_answer(question: str, find: WikiGraphFindResult) -> WikiGraphAnswer:
    bullet_lines: list[str] = []
    citations: list[dict[str, Any]] = []
    for index, ctx in enumerate(find.contexts, start=1):
        bullet_lines.append(
            f"{index}. **{ctx.title}** ({ctx.citation_ref}): {ctx.text.strip()}"
        )
        citations.append({"ref": ctx.citation_ref, "title": ctx.title})
    overview = (
        f"WikiGraphRAG retrieved {len(find.contexts)} context(s) using the "
        f"`{find.method}` method"
    )
    if find.entities:
        overview += " from entities " + ", ".join(find.entities[:5])
    if find.communities:
        overview += " from communities " + ", ".join(find.communities[:3])
    overview += "."
    answer = (
        f"_Provider-free WikiGraphRAG synthesis._\n\n"
        f"{overview}\n\n"
        "### Evidence summary\n\n" + "\n".join(bullet_lines)
    )
    return WikiGraphAnswer(
        method=find.method,
        question=question,
        answer=answer,
        contexts=find.contexts,
        citations=citations,
        trace=[*find.trace, {"step": "answer", "mode": "provider-free"}],
        warnings=["provider-free"],
        insufficient_evidence=False,
        provider_status={"mode": "provider-free"},
    )


# --------------------------------------------------------------------------- #
# Provider-backed synthesis                                                   #
# --------------------------------------------------------------------------- #


def _provider_backed_answer(
    question: str,
    find: WikiGraphFindResult,
    *,
    provider: TextProvider,
) -> WikiGraphAnswer:
    prompt = _build_prompt(question, find.contexts)
    try:
        response = provider.generate(
            ProviderRequest(
                prompt=prompt,
                system_prompt=_WIKIGRAPH_SYSTEM_PROMPT,
                max_tokens=4096,
                response_schema=_WIKIGRAPH_RESPONSE_SCHEMA,
                response_schema_name="kb_wikigraph_answer",
                reasoning_effort="low",
            )
        )
    except Exception as exc:
        return WikiGraphAnswer(
            method=find.method,
            question=question,
            answer=(
                "Provider call failed; falling back to provider-free synthesis. "
                f"Reason: {exc}"
            ),
            contexts=find.contexts,
            citations=[
                {"ref": ctx.citation_ref, "title": ctx.title} for ctx in find.contexts
            ],
            trace=[*find.trace, {"step": "answer", "mode": "provider-error"}],
            warnings=["provider-error"],
            insufficient_evidence=True,
            provider_status={"mode": "provider-error", "error": str(exc)},
        )

    try:
        structured = parse_model_payload(
            response.text, _ProviderWikiGraphAnswer, label="WikiGraphRAG answer"
        )
    except StructuredOutputError:
        return _provider_free_answer(question, find).model_copy(
            update={
                "warnings": ["provider-parse-error"],
                "provider_status": {"mode": "provider-parse-error"},
            }
        )

    known_refs = {ctx.citation_ref for ctx in find.contexts}
    valid_claims: list[dict[str, Any]] = []
    used_refs: set[str] = set()
    for claim in structured.claims:
        refs = [ref for ref in claim.citation_refs if ref in known_refs]
        if not refs:
            continue
        valid_claims.append({"text": claim.text.strip(), "citation_refs": refs})
        used_refs.update(refs)

    declared = [
        {"ref": c.ref, "title": c.title}
        for c in structured.citations
        if c.ref in known_refs
    ]
    used_refs.update(c["ref"] for c in declared)
    cited_contexts = [ctx for ctx in find.contexts if ctx.citation_ref in used_refs]

    answer_text = clean_citation_refs(structured.answer_markdown.strip())
    insufficient = structured.insufficient_evidence or (not valid_claims)

    return WikiGraphAnswer(
        method=find.method,
        question=question,
        answer=answer_text or _provider_free_answer(question, find).answer,
        contexts=cited_contexts or find.contexts,
        citations=[
            {"ref": ctx.citation_ref, "title": ctx.title}
            for ctx in (cited_contexts or find.contexts)
        ],
        trace=[
            *find.trace,
            {
                "step": "answer",
                "mode": "provider",
                "valid_claims": len(valid_claims),
                "declared_citations": len(declared),
            },
        ],
        warnings=["insufficient-evidence"] if insufficient else [],
        insufficient_evidence=insufficient,
        provider_status={
            "mode": "provider",
            "model": getattr(response, "model_name", ""),
            "provider": getattr(response, "provider", ""),
        },
    )


def _build_prompt(question: str, contexts: list[WikiGraphRetrievedContext]) -> str:
    evidence_blocks: list[str] = []
    for ctx in contexts:
        header = f"### {ctx.title}\ncitation_ref: {ctx.citation_ref}"
        if ctx.section and ctx.section != ctx.title:
            header += f"\nSection: {ctx.section}"
        evidence_blocks.append(header + "\n" + ctx.text.strip())
    evidence = "\n\n".join(evidence_blocks)
    return (
        "## Evidence (WikiGraphRAG retrieved contexts)\n\n"
        f"{evidence}\n\n"
        "## Output Rules\n\n"
        "Use only the evidence above. Every claim must include at least one "
        "citation_ref drawn verbatim from the evidence bundle. Set "
        "insufficient_evidence=true when the evidence cannot support a "
        "grounded answer.\n\n"
        f"## Question\n\n{question}"
    )
