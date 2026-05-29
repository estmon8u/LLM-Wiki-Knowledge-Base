"""Citation-grounded answer synthesis for the LightRAG backend.

Consumes a :class:`LightRetrievedBundle` (entities + relations + source
excerpts) and produces a :class:`WikiGraphAnswer`. Two modes:

* **Provider-free** — a deterministic evidence summary citing source chunks.
* **Provider-backed** — a structured-output prompt where every claim must cite
  one or more *source excerpts*. Entity/relation profiles are retrieval
  scaffolding, not standalone evidence: citations are validated to map back to a
  returned source chunk (or normalized to one), else the claim is dropped and,
  if nothing valid remains, the answer is marked insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graphwiki_kb.providers import ProviderConfigurationError, UnavailableProvider
from graphwiki_kb.providers.base import ProviderRequest, TextProvider
from graphwiki_kb.providers.structured import (
    StructuredOutputError,
    parse_model_payload,
)
from graphwiki_kb.services.citation_cleanup import clean_citation_refs
from graphwiki_kb.wikigraph.light_models import (
    LightAnswerPayload,
    LightQueryMethod,
    LightRetrievedBundle,
)
from graphwiki_kb.wikigraph.light_query_service import LightQueryEngine
from graphwiki_kb.wikigraph.models import QueryMethod, WikiGraphAnswer

_SYSTEM_PROMPT = (
    "You are a research assistant for a curated knowledge base served by the "
    "LightRAG-style WikiGraphRAG backend. Answer using ONLY the retrieved "
    "entities, relationships, and source excerpts. Every factual claim MUST "
    "cite one or more source excerpts by their citation ref. Entity and "
    "relationship profiles are retrieval summaries, not standalone evidence "
    "unless backed by a source excerpt. If the evidence is insufficient, set "
    "insufficient_evidence to true and omit unsupported claims."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
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
    "required": ["answer", "claims", "citations", "insufficient_evidence"],
}


@dataclass
class LightAnswerService:
    """Synthesis layer over :class:`LightQueryEngine`."""

    engine: LightQueryEngine
    provider: TextProvider | None = None

    def ask(
        self,
        question: str,
        *,
        method: LightQueryMethod = "auto",
        require_provider: bool = False,
    ) -> WikiGraphAnswer:
        """Answer ``question`` with citation-grounded synthesis."""
        bundle = self.engine.find(question, method=method)
        method_value: QueryMethod = bundle.method
        if not bundle.contexts:
            return WikiGraphAnswer(
                method=method_value,
                question=question,
                answer=(
                    "WikiGraphRAG (lightrag) did not match any source evidence "
                    "for this question. Build the index with `kb update "
                    "--wikigraph-mode lightrag` and ingest more sources."
                ),
                contexts=[],
                citations=[],
                trace=[*bundle.trace, {"step": "answer", "mode": "no-context"}],
                warnings=["no_context"],
                insufficient_evidence=True,
            )
        provider = self._maybe_provider()
        if provider is None:
            if require_provider:
                raise ProviderConfigurationError(
                    "WikiGraphRAG (lightrag) ask --require-provider is set but "
                    "no provider is configured."
                )
            return _provider_free_answer(question, bundle, self.engine.using_embeddings)
        return _provider_backed_answer(
            question, bundle, provider=provider, embeddings=self.engine.using_embeddings
        )

    def _maybe_provider(self) -> TextProvider | None:
        if self.provider is None:
            return None
        if isinstance(self.provider, UnavailableProvider):
            try:
                self.provider.ensure_available()
            except ProviderConfigurationError:
                return None
        return self.provider


def _fallback_warnings(embeddings: bool) -> list[str]:
    return [] if embeddings else ["bm25-fallback"]


def _provider_free_answer(
    question: str, bundle: LightRetrievedBundle, embeddings: bool
) -> WikiGraphAnswer:
    method_value: QueryMethod = bundle.method
    lines: list[str] = []
    citations: list[dict[str, Any]] = []
    for i, ctx in enumerate(bundle.contexts, start=1):
        lines.append(f"{i}. **{ctx.title}** ({ctx.citation_ref}): {ctx.text.strip()}")
        citations.append({"ref": ctx.citation_ref, "title": ctx.title})
    overview = (
        f"LightRAG retrieved {len(bundle.entities)} entity(ies), "
        f"{len(bundle.relations)} relationship(s), and {len(bundle.contexts)} "
        f"source excerpt(s) using the `{bundle.method}` method."
    )
    answer = (
        "_Provider-free LightRAG synthesis._\n\n"
        f"{overview}\n\n### Evidence summary\n\n" + "\n".join(lines)
    )
    return WikiGraphAnswer(
        method=method_value,
        question=question,
        answer=answer,
        contexts=bundle.contexts,
        citations=citations,
        trace=[*bundle.trace, {"step": "answer", "mode": "provider-free"}],
        warnings=["provider-free", *_fallback_warnings(embeddings)],
        insufficient_evidence=False,
        provider_status={"mode": "provider-free"},
    )


def _build_prompt(bundle: LightRetrievedBundle) -> str:
    entity_lines: list[str] = ["# Retrieved entities"]
    for i, entity in enumerate(bundle.entities, start=1):
        entity_lines.append(f"[E{i}] {entity.canonical_name} — {entity.type}")
        if entity.description:
            entity_lines.append(entity.description)
    relation_lines: list[str] = ["# Retrieved relationships"]
    name_by_id = {e.id: e.canonical_name for e in bundle.entities}
    for i, relation in enumerate(bundle.relations, start=1):
        src = name_by_id.get(relation.source_entity_id, relation.source_entity_id)
        tgt = name_by_id.get(relation.target_entity_id, relation.target_entity_id)
        relation_lines.append(
            f"[R{i}] {src} {relation.relation_type} {tgt} — {relation.relation_type}"
        )
        if relation.description:
            relation_lines.append(relation.description)
    excerpt_lines: list[str] = ["# Source excerpts"]
    for i, ctx in enumerate(bundle.contexts, start=1):
        excerpt_lines.append(f"[C{i}] {ctx.citation_ref}")
        excerpt_lines.append(ctx.text.strip())
    return (
        "\n".join(entity_lines)
        + "\n\n"
        + "\n".join(relation_lines)
        + "\n\n"
        + "\n".join(excerpt_lines)
        + "\n\n## Output Rules\n\nCite every claim with one or more source "
        "excerpt refs (the citation_ref after each [C#], or the [C#] label). "
        "Set insufficient_evidence=true when the excerpts cannot support a "
        f"grounded answer.\n\n## Question\n\n{bundle.question}"
    )


def _provider_backed_answer(
    question: str,
    bundle: LightRetrievedBundle,
    *,
    provider: TextProvider,
    embeddings: bool,
) -> WikiGraphAnswer:
    method_value: QueryMethod = bundle.method
    try:
        response = provider.generate(
            ProviderRequest(
                prompt=_build_prompt(bundle),
                system_prompt=_SYSTEM_PROMPT,
                max_tokens=4096,
                response_schema=_RESPONSE_SCHEMA,
                response_schema_name="lightrag_answer",
                reasoning_effort="low",
            )
        )
    except Exception as exc:
        fallback = _provider_free_answer(question, bundle, embeddings)
        return fallback.model_copy(
            update={
                "warnings": ["provider-error", *fallback.warnings],
                "provider_status": {"mode": "provider-error", "error": str(exc)},
            }
        )

    try:
        payload = parse_model_payload(
            response.text, LightAnswerPayload, label="LightRAG answer"
        )
    except StructuredOutputError:
        return _provider_free_answer(question, bundle, embeddings).model_copy(
            update={"warnings": ["provider-parse-error"]}
        )

    label_to_ref = {
        f"[c{i}]": ctx.citation_ref for i, ctx in enumerate(bundle.contexts, start=1)
    }
    label_to_ref.update(
        {f"c{i}": ctx.citation_ref for i, ctx in enumerate(bundle.contexts, start=1)}
    )
    known_refs = {ctx.citation_ref for ctx in bundle.contexts}
    path_to_ref: dict[str, str] = {}
    for ctx in bundle.contexts:
        path_only = ctx.citation_ref.split("#", 1)[0]
        path_to_ref.setdefault(path_only, ctx.citation_ref)

    def _normalize(ref: str) -> str | None:
        ref = ref.strip()
        if ref in known_refs:
            return ref
        lowered = ref.casefold()
        if lowered in label_to_ref:
            return label_to_ref[lowered]
        return path_to_ref.get(ref.split("#", 1)[0])

    valid_claims: list[dict[str, Any]] = []
    used_refs: set[str] = set()
    for claim in payload.claims:
        refs = [r for r in (_normalize(ref) for ref in claim.citation_refs) if r]
        if not refs:
            continue
        valid_claims.append({"text": claim.text.strip(), "citation_refs": refs})
        used_refs.update(refs)

    cited = [ctx for ctx in bundle.contexts if ctx.citation_ref in used_refs]
    insufficient = payload.insufficient_evidence or not valid_claims
    answer_text = clean_citation_refs(payload.answer.strip())
    return WikiGraphAnswer(
        method=method_value,
        question=question,
        answer=answer_text
        or _provider_free_answer(question, bundle, embeddings).answer,
        contexts=cited or bundle.contexts,
        citations=[
            {"ref": ctx.citation_ref, "title": ctx.title}
            for ctx in (cited or bundle.contexts)
        ],
        trace=[
            *bundle.trace,
            {
                "step": "answer",
                "mode": "provider",
                "valid_claims": len(valid_claims),
            },
        ],
        warnings=(["insufficient-evidence"] if insufficient else [])
        + _fallback_warnings(embeddings),
        insufficient_evidence=insufficient,
        provider_status={
            "mode": "provider",
            "model": getattr(response, "model_name", ""),
            "provider": getattr(response, "provider", ""),
        },
    )
