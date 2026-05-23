"""High-level LightRAG-style query service used by ``WikiGraphQueryService``.

This is the LightRAG counterpart to
:class:`graphwiki_kb.wikigraph.query_service.WikiGraphQueryEngine`. It
keeps the classic engine's public surface (``find`` returns a
:class:`WikiGraphFindResult`-shaped payload; ``ask`` returns a
:class:`WikiGraphAnswer`) so the rest of the CLI does not need to
branch on backend mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graphwiki_kb.providers.base import (
    ProviderRequest,
    ProviderResponse,
    TextProvider,
)
from graphwiki_kb.wikigraph.light_context_builder import (
    LightContextBuilder,
    LightContextBuilderConfig,
)
from graphwiki_kb.wikigraph.light_models import (
    LightGraphIndex,
    LightQueryMethod,
    LightRetrievedBundle,
    LightRetrievedContext,
)
from graphwiki_kb.wikigraph.models import (
    QueryMethod,
    WikiGraphAnswer,
    WikiGraphFindResult,
    WikiGraphRetrievedContext,
)


def _light_method_to_classic(method: LightQueryMethod) -> QueryMethod:
    """Translate a LightRAG method label into the classic :class:`QueryMethod`.

    The classic models do not have a ``hybrid`` literal, so it is mapped
    to ``drift-lite`` (the closest behavioral analog) for output-shape
    compatibility.
    """
    if method == "hybrid":
        return "drift-lite"
    if method == "auto":
        return "auto"
    return method


def _context_to_classic(ctx: LightRetrievedContext) -> WikiGraphRetrievedContext:
    """Render a :class:`LightRetrievedContext` in the classic context shape.

    LightRAG chunks render their citation_ref with a ``#chunk-N`` anchor,
    so we deliberately use the classic ``"chunk"`` node kind (not
    ``"text_unit"``) to keep the same anchor format end-to-end. This
    matters for the evaluation harness: the provider-backed answer
    citation that the LightAnswerService records uses the
    LightRetrievedContext.citation_ref, and the harness validates it
    against the converted WikiGraphRetrievedContext.citation_ref. Using
    ``"chunk"`` keeps both ends at ``path#chunk-N``.
    """
    if ctx.kind == "chunk":
        node_kind = "chunk"
    elif ctx.kind == "relation":
        node_kind = "claim"
    else:
        node_kind = "entity"
    return WikiGraphRetrievedContext(
        node_id=ctx.id,
        node_kind=node_kind,
        title=ctx.title,
        path=ctx.path,
        text=ctx.text,
        score=float(ctx.score),
        source_ids=list(ctx.source_ids),
        section="",
        chunk_index=ctx.chunk_index,
        trace=list(ctx.trace),
        metadata={"lightrag_kind": ctx.kind, **(ctx.metadata or {})},
    )


@dataclass
class LightGraphQueryEngine:
    """Provider-free retrieval-only wrapper around :class:`LightContextBuilder`."""

    index: LightGraphIndex
    config: LightContextBuilderConfig = field(default_factory=LightContextBuilderConfig)

    def __post_init__(self) -> None:
        self._builder = LightContextBuilder(index=self.index, config=self.config)

    @property
    def builder(self) -> LightContextBuilder:
        """Expose the underlying context builder."""
        return self._builder

    def find(
        self, question: str, *, method: LightQueryMethod = "auto"
    ) -> WikiGraphFindResult:
        """Run dual-level retrieval and return a classic-shaped find result."""
        bundle = self._builder.retrieve(question, method=method)
        classic_contexts = [_context_to_classic(c) for c in bundle.contexts]
        return WikiGraphFindResult(
            query=question,
            method=_light_method_to_classic(bundle.method),
            contexts=classic_contexts,
            entities=[e.canonical_name for e in bundle.entities],
            communities=[],
            trace=bundle.trace,
            diagnostics=[
                *bundle.diagnostics,
                f"lightrag_low_level_keywords={','.join(bundle.low_level_keywords)}",
                f"lightrag_high_level_keywords={','.join(bundle.high_level_keywords)}",
                f"lightrag_method={bundle.method}",
            ],
        )

    def retrieve_bundle(
        self, question: str, *, method: LightQueryMethod = "auto"
    ) -> LightRetrievedBundle:
        """Return the structured LightRAG bundle (used by answer service)."""
        return self._builder.retrieve(question, method=method)


def render_answer_prompt(question: str, bundle: LightRetrievedBundle) -> str:
    """Render the prompt body fed to a provider-backed answerer.

    The prompt cleanly separates entity profiles, relation profiles, and
    source excerpts so the model can tell scaffolding apart from
    grounding evidence. Citations must reference ``[C#]`` source
    excerpts; entity/relation cards are not standalone evidence.
    """
    lines: list[str] = [f"Question: {question}", ""]
    lines.append(
        "Use only the retrieved entities, relationships, and source "
        "excerpts. Every factual claim MUST cite at least one source "
        "excerpt ([C#]). Entity and relationship profiles are retrieval "
        "summaries, not standalone evidence unless backed by a source "
        "excerpt. If the evidence is insufficient, say so."
    )
    # Cap scaffolding bulk so the model has token budget left for
    # reasoning and answer generation. Entities and relations are
    # scaffolding — they help the model orient — but the real evidence
    # lives in the source excerpts.
    lines.extend(["", "# Retrieved entities"])
    for idx, entity in enumerate(bundle.entities[:8], start=1):
        lines.append(f"[E{idx}] {entity.canonical_name} — {entity.type}")
        if entity.description:
            lines.append(entity.description[:240])
        if entity.source_ids:
            lines.append("Sources: " + ", ".join(entity.source_ids[:4]))
        lines.append("")

    lines.append("# Retrieved relationships")
    for idx, relation in enumerate(bundle.relations[:8], start=1):
        lines.append(
            f"[R{idx}] {relation.source_entity_id} {relation.relation_type} "
            f"{relation.target_entity_id}"
        )
        if relation.description:
            lines.append(relation.description[:240])
        if relation.keywords:
            lines.append("Keywords: " + ", ".join(relation.keywords[:6]))
        if relation.source_ids:
            lines.append("Sources: " + ", ".join(relation.source_ids[:4]))
        lines.append("")

    lines.append("# Source excerpts")
    chunk_contexts = [ctx for ctx in bundle.contexts if ctx.kind == "chunk"]
    if not chunk_contexts:
        chunk_contexts = [
            LightRetrievedContext(
                kind="chunk",
                id=c.id,
                title=c.source_title or c.source_slug,
                score=0.0,
                text=c.text[:1000],
                path=c.compiled_page_path or c.normalized_path,
                chunk_index=c.chunk_index,
                source_ids=[c.source_id],
                trace=["bundle_chunks"],
            )
            for c in bundle.chunks
        ]
    for idx, ctx in enumerate(chunk_contexts, start=1):
        lines.append(f"[C{idx}] {ctx.citation_ref}")
        lines.append(ctx.text[:1000])
        lines.append("")
    return "\n".join(lines)


@dataclass
class LightAnswerService:
    """Build a :class:`WikiGraphAnswer` from a LightRAG retrieval bundle.

    Mirrors the existing :class:`WikiGraphAnswerService` contract:

    * Provider-free path returns a deterministic evidence summary that
      cites the chunk anchors verbatim. Citations are always grounded
      in returned ``[C#]`` source excerpts.
    * Provider-backed path forwards a structured prompt and validates
      that every citation maps to a returned chunk before accepting it.
    """

    engine: LightGraphQueryEngine
    provider: TextProvider | None = None

    def ask(
        self,
        question: str,
        *,
        method: LightQueryMethod = "auto",
        require_provider: bool = False,
    ) -> WikiGraphAnswer:
        """Return a :class:`WikiGraphAnswer` for ``question``."""
        bundle = self.engine.retrieve_bundle(question, method=method)
        contexts = [_context_to_classic(c) for c in bundle.contexts]
        classic_method = _light_method_to_classic(bundle.method)

        if self.provider is None:
            if require_provider:
                return WikiGraphAnswer(
                    method=classic_method,
                    question=question,
                    answer=(
                        "No provider configured for LightRAG answer synthesis. "
                        "Re-run with a configured provider (e.g. OPENAI_API_KEY)."
                    ),
                    contexts=contexts,
                    trace=bundle.trace,
                    warnings=["provider_required_but_missing"],
                    insufficient_evidence=True,
                    provider_status={"mode": "provider-required"},
                )
            return _provider_free_answer(question, bundle, contexts, classic_method)

        prompt = render_answer_prompt(question, bundle)
        try:
            # ``max_tokens=4096`` matches the classic backend. Lower
            # budgets get consumed entirely by the OpenAI Responses API
            # reasoning step on long structured prompts and leave no
            # output budget — observed empirically as empty answers in
            # the LightRAG end-to-end evaluation.
            response: ProviderResponse = self.provider.generate(
                ProviderRequest(
                    prompt=prompt,
                    system_prompt=(
                        "You answer questions about a research knowledge base "
                        "using only the retrieved entities, relationships, and "
                        "source excerpts. Cite source excerpts with [C#]."
                    ),
                    max_tokens=4096,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            return WikiGraphAnswer(
                method=classic_method,
                question=question,
                answer=f"Provider error: {exc}",
                contexts=contexts,
                trace=bundle.trace,
                warnings=[f"provider_error:{type(exc).__name__}"],
                insufficient_evidence=True,
                provider_status={"mode": "provider-error"},
            )

        warnings, citations, insufficient = _validate_citations(response.text, bundle)
        return WikiGraphAnswer(
            method=classic_method,
            question=question,
            answer=response.text,
            contexts=contexts,
            citations=citations,
            trace=bundle.trace,
            warnings=warnings,
            insufficient_evidence=insufficient,
            provider_status={
                "mode": "provider",
                "provider": response.provider or self.provider.name,
                "model": response.model_name,
            },
        )


def _provider_free_answer(
    question: str,
    bundle: LightRetrievedBundle,
    contexts: list[WikiGraphRetrievedContext],
    method: QueryMethod,
) -> WikiGraphAnswer:
    # LightRAG chunk contexts come through with node_kind=="chunk"
    # (see _context_to_classic). Fall back to any context when no
    # chunk-kind context was returned so provider-free runs still
    # carry citations rather than silently emitting nothing.
    chunk_contexts = [c for c in contexts if c.node_kind == "chunk"]
    if not chunk_contexts:
        chunk_contexts = contexts
    if not chunk_contexts:
        return WikiGraphAnswer(
            method=method,
            question=question,
            answer=(
                "No supporting evidence found in the LightRAG index for this "
                "question. Try `kb update` or rephrase the question."
            ),
            contexts=contexts,
            trace=bundle.trace,
            warnings=["no_evidence"],
            insufficient_evidence=True,
            provider_status={"mode": "provider-free"},
        )

    bullet_lines = ["## Evidence summary (provider-free)", ""]
    for idx, ctx in enumerate(chunk_contexts[:6], start=1):
        snippet = (ctx.text or "").strip().split("\n", 1)[0][:200]
        bullet_lines.append(f"- [C{idx}] `{ctx.citation_ref}` — {snippet}".rstrip())
    bullet_lines.append("")
    bullet_lines.append(
        "Configure a provider (e.g. set OPENAI_API_KEY) to enable "
        "provider-backed answer synthesis."
    )
    citations = [
        {
            "title": ctx.title,
            "ref": ctx.citation_ref,
            "kind": ctx.node_kind,
        }
        for ctx in chunk_contexts[:6]
    ]
    return WikiGraphAnswer(
        method=method,
        question=question,
        answer="\n".join(bullet_lines),
        contexts=contexts,
        citations=citations,
        trace=bundle.trace,
        warnings=[],
        insufficient_evidence=False,
        provider_status={"mode": "provider-free"},
    )


def _validate_citations(
    answer_text: str, bundle: LightRetrievedBundle
) -> tuple[list[str], list[dict[str, Any]], bool]:
    """Validate ``[C#]`` markers against the bundle's source excerpts.

    Returns ``(warnings, citations, insufficient_evidence)``.
    """
    import re

    chunk_contexts = [c for c in bundle.contexts if c.kind == "chunk"]
    valid_indices = list(range(1, len(chunk_contexts) + 1))
    referenced = {int(m) for m in re.findall(r"\[C(\d+)\]", answer_text)}
    warnings: list[str] = []
    invalid = referenced - set(valid_indices)
    if invalid:
        warnings.append(
            "invalid_citations:" + ",".join(str(i) for i in sorted(invalid))
        )
    citations: list[dict[str, Any]] = []
    for idx in sorted(referenced):
        if idx not in valid_indices:
            continue
        ctx = chunk_contexts[idx - 1]
        citations.append(
            {
                "title": ctx.title,
                "ref": ctx.citation_ref,
                "kind": ctx.kind,
            }
        )
    insufficient = not citations or "insufficient" in answer_text.lower()
    if not citations and chunk_contexts:
        warnings.append("no_valid_citations_in_answer")
    return warnings, citations, insufficient
