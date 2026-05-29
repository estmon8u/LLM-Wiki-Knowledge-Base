"""Uniform answering backends for the RAG evaluation harness.

Every backend implements ``run(question) -> RagSample`` and is given an
*identical* contract (same retrieval budget where applicable, same question)
so the comparison across the four methods is fair:

* ``direct``    — LLM-only, no retrieval (baseline).
* ``legacy``    — deprecated SQLite FTS retrieval + grounded answer.
* ``graphrag``  — Microsoft GraphRAG retrieval + ask controller.
* ``wikigraph`` — custom WikiGraphRAG (classic/lightrag, method-aware).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol, cast

from graphwiki_kb.models.command_models import CommandContext
from graphwiki_kb.providers import build_provider
from graphwiki_kb.providers.base import ProviderRequest, TextProvider
from graphwiki_kb.services import build_services
from graphwiki_kb.services.config_service import ConfigService
from graphwiki_kb.services.project_service import build_project_paths
from graphwiki_kb.wikigraph.models import QueryMethod
from scripts.rag_eval.dataset import EvalQuestion
from scripts.rag_eval.types import RagSample, RetrievedContext

_REFUSAL_HINTS = (
    "insufficient",
    "no information",
    "cannot answer",
    "not aware",
    "don't have",
    "do not have",
    "unable to answer",
    "no evidence",
)


def build_command_context(project_root: Path) -> CommandContext:
    """Construct a CommandContext for evaluator runs."""
    paths = build_project_paths(project_root)
    config_service = ConfigService(paths)
    config = config_service.load()
    schema_text = config_service.load_schema()
    services = build_services(paths, config)
    return CommandContext(
        project_root=paths.root,
        cwd=paths.root,
        config=config,
        schema_text=schema_text,
        services=services,
        verbose=False,
    )


class Backend(Protocol):
    """A uniform answering backend."""

    name: str
    method: str

    def retrieve(self, question: EvalQuestion) -> list[RetrievedContext]:
        """Return retrieved contexts only (no answer generation)."""

    def run(self, question: EvalQuestion) -> RagSample:
        """Run ``question`` and return a normalized :class:`RagSample`."""


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in _REFUSAL_HINTS)


class DirectBackend:
    """LLM-only baseline: answer from parametric knowledge, no retrieval."""

    name = "direct"
    method = "direct"

    def __init__(self, provider: TextProvider) -> None:
        self.provider = provider

    def retrieve(self, question: EvalQuestion) -> list[RetrievedContext]:
        return []

    def run(self, question: EvalQuestion) -> RagSample:
        start = time.perf_counter()
        try:
            response = self.provider.generate(
                ProviderRequest(
                    prompt=(
                        f"Answer the question concisely and factually. If you do "
                        f"not know, say the information is insufficient.\n\n"
                        f"Question: {question.question}"
                    ),
                    system_prompt="You are a knowledgeable research assistant.",
                    max_tokens=1024,
                    reasoning_effort="low",
                )
            )
            answer = response.text.strip()
            return RagSample(
                question_id=question.id,
                question=question.question,
                backend=self.name,
                method=self.method,
                retrieved_contexts=[],
                answer=answer,
                citations=[],
                insufficient_evidence=_looks_like_refusal(answer),
                latency_seconds=time.perf_counter() - start,
                provider_mode="provider",
            )
        except Exception as exc:
            return _error_sample(question, self.name, self.method, start, exc)


class LegacyBackend:
    """Deprecated SQLite FTS retrieval + grounded answer."""

    name = "legacy"
    method = "fts"

    def __init__(self, context: CommandContext, *, top_k: int = 8) -> None:
        self.context = context
        self.top_k = top_k

    def retrieve(self, question: EvalQuestion) -> list[RetrievedContext]:
        results = self.context.services.search.search(
            question.question,
            limit=self.top_k,
            include_analysis=False,
            page_types={"source"},
        )
        return [
            RetrievedContext(
                text=r.snippet or r.title, source_id=str(r.path), ref=r.citation_ref
            )
            for r in results
        ]

    def run(self, question: EvalQuestion) -> RagSample:
        start = time.perf_counter()
        try:
            contexts = self.retrieve(question)
            answer = self.context.services.query.answer_question(question.question)
            citations = [c.citation_ref for c in answer.citations]
            return RagSample(
                question_id=question.id,
                question=question.question,
                backend=self.name,
                method=self.method,
                retrieved_contexts=contexts,
                answer=answer.answer,
                citations=citations,
                insufficient_evidence=answer.insufficient_evidence,
                latency_seconds=time.perf_counter() - start,
                provider_mode="provider",
            )
        except Exception as exc:
            return _error_sample(question, self.name, self.method, start, exc)


class GraphRAGBackend:
    """Microsoft GraphRAG retrieval + ask controller."""

    name = "graphrag"

    def __init__(
        self, context: CommandContext, *, method: str = "auto", top_k: int = 8
    ) -> None:
        self.context = context
        self.method = method
        self.top_k = top_k

    def retrieve(self, question: EvalQuestion) -> list[RetrievedContext]:
        results = self.context.services.graphrag_find.search(
            question.question, limit=self.top_k
        )
        return [
            RetrievedContext(
                text=r.snippet or r.title, source_id=str(r.path), ref=r.citation_ref
            )
            for r in results
        ]

    def run(self, question: EvalQuestion) -> RagSample:
        start = time.perf_counter()
        try:
            contexts = self.retrieve(question)
            answer = self.context.services.graph_ask_controller.ask(
                question.question, method=self.method
            )
            insufficient = (getattr(answer, "claim_support", "") or "").lower() in {
                "no-answer",
                "insufficient-evidence",
                "stale-index",
            }
            return RagSample(
                question_id=question.id,
                question=question.question,
                backend=self.name,
                method=getattr(answer, "method", self.method) or self.method,
                retrieved_contexts=contexts,
                answer=getattr(answer, "answer", "") or "",
                citations=[],
                insufficient_evidence=insufficient,
                latency_seconds=time.perf_counter() - start,
                provider_mode="provider",
            )
        except Exception as exc:
            return _error_sample(question, self.name, self.method, start, exc)


class WikiGraphBackend:
    """Custom WikiGraphRAG backend (classic/lightrag, method-aware).

    When ``mode`` is given, a dedicated :class:`WikiGraphQueryService` is built
    with ``wikigraph.mode`` overridden, so ``wikigraph-classic`` and
    ``wikigraph-lightrag`` can be compared side by side in one process.
    """

    name = "wikigraph"

    def __init__(
        self,
        context: CommandContext,
        *,
        method: str = "auto",
        mode: str | None = None,
        name: str = "wikigraph",
    ) -> None:
        self.context = context
        self.method = method
        self.name = name
        if mode is None:
            self._query = context.services.wikigraph_query
        else:
            from copy import deepcopy

            from graphwiki_kb.services.wikigraph_query_service import (
                WikiGraphQueryService,
            )

            config = deepcopy(context.config)
            config.setdefault("wikigraph", {})["mode"] = mode
            index_service = context.services.wikigraph_index
            self._query = WikiGraphQueryService(
                paths=index_service.paths,
                index_service=index_service,
                provider=context.services.wikigraph_query.provider,
                config=config,
            )

    def retrieve(self, question: EvalQuestion) -> list[RetrievedContext]:
        find = self._query.find(
            question.question, method=cast(QueryMethod, self.method)
        )
        return [
            RetrievedContext(
                text=ctx.text,
                source_id=(ctx.source_ids[0] if ctx.source_ids else (ctx.path or "")),
                ref=ctx.citation_ref,
                score=ctx.score,
            )
            for ctx in find.contexts
        ]

    def run(self, question: EvalQuestion) -> RagSample:
        start = time.perf_counter()
        try:
            contexts = self.retrieve(question)
            answer = self._query.ask(
                question.question, method=cast(QueryMethod, self.method)
            )
            citations = [c.get("ref", "") for c in answer.citations]
            return RagSample(
                question_id=question.id,
                question=question.question,
                backend=self.name,
                method=answer.method,
                retrieved_contexts=contexts,
                answer=answer.answer,
                citations=[c for c in citations if c],
                insufficient_evidence=answer.insufficient_evidence,
                latency_seconds=time.perf_counter() - start,
                provider_mode=str(answer.provider_status.get("mode", "")),
            )
        except Exception as exc:
            return _error_sample(question, self.name, self.method, start, exc)


def _error_sample(
    question: EvalQuestion, backend: str, method: str, start: float, exc: Exception
) -> RagSample:
    return RagSample(
        question_id=question.id,
        question=question.question,
        backend=backend,
        method=method,
        retrieved_contexts=[],
        answer="",
        citations=[],
        insufficient_evidence=True,
        latency_seconds=time.perf_counter() - start,
        error=str(exc),
    )


def build_backends(
    context: CommandContext,
    method_names: list[str],
    *,
    wikigraph_method: str = "auto",
    graphrag_method: str = "auto",
) -> list[Backend]:
    """Construct the requested backends from a CommandContext."""
    backends: list[Backend] = []
    for name in method_names:
        if name == "direct":
            provider = build_provider(context.config)
            if provider is not None:
                backends.append(DirectBackend(provider))
        elif name == "legacy":
            backends.append(LegacyBackend(context))
        elif name == "graphrag":
            backends.append(GraphRAGBackend(context, method=graphrag_method))
        elif name == "wikigraph":
            backends.append(WikiGraphBackend(context, method=wikigraph_method))
        elif name == "wikigraph-classic":
            backends.append(
                WikiGraphBackend(
                    context,
                    method=wikigraph_method,
                    mode="classic",
                    name="wikigraph-classic",
                )
            )
        elif name == "wikigraph-lightrag":
            backends.append(
                WikiGraphBackend(
                    context,
                    method=wikigraph_method,
                    mode="lightrag",
                    name="wikigraph-lightrag",
                )
            )
        else:
            raise ValueError(f"Unknown backend: {name}")
    return backends
