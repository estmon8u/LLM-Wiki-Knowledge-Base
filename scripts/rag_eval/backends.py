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

import json
import time
from pathlib import Path
from typing import Any, Protocol, cast

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
_GRAPH_KIND_TABLE = {
    "source": "text_units",
    "text_unit": "text_units",
    "document": "documents",
    "entity": "entities",
    "relationship": "relationships",
    "community": "communities",
    "community_report": "community_reports",
}
_GRAPH_CONTEXT_TEXT_CAP = 4000


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


def _graph_data_reference_contexts(
    status_service: Any,
    references: list[dict[str, object]],
) -> list[RetrievedContext]:
    """Translate GraphRAG ``[Data: ...]`` references into evaluator evidence."""
    if not references:
        return []
    source_meta = _graph_source_metadata(status_service)
    table_cache: dict[str, list[dict[str, Any]]] = {}
    text_unit_by_id: dict[str, dict[str, Any]] | None = None
    contexts: list[RetrievedContext] = []
    for item in references:
        kind = str(item.get("kind") or "").strip()
        table_name = _GRAPH_KIND_TABLE.get(kind)
        if table_name is None:
            continue
        rows = _graph_table_rows(status_service, table_name, table_cache)
        for raw_id in _reference_ids(item.get("ids")):
            row = _row_for_graph_id(rows, raw_id)
            if row is None:
                continue
            ref = f"graph://{kind}/{raw_id}"
            if kind in {"source", "text_unit"}:
                contexts.append(_text_unit_context(row, ref, source_meta))
            elif kind == "document":
                contexts.append(_document_context(row, ref, source_meta))
            elif kind in {"entity", "relationship"}:
                if text_unit_by_id is None:
                    text_unit_rows = _graph_table_rows(
                        status_service, "text_units", table_cache
                    )
                    text_unit_by_id = {
                        str(r.get("id") or ""): r for r in text_unit_rows
                    }
                contexts.append(
                    _linked_graph_context(row, ref, source_meta, text_unit_by_id)
                )
            else:
                contexts.append(_community_context(row, ref))
    return _dedupe_contexts(contexts)


def _reference_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _graph_table_rows(
    status_service: Any,
    table_name: str,
    table_cache: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if table_name in table_cache:
        return table_cache[table_name]
    rows: list[dict[str, Any]] = []
    table_path = status_service.table_path(table_name)
    if table_path is not None and Path(table_path).exists():
        import pyarrow.parquet as pq

        rows = pq.read_table(Path(table_path)).to_pylist()
    table_cache[table_name] = rows
    return rows


def _row_for_graph_id(rows: list[dict[str, Any]], raw_id: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("human_readable_id")) == raw_id or str(row.get("id")) == raw_id:
            return row
    return None


def _graph_source_metadata(status_service: Any) -> dict[str, dict[str, Any]]:
    input_path = getattr(status_service, "input_path", None)
    if input_path is None or not Path(input_path).exists():
        return {}
    try:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        raw_items = payload.get("sources") or payload.get("documents") or []
        items = raw_items if isinstance(raw_items, list) else []
    else:
        items = []
    metadata: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in (item.get("id"), item.get("source_id")):
            if key:
                metadata[str(key)] = item
    return metadata


def _text_unit_context(
    row: dict[str, Any],
    ref: str,
    source_meta: dict[str, dict[str, Any]],
) -> RetrievedContext:
    text = str(row.get("text") or "")
    header = _source_header_metadata(text)
    meta = source_meta.get(str(row.get("document_id") or "")) or {}
    source_id = _join_unique(
        [
            header.get("slug"),
            header.get("source_id"),
            header.get("raw_path"),
            header.get("normalized_path"),
            meta.get("slug"),
            meta.get("source_id"),
            meta.get("raw_path"),
            meta.get("normalized_path"),
            meta.get("title"),
            str(row.get("document_id") or ""),
        ]
    )
    return RetrievedContext(text=text, source_id=source_id, ref=ref)


def _document_context(
    row: dict[str, Any],
    ref: str,
    source_meta: dict[str, dict[str, Any]],
) -> RetrievedContext:
    meta = source_meta.get(str(row.get("id") or "")) or {}
    title = str(row.get("title") or meta.get("title") or "")
    text = _join_unique([title, str(row.get("text") or "")])
    source_id = _join_unique(
        [
            meta.get("slug"),
            meta.get("source_id"),
            meta.get("raw_path"),
            meta.get("normalized_path"),
            title,
            str(row.get("id") or ""),
        ]
    )
    return RetrievedContext(text=text, source_id=source_id, ref=ref)


def _linked_graph_context(
    row: dict[str, Any],
    ref: str,
    source_meta: dict[str, dict[str, Any]],
    text_unit_by_id: dict[str, dict[str, Any]],
) -> RetrievedContext:
    text_parts = [
        str(row.get("title") or ""),
        str(row.get("source") or ""),
        str(row.get("target") or ""),
        str(row.get("description") or ""),
    ]
    source_ids: list[str] = []
    for text_unit_id in _reference_ids(row.get("text_unit_ids")):
        text_unit = text_unit_by_id.get(text_unit_id)
        if text_unit is None:
            continue
        context = _text_unit_context(text_unit, ref, source_meta)
        source_ids.append(context.source_id)
        text_parts.append(context.text[:_GRAPH_CONTEXT_TEXT_CAP])
    return RetrievedContext(
        text=_join_unique(text_parts),
        source_id=_join_unique(source_ids),
        ref=ref,
    )


def _community_context(row: dict[str, Any], ref: str) -> RetrievedContext:
    title = str(row.get("title") or "")
    text = _join_unique(
        [
            title,
            str(row.get("summary") or ""),
            str(row.get("full_content") or ""),
        ]
    )
    source_id = _join_unique(
        [title, str(row.get("community") or ""), str(row.get("id") or "")]
    )
    return RetrievedContext(text=text, source_id=source_id, ref=ref)


def _source_header_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in text.splitlines()[:12]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip()
        if normalized_key in {"source_id", "slug", "raw_path", "normalized_path"}:
            metadata[normalized_key] = value.strip().rstrip(".")
    return metadata


def _join_unique(values: list[object]) -> str:
    parts: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in parts:
            parts.append(text)
    return " ".join(parts)


def _dedupe_contexts(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
    seen: set[str] = set()
    deduped: list[RetrievedContext] = []
    for context in contexts:
        key = context.ref or f"{context.source_id}\n{context.text[:80]}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(context)
    return deduped


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
            graph_contexts = _graph_data_reference_contexts(
                self.context.services.graphrag_status,
                getattr(answer, "graph_data_references", []) or [],
            )
            citations = [ctx.ref for ctx in graph_contexts if ctx.ref]
            return RagSample(
                question_id=question.id,
                question=question.question,
                backend=self.name,
                method=getattr(answer, "method", self.method) or self.method,
                retrieved_contexts=_dedupe_contexts([*graph_contexts, *contexts]),
                answer=getattr(answer, "answer", "") or "",
                citations=citations,
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
