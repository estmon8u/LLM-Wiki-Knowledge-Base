"""WikiGraphRAG answer synthesis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphwiki_kb.providers.base import ProviderRequest, TextProvider
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
    utc_now_iso,
)
from graphwiki_kb.wikigraph.models import WikiGraphAnswer, WikiGraphRetrievedContext
from graphwiki_kb.wikigraph.query_service import WikiGraphMethod, WikiGraphQueryService

_ANSWER_SYSTEM = (
    "You answer questions using only the provided WikiGraphRAG evidence. "
    "Cite sources by title. If evidence is insufficient, say so clearly."
)


class WikiGraphAnswerService:
    """Synthesize WikiGraphRAG answers with optional provider support."""

    def __init__(
        self,
        paths: ProjectPaths,
        query_service: WikiGraphQueryService,
        *,
        provider: TextProvider | None = None,
    ) -> None:
        self.paths = paths
        self.query_service = query_service
        self.provider = provider

    def answer(
        self,
        question: str,
        *,
        method: WikiGraphMethod = "auto",
        save: bool = False,
    ) -> WikiGraphAnswer:
        """Answer a question using retrieved WikiGraphRAG contexts."""
        contexts, trace, warnings = self.query_service.retrieve(question, method=method)
        resolved_method = str(
            next(
                (item["value"] for item in trace if item.get("step") == "method"),
                method,
            )
        )
        citations = _build_citations(contexts)
        if self.provider is not None:
            answer_text = self._provider_answer(question, contexts)
        else:
            answer_text = _extractive_answer(question, contexts)
        result = WikiGraphAnswer(
            method=resolved_method,  # type: ignore[arg-type]
            question=question,
            answer=answer_text,
            contexts=contexts,
            citations=citations,
            trace=trace,
            warnings=warnings,
        )
        self._persist_run(result)
        if save:
            self._save_analysis_page(result)
        return result

    def _provider_answer(
        self,
        question: str,
        contexts: list[WikiGraphRetrievedContext],
    ) -> str:
        assert self.provider is not None
        evidence = "\n\n".join(
            f"[{index + 1}] {context.title}\n{context.text[:1200]}"
            for index, context in enumerate(contexts[:8])
        )
        request = ProviderRequest(
            system_prompt=_ANSWER_SYSTEM,
            prompt=(
                f"Question: {question}\n\nEvidence:\n{evidence}\n\n"
                "Write a concise markdown answer with inline [Title] citations."
            ),
        )
        response = self.provider.generate(request)
        return response.text.strip()

    def _persist_run(self, answer: WikiGraphAnswer) -> None:
        runs_dir = self.paths.graph_dir / "runs" / "wikigraph"
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = utc_now_iso().replace(":", "").replace("+", "")
        path = runs_dir / f"query-{stamp}.json"
        atomic_write_text(path, json.dumps(answer.model_dump(), indent=2))

    def _save_analysis_page(self, answer: WikiGraphAnswer) -> Path:
        slug = slugify(answer.question[:60]) or "wikigraph-answer"
        filename = f"wikigraph-answer-{slug}.md"
        path = self.paths.wiki_analysis_dir / filename
        lines = [
            "---",
            "type: analysis",
            f'title: "WikiGraphRAG: {answer.question}"',
            f'question: "{answer.question}"',
            "retrieval_backend: wikigraph",
            f'method: "{answer.method}"',
            f'generated_at: "{utc_now_iso()}"',
            "---",
            "",
            f"# {answer.question}",
            "",
            answer.answer,
            "",
            "## Retrieved Contexts",
            "",
        ]
        for context in answer.contexts:
            lines.append(
                f"- **{context.title}** ({context.node_kind}, score={context.score:.2f})"
            )
            if context.path:
                lines.append(f"  - path: `{context.path}`")
        atomic_write_text(path, "\n".join(lines))
        return path


def _build_citations(contexts: list[WikiGraphRetrievedContext]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for context in contexts:
        key = context.title
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "title": context.title,
                "path": context.path,
                "node_id": context.node_id,
                "source_ids": context.source_ids,
            }
        )
    return citations


def _extractive_answer(
    question: str,
    contexts: list[WikiGraphRetrievedContext],
) -> str:
    if not contexts:
        return (
            "Insufficient evidence: the WikiGraphRAG index did not retrieve "
            "relevant wiki contexts for this question."
        )
    paragraphs: list[str] = []
    for context in contexts[:6]:
        snippet = context.text.strip().split("\n")[0][:400]
        cite = f"[{context.title}]"
        paragraphs.append(f"- {snippet} {cite}")
    joined = "\n".join(paragraphs)
    return (
        f"Based on the indexed wiki artifacts, here is what the knowledge base "
        f"supports for **{question.strip()}**:\n\n{joined}"
    )
