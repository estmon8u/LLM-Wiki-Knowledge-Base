from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import yaml

from src.models.wiki_models import SearchResult
from src.providers.base import ProviderRequest, TextProvider
from src.services.project_service import ProjectPaths, slugify, utc_now_iso
from src.services.search_service import SearchService

logger = logging.getLogger(__name__)

_QUERY_SYSTEM_PROMPT = (
    "You are a research assistant for a curated markdown knowledge base. "
    "Answer the user's question using ONLY the evidence provided below. "
    "Cite each claim by referencing the source title in square brackets, "
    "e.g. [Source Title]. If the evidence is insufficient, say so."
)


@dataclass
class QueryAnswer:
    answer: str
    citations: list[SearchResult]
    saved_path: str | None = None
    mode: str = "heuristic"


class QueryService:
    def __init__(
        self,
        paths: ProjectPaths,
        search_service: SearchService,
        *,
        provider: Optional[TextProvider] = None,
    ) -> None:
        self.paths = paths
        self.search_service = search_service
        self.provider = provider

    def answer_question(self, question: str, *, limit: int = 3) -> QueryAnswer:
        matches = self.search_service.search(question, limit=limit)
        if not matches:
            return QueryAnswer(
                answer="No compiled wiki pages matched that question yet. Ingest more sources or re-run compile.",
                citations=[],
            )

        if self.provider is not None:
            return self._provider_answer(question, matches)

        evidence_lines = [f"{match.title}: {match.snippet}" for match in matches]
        answer = " ".join(evidence_lines)
        return QueryAnswer(answer=answer, citations=matches, mode="heuristic")

    def _provider_answer(
        self, question: str, matches: list[SearchResult]
    ) -> QueryAnswer:
        evidence_block = "\n\n".join(
            f"### {m.title} ({m.path})\n{m.snippet}" for m in matches
        )
        prompt = f"## Evidence\n\n{evidence_block}\n\n" f"## Question\n\n{question}"
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
