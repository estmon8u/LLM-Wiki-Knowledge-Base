from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import yaml

from src.models.wiki_models import SearchResult
from src.providers import (
    ProviderConfigurationError,
    ProviderExecutionError,
    UnavailableProvider,
)
from src.providers.base import ProviderRequest, TextProvider
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
    "Cite each claim by referencing the source title in square brackets, "
    "e.g. [Source Title]. If the evidence is insufficient, say so."
)


@dataclass
class QueryAnswer:
    answer: str
    citations: list[SearchResult]
    saved_path: str | None = None
    mode: str = ""


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
        try:
            response = provider.generate(
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
            raise ProviderExecutionError(f"Provider query failed: {exc}") from exc

    def _build_prompt(self, question: str, matches: list[SearchResult]) -> str:
        evidence_block = "\n\n".join(
            self._format_prompt_match(match) for match in matches
        )
        return (
            f"## Evidence\n\n{evidence_block}\n\n"
            "## Output Rules\n\n"
            "Use only the evidence above. Keep claims concise and cite factual sentences with [Source Title]. "
            "If the evidence is insufficient, say so explicitly.\n\n"
            f"## Question\n\n{question}"
        )

    def _format_prompt_match(self, match: SearchResult) -> str:
        lines = [f"### {match.title} ({match.citation_ref})"]
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
        }
        yaml_block = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        citation_lines = ""
        if answer.citations:
            citation_lines = "\n".join(
                self._format_saved_citation(c) for c in answer.citations
            )
        page_text = (
            f"---\n{yaml_block}\n---\n\n"
            f"# {question}\n\n"
            "## Answer\n\n"
            f"{answer.answer}\n\n"
            "## Citations\n\n"
            f"{citation_lines or 'No citations.'}\n"
        )
        dest = self.paths.wiki_analysis_dir / f"{safe_slug}.md"
        atomic_write_text(dest, page_text)
        self.search_service.refresh_file(dest)
        return dest.relative_to(self.paths.root).as_posix()

    def _format_saved_citation(self, citation: SearchResult) -> str:
        line = f"- [[{citation.title}]] (`{citation.citation_ref}`)"
        if citation.section and citation.section != citation.title:
            line += f" - Section: {citation.section}"
        return line
