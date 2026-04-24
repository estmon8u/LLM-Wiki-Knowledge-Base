from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Callable, Optional

import nltk
import nltk.data
from pydantic import BaseModel, Field
import yaml

from src.models.source_models import RawSourceRecord
from src.providers import (
    ProviderConfigurationError,
    UnavailableProvider,
)
from src.providers.base import ProviderRequest, TextProvider
from src.providers.structured import StructuredOutputError, parse_model_payload
from src.services.config_service import schema_excerpt
from src.services.markdown_document import (
    headings as markdown_headings,
    inline_text as markdown_inline_text,
    is_content_paragraph as markdown_is_content_paragraph,
    is_link_only_inline as markdown_is_link_only_inline,
    normalize_newlines as markdown_normalize_newlines,
    paragraphs as markdown_paragraphs,
    parse_frontmatter as markdown_parse_frontmatter,
    plain_text as markdown_plain_text,
    section_paragraphs as markdown_section_paragraphs,
    strip_frontmatter as markdown_strip_frontmatter,
)
from src.services.manifest_service import ManifestService
from src.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    unique_markdown_heading,
    utc_now_iso,
)
from src.storage.compile_run_store import CompileRunStore

logger = logging.getLogger(__name__)


_SUMMARY_SYSTEM_PROMPT = (
    "You are a research assistant summarizing documents for a curated knowledge base. "
    "Return only JSON matching the provided schema. Write a concise 2-4 sentence "
    "summary of the document below. "
    "Focus on the core thesis, methods, findings, and open questions. "
    "Do not include author names, affiliations, or publication metadata. "
    "Write in plain text without markdown formatting."
)

_SUMMARY_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "title_suggestion": {"type": "string"},
    },
    "required": ["summary", "key_points", "open_questions", "title_suggestion"],
}

_SUMMARY_CONTENT_LIMIT = 12000
SOURCE_PAGE_CONTRACT_VERSION_KEY = "source_page_contract_version"
SOURCE_PAGE_CONTRACT_VERSION = 2
_PLACEHOLDER_SUMMARIES = {
    "no summary available yet.",
    "summary unavailable.",
    "summary not available.",
}
_SUMMARY_PROMPT_ECHO_PATTERN = re.compile(
    r"(?:^|\s)(source_id|raw_path|content_hash|source_hash|compiled_at|summary)\s*:",
    re.IGNORECASE,
)


class _ProviderSummary(BaseModel):
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    title_suggestion: str = ""


@dataclass
class _SummaryResult:
    summary: str
    key_points: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    title_suggestion: str = ""


@dataclass
class CompileResult:
    compiled_count: int
    skipped_count: int
    compiled_paths: list[str]
    resumed_from_run_id: str | None = None


@dataclass
class CompilePlan:
    pending_sources: list[RawSourceRecord]
    skipped_count: int

    @property
    def pending_count(self) -> int:
        return len(self.pending_sources)


class CompileService:
    def __init__(
        self,
        paths: ProjectPaths,
        config: dict[str, Any],
        manifest_service: ManifestService,
        *,
        provider: Optional[TextProvider] = None,
        compile_run_store: Optional[CompileRunStore] = None,
        schema_text: str = "",
    ) -> None:
        self.paths = paths
        self.config = config
        self.manifest_service = manifest_service
        self.provider = provider
        self.compile_run_store = compile_run_store or CompileRunStore(
            self.paths.graph_exports_dir / "compile_runs.json"
        )
        self.schema_text = schema_text

    def _require_provider(self) -> TextProvider:
        if self.provider is None:
            raise ProviderConfigurationError(
                "kb update requires a configured provider. Add a provider section "
                "to kb.config.yaml and set the matching API key environment variable."
            )
        if isinstance(self.provider, UnavailableProvider):
            self.provider.ensure_available()
        return self.provider

    def plan(self, *, force: bool = False, resume: bool = False) -> CompilePlan:
        if force and resume:
            raise ValueError("--resume cannot be combined with --force.")
        if resume and self.compile_run_store.resume_candidate() is None:
            raise ValueError("No failed compile run is available to resume.")
        pending_sources: list[RawSourceRecord] = []
        skipped_count = 0
        sources = _sorted_sources(self.manifest_service.list_sources())
        for source in sources:
            article_path = self.paths.wiki_sources_dir / f"{source.slug}.md"
            if (
                not force
                and source.compiled_from_hash == source.content_hash
                and article_path.exists()
            ):
                skipped_count += 1
                continue
            pending_sources.append(source)
        return CompilePlan(
            pending_sources=pending_sources,
            skipped_count=skipped_count,
        )

    def compile(
        self,
        *,
        force: bool = False,
        resume: bool = False,
        progress_callback: Optional[Callable[[RawSourceRecord], None]] = None,
    ) -> CompileResult:
        compiled_paths: list[str] = []
        compiled_count = 0
        resume_record = self.compile_run_store.resume_candidate() if resume else None
        plan = self.plan(force=force, resume=resume)
        run_record = self.compile_run_store.start_run(
            plan.pending_sources,
            force=force,
            resumed_from_run_id=resume_record.run_id
            if resume_record is not None
            else "",
        )
        current_source: RawSourceRecord | None = None
        try:
            for source in plan.pending_sources:
                current_source = source
                article_path = self.paths.wiki_sources_dir / f"{source.slug}.md"

                canonical_path = self.paths.root / (
                    source.normalized_path or source.raw_path
                )
                if not canonical_path.exists():
                    raise FileNotFoundError(
                        "Normalized or raw source file does not exist for "
                        f"{source.source_id}: {canonical_path}"
                    )
                contents = canonical_path.read_text(encoding="utf-8")
                compiled_at = utc_now_iso()
                article_text = self._render_source_page(source, contents, compiled_at)
                atomic_write_text(article_path, article_text)

                source.compiled_at = compiled_at
                source.compiled_from_hash = source.content_hash
                source.metadata = dict(source.metadata or {})
                source.metadata[
                    SOURCE_PAGE_CONTRACT_VERSION_KEY
                ] = SOURCE_PAGE_CONTRACT_VERSION
                self.manifest_service.save_source(source)
                self.compile_run_store.mark_source_compiled(run_record.run_id, source)

                compiled_count += 1
                compiled_paths.append(
                    article_path.relative_to(self.paths.root).as_posix()
                )
                if progress_callback is not None:
                    progress_callback(source)

            current_source = None
            self._write_index(_sorted_sources(self.manifest_service.list_sources()))
            self._append_log(
                compiled_count,
                plan.skipped_count,
                force,
                resumed=resume_record is not None,
            )
            self.compile_run_store.mark_completed(run_record.run_id)
            return CompileResult(
                compiled_count=compiled_count,
                skipped_count=plan.skipped_count,
                compiled_paths=compiled_paths,
                resumed_from_run_id=(
                    resume_record.run_id if resume_record is not None else None
                ),
            )
        except Exception as exc:
            self.compile_run_store.mark_failed(
                run_record.run_id,
                error=str(exc),
                failed_source=current_source,
            )
            raise

    def refresh_index(self) -> None:
        """Rewrite the wiki index from the current source and concept pages."""
        self._write_index(_sorted_sources(self.manifest_service.list_sources()))

    def _render_source_page(
        self, source: RawSourceRecord, contents: str, compiled_at: str
    ) -> str:
        summary_result = self._extract_summary_result(contents)
        summary = summary_result.summary
        excerpt = self._extract_excerpt(contents)
        frontmatter = {
            "title": source.title,
            "summary": summary,
            "type": "source",
            "source_id": source.source_id,
            "source_hash": source.content_hash,
            "raw_path": source.raw_path,
            "origin": source.origin,
            "compiled_at": compiled_at,
            "ingested_at": source.ingested_at,
            "tags": [],
        }
        if summary_result.key_points:
            frontmatter["key_points"] = summary_result.key_points
        if summary_result.open_questions:
            frontmatter["open_questions"] = summary_result.open_questions
        if summary_result.title_suggestion:
            frontmatter["title_suggestion"] = summary_result.title_suggestion
        if source.normalized_path is not None:
            frontmatter["normalized_path"] = source.normalized_path
        yaml_frontmatter = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        canonical_file_line = ""
        if source.normalized_path is not None:
            canonical_file_line = f"- Canonical file: `{source.normalized_path}`\n"
        structured_sections = _render_summary_detail_sections(summary_result)
        return (
            f"---\n{yaml_frontmatter}\n---\n\n"
            f"# {source.title}\n\n"
            "## Summary\n\n"
            f"{summary}\n\n"
            f"{structured_sections}"
            "## Source Details\n\n"
            f"- Source ID: `{source.source_id}`\n"
            f"- Raw file: `{source.raw_path}`\n"
            f"{canonical_file_line}"
            f"- Origin: `{source.origin}`\n"
            f"- Ingested at: `{source.ingested_at}`\n\n"
            "## Key Excerpt\n\n"
            f"{excerpt}\n"
        )

    def _extract_summary(self, contents: str) -> str:
        return self._extract_summary_result(contents).summary

    def _extract_summary_result(self, contents: str) -> _SummaryResult:
        provider = self._require_provider()
        truncated = _plain_text_fallback(contents)[:_SUMMARY_CONTENT_LIMIT]
        if not truncated.strip():
            return _SummaryResult("No content available for summarization.")
        system_prompt = _SUMMARY_SYSTEM_PROMPT
        if self.schema_text:
            excerpt = schema_excerpt(self.schema_text, ["Source Pages"])
            if excerpt:
                system_prompt = f"{system_prompt}\n\n{excerpt}"
        try:
            response = provider.generate(
                ProviderRequest(
                    prompt=truncated,
                    system_prompt=system_prompt,
                    max_tokens=512,
                    response_schema=_SUMMARY_RESPONSE_SCHEMA,
                    response_schema_name="kb_compile_summary",
                    reasoning_effort="low",
                )
            )
            structured = _parse_provider_summary(response.text)
            if structured is not None:
                if not _is_weak_summary(structured.summary):
                    return structured
                return _SummaryResult(_deterministic_summary(contents))
            summary = response.text.strip()
            if _is_weak_summary(summary):
                return _SummaryResult(_deterministic_summary(contents))
            return _SummaryResult(summary)
        except Exception as exc:
            logger.warning(
                "Provider summary generation failed; using deterministic fallback: %s",
                exc,
            )
            return _SummaryResult(_deterministic_summary(contents))

    def _extract_excerpt(self, contents: str) -> str:
        abstract_paragraphs = _abstract_paragraphs(contents)
        if abstract_paragraphs:
            clean = "\n".join(abstract_paragraphs).strip()
        else:
            clean = "\n".join(_markdown_paragraphs(contents)).strip()
        if not clean:
            clean = _plain_text_fallback(contents)
        character_limit = _safe_int(
            self._compile_config().get("excerpt_character_limit"),
            default=900,
            minimum=1,
        )
        excerpt = _truncate_with_boundary(
            clean,
            character_limit,
            add_ellipsis=True,
        )
        return excerpt or "No excerpt available yet."

    def _compile_config(self) -> dict[str, Any]:
        compile_config = self.config.get("compile", {})
        return compile_config if isinstance(compile_config, dict) else {}

    def _write_index(self, sources: list[RawSourceRecord]) -> None:
        concept_entries = _discover_concept_pages(self.paths)
        analysis_entries = _discover_analysis_pages(self.paths)
        index_payload = {
            "generated_at": utc_now_iso(),
            "source_pages": [
                {
                    "title": source.title,
                    "slug": source.slug,
                    "path": f"wiki/sources/{source.slug}.md",
                    "compiled_at": source.compiled_at,
                }
                for source in _sorted_sources(sources)
            ],
            "concept_pages": concept_entries,
            "analysis_pages": analysis_entries,
        }
        self.paths.wiki_index_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.paths.wiki_index_file,
            json.dumps(index_payload, indent=2, sort_keys=True),
        )

        lines = [
            "# Knowledge Base Index",
            "",
            f"Generated: {index_payload['generated_at']}",
            "",
        ]
        if index_payload["source_pages"]:
            lines.extend(["## Source Pages", ""])
            for page in index_payload["source_pages"]:
                lines.append(f"- [[{page['slug']}]]")
            lines.append("")
        else:
            lines.extend(["## Source Pages", "", "- No source pages compiled yet.", ""])

        if concept_entries:
            lines.extend(["## Concept Pages", ""])
            for page in concept_entries:
                lines.append(f"- [[{page['slug']}]]")
            lines.append("")
        else:
            lines.extend(
                ["## Concept Pages", "", "- No concept pages compiled yet.", ""]
            )

        if analysis_entries:
            lines.extend(["## Analysis Pages", ""])
            for page in analysis_entries:
                lines.append(f"- [[{page['slug']}]]")
            lines.append("")
        else:
            lines.extend(
                ["## Analysis Pages", "", "- No analysis pages saved yet.", ""]
            )
        atomic_write_text(self.paths.wiki_index_markdown, "\n".join(lines))

    def _append_log(
        self,
        compiled_count: int,
        skipped_count: int,
        force: bool,
        *,
        resumed: bool = False,
    ) -> None:
        timestamp = utc_now_iso()
        current = "# Activity Log\n"
        if self.paths.wiki_log_file.exists():
            current = self.paths.wiki_log_file.read_text(encoding="utf-8")
        if not current.endswith("\n"):
            current += "\n"
        flags = []
        if force:
            flags.append("force")
        if resumed:
            flags.append("resume")
        flag_str = f" ({', '.join(flags)})" if flags else ""
        heading = unique_markdown_heading(
            current,
            f"## [{timestamp}] update | "
            f"{compiled_count} compiled, {skipped_count} skipped{flag_str}",
        )
        current += f"\n{heading}\n"
        atomic_write_text(self.paths.wiki_log_file, current)


def _deterministic_summary(contents: str) -> str:
    abstract_paragraphs = _abstract_paragraphs(contents)
    if abstract_paragraphs:
        source_text = " ".join(abstract_paragraphs)
    else:
        paragraphs = _markdown_paragraphs(contents)
        source_text = (
            " ".join(paragraphs) if paragraphs else _plain_text_fallback(contents)
        )

    sentences = _split_sentences(source_text)
    if sentences:
        summary = " ".join(sentences[:3])
    else:
        summary = source_text

    summary = _truncate_with_boundary(summary, 700, add_ellipsis=True)
    return summary or "No summary available yet."


def _parse_provider_summary(raw: str) -> _SummaryResult | None:
    try:
        summary = parse_model_payload(
            raw,
            _ProviderSummary,
            label="Provider summary response",
        )
    except (StructuredOutputError, TypeError, ValueError):
        return None
    return _SummaryResult(
        summary=summary.summary.strip(),
        key_points=_clean_summary_items(summary.key_points, limit=6),
        open_questions=_clean_summary_items(summary.open_questions, limit=6),
        title_suggestion=summary.title_suggestion.strip(),
    )


def _clean_summary_items(items: list[str], *, limit: int) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = " ".join(str(item).split()).strip()
        if not value or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        cleaned.append(value)
        if len(cleaned) == limit:
            break
    return cleaned


def _render_summary_detail_sections(summary: _SummaryResult) -> str:
    sections: list[str] = []
    if summary.key_points:
        sections.extend(
            [
                "## Key Points",
                "",
                "\n".join(f"- {point}" for point in summary.key_points),
                "",
            ]
        )
    if summary.open_questions:
        sections.extend(
            [
                "## Open Questions",
                "",
                "\n".join(f"- {question}" for question in summary.open_questions),
                "",
            ]
        )
    if not sections:
        return ""
    return "\n".join(sections) + "\n"


def _discover_concept_pages(paths: ProjectPaths) -> list[dict[str, str]]:
    return _discover_markdown_pages(paths, paths.wiki_concepts_dir)


def _discover_analysis_pages(paths: ProjectPaths) -> list[dict[str, str]]:
    return _discover_markdown_pages(paths, paths.wiki_analysis_dir)


def _discover_markdown_pages(
    paths: ProjectPaths,
    page_dir: Any,
) -> list[dict[str, str]]:
    if not page_dir.exists():
        return []

    pages: list[dict[str, str]] = []
    for page_path in sorted(page_dir.glob("*.md")):
        try:
            contents = page_path.read_text(encoding="utf-8")
        except OSError:
            continue

        frontmatter = _parse_frontmatter(contents)
        title = str(frontmatter.get("title", "")).strip()
        if not title:
            for heading in markdown_headings(contents):
                title = heading.title
                break
        if not title:
            title = page_path.stem.replace("-", " ").title()

        pages.append(
            {
                "title": title,
                "slug": page_path.stem,
                "path": page_path.relative_to(paths.root).as_posix(),
            }
        )

    return pages


def _markdown_paragraphs(contents: str) -> list[str]:
    return markdown_paragraphs(contents)


def _inline_text(token: Any) -> str:
    return markdown_inline_text(token)


def _is_link_only_inline(token: Any) -> bool:
    return markdown_is_link_only_inline(token)


def _is_content_paragraph(paragraph: str) -> bool:
    return markdown_is_content_paragraph(paragraph)


def _parse_frontmatter(contents: str) -> dict[str, Any]:
    return markdown_parse_frontmatter(contents)


def _strip_frontmatter(contents: str) -> str:
    return markdown_strip_frontmatter(contents)


def _abstract_paragraphs(contents: str) -> list[str]:
    return markdown_section_paragraphs(contents, "abstract")


def _is_heading_line(line: str) -> bool:
    return bool(markdown_headings(line))


def _normalize_newlines(contents: str) -> str:
    return markdown_normalize_newlines(contents).lstrip("\ufeff")


def _plain_text_fallback(contents: str) -> str:
    return markdown_plain_text(contents)


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return []
    try:
        return [s.strip() for s in nltk.sent_tokenize(normalized) if s.strip()]
    except LookupError:
        # Fallback if punkt_tab data is unavailable at runtime.
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]


def _is_weak_summary(summary: str) -> bool:
    normalized = summary.strip()
    if not normalized:
        return True
    if normalized.casefold() in _PLACEHOLDER_SUMMARIES:
        return True
    if _SUMMARY_PROMPT_ECHO_PATTERN.search(normalized):
        return True
    return len(normalized.split()) < 5


def _truncate_with_boundary(text: str, limit: int, *, add_ellipsis: bool) -> str:
    clean = " ".join(text.split()).strip()
    if not clean or limit <= 0:
        return ""
    if len(clean) <= limit:
        return clean

    ellipsis = "..." if add_ellipsis and limit > 3 else ""
    effective_limit = limit - len(ellipsis)
    window = clean[:effective_limit]
    sentence_start = max(0, effective_limit - 160)
    sentence_window = window[sentence_start:]
    sentence_matches = list(re.finditer(r"[.!?][\"')\]]?(?:\s|$)", sentence_window))
    if sentence_matches:
        cut = sentence_start + sentence_matches[-1].end()
        trimmed = window[:cut].rstrip()
    else:
        whitespace_index = window.rfind(" ")
        if whitespace_index > max(20, effective_limit // 3):
            trimmed = window[:whitespace_index].rstrip()
        else:
            trimmed = window.rstrip()

    if ellipsis and trimmed and trimmed != clean:
        trimmed = trimmed.rstrip(" ,;:") + ellipsis
    return trimmed


def _safe_int(value: Any, *, default: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _sorted_sources(sources: list[RawSourceRecord]) -> list[RawSourceRecord]:
    return sorted(sources, key=lambda source: (source.slug, source.source_id))
