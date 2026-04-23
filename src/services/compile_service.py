from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Callable, Optional

from markdown_it import MarkdownIt
import nltk
import nltk.data
import yaml

from src.models.source_models import RawSourceRecord
from src.providers import (
    ProviderConfigurationError,
    UnavailableProvider,
)
from src.providers.base import ProviderRequest, TextProvider
from src.services.config_service import schema_excerpt
from src.services.manifest_service import ManifestService
from src.services.project_service import ProjectPaths, atomic_write_text, utc_now_iso
from src.storage.compile_run_store import CompileRunStore

logger = logging.getLogger(__name__)

_MD_PARSER = MarkdownIt()

_SUMMARY_SYSTEM_PROMPT = (
    "You are a research assistant summarizing documents for a curated knowledge base. "
    "Write a concise 2-4 sentence summary of the document below. "
    "Focus on the core thesis, methods, findings, and open questions. "
    "Do not include author names, affiliations, or publication metadata. "
    "Write in plain text without markdown formatting."
)

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
        summary = self._extract_summary(contents)
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
        if source.normalized_path is not None:
            frontmatter["normalized_path"] = source.normalized_path
        yaml_frontmatter = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        canonical_file_line = ""
        if source.normalized_path is not None:
            canonical_file_line = f"- Canonical file: `{source.normalized_path}`\n"
        return (
            f"---\n{yaml_frontmatter}\n---\n\n"
            f"# {source.title}\n\n"
            "## Summary\n\n"
            f"{summary}\n\n"
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
        provider = self._require_provider()
        truncated = _plain_text_fallback(contents)[:_SUMMARY_CONTENT_LIMIT]
        if not truncated.strip():
            return "No content available for summarization."
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
                )
            )
            summary = response.text.strip()
            if _is_weak_summary(summary):
                return _deterministic_summary(contents)
            return summary
        except Exception as exc:
            logger.warning(
                "Provider summary generation failed; using deterministic fallback: %s",
                exc,
            )
            return _deterministic_summary(contents)

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
        timestamp = utc_now_iso()[:10]
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
        current += (
            f"\n## [{timestamp}] update | "
            f"{compiled_count} compiled, {skipped_count} skipped{flag_str}\n"
        )
        atomic_write_text(self.paths.wiki_log_file, current)


def _markdown_paragraphs(contents: str) -> list[str]:
    """Extract content paragraphs from markdown using markdown-it-py AST."""
    normalized = _strip_frontmatter(contents)
    tokens = _MD_PARSER.parse(normalized)
    paragraphs: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == "paragraph_open":
            # Next token should be the inline content
            if index + 1 < len(tokens) and tokens[index + 1].type == "inline":
                inline = tokens[index + 1]
                if not _is_link_only_inline(inline):
                    text = _inline_text(inline)
                    if text:
                        paragraphs.append(text)
            index += 3  # skip paragraph_open, inline, paragraph_close
        else:
            index += 1
    filtered = [p for p in paragraphs if _is_content_paragraph(p)]
    return _trim_leading_boilerplate(filtered)


def _inline_text(token: Any) -> str:
    """Extract plain text from a markdown-it inline token."""
    if not token.children:
        return token.content.strip()
    parts: list[str] = []
    for child in token.children:
        if child.type in ("text", "code_inline"):
            parts.append(child.content)
        elif child.type == "softbreak":
            parts.append(" ")
    return " ".join("".join(parts).split()).strip()


def _is_link_only_inline(token: Any) -> bool:
    """Return True if every content-bearing child is inside a link or image."""
    if not token.children:
        return False
    in_link = 0
    for child in token.children:
        if child.type in ("link_open", "image"):
            in_link += 1
        elif child.type == "link_close":
            in_link = max(0, in_link - 1)
        elif child.type == "text":
            if not in_link and child.content.strip():
                return False
    return True


def _trim_leading_boilerplate(paragraphs: list[str]) -> list[str]:
    if not paragraphs:
        return []
    toc_index = next(
        (
            index
            for index, paragraph in enumerate(paragraphs[:8])
            if paragraph.casefold().strip() == "table of contents"
        ),
        None,
    )
    if toc_index is None:
        return paragraphs
    trimmed = paragraphs[toc_index + 1 :]
    return trimmed or paragraphs


def _is_content_paragraph(paragraph: str) -> bool:
    """Return False for image-only, link-dominated, or navigation paragraphs."""
    # Skip very short fragments (single words, syntax diagram tokens)
    words = paragraph.split()
    if len(words) <= 1 and len(paragraph) < 15:
        return False
    # Remove linked images: [![alt](img)](url) and plain images: ![alt](url)
    stripped = re.sub(r"\[?!\[[^\]]*\]\([^)]*\)\]?(?:\([^)]*\))?", "", paragraph)
    # Remove markdown links: [text](url)
    stripped = re.sub(r"\[[^\]]*\]\([^)]*\)", "", stripped)
    # Remove residual markdown syntax and list markers
    stripped = re.sub(r"[*\[\]()\-]+", " ", stripped)
    cleaned = " ".join(stripped.split()).strip()
    if not cleaned:
        return False
    if len(paragraph) > 30 and len(cleaned) < 15:
        return False
    # Paragraph is entirely fragment-only links (TOC patterns like [text](#anchor))
    toc_stripped = re.sub(r"\[[^\]]*\]\(#[^)]*\)", "", paragraph)
    toc_cleaned = " ".join(toc_stripped.split()).strip()
    if not toc_cleaned:
        return False
    return True


def _discover_concept_pages(paths: ProjectPaths) -> list[dict[str, str]]:
    """Scan wiki/concepts/ for saved analysis pages."""
    entries: list[dict[str, str]] = []
    if not paths.wiki_concepts_dir.exists():
        return entries
    for cp in sorted(paths.wiki_concepts_dir.glob("*.md")):
        try:
            text = cp.read_text(encoding="utf-8")
            fm = _parse_frontmatter(text)
            entries.append(
                {
                    "title": fm.get("title", cp.stem.replace("-", " ").title()),
                    "slug": cp.stem,
                    "path": f"wiki/concepts/{cp.name}",
                }
            )
        except Exception:
            continue
    return entries


def _discover_analysis_pages(paths: ProjectPaths) -> list[dict[str, str]]:
    """Scan wiki/analysis/ for saved analysis pages."""
    entries: list[dict[str, str]] = []
    if not paths.wiki_analysis_dir.exists():
        return entries
    for ap in sorted(paths.wiki_analysis_dir.glob("*.md")):
        try:
            text = ap.read_text(encoding="utf-8")
            fm = _parse_frontmatter(text)
            entries.append(
                {
                    "title": fm.get("title", ap.stem.replace("-", " ").title()),
                    "slug": ap.stem,
                    "path": f"wiki/analysis/{ap.name}",
                }
            )
        except Exception:
            continue
    return entries


def _parse_frontmatter(contents: str) -> dict:
    normalized = _normalize_newlines(contents)
    if not normalized.startswith("---\n"):
        return {}
    marker = normalized.find("\n---\n", 4)
    if marker == -1:
        return {}
    try:
        payload = yaml.safe_load(normalized[4:marker]) or {}
    except yaml.YAMLError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _strip_frontmatter(contents: str) -> str:
    normalized = _normalize_newlines(contents)
    if not normalized.startswith("---\n"):
        return normalized
    marker = normalized.find("\n---\n", 4)
    if marker == -1:
        return normalized
    return normalized[marker + 5 :]


def _abstract_paragraphs(contents: str) -> list[str]:
    """Extract paragraphs from the first Abstract section, if present."""
    normalized = _strip_frontmatter(contents)
    abstract_start: int | None = None
    for match in re.finditer(r"(?m)^#{1,6}\s+(.+)", normalized):
        heading_text = match.group(1).strip().rstrip("#").strip()
        if heading_text.casefold() == "abstract":
            abstract_start = match.end()
            break
    if abstract_start is None:
        return []
    rest = normalized[abstract_start:]
    next_heading = re.search(r"(?m)^#{1,6}\s+", rest)
    section = rest[: next_heading.start()] if next_heading else rest
    paragraphs: list[str] = []
    current: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if _is_heading_line(stripped):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))
    return [p for p in paragraphs if _is_content_paragraph(p)]


def _is_heading_line(line: str) -> bool:
    return bool(re.match(r"^#{1,6}\s+\S", line))


def _normalize_newlines(contents: str) -> str:
    return contents.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def _plain_text_fallback(contents: str) -> str:
    """Extract plain text from markdown using markdown-it-py AST."""
    text = _strip_frontmatter(contents)
    tokens = _MD_PARSER.parse(text)
    parts: list[str] = []
    for token in tokens:
        if token.type == "inline" and token.children:
            for child in token.children:
                if child.type in ("text", "code_inline"):
                    parts.append(child.content)
                elif child.type == "softbreak":
                    parts.append(" ")
        elif token.type in ("code_block", "fence"):
            continue
        elif token.type == "html_block":
            continue
    return " ".join(" ".join(parts).split()).strip()


def _deterministic_summary(contents: str) -> str:
    summary_source = " ".join(_abstract_paragraphs(contents)).strip()
    if not summary_source:
        paragraphs = _markdown_paragraphs(contents)
        summary_source = " ".join(paragraphs[:2]).strip()
    if not summary_source:
        summary_source = _plain_text_fallback(contents)
    if not summary_source:
        return "No content available for summarization."

    sentences = _split_sentences(summary_source)
    if len(sentences) >= 2:
        return " ".join(sentences[:2]).strip()
    if sentences:
        return sentences[0].strip()
    return _truncate_with_boundary(summary_source, 280, add_ellipsis=True)


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
