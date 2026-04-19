from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Callable, Optional

import yaml

from src.models.source_models import RawSourceRecord
from src.providers import (
    ProviderConfigurationError,
    ProviderExecutionError,
    UnavailableProvider,
)
from src.providers.base import ProviderRequest, TextProvider
from src.services.manifest_service import ManifestService
from src.services.project_service import ProjectPaths, utc_now_iso

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM_PROMPT = (
    "You are a research assistant summarizing documents for a curated knowledge base. "
    "Write a concise 2-4 sentence summary of the document below. "
    "Focus on the core thesis, methods, findings, and open questions. "
    "Do not include author names, affiliations, or publication metadata. "
    "Write in plain text without markdown formatting."
)

_SUMMARY_CONTENT_LIMIT = 12000


@dataclass
class CompileResult:
    compiled_count: int
    skipped_count: int
    compiled_paths: list[str]


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
    ) -> None:
        self.paths = paths
        self.config = config
        self.manifest_service = manifest_service
        self.provider = provider

    def _require_provider(self) -> TextProvider:
        if self.provider is None:
            raise ProviderConfigurationError(
                "kb compile requires a configured provider. Add a provider section "
                "to kb.config.yaml and set the matching API key environment variable."
            )
        if isinstance(self.provider, UnavailableProvider):
            self.provider.ensure_available()
        return self.provider

    def plan(self, *, force: bool = False) -> CompilePlan:
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
        progress_callback: Optional[Callable[[RawSourceRecord], None]] = None,
    ) -> CompileResult:
        compiled_paths: list[str] = []
        compiled_count = 0
        plan = self.plan(force=force)
        for source in plan.pending_sources:
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
            article_path.parent.mkdir(parents=True, exist_ok=True)
            article_path.write_text(article_text, encoding="utf-8")

            source.compiled_at = compiled_at
            source.compiled_from_hash = source.content_hash
            self.manifest_service.save_source(source)

            compiled_count += 1
            compiled_paths.append(article_path.relative_to(self.paths.root).as_posix())
            if progress_callback is not None:
                progress_callback(source)

        self._write_index(_sorted_sources(self.manifest_service.list_sources()))
        self._append_log(compiled_count, plan.skipped_count, force)
        return CompileResult(
            compiled_count=compiled_count,
            skipped_count=plan.skipped_count,
            compiled_paths=compiled_paths,
        )

    def _render_source_page(
        self, source: RawSourceRecord, contents: str, compiled_at: str
    ) -> str:
        summary = self._extract_summary(contents)
        excerpt = self._extract_excerpt(contents)
        frontmatter = {
            "title": source.title,
            "summary": summary,
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
        try:
            response = provider.generate(
                ProviderRequest(
                    prompt=truncated,
                    system_prompt=_SUMMARY_SYSTEM_PROMPT,
                    max_tokens=512,
                )
            )
            summary = response.text.strip()
            return summary or "No summary available yet."
        except Exception as exc:
            raise ProviderExecutionError(
                f"Provider summary generation failed: {exc}"
            ) from exc

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
        excerpt = clean[:character_limit].rstrip()
        return excerpt or "No excerpt available yet."

    def _compile_config(self) -> dict[str, Any]:
        compile_config = self.config.get("compile", {})
        return compile_config if isinstance(compile_config, dict) else {}

    def _write_index(self, sources: list[RawSourceRecord]) -> None:
        concept_entries = _discover_concept_pages(self.paths)
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
        }
        self.paths.wiki_index_file.parent.mkdir(parents=True, exist_ok=True)
        self.paths.wiki_index_file.write_text(
            json.dumps(index_payload, indent=2, sort_keys=True),
            encoding="utf-8",
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
        self.paths.wiki_index_markdown.parent.mkdir(parents=True, exist_ok=True)
        self.paths.wiki_index_markdown.write_text("\n".join(lines), encoding="utf-8")

    def _append_log(self, compiled_count: int, skipped_count: int, force: bool) -> None:
        timestamp = utc_now_iso()
        self.paths.wiki_log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.paths.wiki_log_file.exists():
            self.paths.wiki_log_file.write_text("# Activity Log\n\n", encoding="utf-8")
        with self.paths.wiki_log_file.open("a", encoding="utf-8") as handle:
            handle.write(
                f"- {timestamp}: compiled {compiled_count} source page(s), "
                f"skipped {skipped_count}, force={str(force).lower()}\n"
            )


def _markdown_paragraphs(contents: str) -> list[str]:
    normalized = _strip_frontmatter(contents)
    paragraphs: list[str] = []
    current: list[str] = []
    in_fenced_code = False
    in_html_comment = False
    active_fence: str | None = None

    def flush_current() -> None:
        nonlocal current
        if current:
            paragraphs.append(" ".join(current).strip())
            current = []

    for line in normalized.splitlines():
        stripped = line.strip()

        if in_html_comment:
            if "-->" in stripped:
                in_html_comment = False
            continue

        if stripped.startswith("<!--"):
            flush_current()
            if "-->" not in stripped:
                in_html_comment = True
            continue

        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush_current()
            fence_marker = stripped[:3]
            if not in_fenced_code:
                in_fenced_code = True
                active_fence = fence_marker
            elif active_fence == fence_marker:
                in_fenced_code = False
                active_fence = None
            continue

        if in_fenced_code:
            continue

        if not stripped or re.fullmatch(r"[-*_]{3,}", stripped):
            flush_current()
            continue

        if _is_heading_line(stripped):
            flush_current()
            continue

        current.append(stripped)

    flush_current()
    filtered = [p for p in paragraphs if _is_content_paragraph(p)]
    return _trim_leading_boilerplate(filtered)


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
    text = _strip_frontmatter(contents)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"~~~.*?~~~", " ", text, flags=re.DOTALL)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*`_~>\[\](){}#+\-|:]+", " ", text)
    return " ".join(text.split()).strip()


def _safe_int(value: Any, *, default: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _sorted_sources(sources: list[RawSourceRecord]) -> list[RawSourceRecord]:
    return sorted(sources, key=lambda source: (source.slug, source.source_id))
