from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import yaml

from src.models.source_models import RawSourceRecord
from src.services.manifest_service import ManifestService
from src.services.project_service import ProjectPaths, utc_now_iso


@dataclass
class CompileResult:
    compiled_count: int
    skipped_count: int
    compiled_paths: list[str]


class CompileService:
    def __init__(
        self,
        paths: ProjectPaths,
        config: dict[str, Any],
        manifest_service: ManifestService,
    ) -> None:
        self.paths = paths
        self.config = config
        self.manifest_service = manifest_service

    def compile(self, *, force: bool = False) -> CompileResult:
        compiled_paths: list[str] = []
        compiled_count = 0
        skipped_count = 0
        sources = self.manifest_service.list_sources()
        for source in sources:
            article_path = self.paths.wiki_sources_dir / f"{source.slug}.md"
            if (
                not force
                and source.compiled_from_hash == source.content_hash
                and article_path.exists()
            ):
                skipped_count += 1
                continue

            canonical_path = self.paths.root / (
                source.normalized_path or source.raw_path
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

        self._write_index(self.manifest_service.list_sources())
        self._append_log(compiled_count, skipped_count, force)
        return CompileResult(
            compiled_count=compiled_count,
            skipped_count=skipped_count,
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
        paragraphs = _markdown_paragraphs(contents)
        limit = self.config["compile"].get("summary_paragraph_limit", 2)
        if paragraphs:
            return "\n\n".join(paragraphs[:limit])
        cleaned = "\n".join(_markdown_paragraphs(_strip_frontmatter(contents))).strip()
        return cleaned[:280] or "No summary available yet."

    def _extract_excerpt(self, contents: str) -> str:
        clean = "\n".join(_markdown_paragraphs(contents))
        character_limit = self.config["compile"].get("excerpt_character_limit", 900)
        excerpt = clean[:character_limit].rstrip()
        return excerpt or "No excerpt available yet."

    def _write_index(self, sources: list[RawSourceRecord]) -> None:
        index_payload = {
            "generated_at": utc_now_iso(),
            "source_pages": [
                {
                    "title": source.title,
                    "slug": source.slug,
                    "path": f"wiki/sources/{source.slug}.md",
                    "compiled_at": source.compiled_at,
                }
                for source in sources
            ],
            "concept_pages": [],
        }
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

        lines.extend(["## Concept Pages", "", "- No concept pages compiled yet.", ""])
        self.paths.wiki_index_markdown.write_text("\n".join(lines), encoding="utf-8")

    def _append_log(self, compiled_count: int, skipped_count: int, force: bool) -> None:
        timestamp = utc_now_iso()
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
    for line in normalized.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        if stripped.startswith("#"):
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current).strip())
    return paragraphs


def _strip_frontmatter(contents: str) -> str:
    if not contents.startswith("---\n"):
        return contents
    marker = contents.find("\n---\n", 4)
    if marker == -1:
        return contents
    return contents[marker + 5 :]
