from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

import yaml

from src.models.source_models import RawSourceRecord
from src.models.wiki_models import LintIssue, LintReport
from src.services.manifest_service import ManifestService
from src.services.project_service import ProjectPaths, slugify


WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]|#]+)")


class LintService:
    def __init__(
        self,
        paths: ProjectPaths,
        config: dict[str, Any],
        manifest_service: ManifestService,
    ) -> None:
        self.paths = paths
        self.config = config
        self.manifest_service = manifest_service

    def lint(self) -> LintReport:
        issues: list[LintIssue] = []
        markdown_files = sorted(self.paths.wiki_dir.rglob("*.md"))
        available_slugs = {file_path.stem for file_path in markdown_files}
        incoming_links: dict[str, int] = {slug: 0 for slug in available_slugs}

        for file_path in markdown_files:
            relative_path = file_path.relative_to(self.paths.root).as_posix()
            text = file_path.read_text(encoding="utf-8")
            frontmatter, _ = _split_frontmatter(text)
            if file_path.parent in {
                self.paths.wiki_sources_dir,
                self.paths.wiki_concepts_dir,
            }:
                if frontmatter is None:
                    issues.append(
                        LintIssue(
                            severity="error",
                            code="missing-frontmatter",
                            path=relative_path,
                            message="Compiled wiki pages must include YAML frontmatter.",
                        )
                    )
                else:
                    for field_name in self.config["lint"][
                        "required_frontmatter_fields"
                    ]:
                        if field_name not in frontmatter:
                            issues.append(
                                LintIssue(
                                    severity="error",
                                    code="missing-field",
                                    path=relative_path,
                                    message=f"Missing required frontmatter field: {field_name}",
                                )
                            )
                    if not str(frontmatter.get("summary", "")).strip():
                        issues.append(
                            LintIssue(
                                severity="warning",
                                code="empty-summary",
                                path=relative_path,
                                message="Summary field is empty.",
                            )
                        )

            for raw_target in WIKI_LINK_PATTERN.findall(text):
                target = slugify(raw_target)
                if target not in available_slugs:
                    issues.append(
                        LintIssue(
                            severity="error",
                            code="broken-link",
                            path=relative_path,
                            message=f"Wiki link target not found: [[{raw_target}]]",
                        )
                    )
                else:
                    incoming_links[target] += 1

        for file_path in markdown_files:
            if file_path.parent not in {
                self.paths.wiki_sources_dir,
                self.paths.wiki_concepts_dir,
            }:
                continue
            slug = file_path.stem
            relative_path = file_path.relative_to(self.paths.root).as_posix()
            if incoming_links.get(slug, 0) == 0:
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="orphan-page",
                        path=relative_path,
                        message="Page has no inbound wiki links.",
                    )
                )

        for source in self.manifest_service.list_sources():
            issues.extend(self._lint_manifest_state(source))

        return LintReport(issues=issues)

    def _lint_manifest_state(self, source: RawSourceRecord) -> list[LintIssue]:
        issues: list[LintIssue] = []
        article_path = self.paths.wiki_sources_dir / f"{source.slug}.md"
        if source.compiled_from_hash != source.content_hash:
            issues.append(
                LintIssue(
                    severity="warning",
                    code="stale-source-page",
                    path=f"wiki/sources/{source.slug}.md",
                    message="Source page is stale and should be recompiled.",
                )
            )
        if (
            source.compiled_from_hash == source.content_hash
            and not article_path.exists()
        ):
            issues.append(
                LintIssue(
                    severity="error",
                    code="missing-compiled-page",
                    path=f"wiki/sources/{source.slug}.md",
                    message="Manifest says the source was compiled, but the source page is missing.",
                )
            )
        return issues


def _split_frontmatter(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    if not text.startswith("---\n"):
        return None, text
    marker = text.find("\n---\n", 4)
    if marker == -1:
        return None, text
    payload = text[4:marker]
    content = text[marker + 5 :]
    return yaml.safe_load(payload) or {}, content
