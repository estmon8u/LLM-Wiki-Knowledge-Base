from __future__ import annotations

import hashlib
import os
from typing import Any

from src.providers import resolve_provider_settings
from src.models.wiki_models import StatusSnapshot
from src.services.manifest_service import ManifestService
from src.services.project_service import ProjectPaths


class StatusService:
    def __init__(
        self,
        paths: ProjectPaths,
        manifest_service: ManifestService,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.paths = paths
        self.manifest_service = manifest_service
        self._config = config or {}

    def snapshot(self, *, initialized: bool) -> StatusSnapshot:
        sources = (
            self.manifest_service.list_sources()
            if self.paths.raw_manifest_file.exists()
            else []
        )
        compiled_sources = sum(
            1
            for source in sources
            if source.compiled_from_hash is not None
            and source.compiled_from_hash == self._current_content_hash(source)
        )

        concept_count = 0
        analysis_count = 0
        if self.paths.wiki_concepts_dir.exists():
            for page in self.paths.wiki_concepts_dir.glob("*.md"):
                try:
                    text = page.read_text(encoding="utf-8")
                except OSError:
                    continue
                if "type: analysis" in text:
                    analysis_count += 1
                else:
                    concept_count += 1

        # Also count analysis pages in wiki/analysis/
        if self.paths.wiki_analysis_dir.exists():
            analysis_count += len(list(self.paths.wiki_analysis_dir.glob("*.md")))

        last_compile_at = max(
            (source.compiled_at for source in sources if source.compiled_at),
            default=None,
        )

        return StatusSnapshot(
            initialized=initialized,
            source_count=len(sources),
            compiled_source_count=compiled_sources,
            concept_page_count=concept_count,
            analysis_page_count=analysis_count,
            last_compile_at=last_compile_at,
            provider_summary=self._provider_summary(),
            index_status=self._index_status(),
            export_status=self._export_status(last_compile_at),
        )

    def _provider_summary(self) -> str:
        resolved = resolve_provider_settings(
            self._config,
        )
        if resolved is None:
            return "not configured"
        name, provider_config = resolved
        env_var = provider_config.get("api_key_env", f"{name.upper()}_API_KEY")
        key_set = bool(os.environ.get(env_var))
        model = provider_config.get("model", "")
        parts = [f"{name} configured"]
        if model:
            parts.append(f"model={model}")
        parts.append(f"{env_var} {'set' if key_set else 'NOT SET'}")
        return ", ".join(parts)

    def _index_status(self) -> str:
        index_path = self.paths.graph_exports_dir / "search_index.sqlite3"
        if not index_path.exists():
            return "not built"
        return "available"

    def _export_status(self, last_compile_at: str | None) -> str:
        if not self.paths.vault_obsidian_dir.exists():
            return "not exported"
        vault_files = list(self.paths.vault_obsidian_dir.rglob("*.md"))
        if not vault_files:
            return "empty"
        if last_compile_at:
            # If any wiki page is newer than the oldest vault file, export is stale
            vault_mtime = min(f.stat().st_mtime for f in vault_files)
            wiki_files = list(self.paths.wiki_dir.rglob("*.md"))
            if wiki_files:
                wiki_mtime = max(f.stat().st_mtime for f in wiki_files)
                if wiki_mtime > vault_mtime:
                    return "stale"
        return "current"

    def _current_content_hash(self, source) -> str:
        norm_path = source.normalized_path or source.raw_path
        full_path = self.paths.root / norm_path
        try:
            text = full_path.read_text(encoding="utf-8")
            return hashlib.sha256(text.encode("utf-8")).hexdigest()
        except OSError:
            return source.content_hash
