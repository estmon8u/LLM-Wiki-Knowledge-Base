"""Diagnostic checks for project health."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from src.providers import _DEFAULT_API_KEY_ENVS
from src.services.project_service import ProjectPaths


@dataclass
class DoctorCheck:
    name: str
    ok: bool
    detail: str


@dataclass
class DoctorReport:
    checks: list[DoctorCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.ok)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.ok)


class DoctorService:
    def __init__(
        self,
        paths: ProjectPaths,
        config: dict[str, Any],
        provider: Optional[Any] = None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.provider = provider

    def diagnose(self) -> DoctorReport:
        checks: list[DoctorCheck] = []
        checks.append(self._check_project_structure())
        checks.append(self._check_config_file())
        checks.append(self._check_schema_file())
        checks.append(self._check_manifest())
        checks.append(self._check_provider_config())
        checks.append(self._check_api_key())
        checks.append(self._check_converters())
        checks.append(self._check_run_store_db())
        return DoctorReport(checks=checks)

    def _check_project_structure(self) -> DoctorCheck:
        required = [
            self.paths.raw_dir,
            self.paths.raw_sources_dir,
            self.paths.raw_normalized_dir,
            self.paths.wiki_dir,
            self.paths.wiki_sources_dir,
            self.paths.vault_dir,
            self.paths.vault_obsidian_dir,
            self.paths.graph_dir,
            self.paths.graph_exports_dir,
        ]
        missing = [str(d) for d in required if not d.exists()]
        if missing:
            return DoctorCheck(
                name="project_structure",
                ok=False,
                detail=f"Missing directories: {', '.join(missing)}",
            )
        return DoctorCheck(
            name="project_structure", ok=True, detail="All project directories exist."
        )

    def _check_config_file(self) -> DoctorCheck:
        if not self.paths.config_file.exists():
            return DoctorCheck(
                name="config_file",
                ok=False,
                detail="kb.config.yaml not found. Run 'kb init'.",
            )
        return DoctorCheck(
            name="config_file", ok=True, detail="kb.config.yaml present."
        )

    def _check_schema_file(self) -> DoctorCheck:
        if not self.paths.schema_file.exists():
            return DoctorCheck(
                name="schema_file",
                ok=False,
                detail="kb.schema.md not found. Run 'kb init'.",
            )
        return DoctorCheck(name="schema_file", ok=True, detail="kb.schema.md present.")

    def _check_manifest(self) -> DoctorCheck:
        if not self.paths.raw_manifest_file.exists():
            return DoctorCheck(
                name="manifest",
                ok=False,
                detail="raw/_manifest.json not found. Run 'kb init'.",
            )
        return DoctorCheck(name="manifest", ok=True, detail="Manifest file present.")

    def _check_provider_config(self) -> DoctorCheck:
        provider_cfg = self.config.get("provider") or {}
        name = provider_cfg.get("name", "")
        if not name:
            return DoctorCheck(
                name="provider_config",
                ok=False,
                detail="No provider configured in kb.config.yaml.",
            )
        supported = {"openai", "anthropic", "gemini"}
        if name not in supported:
            return DoctorCheck(
                name="provider_config",
                ok=False,
                detail=f"Unknown provider '{name}'. Supported: {', '.join(sorted(supported))}.",
            )
        model = provider_cfg.get("model", "(default)")
        return DoctorCheck(
            name="provider_config",
            ok=True,
            detail=f"Provider '{name}' configured with model '{model}'.",
        )

    def _check_api_key(self) -> DoctorCheck:
        provider_cfg = self.config.get("provider") or {}
        name = provider_cfg.get("name", "")
        if not name:
            return DoctorCheck(
                name="api_key",
                ok=False,
                detail="No provider configured — cannot check API key.",
            )
        env_var = provider_cfg.get("api_key_env", _DEFAULT_API_KEY_ENVS.get(name, ""))
        if not env_var:
            return DoctorCheck(
                name="api_key",
                ok=False,
                detail=f"No API key env variable known for provider '{name}'.",
            )
        if os.environ.get(env_var):
            return DoctorCheck(
                name="api_key",
                ok=True,
                detail=f"Environment variable {env_var} is set.",
            )
        return DoctorCheck(
            name="api_key",
            ok=False,
            detail=f"Environment variable {env_var} is not set.",
        )

    def _check_converters(self) -> DoctorCheck:
        available: list[str] = []
        missing: list[str] = []
        try:
            import markitdown  # noqa: F401

            available.append("MarkItDown")
        except ImportError:
            missing.append("MarkItDown")
        try:
            import docling  # noqa: F401

            available.append("Docling")
        except ImportError:
            missing.append("Docling")
        if missing:
            return DoctorCheck(
                name="converters",
                ok=False,
                detail=f"Available: {', '.join(available) or 'none'}. "
                f"Missing: {', '.join(missing)}.",
            )
        return DoctorCheck(
            name="converters",
            ok=True,
            detail=f"All converters available: {', '.join(available)}.",
        )

    def _check_run_store_db(self) -> DoctorCheck:
        db_path = self.paths.graph_exports_dir / "run_artifacts.sqlite3"
        if not db_path.exists():
            return DoctorCheck(
                name="run_store",
                ok=True,
                detail="Run-artifact database not yet created (OK for new projects).",
            )
        try:
            size = db_path.stat().st_size
            return DoctorCheck(
                name="run_store",
                ok=True,
                detail=f"Run-artifact database present ({size} bytes).",
            )
        except OSError as exc:
            return DoctorCheck(
                name="run_store",
                ok=False,
                detail=f"Cannot read run-artifact database: {exc}",
            )
