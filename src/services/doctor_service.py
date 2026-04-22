"""Diagnostic checks for project health."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from src.providers import resolve_provider_settings, supported_provider_names
from src.services.project_service import ProjectPaths


@dataclass
class DoctorCheck:
    name: str
    ok: bool
    detail: str
    severity: str = "error"  # "ok", "warning", or "error"


@dataclass
class DoctorReport:
    checks: list[DoctorCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.severity != "error" for c in self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == "ok")

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == "warning")

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == "error")


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

    def diagnose(self, *, strict: bool = False) -> DoctorReport:
        checks: list[DoctorCheck] = []
        checks.append(self._check_project_structure())
        checks.append(self._check_config_file())
        checks.append(self._check_schema_file())
        checks.append(self._check_manifest())
        checks.append(self._check_provider_config(strict=strict))
        checks.append(self._check_api_key(strict=strict))
        checks.append(self._check_converters())
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
                severity="error",
            )
        return DoctorCheck(
            name="project_structure",
            ok=True,
            detail="All project directories exist.",
            severity="ok",
        )

    def _check_config_file(self) -> DoctorCheck:
        if not self.paths.config_file.exists():
            return DoctorCheck(
                name="config_file",
                ok=False,
                detail="kb.config.yaml not found. Run 'kb init'.",
                severity="error",
            )
        return DoctorCheck(
            name="config_file",
            ok=True,
            detail="kb.config.yaml present.",
            severity="ok",
        )

    def _check_schema_file(self) -> DoctorCheck:
        if not self.paths.schema_file.exists():
            return DoctorCheck(
                name="schema_file",
                ok=False,
                detail="kb.schema.md not found. Run 'kb init'.",
                severity="error",
            )
        return DoctorCheck(
            name="schema_file",
            ok=True,
            detail="kb.schema.md present.",
            severity="ok",
        )

    def _check_manifest(self) -> DoctorCheck:
        if not self.paths.raw_manifest_file.exists():
            return DoctorCheck(
                name="manifest",
                ok=False,
                detail="raw/_manifest.json not found. Run 'kb init'.",
                severity="error",
            )
        return DoctorCheck(
            name="manifest",
            ok=True,
            detail="Manifest file present.",
            severity="ok",
        )

    def _check_provider_config(self, *, strict: bool = False) -> DoctorCheck:
        provider_cfg = self.config.get("provider") or {}
        name = str(provider_cfg.get("name", "")).strip().lower()
        if not name:
            sev = "error" if strict else "warning"
            return DoctorCheck(
                name="provider_config",
                ok=False,
                detail="No provider configured. Required for update, ask, and review.",
                severity=sev,
            )
        supported = set(supported_provider_names(self.config.get("providers")))
        if name not in supported:
            return DoctorCheck(
                name="provider_config",
                ok=False,
                detail=f"Unknown provider '{name}'. Supported: {', '.join(sorted(supported))}.",
                severity="error",
            )
        _, resolved_cfg = resolve_provider_settings(
            self.config,
        ) or (name, {})
        model = resolved_cfg.get("model", "(default)")
        return DoctorCheck(
            name="provider_config",
            ok=True,
            detail=f"Provider '{name}' configured with model '{model}'.",
            severity="ok",
        )

    def _check_api_key(self, *, strict: bool = False) -> DoctorCheck:
        provider_cfg = self.config.get("provider") or {}
        name = str(provider_cfg.get("name", "")).strip().lower()
        if not name:
            sev = "error" if strict else "warning"
            return DoctorCheck(
                name="api_key",
                ok=False,
                detail="Cannot check API key until a provider is selected.",
                severity=sev,
            )
        _, resolved_cfg = resolve_provider_settings(
            self.config,
        ) or (name, {})
        env_var = resolved_cfg.get("api_key_env", "")
        if not env_var:
            return DoctorCheck(
                name="api_key",
                ok=False,
                detail=f"No API key env variable known for provider '{name}'.",
                severity="error",
            )
        if os.environ.get(env_var):
            return DoctorCheck(
                name="api_key",
                ok=True,
                detail=f"Environment variable {env_var} is set.",
                severity="ok",
            )
        sev = "error" if strict else "warning"
        return DoctorCheck(
            name="api_key",
            ok=False,
            detail=f"Environment variable {env_var} is not set.",
            severity=sev,
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
                severity="warning",
            )
        return DoctorCheck(
            name="converters",
            ok=True,
            detail=f"All converters available: {', '.join(available)}.",
            severity="ok",
        )
