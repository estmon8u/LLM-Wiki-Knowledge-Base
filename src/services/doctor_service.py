"""Diagnostic checks for project health."""

from __future__ import annotations

import os
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Optional

from src.providers import resolve_provider_settings, supported_provider_names
from src.services.config_service import resolve_graph_config
from src.services.graphrag_defaults import env_file_has_key
from src.services.graphrag_status_service import GraphRAGStatusService
from src.services.normalization_service import resolve_wkhtmltopdf_binary
from src.services.project_service import ProjectPaths


@dataclass
class DoctorCheck:
    """Represents doctor check behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    name: str
    ok: bool
    detail: str
    severity: str = "error"  # "ok", "warning", or "error"


@dataclass
class DoctorReport:
    """Stores doctor report data.

    Attributes:
        See annotated class attributes for stored values.
    """

    checks: list[DoctorCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Ok.

        Returns:
            bool produced by the operation.
        """
        return all(c.severity != "error" for c in self.checks)

    @property
    def passed_count(self) -> int:
        """Passed count.

        Returns:
            int produced by the operation.
        """
        return sum(1 for c in self.checks if c.severity == "ok")

    @property
    def warning_count(self) -> int:
        """Warning count.

        Returns:
            int produced by the operation.
        """
        return sum(1 for c in self.checks if c.severity == "warning")

    @property
    def failed_count(self) -> int:
        """Failed count.

        Returns:
            int produced by the operation.
        """
        return sum(1 for c in self.checks if c.severity == "error")


class DoctorService:
    """Coordinates doctor operations.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        config: dict[str, Any],
        provider: Optional[Any] = None,
        graphrag_status_service: GraphRAGStatusService | None = None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.provider = provider
        self.graphrag_status_service = graphrag_status_service

    def diagnose(self, *, strict: bool = False) -> DoctorReport:
        """Diagnose.

        Args:
            strict: Strict value used by the operation.

        Returns:
            DoctorReport produced by the operation.
        """
        checks: list[DoctorCheck] = []
        checks.append(self._check_project_structure())
        checks.append(self._check_config_file())
        checks.append(self._check_schema_file())
        checks.append(self._check_manifest())
        checks.append(self._check_provider_config(strict=strict))
        checks.append(self._check_api_key(strict=strict))
        checks.append(self._check_mistral_api_key(strict=strict))
        checks.append(self._check_html_renderer(strict=strict))
        checks.append(self._check_converters())
        checks.extend(self._check_graphrag(strict=strict))
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
        optional_available: list[str] = []
        try:
            import mistralai  # noqa: F401

            available.append("Mistral SDK")
        except ImportError:
            missing.append("Mistral SDK")
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
        try:
            import pdfkit  # noqa: F401

            available.append("pdfkit")
        except ImportError:
            missing.append("pdfkit")
        try:
            import xhtml2pdf  # noqa: F401

            optional_available.append("xhtml2pdf")
        except ImportError:
            pass  # xhtml2pdf is optional; absence is not a problem
        all_names = available + optional_available
        if missing:
            detail = f"Available: {', '.join(all_names) or 'none'}. "
            detail += f"Missing: {', '.join(missing)}."
            return DoctorCheck(
                name="converters",
                ok=False,
                detail=detail,
                severity="warning",
            )
        detail = f"All converters available: {', '.join(all_names)}."
        return DoctorCheck(
            name="converters",
            ok=True,
            detail=detail,
            severity="ok",
        )

    def _check_mistral_api_key(self, *, strict: bool = False) -> DoctorCheck:
        conversion = self.config.get("conversion") or {}
        mistral_ocr = (
            conversion.get("mistral_ocr") if isinstance(conversion, dict) else {}
        )
        env_var = ""
        if isinstance(mistral_ocr, dict):
            env_var = str(mistral_ocr.get("api_key_env", "")).strip()
        env_var = env_var or "MISTRAL_API_KEY"
        if os.environ.get(env_var):
            return DoctorCheck(
                name="mistral_api_key",
                ok=True,
                detail=f"Environment variable {env_var} is set for Mistral OCR ingest.",
                severity="ok",
            )
        sev = "error" if strict else "warning"
        return DoctorCheck(
            name="mistral_api_key",
            ok=False,
            detail=(
                f"Environment variable {env_var} is not set. Required for the default "
                "OCR ingest routes for PDF, DOCX, PPTX, and image inputs."
            ),
            severity=sev,
        )

    def _check_html_renderer(self, *, strict: bool = False) -> DoctorCheck:
        binary = resolve_wkhtmltopdf_binary(self.config)
        from src.services.normalization_service import Xhtml2pdfRenderer

        xhtml2pdf_ok = Xhtml2pdfRenderer.available()
        parts: list[str] = []
        if binary:
            parts.append(f"wkhtmltopdf available at {binary}")
        if xhtml2pdf_ok:
            parts.append("xhtml2pdf available")
        if parts:
            return DoctorCheck(
                name="html_renderer",
                ok=True,
                detail="; ".join(parts) + ".",
                severity="ok",
            )
        sev = "error" if strict else "warning"
        return DoctorCheck(
            name="html_renderer",
            ok=False,
            detail=(
                "Neither wkhtmltopdf nor xhtml2pdf is available. At least one is "
                "required for the HTML/HTM render-to-PDF OCR route."
            ),
            severity=sev,
        )

    def _check_graphrag(self, *, strict: bool = False) -> list[DoctorCheck]:
        sev = "error" if strict else "warning"
        checks: list[DoctorCheck] = []
        checks.append(
            DoctorCheck(
                name="graphrag_dependency",
                ok=importlib.util.find_spec("graphrag") is not None,
                detail=(
                    "GraphRAG dependency is importable."
                    if importlib.util.find_spec("graphrag") is not None
                    else "GraphRAG dependency is missing. Run `poetry install`."
                ),
                severity=(
                    "ok"
                    if importlib.util.find_spec("graphrag") is not None
                    else "error"
                ),
            )
        )
        status = (
            self.graphrag_status_service.status()
            if self.graphrag_status_service is not None
            else None
        )
        workspace_ok = bool(status and status.workspace_initialized)
        checks.append(
            DoctorCheck(
                name="graphrag_workspace",
                ok=workspace_ok,
                detail=(
                    "GraphRAG workspace initialized."
                    if workspace_ok
                    else "GraphRAG workspace missing. Run `kb init` or `kb update`."
                ),
                severity="ok" if workspace_ok else sev,
            )
        )
        checks.append(self._check_graphrag_api_key(strict=strict))
        input_ok = bool(
            status and status.input_exists and status.input_document_count > 0
        )
        checks.append(
            DoctorCheck(
                name="graphrag_input",
                ok=input_ok,
                detail=(
                    f"GraphRAG input has {status.input_document_count} document(s)."
                    if input_ok and status
                    else "GraphRAG input is missing or empty. Run `kb update`."
                ),
                severity="ok" if input_ok else sev,
            )
        )
        index_ok = bool(status and status.output_present)
        checks.append(
            DoctorCheck(
                name="graphrag_index",
                ok=index_ok,
                detail=(
                    "GraphRAG index output is present."
                    if index_ok
                    else "GraphRAG index output is missing. Run `kb update`."
                ),
                severity="ok" if index_ok else sev,
            )
        )
        export_ok = bool(status and status.wiki_export_present)
        checks.append(
            DoctorCheck(
                name="graphrag_wiki_export",
                ok=export_ok,
                detail=(
                    "GraphRAG wiki export is present."
                    if export_ok
                    else "GraphRAG wiki export is missing. Run `kb update` or `kb export`."
                ),
                severity="ok" if export_ok else sev,
            )
        )
        return checks

    def _check_graphrag_api_key(self, *, strict: bool = False) -> DoctorCheck:
        sev = "error" if strict else "warning"
        try:
            graph_config = resolve_graph_config(self.config)
        except ValueError as exc:
            return DoctorCheck(
                name="graphrag_api_key",
                ok=False,
                detail=str(exc),
                severity="error",
            )
        dot_env = self.paths.graph_dir / "graphrag" / ".env"
        missing = []
        for key in dict.fromkeys(
            (graph_config.api_key_env, graph_config.embedding_api_key_env)
        ):
            if os.environ.get(key) or env_file_has_key(dot_env, key):
                continue
            missing.append(key)
        if not missing:
            return DoctorCheck(
                name="graphrag_api_key",
                ok=True,
                detail="GraphRAG provider API key environment is configured.",
                severity="ok",
            )
        return DoctorCheck(
            name="graphrag_api_key",
            ok=False,
            detail=(
                "Missing GraphRAG API key environment variable(s): "
                + ", ".join(missing)
                + ". Set them before provider-backed graph indexing or asking."
            ),
            severity=sev,
        )
