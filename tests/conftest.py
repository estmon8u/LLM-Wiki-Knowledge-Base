"""Tests for conftest.

This module belongs to `tests.conftest` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import os

# Set a wide terminal width so Rich tables do not wrap cell content in tests.
os.environ.setdefault("COLUMNS", "300")

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from graphwiki_kb.models.command_models import CommandContext
from graphwiki_kb.providers.base import ProviderRequest, ProviderResponse, TextProvider
from graphwiki_kb.services import build_services
from graphwiki_kb.services.container import ServiceContainer
from graphwiki_kb.services.config_service import ConfigService
from graphwiki_kb.services.manifest_service import ManifestService
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    ProjectService,
    build_project_paths,
)


@dataclass
class TestProject:
    """Represents test project behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    root: Path
    paths: ProjectPaths
    config: dict[str, Any]
    schema_text: str
    services: ServiceContainer
    command_context: CommandContext

    def write_file(self, relative_path: str, content: str) -> Path:
        """Writes file.

        Args:
            relative_path: Relative path value used by the operation.
            content: Content value used by the operation.

        Returns:
            Path produced by the operation.
        """
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


class _StubProvider(TextProvider):
    """Deterministic provider for tests — echoes a canned summary."""

    name = "stub"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate.

        Args:
            request: Request value used by the operation.

        Returns:
            ProviderResponse produced by the operation.
        """
        if request.response_schema_name == "kb_review_report":
            return ProviderResponse(text='{"issues": []}', model_name="stub-1")
        return ProviderResponse(
            text="Stub summary of the document.", model_name="stub-1"
        )


def create_test_project(root: Path, *, initialized: bool) -> TestProject:
    """Create test project.

    Args:
        root: Root path used for discovery or relative path resolution.
        initialized: Initialized value used by the operation.

    Returns:
        TestProject produced by the operation.
    """
    paths = build_project_paths(root)
    project_service = ProjectService(paths)
    config_service = ConfigService(paths)
    manifest_service = ManifestService(paths)

    if initialized:
        project_service.ensure_structure()
        config_service.ensure_files()
        manifest_service.ensure_manifest()

    config = config_service.load()
    schema_text = config_service.load_schema()
    services = build_services(paths, config)
    services["compile"].provider = _StubProvider()
    services["review"].provider = _StubProvider()
    command_context = CommandContext(
        project_root=paths.root,
        cwd=paths.root,
        config=config,
        schema_text=schema_text,
        services=services,
        verbose=False,
    )
    return TestProject(
        root=paths.root,
        paths=paths,
        config=config,
        schema_text=schema_text,
        services=services,
        command_context=command_context,
    )


@pytest.fixture
def test_project(tmp_path: Path) -> TestProject:
    """Verifies that project.

    Args:
        tmp_path: Tmp path value used by the operation.

    Returns:
        TestProject produced by the operation.
    """
    return create_test_project(tmp_path, initialized=True)


@pytest.fixture
def uninitialized_project(tmp_path: Path) -> TestProject:
    """Uninitialized project.

    Args:
        tmp_path: Tmp path value used by the operation.

    Returns:
        TestProject produced by the operation.
    """
    return create_test_project(tmp_path, initialized=False)
