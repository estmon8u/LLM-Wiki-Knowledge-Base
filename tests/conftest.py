from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from kb.models.command_models import CommandContext
from kb.services import build_services
from kb.services.config_service import ConfigService
from kb.services.manifest_service import ManifestService
from kb.services.project_service import (
    ProjectPaths,
    ProjectService,
    build_project_paths,
)


@dataclass
class TestProject:
    root: Path
    paths: ProjectPaths
    config: dict[str, Any]
    schema_text: str
    services: dict[str, Any]
    command_context: CommandContext

    def write_file(self, relative_path: str, content: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


def create_test_project(root: Path, *, initialized: bool) -> TestProject:
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
    return create_test_project(tmp_path, initialized=True)


@pytest.fixture
def uninitialized_project(tmp_path: Path) -> TestProject:
    return create_test_project(tmp_path, initialized=False)
