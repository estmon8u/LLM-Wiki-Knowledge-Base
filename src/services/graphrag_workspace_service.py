from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.services.graphrag_command_service import (
    GraphRAGCommandResult,
    GraphRAGCommandService,
)
from src.services.project_service import ProjectPaths


@dataclass(frozen=True)
class GraphRAGWorkspaceInitResult:
    workspace_dir: Path
    settings_path: Path
    result: GraphRAGCommandResult


class GraphRAGWorkspaceService:
    def __init__(
        self,
        paths: ProjectPaths,
        command_service: GraphRAGCommandService,
    ) -> None:
        self.paths = paths
        self.command_service = command_service
        self.workspace_dir = paths.graph_dir / "graphrag"
        self.settings_path = self.workspace_dir / "settings.yaml"

    def is_initialized(self) -> bool:
        return self.settings_path.exists()

    def init_workspace(
        self,
        *,
        model: str,
        embedding: str,
        force: bool,
    ) -> GraphRAGWorkspaceInitResult:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        result = self.command_service.init_workspace(
            model=model,
            embedding=embedding,
            force=force,
        )
        return GraphRAGWorkspaceInitResult(
            workspace_dir=self.workspace_dir,
            settings_path=self.settings_path,
            result=result,
        )
