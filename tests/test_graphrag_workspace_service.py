from __future__ import annotations

import subprocess

from src.services.graphrag_command_service import GraphRAGCommandService
from src.services.graphrag_workspace_service import GraphRAGWorkspaceService


def test_workspace_service_reports_initialization_state(test_project) -> None:
    def runner(command, *, cwd, capture_output, text):
        settings_path = test_project.paths.graph_dir / "graphrag" / "settings.yaml"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("input:\n  type: json\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    command_service = GraphRAGCommandService(test_project.paths, runner=runner)
    workspace_service = GraphRAGWorkspaceService(test_project.paths, command_service)

    assert workspace_service.is_initialized() is False

    result = workspace_service.init_workspace(
        model="chat-model",
        embedding="embed-model",
        force=True,
    )

    assert result.workspace_dir == test_project.paths.graph_dir / "graphrag"
    assert result.settings_path.exists()
    assert workspace_service.is_initialized() is True
