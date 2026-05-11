from __future__ import annotations

import subprocess

import yaml

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
    assert result.model == "chat-model"
    assert result.embedding_provider == "openai"
    assert result.embedding_model == "embed-model"
    assert result.api_key_env == "OPENAI_API_KEY"
    assert result.embedding_api_key_env == "OPENAI_API_KEY"
    settings = yaml.safe_load(result.settings_path.read_text(encoding="utf-8"))
    assert (
        settings["completion_models"]["default_completion_model"]["model"]
        == "chat-model"
    )
    assert (
        settings["embedding_models"]["default_embedding_model"]["model"]
        == "embed-model"
    )
    assert (
        settings["completion_models"]["default_completion_model"]["api_key"]
        == "${OPENAI_API_KEY}"
    )
    assert (
        settings["embedding_models"]["default_embedding_model"]["api_key"]
        == "${OPENAI_API_KEY}"
    )
    assert workspace_service.is_initialized() is True


def test_workspace_service_syncs_graph_config_defaults(test_project) -> None:
    test_project.config["graph"] = {
        "provider": "openai",
        "model": "configured-chat",
        "embedding_model": "configured-embedding",
        "api_key_env": "OPENAI_GRAPH_KEY",
    }

    def runner(command, *, cwd, capture_output, text):
        settings_path = test_project.paths.graph_dir / "graphrag" / "settings.yaml"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("input:\n  type: json\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    command_service = GraphRAGCommandService(test_project.paths, runner=runner)
    workspace_service = GraphRAGWorkspaceService(
        test_project.paths,
        command_service,
        config=test_project.config,
    )

    result = workspace_service.init_workspace(force=True)

    settings = yaml.safe_load(result.settings_path.read_text(encoding="utf-8"))
    completion = settings["completion_models"]["default_completion_model"]
    embedding = settings["embedding_models"]["default_embedding_model"]
    assert completion["model_provider"] == "openai"
    assert completion["model"] == "configured-chat"
    assert completion["api_key"] == "${OPENAI_GRAPH_KEY}"
    assert embedding["model_provider"] == "openai"
    assert embedding["model"] == "configured-embedding"
    assert embedding["api_key"] == "${OPENAI_GRAPH_KEY}"


def test_workspace_service_syncs_separate_embedding_provider(test_project) -> None:
    test_project.config["graph"] = {
        "provider": "openai",
        "model": "configured-chat",
        "embedding_provider": "gemini",
        "embedding_model": "gemini-embedding-001",
    }

    def runner(command, *, cwd, capture_output, text):
        settings_path = test_project.paths.graph_dir / "graphrag" / "settings.yaml"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("input:\n  type: json\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    command_service = GraphRAGCommandService(test_project.paths, runner=runner)
    workspace_service = GraphRAGWorkspaceService(
        test_project.paths,
        command_service,
        config=test_project.config,
    )

    result = workspace_service.init_workspace(force=True)

    settings = yaml.safe_load(result.settings_path.read_text(encoding="utf-8"))
    completion = settings["completion_models"]["default_completion_model"]
    embedding = settings["embedding_models"]["default_embedding_model"]
    assert result.api_key_env == "OPENAI_API_KEY"
    assert result.embedding_api_key_env == "GEMINI_API_KEY"
    assert completion["api_key"] == "${OPENAI_API_KEY}"
    assert embedding["model_provider"] == "gemini"
    assert embedding["model"] == "gemini-embedding-001"
    assert embedding["api_key"] == "${GEMINI_API_KEY}"
