"""Tests for test graphrag workspace service.

This module belongs to `tests.test_graphrag_workspace_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib

import yaml

from src.services.graphrag_command_service import GraphRAGCommandService
from src.services.graphrag_defaults import DEFAULT_GRAPHRAG_ENCODING_MODEL
from src.services.graphrag_workspace_service import GraphRAGWorkspaceService


def test_workspace_service_reports_initialization_state(test_project) -> None:
    """Verifies that workspace service reports initialization state.

    Args:
        test_project: Test project value used by the operation.
    """

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
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
    assert result.result.command[1:4] == ("-m", "graphrag", "init")
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
    """Verifies that workspace service syncs graph config defaults.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.config["graph"] = {
        "provider": "openai",
        "model": "configured-chat",
        "embedding_model": "configured-embedding",
        "api_key_env": "OPENAI_GRAPH_KEY",
    }

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
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


def test_workspace_service_preserves_user_settings_when_syncing_managed_fields(
    test_project,
) -> None:
    """Regression: defaults should not overwrite user-owned GraphRAG tuning."""
    settings_path = test_project.paths.graph_dir / "graphrag" / "settings.yaml"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        "chunking:\n"
        "  type: tokens\n"
        "  size: 2222\n"
        "vector_store:\n"
        "  type: lancedb\n"
        "  db_uri: custom/lancedb\n"
        "completion_models:\n"
        "  default_completion_model:\n"
        "    model: stale-model\n",
        encoding="utf-8",
    )
    command_service = GraphRAGCommandService(test_project.paths)
    workspace_service = GraphRAGWorkspaceService(
        test_project.paths,
        command_service,
        config=test_project.config,
    )

    workspace_service.sync_settings()

    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    assert settings["chunking"]["size"] == 2222
    assert settings["vector_store"]["db_uri"] == "custom/lancedb"
    assert (
        settings["completion_models"]["default_completion_model"]["model"]
        == "gpt-5.4-nano"
    )
    assert settings["input"]["type"] == "json"
    assert (
        settings["drift_search"]["reduce_prompt"] == "prompts/drift_reduce_prompt.txt"
    )


def test_workspace_service_normalizes_stock_windows_vector_store_path(
    test_project,
) -> None:
    """Verifies the stock GraphRAG Windows path is rewritten portably."""
    settings_path = test_project.paths.graph_dir / "graphrag" / "settings.yaml"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        "vector_store:\n" "  type: lancedb\n" "  db_uri: output\\lancedb\n",
        encoding="utf-8",
    )
    command_service = GraphRAGCommandService(test_project.paths)
    workspace_service = GraphRAGWorkspaceService(
        test_project.paths,
        command_service,
        config=test_project.config,
    )

    workspace_service.sync_settings()

    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    assert settings["vector_store"]["db_uri"] == "output/lancedb"


def test_workspace_service_syncs_separate_embedding_provider(test_project) -> None:
    """Verifies that workspace service syncs separate embedding provider.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.config["graph"] = {
        "provider": "openai",
        "model": "configured-chat",
        "embedding_provider": "gemini",
        "embedding_model": "gemini-embedding-001",
    }

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
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


def test_pyproject_includes_graphrag_prompt_templates() -> None:
    """Verifies packaged builds include bundled GraphRAG prompt templates."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert "graph/graphrag/prompts/*.txt" in payload["tool"]["poetry"]["include"]


def test_tracked_graphrag_settings_use_portable_defaults() -> None:
    """Verifies tracked GraphRAG settings match portable project defaults."""
    settings_path = (
        Path(__file__).resolve().parents[1] / "graph" / "graphrag" / "settings.yaml"
    )

    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))

    assert settings["chunking"]["encoding_model"] == DEFAULT_GRAPHRAG_ENCODING_MODEL
    assert settings["vector_store"]["db_uri"] == "output/lancedb"
