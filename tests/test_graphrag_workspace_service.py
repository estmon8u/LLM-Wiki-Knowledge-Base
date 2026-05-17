"""Tests for test graphrag workspace service.

This module belongs to `tests.test_graphrag_workspace_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import yaml

from graphwiki_kb.services.graphrag_command_service import GraphRAGCommandService
from graphwiki_kb.services.graphrag_defaults import DEFAULT_GRAPHRAG_ENCODING_MODEL
from graphwiki_kb.services.graphrag_workspace_service import (
    GraphRAGWorkspaceService,
    _bundled_prompt_dirs,
)


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
    assert result.result.command[:2] == ("graphrag.api", "initialize_project_at")
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


def test_workspace_service_writes_changed_prompt_templates_as_new_files(
    test_project,
) -> None:
    """User-tuned prompt files must not be overwritten by bundled templates."""
    prompt_path = (
        test_project.paths.graph_dir / "graphrag" / "prompts" / "extract_graph.txt"
    )
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("custom tuned prompt", encoding="utf-8")
    command_service = GraphRAGCommandService(test_project.paths)
    workspace_service = GraphRAGWorkspaceService(
        test_project.paths,
        command_service,
        config=test_project.config,
    )

    created = workspace_service._ensure_prompt_templates()

    assert prompt_path.read_text(encoding="utf-8") == "custom tuned prompt"
    candidate = prompt_path.with_suffix(".txt.new")
    assert candidate.exists()
    assert candidate.relative_to(test_project.root).as_posix() in created


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

    includes = payload["tool"]["poetry"]["include"]
    prompt_include = next(
        item for item in includes if item["path"] == "graph/graphrag/prompts/*.txt"
    )
    assert prompt_include["format"] == ["sdist", "wheel"]


def test_bundled_prompt_dirs_finds_repo_root_templates() -> None:
    """Regression: prompt discovery must find repository-root templates."""
    candidates = _bundled_prompt_dirs()

    assert any(
        candidate.parts[-3:] == ("graph", "graphrag", "prompts")
        and (candidate / "extract_graph.txt").exists()
        for candidate in candidates
    )


def test_bundled_prompt_dirs_finds_installed_wheel_root_templates(tmp_path) -> None:
    """Regression: prompt discovery works after package installation."""
    module_file = (
        tmp_path
        / "site-packages"
        / "graphwiki_kb"
        / "services"
        / "graphrag_workspace_service.py"
    )
    module_file.parent.mkdir(parents=True)
    prompt_dir = tmp_path / "site-packages" / "graph" / "graphrag" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "extract_graph.txt").write_text("Prompt", encoding="utf-8")

    candidates = _bundled_prompt_dirs(module_file)

    assert candidates[0] == prompt_dir


def test_tracked_graphrag_settings_use_portable_defaults() -> None:
    """Verifies tracked GraphRAG settings match portable project defaults."""
    settings_path = (
        Path(__file__).resolve().parents[1] / "graph" / "graphrag" / "settings.yaml"
    )

    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))

    assert settings["chunking"]["encoding_model"] == DEFAULT_GRAPHRAG_ENCODING_MODEL
    assert settings["vector_store"]["db_uri"] == "output/lancedb"
