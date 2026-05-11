from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.services.graphrag_command_service import (
    GraphRAGCommandResult,
    GraphRAGCommandService,
)
from src.services.config_service import GraphRAGRuntimeConfig, resolve_graph_config
from src.services.project_service import ProjectPaths, atomic_write_text
import yaml


@dataclass(frozen=True)
class GraphRAGWorkspaceInitResult:
    workspace_dir: Path
    settings_path: Path
    result: GraphRAGCommandResult
    provider: str
    model: str
    embedding_provider: str
    embedding_model: str
    api_key_env: str
    embedding_api_key_env: str


class GraphRAGWorkspaceService:
    def __init__(
        self,
        paths: ProjectPaths,
        command_service: GraphRAGCommandService,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.paths = paths
        self.command_service = command_service
        self.config = config or {}
        self.workspace_dir = paths.graph_dir / "graphrag"
        self.settings_path = self.workspace_dir / "settings.yaml"

    def is_initialized(self) -> bool:
        return self.settings_path.exists()

    def init_workspace(
        self,
        *,
        model: str | None = None,
        embedding: str | None = None,
        force: bool,
    ) -> GraphRAGWorkspaceInitResult:
        graph_config = resolve_graph_config(self.config)
        model_name = model or graph_config.model
        embedding_model = embedding or graph_config.embedding_model
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        result = self.command_service.init_workspace(
            model=model_name,
            embedding=embedding_model,
            force=force,
        )
        self.sync_settings(
            GraphRAGRuntimeConfig(
                provider=graph_config.provider,
                model=model_name,
                embedding_provider=graph_config.embedding_provider,
                embedding_model=embedding_model,
                api_key_env=graph_config.api_key_env,
                embedding_api_key_env=graph_config.embedding_api_key_env,
            )
        )
        return GraphRAGWorkspaceInitResult(
            workspace_dir=self.workspace_dir,
            settings_path=self.settings_path,
            result=result,
            provider=graph_config.provider,
            model=model_name,
            embedding_provider=graph_config.embedding_provider,
            embedding_model=embedding_model,
            api_key_env=graph_config.api_key_env,
            embedding_api_key_env=graph_config.embedding_api_key_env,
        )

    def sync_settings(self, graph_config: GraphRAGRuntimeConfig | None = None) -> None:
        graph_config = graph_config or resolve_graph_config(self.config)
        settings = self._load_settings()
        completion_models = settings.setdefault("completion_models", {})
        completion = completion_models.setdefault("default_completion_model", {})
        completion["model_provider"] = graph_config.provider
        completion["model"] = graph_config.model
        completion["auth_method"] = "api_key"
        completion["api_key"] = f"${{{graph_config.api_key_env}}}"

        embedding_models = settings.setdefault("embedding_models", {})
        embedding = embedding_models.setdefault("default_embedding_model", {})
        embedding["model_provider"] = graph_config.embedding_provider
        embedding["model"] = graph_config.embedding_model
        embedding["auth_method"] = "api_key"
        embedding["api_key"] = f"${{{graph_config.embedding_api_key_env}}}"

        atomic_write_text(
            self.settings_path,
            yaml.safe_dump(settings, sort_keys=False),
        )

    def _load_settings(self) -> dict[str, Any]:
        if not self.settings_path.exists():
            return {}
        payload = yaml.safe_load(self.settings_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            return {}
        return payload
