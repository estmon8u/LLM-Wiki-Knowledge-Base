"""Project-managed GraphRAG workspace setup and settings synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
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
    """Stores graph ragworkspace init result data.

    Attributes:
        See annotated class attributes for stored values.
    """

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
    """Coordinates graph ragworkspace operations.

    Attributes:
        See annotated class attributes for stored values.
    """

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
        """Is initialized.

        Returns:
            bool produced by the operation.
        """
        return self.settings_path.exists()

    def ensure_workspace(self) -> list[str]:
        """Ensure workspace.

        Returns:
            list[str] produced by the operation.
        """
        created: list[str] = []
        for directory in (
            self.workspace_dir,
            self.workspace_dir / "input",
            self.workspace_dir / "prompts",
        ):
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created.append(directory.relative_to(self.paths.root).as_posix())
        if not self.settings_path.exists():
            atomic_write_text(
                self.settings_path,
                yaml.safe_dump(_default_settings(), sort_keys=False),
            )
            created.append(self.settings_path.relative_to(self.paths.root).as_posix())
        created.extend(self._ensure_prompt_templates())
        self.sync_settings()
        return created

    def init_workspace(
        self,
        *,
        model: str | None = None,
        embedding: str | None = None,
        force: bool,
    ) -> GraphRAGWorkspaceInitResult:
        """Init workspace.

        Args:
            model: Model value used by the operation.
            embedding: Embedding value used by the operation.
            force: Force value used by the operation.

        Returns:
            GraphRAGWorkspaceInitResult produced by the operation.
        """
        graph_config = resolve_graph_config(self.config)
        model_name = model or graph_config.model
        embedding_model = embedding or graph_config.embedding_model
        if force and self.settings_path.exists():
            atomic_write_text(
                self.settings_path,
                yaml.safe_dump(_default_settings(), sort_keys=False),
            )
        created = self.ensure_workspace()
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
        result = GraphRAGCommandResult(
            command=("kb", "internal", "graphrag", "init"),
            cwd=self.paths.root,
            returncode=0,
            stdout="\n".join(created),
            stderr="",
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
        """Sync settings.

        Args:
            graph_config: Graph config value used by the operation.
        """
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

    def _ensure_prompt_templates(self) -> list[str]:
        bundled_prompts = _bundled_prompts_dir()
        prompt_dir = self.workspace_dir / "prompts"
        if not bundled_prompts.exists():
            return []
        created: list[str] = []
        for source in sorted(bundled_prompts.glob("*.txt")):
            target = prompt_dir / source.name
            if target.exists():
                continue
            if source.resolve() == target.resolve():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            created.append(target.relative_to(self.paths.root).as_posix())
        return created


def _default_settings() -> dict[str, Any]:
    return {
        "input": {
            "type": "json",
            "encoding": "utf-8",
            "file_pattern": ".*\\.json\\Z",
            "id_column": "id",
            "title_column": "title",
            "text_column": "text",
        },
        "input_storage": {
            "type": "file",
            "base_dir": "input",
        },
        "chunking": {
            "type": "tokens",
            "size": 1200,
            "overlap": 100,
            "encoding_model": "o200k_base",
        },
        "output_storage": {
            "type": "file",
            "base_dir": "output",
        },
        "reporting": {
            "type": "file",
            "base_dir": "logs",
        },
        "cache": {
            "type": "json",
            "storage": {
                "type": "file",
                "base_dir": "cache",
            },
        },
        "vector_store": {
            "type": "lancedb",
            "db_uri": "output/lancedb",
        },
        "embed_text": {
            "embedding_model_id": "default_embedding_model",
        },
        "extract_graph": {
            "completion_model_id": "default_completion_model",
            "prompt": "prompts/extract_graph.txt",
            "entity_types": ["organization", "person", "geo", "event"],
            "max_gleanings": 1,
        },
        "summarize_descriptions": {
            "completion_model_id": "default_completion_model",
            "prompt": "prompts/summarize_descriptions.txt",
            "max_length": 500,
        },
        "extract_graph_nlp": {
            "text_analyzer": {
                "extractor_type": "regex_english",
            },
        },
        "cluster_graph": {
            "max_cluster_size": 10,
        },
        "extract_claims": {
            "enabled": False,
            "completion_model_id": "default_completion_model",
            "prompt": "prompts/extract_claims.txt",
            "description": "Any claims or facts that could be relevant to information discovery.",
            "max_gleanings": 1,
        },
        "community_reports": {
            "completion_model_id": "default_completion_model",
            "graph_prompt": "prompts/community_report_graph.txt",
            "text_prompt": "prompts/community_report_text.txt",
            "max_length": 2000,
            "max_input_length": 8000,
        },
        "snapshots": {
            "graphml": False,
            "embeddings": False,
        },
        "local_search": {
            "completion_model_id": "default_completion_model",
            "embedding_model_id": "default_embedding_model",
            "prompt": "prompts/local_search_system_prompt.txt",
        },
        "global_search": {
            "completion_model_id": "default_completion_model",
            "map_prompt": "prompts/global_search_map_system_prompt.txt",
            "reduce_prompt": "prompts/global_search_reduce_system_prompt.txt",
            "knowledge_prompt": "prompts/global_search_knowledge_system_prompt.txt",
        },
        "drift_search": {
            "completion_model_id": "default_completion_model",
            "embedding_model_id": "default_embedding_model",
            "prompt": "prompts/drift_search_system_prompt.txt",
            "reduce_prompt": "prompts/drift_search_reduce_prompt.txt",
        },
        "basic_search": {
            "completion_model_id": "default_completion_model",
            "embedding_model_id": "default_embedding_model",
            "prompt": "prompts/basic_search_system_prompt.txt",
        },
    }


def _bundled_prompts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "graph" / "graphrag" / "prompts"
