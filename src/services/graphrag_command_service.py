from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Sequence

from src.services.project_service import ProjectPaths


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class GraphRAGCommandResult:
    command: tuple[str, ...]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str


class GraphRAGCommandError(RuntimeError):
    def __init__(self, message: str, *, result: GraphRAGCommandResult | None = None):
        super().__init__(message)
        self.result = result


class GraphRAGCommandService:
    def __init__(
        self,
        paths: ProjectPaths,
        *,
        runner: Runner | None = None,
    ) -> None:
        self.paths = paths
        self.runner = runner or subprocess.run
        self.workspace_dir = paths.graph_dir / "graphrag"

    def init_workspace(
        self,
        *,
        model: str,
        embedding: str,
        force: bool,
    ) -> GraphRAGCommandResult:
        args = [
            "init",
            "--root",
            str(self.workspace_dir),
            "--model",
            model,
            "--embedding",
            embedding,
        ]
        if force:
            args.append("--force")
        return self._run(args)

    def index(
        self,
        *,
        method: str,
        dry_run: bool,
        cache: bool,
        skip_validation: bool,
        verbose: bool = False,
    ) -> GraphRAGCommandResult:
        args = [
            "index",
            "--root",
            str(self.workspace_dir),
            "--method",
            method,
        ]
        if dry_run:
            args.append("--dry-run")
        if not cache:
            args.append("--no-cache")
        if skip_validation:
            args.append("--skip-validation")
        if verbose:
            args.append("--verbose")
        return self._run(args)

    def query(
        self,
        question: str,
        *,
        method: str,
        community_level: int | None = None,
        dynamic_community_selection: bool | None = None,
        response_type: str | None = None,
        verbose: bool = False,
    ) -> GraphRAGCommandResult:
        args = [
            "query",
            "--root",
            str(self.workspace_dir),
            "--method",
            method,
        ]
        if community_level is not None:
            args.extend(["--community-level", str(community_level)])
        if dynamic_community_selection is True:
            args.append("--dynamic-community-selection")
        elif dynamic_community_selection is False:
            args.append("--no-dynamic-selection")
        if response_type:
            args.extend(["--response-type", response_type])
        if verbose:
            args.append("--verbose")
        args.append(question)
        return self._run(args)

    def _run(self, args: Sequence[str]) -> GraphRAGCommandResult:
        command = (sys.executable, "-m", "graphrag", *args)
        try:
            completed = self.runner(
                command,
                cwd=self.paths.root,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise GraphRAGCommandError(
                "Unable to run GraphRAG because the Python executable was not found."
            ) from exc

        result = self._to_result(command, completed)
        if result.returncode != 0:
            detail = _last_non_empty_line(result.stderr) or _last_non_empty_line(
                result.stdout
            )
            message = "GraphRAG command failed"
            if detail:
                message = f"{message}: {detail}"
            raise GraphRAGCommandError(message, result=result)
        return result

    def _to_result(
        self,
        command: tuple[str, ...],
        completed: Any,
    ) -> GraphRAGCommandResult:
        return GraphRAGCommandResult(
            command=command,
            cwd=self.paths.root,
            returncode=int(completed.returncode),
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )


def _last_non_empty_line(value: str) -> str:
    for line in reversed(value.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""
