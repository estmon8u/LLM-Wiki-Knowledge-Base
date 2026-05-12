from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Sequence

from src.services.project_service import ProjectPaths


Runner = Callable[..., subprocess.CompletedProcess[str]]
StatusCallback = Callable[[str], None] | None


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
        status_callback: StatusCallback = None,
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
        if status_callback is not None:
            return self._run_streaming(args, status_callback)
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

    def _run_streaming(
        self, args: Sequence[str], status_callback: Callable[[str], None]
    ) -> GraphRAGCommandResult:
        """Run a GraphRAG command while streaming stderr to *status_callback*."""
        command = (sys.executable, "-m", "graphrag", *args)
        try:
            proc = subprocess.Popen(
                command,
                cwd=self.paths.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise GraphRAGCommandError(
                "Unable to run GraphRAG because the Python executable was not found."
            ) from exc

        stderr_lines: list[str] = []
        stdout_data = ""
        try:
            assert proc.stderr is not None  # guaranteed by PIPE
            for raw_line in proc.stderr:
                line = raw_line.rstrip()
                stderr_lines.append(raw_line)
                if line:
                    display = _extract_progress_label(line)
                    if display:
                        status_callback(display)
            if proc.stdout is not None:
                stdout_data = proc.stdout.read()
        finally:
            proc.wait()

        stderr_text = "".join(stderr_lines)
        completed = subprocess.CompletedProcess(
            command, proc.returncode, stdout=stdout_data, stderr=stderr_text
        )
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
            stderr=_sanitize_stderr(command, completed.stderr or ""),
        )


def _last_non_empty_line(value: str) -> str:
    for line in reversed(value.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _sanitize_stderr(command: tuple[str, ...], stderr: str) -> str:
    if "--dry-run" not in command:
        return stderr
    if _is_known_graphrag_dry_run_logging_error(stderr):
        return ""
    return stderr


def _extract_progress_label(line: str) -> str:
    """Extract a human-readable status fragment from a GraphRAG log line."""
    # GraphRAG logs lines like:
    #   "graphrag.index ... Running workflow: extract_graph"
    #   "graphrag.index ... 50% complete"
    #   "Running ... verb: extract_graph"
    lowered = line.lower()
    # Skip noisy debug lines
    if any(skip in lowered for skip in ("debug", "traceback", "warning:")):
        return ""
    # Try to extract workflow step name
    for marker in ("running workflow:", "running verb:", "running step:"):
        idx = lowered.find(marker)
        if idx >= 0:
            return line[idx:].strip()
    # Show percentage lines directly
    if "%" in line and any(c.isdigit() for c in line):
        return line.strip()
    # Pass through other substantive lines (but cap length)
    stripped = line.strip()
    if len(stripped) > 120:
        return stripped[:117] + "..."
    return stripped


def _is_known_graphrag_dry_run_logging_error(stderr: str) -> bool:
    stripped = stderr.strip()
    return (
        stripped.startswith("--- Logging error ---")
        and 'logger.info("Dry run complete, exiting...", True)' in stripped
        and "Message: 'Dry run complete, exiting...'" in stripped
        and "Arguments: (True,)" in stripped
    )
