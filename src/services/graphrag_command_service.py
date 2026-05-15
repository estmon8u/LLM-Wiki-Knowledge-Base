"""Graphrag command service service behavior for the knowledge-base workflow.

This module belongs to `src.services.graphrag_command_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
from typing import Any, Callable, Sequence

from src.services.project_service import ProjectPaths


Runner = Callable[..., subprocess.CompletedProcess[str]]
StatusCallback = Callable[[str], None] | None
MAX_PROGRESS_LABEL_LENGTH = 120


@dataclass(frozen=True)
class GraphRAGCommandResult:
    """Stores graph ragcommand result data.

    Attributes:
        See annotated class attributes for stored values.
    """

    command: tuple[str, ...]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str


class GraphRAGCommandError(RuntimeError):
    """Error raised for graph ragcommand failures.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(self, message: str, *, result: GraphRAGCommandResult | None = None):
        super().__init__(message)
        self.result = result


class GraphRAGCommandService:
    """Coordinates graph ragcommand operations.

    Attributes:
        See annotated class attributes for stored values.
    """

    @staticmethod
    def _default_runner(
        command: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        kwargs.setdefault("encoding", "utf-8")
        return subprocess.run(command, **kwargs)

    def __init__(
        self,
        paths: ProjectPaths,
        *,
        runner: Runner | None = None,
    ) -> None:
        self.paths = paths
        self.runner = runner or self._default_runner
        self.workspace_dir = paths.graph_dir / "graphrag"

    def init_workspace(
        self,
        *,
        model: str,
        embedding: str,
        force: bool,
    ) -> GraphRAGCommandResult:
        """Init workspace.

        Args:
            model: Model value used by the operation.
            embedding: Embedding value used by the operation.
            force: Force value used by the operation.

        Returns:
            GraphRAGCommandResult produced by the operation.
        """
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
        """Index.

        Args:
            method: Method value used by the operation.
            dry_run: Dry run value used by the operation.
            cache: Cache value used by the operation.
            skip_validation: Skip validation value used by the operation.
            verbose: Whether to emit verbose command output.
            status_callback: Status callback value used by the operation.

        Returns:
            GraphRAGCommandResult produced by the operation.
        """
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
        data_dir: Path | None = None,
        community_level: int | None = None,
        dynamic_community_selection: bool | None = None,
        response_type: str | None = None,
        streaming: bool | None = None,
        verbose: bool = False,
    ) -> GraphRAGCommandResult:
        """Query.

        Args:
            question: User question to answer from available evidence.
            method: Method value used by the operation.
            data_dir: Active GraphRAG output directory containing parquet files.
            community_level: Community level value used by the operation.
            dynamic_community_selection: Dynamic community selection value used by the operation.
            response_type: Response type value used by the operation.
            streaming: GraphRAG streaming flag forwarded to the query CLI.
            verbose: Whether to emit verbose command output.

        Returns:
            GraphRAGCommandResult produced by the operation.
        """
        args = [
            "query",
            "--root",
            str(self.workspace_dir),
            "--method",
            method,
        ]
        if data_dir is not None:
            args.extend(["--data", str(data_dir)])
        if community_level is not None:
            args.extend(["--community-level", str(community_level)])
        if dynamic_community_selection is True:
            args.append("--dynamic-community-selection")
        elif dynamic_community_selection is False:
            args.append("--no-dynamic-selection")
        if response_type:
            args.extend(["--response-type", response_type])
        if streaming is True:
            args.append("--streaming")
        elif streaming is False:
            args.append("--no-streaming")
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
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            proc = subprocess.Popen(
                command,
                cwd=self.paths.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
        except FileNotFoundError as exc:
            raise GraphRAGCommandError(
                "Unable to run GraphRAG because the Python executable was not found."
            ) from exc

        stderr_lines: list[str] = []
        stdout_lines: list[str] = []
        status_lines: Queue[str] = Queue()
        threads: list[threading.Thread] = []

        def read_stream(stream: Any, sink: list[str]) -> None:
            """Read stream.

            Args:
                stream: Stream value used by the operation.
                sink: Sink value used by the operation.
            """
            try:
                for raw_line in iter(stream.readline, ""):
                    sink.append(raw_line)
                    status_lines.put(raw_line)
            finally:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()

        try:
            status_callback("starting graph index")
            if proc.stdout is not None:
                stdout_thread = threading.Thread(
                    target=read_stream,
                    args=(proc.stdout, stdout_lines),
                    daemon=True,
                )
                threads.append(stdout_thread)
                stdout_thread.start()
            if proc.stderr is not None:
                stderr_thread = threading.Thread(
                    target=read_stream,
                    args=(proc.stderr, stderr_lines),
                    daemon=True,
                )
                threads.append(stderr_thread)
                stderr_thread.start()

            while (
                any(thread.is_alive() for thread in threads) or not status_lines.empty()
            ):
                try:
                    raw_line = status_lines.get(timeout=0.1)
                except Empty:
                    continue
                line = raw_line.rstrip()
                if line:
                    display = _extract_progress_label(line)
                    if display:
                        status_callback(display)
        finally:
            proc.wait()
            for thread in threads:
                thread.join()

        stdout_data = "".join(stdout_lines)
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
    if len(stripped) > MAX_PROGRESS_LABEL_LENGTH:
        return stripped[: MAX_PROGRESS_LABEL_LENGTH - 3] + "..."
    return stripped


def _is_known_graphrag_dry_run_logging_error(stderr: str) -> bool:
    stripped = stderr.strip()
    normalized = " ".join(stripped.split())
    return (
        stripped.startswith("--- Logging error ---")
        and "Dry run complete, exiting..." in stripped
        and "Arguments: (True,)" in stripped
        and ("logger.info" in normalized or "logging" in normalized.casefold())
    )
