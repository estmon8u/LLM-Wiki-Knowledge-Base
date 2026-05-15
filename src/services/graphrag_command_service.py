"""GraphRAG Python entrypoint adapter for indexing and querying."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
from pathlib import Path
import subprocess
from typing import Any, Callable, Sequence

from src.services.project_service import ProjectPaths


Runner = Callable[..., subprocess.CompletedProcess[str]]
StatusCallback = Callable[[str], None] | None
MAX_PROGRESS_LABEL_LENGTH = 120
DEFAULT_QUERY_RESPONSE_TYPE = "Multiple Paragraphs"
DEFAULT_QUERY_COMMUNITY_LEVEL = 2


@dataclass(frozen=True)
class GraphRAGCommandResult:
    """Result metadata kept for GraphRAG run history and diagnostics."""

    command: tuple[str, ...]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str


class GraphRAGCommandError(RuntimeError):
    """Error raised for GraphRAG library API failures."""

    def __init__(self, message: str, *, result: GraphRAGCommandResult | None = None):
        super().__init__(message)
        self.result = result


class GraphRAGCommandService:
    """Runs GraphRAG init, index, and query without spawning `python -m graphrag`."""

    @staticmethod
    def _default_runner(
        command: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        """Compatibility helper retained for older subprocess-focused tests."""
        kwargs.setdefault("encoding", "utf-8")
        return subprocess.run(command, **kwargs)

    def __init__(
        self,
        paths: ProjectPaths,
        *,
        runner: Runner | None = None,
        api_backend: "GraphRAGApiBackend | None" = None,
    ) -> None:
        self.paths = paths
        self.runner = runner or self._default_runner
        if api_backend is not None:
            self.api_backend = api_backend
        elif runner is not None:
            self.api_backend = _RunnerGraphRAGApiBackend(paths, runner)
        else:
            self.api_backend = GraphRAGApiBackend(paths)
        self.workspace_dir = paths.graph_dir / "graphrag"

    def init_workspace(
        self,
        *,
        model: str,
        embedding: str,
        force: bool,
    ) -> GraphRAGCommandResult:
        """Initialize the GraphRAG workspace through GraphRAG's Python helper."""
        result = self.api_backend.init_workspace(
            workspace_dir=self.workspace_dir,
            model=model,
            embedding=embedding,
            force=force,
        )
        self._raise_if_failed(result)
        return result

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
        """Run GraphRAG indexing through installed Python entrypoints."""
        result = self.api_backend.index(
            workspace_dir=self.workspace_dir,
            method=method,
            dry_run=dry_run,
            cache=cache,
            skip_validation=skip_validation,
            verbose=verbose,
            status_callback=status_callback,
        )
        self._raise_if_failed(result)
        return result

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
        """Run a GraphRAG search query through installed Python entrypoints."""
        result = self.api_backend.query(
            workspace_dir=self.workspace_dir,
            question=question,
            method=method,
            data_dir=data_dir,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            streaming=streaming,
            verbose=verbose,
        )
        self._raise_if_failed(result)
        return result

    def _raise_if_failed(self, result: GraphRAGCommandResult) -> None:
        if result.returncode == 0:
            return
        detail = _last_non_empty_line(result.stderr) or _last_non_empty_line(
            result.stdout
        )
        message = "GraphRAG command failed"
        if detail:
            message = f"{message}: {detail}"
        raise GraphRAGCommandError(message, result=result)


class GraphRAGApiBackend:
    """Small adapter over GraphRAG's installed Python command entrypoints."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def init_workspace(
        self,
        *,
        workspace_dir: Path,
        model: str,
        embedding: str,
        force: bool,
    ) -> GraphRAGCommandResult:  # pragma: no cover - exercised by GraphRAG package
        command = _init_command(workspace_dir, model, embedding, force)
        try:
            from graphrag.cli.initialize import initialize_project_at

            initialize_project_at(
                workspace_dir,
                force=force,
                model=model,
                embedding_model=embedding,
            )
        except Exception as exc:  # noqa: BLE001
            return _api_result(command, self.paths.root, returncode=1, stderr=str(exc))
        return _api_result(
            command,
            self.paths.root,
            stdout=f"Initialized GraphRAG workspace at {workspace_dir}\n",
        )

    def index(
        self,
        *,
        workspace_dir: Path,
        method: str,
        dry_run: bool,
        cache: bool,
        skip_validation: bool,
        verbose: bool,
        status_callback: StatusCallback,
    ) -> GraphRAGCommandResult:  # pragma: no cover - exercised by GraphRAG package
        command = _index_command(
            workspace_dir,
            method=method,
            dry_run=dry_run,
            cache=cache,
            skip_validation=skip_validation,
            verbose=verbose,
        )
        if status_callback is not None:
            status_callback("starting graph index")
        stdout = _ProgressCapture(status_callback)
        stderr = _ProgressCapture(status_callback)
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                _run_index_entrypoint(
                    workspace_dir=workspace_dir,
                    method=method,
                    dry_run=dry_run,
                    cache=cache,
                    skip_validation=skip_validation,
                    verbose=verbose,
                )
        except SystemExit as exc:
            stdout.flush()
            stderr.flush()
            returncode = _system_exit_code(exc)
            stderr_text = stderr.getvalue()
            if returncode and not stderr_text:
                stderr_text = str(exc)
            return _api_result(
                command,
                self.paths.root,
                returncode=returncode,
                stdout=stdout.getvalue(),
                stderr=stderr_text,
            )
        except Exception as exc:  # noqa: BLE001
            stdout.flush()
            stderr.flush()
            stderr_text = _append_error(stderr.getvalue(), str(exc))
            return _api_result(
                command,
                self.paths.root,
                returncode=1,
                stdout=stdout.getvalue(),
                stderr=stderr_text,
            )
        stdout.flush()
        stderr.flush()
        return _api_result(
            command,
            self.paths.root,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
        )

    def query(
        self,
        *,
        workspace_dir: Path,
        question: str,
        method: str,
        data_dir: Path | None,
        community_level: int | None,
        dynamic_community_selection: bool | None,
        response_type: str | None,
        streaming: bool | None,
        verbose: bool,
    ) -> GraphRAGCommandResult:  # pragma: no cover - exercised by GraphRAG package
        command = _query_command(
            workspace_dir,
            question,
            method=method,
            data_dir=data_dir,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            streaming=streaming,
            verbose=verbose,
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                _run_query_entrypoint(
                    workspace_dir=workspace_dir,
                    data_dir=data_dir,
                    method=method,
                    community_level=community_level,
                    dynamic_community_selection=dynamic_community_selection,
                    response_type=response_type,
                    streaming=bool(streaming),
                    question=question,
                    verbose=verbose,
                )
        except Exception as exc:  # noqa: BLE001
            stderr_text = _append_error(stderr.getvalue(), str(exc))
            return _api_result(
                command,
                self.paths.root,
                returncode=1,
                stdout=stdout.getvalue(),
                stderr=stderr_text,
            )
        return _api_result(
            command,
            self.paths.root,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
        )


class _RunnerGraphRAGApiBackend:
    """Compatibility backend for tests that inject a fake process runner."""

    def __init__(self, paths: ProjectPaths, runner: Runner) -> None:
        self.paths = paths
        self.runner = runner

    def init_workspace(
        self,
        *,
        workspace_dir: Path,
        model: str,
        embedding: str,
        force: bool,
    ) -> GraphRAGCommandResult:
        command = _init_command(workspace_dir, model, embedding, force)
        completed = self.runner(
            command,
            cwd=self.paths.root,
            capture_output=True,
            text=True,
        )
        return _completed_process_result(command, self.paths.root, completed)

    def index(
        self,
        *,
        workspace_dir: Path,
        method: str,
        dry_run: bool,
        cache: bool,
        skip_validation: bool,
        verbose: bool,
        status_callback: StatusCallback,
    ) -> GraphRAGCommandResult:
        command = _index_command(
            workspace_dir,
            method=method,
            dry_run=dry_run,
            cache=cache,
            skip_validation=skip_validation,
            verbose=verbose,
        )
        if status_callback is not None:
            status_callback("starting graph index")
        completed = self.runner(
            command,
            cwd=self.paths.root,
            capture_output=True,
            text=True,
        )
        _emit_progress_lines(completed.stdout or "", status_callback)
        _emit_progress_lines(completed.stderr or "", status_callback)
        return _completed_process_result(command, self.paths.root, completed)

    def query(
        self,
        *,
        workspace_dir: Path,
        question: str,
        method: str,
        data_dir: Path | None,
        community_level: int | None,
        dynamic_community_selection: bool | None,
        response_type: str | None,
        streaming: bool | None,
        verbose: bool,
    ) -> GraphRAGCommandResult:
        command = _query_command(
            workspace_dir,
            question,
            method=method,
            data_dir=data_dir,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            streaming=streaming,
            verbose=verbose,
        )
        completed = self.runner(
            command,
            cwd=self.paths.root,
            capture_output=True,
            text=True,
        )
        return _completed_process_result(command, self.paths.root, completed)


def _run_index_entrypoint(
    *,
    workspace_dir: Path,
    method: str,
    dry_run: bool,
    cache: bool,
    skip_validation: bool,
    verbose: bool,
) -> None:  # pragma: no cover - thin wrapper around GraphRAG package
    from graphrag.cli.index import index_cli, update_cli
    from graphrag.config.enums import IndexingMethod

    index_method, is_update_run = _split_index_method(method)
    enum_method = IndexingMethod(index_method)
    if is_update_run:
        update_cli(
            root_dir=workspace_dir,
            method=enum_method,
            verbose=verbose,
            cache=cache,
            skip_validation=skip_validation,
        )
        return
    index_cli(
        root_dir=workspace_dir,
        method=enum_method,
        verbose=verbose,
        cache=cache,
        dry_run=dry_run,
        skip_validation=skip_validation,
    )


def _run_query_entrypoint(
    *,
    workspace_dir: Path,
    data_dir: Path | None,
    method: str,
    community_level: int | None,
    dynamic_community_selection: bool | None,
    response_type: str | None,
    streaming: bool,
    question: str,
    verbose: bool,
) -> None:  # pragma: no cover - thin wrapper around GraphRAG package
    from graphrag.cli.query import (
        run_basic_search,
        run_drift_search,
        run_global_search,
        run_local_search,
    )

    response_type = response_type or DEFAULT_QUERY_RESPONSE_TYPE
    community_level = community_level or DEFAULT_QUERY_COMMUNITY_LEVEL
    dynamic_community_selection = bool(dynamic_community_selection)
    if method == "global":
        run_global_search(
            data_dir=data_dir,
            root_dir=workspace_dir,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            streaming=streaming,
            query=question,
            verbose=verbose,
        )
    elif method == "local":
        run_local_search(
            data_dir=data_dir,
            root_dir=workspace_dir,
            community_level=community_level,
            response_type=response_type,
            streaming=streaming,
            query=question,
            verbose=verbose,
        )
    elif method == "drift":
        run_drift_search(
            data_dir=data_dir,
            root_dir=workspace_dir,
            community_level=community_level,
            response_type=response_type,
            streaming=streaming,
            query=question,
            verbose=verbose,
        )
    elif method == "basic":
        run_basic_search(
            data_dir=data_dir,
            root_dir=workspace_dir,
            response_type=response_type,
            streaming=streaming,
            query=question,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Invalid GraphRAG query method: {method}")


def _api_result(
    command: tuple[str, ...],
    cwd: Path,
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> GraphRAGCommandResult:
    return GraphRAGCommandResult(
        command=command,
        cwd=cwd,
        returncode=returncode,
        stdout=stdout,
        stderr=_sanitize_stderr(command, stderr),
    )


def _completed_process_result(
    command: tuple[str, ...],
    cwd: Path,
    completed: subprocess.CompletedProcess[str],
) -> GraphRAGCommandResult:
    return _api_result(
        command,
        cwd,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


class _ProgressCapture(io.StringIO):
    """Capture GraphRAG output and surface status-worthy lines as callbacks."""

    def __init__(self, status_callback: StatusCallback = None) -> None:
        super().__init__()
        self.status_callback = status_callback
        self._pending_line = ""

    def write(self, value: str) -> int:
        written = super().write(value)
        if self.status_callback is not None:
            self._pending_line = _emit_progress_chunks(
                self._pending_line,
                value,
                self.status_callback,
            )
        return written

    def flush(self) -> None:
        if self.status_callback is not None and self._pending_line.strip():
            _emit_progress_line(self._pending_line, self.status_callback)
            self._pending_line = ""
        super().flush()


def _emit_progress_lines(value: str, status_callback: StatusCallback) -> None:
    if status_callback is None:
        return
    pending_line = _emit_progress_chunks("", value, status_callback)
    if pending_line.strip():
        _emit_progress_line(pending_line, status_callback)


def _emit_progress_chunks(
    pending_line: str,
    value: str,
    status_callback: Callable[[str], None],
) -> str:
    buffer = pending_line + value.replace("\r", "\n")
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        _emit_progress_line(line, status_callback)
    return buffer


def _emit_progress_line(line: str, status_callback: Callable[[str], None]) -> None:
    label = _extract_progress_label(line)
    if label:
        status_callback(label)


def _append_error(stderr: str, detail: str) -> str:
    if not stderr:
        return detail
    if not detail:
        return stderr
    return f"{stderr.rstrip()}\n{detail}"


def _init_command(
    workspace_dir: Path,
    model: str,
    embedding: str,
    force: bool,
) -> tuple[str, ...]:
    args = [
        "graphrag.api",
        "initialize_project_at",
        "--root",
        str(workspace_dir),
        "--model",
        model,
        "--embedding",
        embedding,
    ]
    if force:
        args.append("--force")
    return tuple(args)


def _index_command(
    workspace_dir: Path,
    *,
    method: str,
    dry_run: bool,
    cache: bool,
    skip_validation: bool,
    verbose: bool,
) -> tuple[str, ...]:
    args = [
        "graphrag.api",
        "build_index",
        "--root",
        str(workspace_dir),
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
    return tuple(args)


def _query_command(
    workspace_dir: Path,
    question: str,
    *,
    method: str,
    data_dir: Path | None,
    community_level: int | None,
    dynamic_community_selection: bool | None,
    response_type: str | None,
    streaming: bool | None,
    verbose: bool,
) -> tuple[str, ...]:
    args = [
        "graphrag.api",
        f"{method}_search",
        "--root",
        str(workspace_dir),
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
    return tuple(args)


def _split_index_method(method: str) -> tuple[str, bool]:
    if method.endswith("-update"):
        return method.removesuffix("-update"), True
    return method, False


def _system_exit_code(exc: SystemExit) -> int:
    code = exc.code
    if isinstance(code, bool):
        return int(code)
    if isinstance(code, int):
        return code
    if code is None:
        return 0
    if isinstance(code, str) and not code:
        return 0
    return 1


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
    lowered = line.lower()
    if any(skip in lowered for skip in ("debug", "traceback", "warning:")):
        return ""
    for marker in ("running workflow:", "running verb:", "running step:"):
        idx = lowered.find(marker)
        if idx >= 0:
            return line[idx:].strip()
    if "%" in line and any(c.isdigit() for c in line):
        return line.strip()
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
