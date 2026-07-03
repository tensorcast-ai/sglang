"""Local resource provider.

This provider treats the driver host itself as a benchmark worker. It is
intended for single-node smoke runs on machines such as an 8xH800 host. The
service layer still talks through the normal Worker protocol, so SGLang,
Tensorcast, gateway wrappers, and workload code do not need local special cases.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from tensorcast_benchmark.kv.tc_router.resource.base import (
    ClusterConfig,
    RemoteProcess,
    WorkerConfig,
    load_cluster_config,
)


class LocalProcessError(RuntimeError):
    """Raised when a local command exits non-zero with `check=True`."""


@dataclass(frozen=True)
class LocalCompletedProcess:
    """Completed local command result matching the RemoteProcess protocol."""

    completed_pid: int | None
    completed_returncode: int
    completed_stdout: str
    completed_stderr: str

    @property
    def pid(self) -> int | None:
        return self.completed_pid

    @property
    def returncode(self) -> int | None:
        return self.completed_returncode

    @property
    def stdout(self) -> str:
        return self.completed_stdout

    @property
    def stderr(self) -> str:
        return self.completed_stderr

    async def wait(self) -> int:
        return self.completed_returncode

    async def kill(self) -> None:
        return None


@dataclass
class LocalWorker:
    """A Worker implementation backed by local subprocesses."""

    config: WorkerConfig
    _background: dict[str, tuple[subprocess.Popen, IO[bytes]]] = field(
        default_factory=dict
    )

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def address(self) -> str:
        return self.config.address

    @property
    def node(self) -> str:
        return self.config.node

    @property
    def process_handle(self) -> str:
        return self.config.process_handle

    @property
    def gpu_indices(self) -> tuple[int, ...]:
        return self.config.gpu_indices

    @property
    def scratch_dir(self) -> str:
        return self.config.scratch_dir

    @property
    def base_env(self) -> dict[str, str]:
        return dict(self.config.base_env)

    def _merged_env(self, env: dict[str, str] | None) -> dict[str, str]:
        merged = os.environ.copy()
        merged.update(self.config.base_env)
        self._apply_env_path_prepend(merged)
        if env is not None:
            merged.update(env)
        self._apply_env_unset(merged)
        return merged

    def _apply_env_path_prepend(self, env: dict[str, str]) -> None:
        for key, paths in self.config.env_path_prepend.items():
            current = env.get(key, "")
            values = [path for path in paths if path]
            if current:
                values.append(current)
            env[key] = os.pathsep.join(values)

    def _apply_env_unset(self, env: dict[str, str]) -> None:
        for key in self.config.env_unset:
            env.pop(key, None)

    def _argv(self, cmd: list[str] | str) -> list[str]:
        if isinstance(cmd, str):
            return ["bash", "-lc", cmd]
        return list(cmd)

    async def run(
        self,
        cmd: list[str] | str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_s: float | None = None,
        check: bool = True,
        as_user: bool = True,
    ) -> RemoteProcess:
        """Run a local command and return its completed process result.

        `as_user` is accepted for protocol compatibility. Local execution
        already runs as the current user.
        """
        del as_user

        proc = await asyncio.create_subprocess_exec(
            *self._argv(cmd),
            cwd=cwd,
            env=self._merged_env(env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            self._kill_process_group(proc.pid, signal.SIGKILL)
            with suppress(ProcessLookupError):
                await proc.wait()
            raise TimeoutError(f"local command timed out after {timeout_s}s") from None

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        returncode = int(proc.returncode or 0)
        completed = LocalCompletedProcess(
            completed_pid=proc.pid,
            completed_returncode=returncode,
            completed_stdout=stdout,
            completed_stderr=stderr,
        )
        if check and returncode != 0:
            raise LocalProcessError(
                f"local command exited with rc={returncode}: stderr={stderr[-500:]}"
            )
        return completed

    async def start_background(
        self,
        cmd: str,
        *,
        name: str,
        log_path: str,
        pid_path: str,
        env: dict[str, str] | None = None,
    ) -> int:
        """Start a local background process and manage it with a PID file."""
        del name

        log_file = Path(log_path)
        pid_file = Path(pid_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.parent.mkdir(parents=True, exist_ok=True)

        log_fh = log_file.open("ab")
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=self._merged_env(env),
            start_new_session=True,
        )
        pid_file.write_text(str(proc.pid), encoding="utf-8")
        self._background[str(pid_file)] = (proc, log_fh)
        return int(proc.pid)

    async def stop_background(self, *, pid_path: str) -> None:
        """Stop a process previously launched with `start_background`."""
        pid_file = Path(pid_path)
        if not pid_file.exists():
            return

        pid_text = pid_file.read_text(encoding="utf-8").strip()
        if not pid_text:
            pid_file.unlink(missing_ok=True)
            return
        pid = int(pid_text)

        proc_entry = self._background.pop(str(pid_file), None)
        if proc_entry is not None:
            proc, log_fh = proc_entry
            await self._terminate_popen(proc)
            log_fh.close()
        else:
            self._kill_process_group(pid, signal.SIGTERM)
            exited = await self._wait_pid_exit(pid, timeout_s=10.0)
            if not exited:
                self._kill_process_group(pid, signal.SIGKILL)
                await self._wait_pid_exit(pid, timeout_s=5.0)

        pid_file.unlink(missing_ok=True)

    async def read_file(
        self, remote_path: str, *, max_bytes: int | None = None
    ) -> bytes:
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        with Path(remote_path).open("rb") as fh:
            if max_bytes is None:
                return fh.read()
            return fh.read(max_bytes)

    async def put_file(self, local: str | Path, remote: str) -> None:
        src = Path(local)
        dst = Path(remote)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == dst.resolve():
            return
        shutil.copy2(src, dst)

    async def get_file(self, remote: str, local: str | Path) -> None:
        src = Path(remote)
        dst = Path(local)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == dst.resolve():
            return
        shutil.copy2(src, dst)

    async def _terminate_popen(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        self._kill_process_group(int(proc.pid), signal.SIGTERM)
        try:
            await asyncio.to_thread(proc.wait, timeout=10.0)
            return
        except subprocess.TimeoutExpired:
            pass
        self._kill_process_group(int(proc.pid), signal.SIGKILL)
        with suppress(subprocess.TimeoutExpired):
            await asyncio.to_thread(proc.wait, timeout=5.0)

    def _kill_process_group(self, pid: int, sig: signal.Signals) -> None:
        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(pid), sig)

    def _pid_exists(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    async def _wait_pid_exit(self, pid: int, *, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._pid_exists(pid):
                return True
            await asyncio.sleep(0.05)
        return not self._pid_exists(pid)


class LocalProvider:
    """ResourceProvider that exposes local configured workers."""

    def __init__(self, cluster_config: ClusterConfig) -> None:
        self._config = cluster_config
        self._workers: list[LocalWorker] | None = None

    @classmethod
    def from_cluster_config(cls, path: str | Path) -> "LocalProvider":
        return cls(load_cluster_config(path))

    def workers(self) -> list[LocalWorker]:
        if self._workers is None:
            self._workers = [LocalWorker(w) for w in self._config.workers]
        return list(self._workers)

    async def health_check(self) -> None:
        Path(self._config.mount.path).mkdir(parents=True, exist_ok=True)
        Path(self._config.driver_host.scratch_dir).mkdir(parents=True, exist_ok=True)
        for worker in self.workers():
            Path(worker.scratch_dir).mkdir(parents=True, exist_ok=True)
            await worker.run(["true"], check=True)
