"""Static resource provider backed by local subprocesses and SSH.

StaticProvider adapts an operator-supplied inventory to the same Worker
interface used by the rest of tc_router. It never acquires hosts. Command
execution is either local or SSH; files are shared through /mnt/data, so file
operations are always driver-side filesystem copies.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path

from tensorcast_benchmark.kv.tc_router.resource.base import (
    ClusterConfig,
    RemoteProcess,
    WorkerConfig,
    load_cluster_config,
)


class StaticConfigError(RuntimeError):
    """Raised when a static cluster config is internally inconsistent."""


class StaticHealthCheckError(RuntimeError):
    """Raised when a static worker fails health checks."""


class StaticProcessError(RuntimeError):
    """Raised when a static worker command exits non-zero with `check=True`."""


@dataclass(frozen=True)
class StaticCompletedProcess:
    """Completed command result matching the RemoteProcess protocol."""

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
class BaseStaticWorker:
    """Shared Worker behavior for local and SSH static workers."""

    config: WorkerConfig

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
        raise NotImplementedError

    async def start_background(
        self,
        cmd: str,
        *,
        name: str,
        log_path: str,
        pid_path: str,
        env: dict[str, str] | None = None,
    ) -> int:
        """Start a background service via PID-file contract."""
        del name

        start_script = self._background_start_script(cmd, log_path, pid_path)
        proc = await self.run(
            start_script,
            env=env,
            timeout_s=10.0,
            check=True,
        )
        pid_text = proc.stdout.strip().splitlines()[-1]
        return int(pid_text)

    async def stop_background(self, *, pid_path: str) -> None:
        """Stop a PID-file-managed background service."""
        await self.run(
            self._background_stop_script(pid_path),
            timeout_s=20.0,
            check=True,
        )

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

    def _shell_command(self, cmd: list[str] | str) -> str:
        if isinstance(cmd, str):
            return cmd
        return shlex.join(cmd)

    def _remote_env_exports(self, env: dict[str, str] | None) -> list[str]:
        lines: list[str] = []
        for key, value in self.config.base_env.items():
            lines.append(self._export_assignment(key, value))
        for key, paths in self.config.env_path_prepend.items():
            if paths:
                prefix = ":".join(shlex.quote(path) for path in paths)
                lines.append(f"export {key}={prefix}:${{{key}:-}}")
        if env is not None:
            for key, value in env.items():
                lines.append(self._export_assignment(key, value))
        if self.config.env_unset:
            lines.append("unset " + " ".join(self.config.env_unset))
        return lines

    def _command_script(
        self,
        cmd: list[str] | str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> str:
        command = self._shell_command(cmd)
        if cwd is not None:
            command = f"cd {shlex.quote(cwd)} && {command}"
        env_exports = self._remote_env_exports(env)
        if not env_exports:
            return command
        return "; ".join(env_exports + [command])

    def _background_start_script(self, cmd: str, log_path: str, pid_path: str) -> str:
        log_dir = str(Path(log_path).parent)
        pid_dir = str(Path(pid_path).parent)
        return "\n".join(
            [
                f"mkdir -p {shlex.quote(log_dir)} {shlex.quote(pid_dir)}",
                f"rm -f {shlex.quote(pid_path)}",
                (
                    f"nohup setsid bash --noprofile --norc -lc {shlex.quote(cmd)} "
                    f"> {shlex.quote(log_path)} 2>&1 < /dev/null &"
                ),
                "pid=$!",
                f"printf '%s\\n' \"$pid\" > {shlex.quote(pid_path)}",
                "printf '%s\\n' \"$pid\"",
            ]
        )

    def _background_stop_script(self, pid_path: str) -> str:
        quoted_pid_path = shlex.quote(pid_path)
        return " ".join(
            [
                f"if [ -f {quoted_pid_path} ]; then",
                f"pid=$(cat {quoted_pid_path});",
                'if [ -n "$pid" ]; then',
                'kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true;',
                "for _ in $(seq 1 100); do",
                'kill -0 "$pid" 2>/dev/null || break;',
                "sleep 0.1;",
                "done;",
                'kill -0 "$pid" 2>/dev/null &&',
                'kill -KILL -- "-$pid" 2>/dev/null ||',
                'kill -KILL "$pid" 2>/dev/null || true;',
                "fi;",
                f"rm -f {quoted_pid_path};",
                "fi",
            ]
        )

    def _export_assignment(self, key: str, value: str) -> str:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise StaticConfigError(f"invalid shell environment variable name: {key!r}")
        return f"export {key}={shlex.quote(value)}"


@dataclass
class LocalStaticWorker(BaseStaticWorker):
    """Static worker backed by local subprocesses."""

    def _merged_local_env(self, env: dict[str, str] | None) -> dict[str, str]:
        merged = os.environ.copy()
        merged.update(self.config.base_env)
        for key, paths in self.config.env_path_prepend.items():
            current = merged.get(key, "")
            values = [path for path in paths if path]
            if current:
                values.append(current)
            merged[key] = os.pathsep.join(values)
        if env is not None:
            merged.update(env)
        for key in self.config.env_unset:
            merged.pop(key, None)
        return merged

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
        """Run a local command through `bash -lc`."""
        del as_user

        proc = await asyncio.create_subprocess_exec(
            "bash",
            "--noprofile",
            "--norc",
            "-lc",
            self._shell_command(cmd),
            cwd=cwd,
            env=self._merged_local_env(env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        completed = await _communicate(proc, timeout_s, "local static command")
        if check and completed.returncode != 0:
            raise StaticProcessError(
                f"local static command exited with rc={completed.returncode}: "
                f"stderr={completed.stderr[-500:]}"
            )
        return completed


@dataclass
class SshStaticWorker(BaseStaticWorker):
    """Static worker backed by SSH command execution."""

    def _ssh_target(self) -> str:
        return f"{self.config.ssh_user}@{self.config.ssh_host}"

    def _ssh_argv(self, remote_script: str) -> list[str]:
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"ConnectTimeout={int(self.config.connect_timeout_s)}",
            "-p",
            str(self.config.ssh_port),
        ]
        if self.config.identity_file:
            argv.extend(["-i", self.config.identity_file])
        argv.extend(
            [
                self._ssh_target(),
                "bash",
                "--noprofile",
                "--norc",
                "-lc",
                shlex.quote(remote_script),
            ]
        )
        return argv

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
        """Run a command on the static worker via SSH."""
        del as_user

        remote_script = self._command_script(cmd, env=env, cwd=cwd)
        proc = await asyncio.create_subprocess_exec(
            *self._ssh_argv(remote_script),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        completed = await _communicate(proc, timeout_s, "ssh static command")
        if check and completed.returncode != 0:
            raise StaticProcessError(
                f"ssh static command on {self._ssh_target()} exited with "
                f"rc={completed.returncode}: stderr={completed.stderr[-500:]}"
            )
        return completed


class StaticProvider:
    """ResourceProvider for operator-supplied local + SSH workers."""

    def __init__(self, cluster_config: ClusterConfig) -> None:
        self._config = cluster_config
        self._workers: list[BaseStaticWorker] | None = None
        self._validate_config()

    @classmethod
    def from_cluster_config(cls, path: str | Path) -> "StaticProvider":
        return cls(load_cluster_config(path))

    def workers(self) -> list[BaseStaticWorker]:
        if self._workers is None:
            workers: list[BaseStaticWorker] = []
            for worker_config in self._config.workers:
                if worker_config.execution == "local":
                    workers.append(LocalStaticWorker(worker_config))
                elif worker_config.execution == "ssh":
                    workers.append(SshStaticWorker(worker_config))
                else:
                    raise StaticConfigError(
                        f"worker {worker_config.id!r}: unsupported execution "
                        f"{worker_config.execution!r}"
                    )
            self._workers = workers
        return list(self._workers)

    async def health_check(self) -> None:
        Path(self._config.driver_host.scratch_dir).mkdir(parents=True, exist_ok=True)
        for worker in self.workers():
            await self._health_check_worker(worker)

    def _validate_config(self) -> None:
        if self._config.provider.kind != "static":
            raise StaticConfigError(
                f"StaticProvider requires provider.kind='static', got "
                f"{self._config.provider.kind!r}"
            )

    async def _health_check_worker(self, worker: BaseStaticWorker) -> None:
        await self._checked_run(worker, "hostname", "hostname")
        await self._checked_run(worker, "test -d /mnt/data", "shared /mnt/data")
        home_proc = await self._checked_run(
            worker,
            "readlink -f /home/yuhan",
            "/home/yuhan symlink",
        )
        home_lines = [
            line.strip() for line in home_proc.stdout.splitlines() if line.strip()
        ]
        home_target = home_lines[-1] if home_lines else ""
        if home_target != "/mnt/data":
            raise StaticHealthCheckError(
                f"worker {worker.id}: /home/yuhan resolves to {home_target!r}, "
                "expected '/mnt/data'"
            )

        await self._checked_run(
            worker,
            f"mkdir -p {shlex.quote(worker.scratch_dir)}",
            "scratch_dir mkdir",
        )
        await self._checked_run(
            worker,
            f"test -d {shlex.quote(worker.scratch_dir)}",
            "scratch_dir exists",
        )
        await self._checked_run(
            worker,
            "test -d /mnt/data/tot",
            "repo root exists",
        )
        gpu_proc = await self._checked_run(worker, "nvidia-smi -L", "nvidia-smi")
        gpu_lines = [
            line
            for line in gpu_proc.stdout.splitlines()
            if line.strip().startswith("GPU ")
        ]
        if not gpu_lines:
            raise StaticHealthCheckError(
                f"worker {worker.id}: nvidia-smi reported no visible GPUs"
            )
        highest_index = max(worker.gpu_indices)
        if highest_index >= len(gpu_lines):
            raise StaticHealthCheckError(
                f"worker {worker.id}: gpu_indices={worker.gpu_indices} exceed "
                f"visible GPU count {len(gpu_lines)}"
            )

    async def _checked_run(
        self,
        worker: BaseStaticWorker,
        cmd: str,
        label: str,
    ) -> RemoteProcess:
        try:
            return await worker.run(cmd, timeout_s=30.0, check=True)
        except Exception as exc:
            raise StaticHealthCheckError(
                f"worker {worker.id}: {label} failed while running {cmd!r}: {exc}"
            ) from exc


async def _communicate(
    proc: asyncio.subprocess.Process,
    timeout_s: float | None,
    label: str,
) -> StaticCompletedProcess:
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        _kill_process_group(int(proc.pid), signal.SIGKILL)
        await proc.wait()
        raise TimeoutError(f"{label} timed out after {timeout_s}s") from None

    return StaticCompletedProcess(
        completed_pid=proc.pid,
        completed_returncode=int(proc.returncode or 0),
        completed_stdout=stdout_b.decode("utf-8", errors="replace"),
        completed_stderr=stderr_b.decode("utf-8", errors="replace"),
    )


def _kill_process_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except ProcessLookupError:
        return
