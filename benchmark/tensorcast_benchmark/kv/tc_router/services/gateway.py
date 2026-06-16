"""sgl-model-gateway launcher.

The gateway runs on the **driver host** (not on a worker), per arch § 7.2.
We spawn it as a local subprocess and manage its lifecycle through a PID
file, mirroring the worker-side `start_background` / `stop_background`
pattern so the rest of the codebase stays uniform.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional

import aiohttp

from .base import Service


@dataclass(frozen=True)
class GatewayLaunchSpec:
    """Inputs to `sglang_router.launch_router`."""

    worker_urls: tuple[str, ...]
    policy: Literal["power_of_two", "cache_aware", "random", "round_robin", "bucket"]
    host: str = "127.0.0.1"
    port: int = 30100
    log_dir: Optional[str] = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    workspace_root: str = "/home/i-zhouyuhan/tot"
    uv_bin: str = "/home/i-zhouyuhan/.local/bin/uv"


def build_gateway_command(spec: GatewayLaunchSpec) -> str:
    """Build the shell command that launches the gateway via `python -m sglang_router.launch_router`.

    Like the SGLang launcher, we activate the workspace venv first so
    `uv run --active` picks up the right Python.
    """
    if not spec.worker_urls:
        raise ValueError("GatewayLaunchSpec.worker_urls must be non-empty")

    workspace = spec.workspace_root.rstrip("/")
    sglang_root = f"{workspace}/thirdparty/sglang"
    venv_activate = f"{workspace}/.venv/bin/activate"
    pythonpath = f"{sglang_root}/python"

    prefix = (
        f"cd {shlex.quote(sglang_root)}; "
        f"source {shlex.quote(venv_activate)}; "
        f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}; "
    )

    parts: list[str] = [
        shlex.quote(spec.uv_bin),
        "run",
        "--active",
        "--no-project",
        "--offline",
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        "--policy",
        spec.policy,
        "--worker-urls",
        *(shlex.quote(u) for u in spec.worker_urls),
    ]
    parts.extend(spec.extra_args)
    return prefix + " ".join(parts)


# --- Local-host process management ------------------------------------------
#
# We don't reuse `Worker.start_background` because the gateway lives on the
# driver, not on a brainctl-managed worker. A small subprocess.Popen wrapper
# with a PID file gives us symmetric lifecycle semantics.


class GatewayLauncher:
    """Spawns the gateway as a local subprocess on the driver host."""

    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen] = {}
        self._log_handles: dict[str, object] = {}

    async def launch(self, spec: GatewayLaunchSpec) -> Service:
        log_dir = Path(
            spec.log_dir or f"/tmp/tc_router_gateway_{spec.port}_{int(time.time())}"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "gateway.log"
        pid_path = log_dir / "gateway.pid"

        cmd = build_gateway_command(spec)
        log_fh = log_path.open("ab")
        # Use bash -lc so the `cd ...; source .venv/bin/activate; ...` prefix
        # works. preexec_fn=os.setsid puts us in a new process group so
        # killpg can take down children cleanly.
        proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid,
            env=_local_env_no_proxy(),
        )
        pid_path.write_text(str(proc.pid))

        self._procs[str(pid_path)] = proc
        self._log_handles[str(pid_path)] = log_fh

        return Service(
            name=f"gateway_{spec.port}",
            worker_id="driver_host",
            endpoints={
                "openai_http": f"http://{spec.host}:{spec.port}",
            },
            pid=proc.pid,
            pid_path=str(pid_path),
            log_path=str(log_path),
            metadata={"policy": spec.policy, "n_workers": str(len(spec.worker_urls))},
        )

    async def wait_ready(
        self,
        service: Service,
        *,
        timeout_s: float = 120.0,
        poll_interval_s: float = 2.0,
    ) -> None:
        url = f"{service.endpoints['openai_http']}/v1/models"
        deadline = time.monotonic() + timeout_s
        last_detail = "no probe yet"
        async with aiohttp.ClientSession(trust_env=False) as session:
            while time.monotonic() < deadline:
                # If the subprocess died early, fail fast.
                proc = self._procs.get(service.pid_path)
                if proc is not None and proc.poll() is not None:
                    raise RuntimeError(
                        f"gateway subprocess exited early with rc={proc.returncode}; "
                        f"check {service.log_path}"
                    )
                try:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=5.0),
                        proxy=None,
                    ) as resp:
                        if resp.status == 200:
                            return
                        last_detail = f"HTTP {resp.status}"
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    last_detail = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(poll_interval_s)
        raise TimeoutError(
            f"gateway at {url} not ready within {timeout_s}s: {last_detail}"
        )

    async def stop(self, service: Service) -> None:
        proc = self._procs.pop(service.pid_path, None)
        log_fh = self._log_handles.pop(service.pid_path, None)
        if proc is not None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if log_fh is not None:
            try:
                log_fh.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        # Remove pid file (best effort).
        try:
            Path(service.pid_path).unlink(missing_ok=True)
        except Exception:
            pass


# --- proxy stripping (consistent with brainctl side) ------------------------


_PROXY_ENV_KEYS: Iterable[str] = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "https_proxy",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def _local_env_no_proxy() -> dict[str, str]:
    env = os.environ.copy()
    for key in _PROXY_ENV_KEYS:
        env.pop(key, None)
    return env
