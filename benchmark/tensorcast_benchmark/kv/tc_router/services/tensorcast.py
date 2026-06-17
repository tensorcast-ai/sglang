"""Tensorcast global-store + per-worker daemon launcher.

Patterned after `kv/share_remote.run_benchmark.{build_tensorcast_configs,
start_global_store,start_tensorcast_daemon}`. We render per-run config
files by patching the request_transfer templates, then drive the
`scripts/tensorcast_service.sh` shell wrapper through `Worker.start_background`.

Phase 7 stub: TcRouter connects to the daemon (proves wiring) but issues
no plans (NeverRebalance), so the global store + daemon are running but
idle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .base import Service


logger = logging.getLogger(__name__)


# Path to scripts/templates relative to the tc_router package root.
_TC_ROUTER_DIR = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _TC_ROUTER_DIR / "scripts" / "tensorcast_service.sh"
_GLOBAL_STORE_TEMPLATE = _TC_ROUTER_DIR / "configs" / "global_store_config_template.yaml"
_DAEMON_TEMPLATE = _TC_ROUTER_DIR / "configs" / "store_daemon_config_template.yaml"


@dataclass(frozen=True)
class TensorcastEndpoints:
    """Network coordinates for the running services. Exposed to TcRouter."""

    global_store_address: str   # "<host>:<port>" for daemon to connect to
    daemon_address: str         # "<host>:<port>" for `tc.connect` from driver


@dataclass(frozen=True)
class TensorcastLaunchSpec:
    """Inputs to a tensorcast (global store + daemon) launch.

    All paths must live on the shared mount visible from driver host AND
    every worker (per arch § 7 / Phase 1 finding about /home/<user> NFS).
    """

    namespace: str
    global_store_port: int = 61050
    daemon_port: int = 61053
    daemon_p2p_port: int = 61090
    # Daemon needs CUDA libs. Defaults match share_remote's working setup.
    cuda_home: str = "/usr/local/cuda-12.4"
    nvidia_lib_dirs: str = "/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64"
    # Tensorcast tuning — keep small for smoke; share_remote uses 64GB.
    daemon_stable_bytes: str = "16GB"
    enable_rdma: bool = False
    # Where to dump rendered configs + logs (driver-host scratch path).
    config_dir: str = ""
    log_dir: str = ""
    # Per-run state dir for tensorcast (TENSORCAST_HOME). One per service.
    runtime_home_root: str = ""
    workspace_root: str = "/"
    uv_bin: str = "/usr/local/bin/uv"
    # Service ready timeouts.
    service_ready_timeout_s: float = 240.0
    service_poll_interval_s: float = 2.0


def render_global_store_config(
    spec: TensorcastLaunchSpec,
    *,
    advertise_host: str,
    log_path: str,
) -> dict:
    """Render the global-store YAML config by patching the template."""
    with _GLOBAL_STORE_TEMPLATE.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg = deepcopy(cfg)
    cfg.setdefault("meta", {})["description"] = f"tc_router-{spec.namespace}"
    cfg["server"]["listen"]["host"] = "0.0.0.0"
    cfg["server"]["listen"]["port"] = spec.global_store_port
    cfg["server"]["advertise"]["host"] = advertise_host
    cfg["server"]["advertise"]["port"] = spec.global_store_port
    # Disable metrics_port to avoid collisions; default 18000 may conflict.
    cfg["server"]["metrics_port"] = 0
    cfg.setdefault("observability", {}).setdefault("logging", {})["file"] = log_path
    return cfg


def render_daemon_config(
    spec: TensorcastLaunchSpec,
    *,
    advertise_host: str,
    global_store_endpoint: tuple[str, int],
    log_path: str,
    capability_token_secret: str,
) -> dict:
    """Render a daemon YAML config by patching the template."""
    with _DAEMON_TEMPLATE.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg = deepcopy(cfg)
    cfg["server"]["listen"]["host"] = "0.0.0.0"
    cfg["server"]["listen"]["port"] = spec.daemon_port
    cfg["server"]["advertise"]["host"] = advertise_host
    cfg["server"]["p2p_listen"]["host"] = "0.0.0.0"
    cfg["server"]["p2p_listen"]["port"] = spec.daemon_p2p_port

    cfg.setdefault("engine", {}).setdefault("memory_tiers", {})["stable_bytes"] = (
        spec.daemon_stable_bytes
    )

    ha = cfg.setdefault("high_availability", {})
    ha["enabled"] = True
    ha["global_store_endpoints"] = [
        {"host": global_store_endpoint[0], "port": global_store_endpoint[1]}
    ]
    ha["heartbeat_interval"] = "10s"

    cfg.setdefault("observability", {}).setdefault("logging", {})["file"] = log_path

    cfg.setdefault("communicator", {})["enable_rdma"] = bool(spec.enable_rdma)

    # Capability-token rotation needs *some* secret; supply a per-run one
    # rather than relying on whatever the template ships with.
    caps = cfg.setdefault("capability_tokens", {})
    active = caps.setdefault("active", {})
    active["version"] = int(active.get("version", 1) or 1)
    if not str(active.get("secret", "")).strip():
        active["secret"] = capability_token_secret
    return cfg


def _runtime_home(spec: TensorcastLaunchSpec, *, name: str) -> str:
    base = spec.runtime_home_root or f"{spec.config_dir.rstrip('/')}/runtime"
    return f"{base.rstrip('/')}/{name}"


def _service_cmd(
    spec: TensorcastLaunchSpec,
    *,
    runtime_home: str,
    subcommand: str,
    args: list[str],
) -> str:
    """Build the bash invocation that runs `tensorcast_service.sh <subcmd>`."""
    workspace = spec.workspace_root.rstrip("/")
    venv_activate = f"{workspace}/.venv/bin/activate"
    joined_args = " ".join(shlex.quote(a) for a in args)
    return (
        f"cd {shlex.quote(workspace)}; "
        f"source {shlex.quote(venv_activate)}; "
        f"export TENSORCAST_HOME={shlex.quote(runtime_home)}; "
        f"export UV_BIN={shlex.quote(spec.uv_bin)}; "
        f"bash {shlex.quote(str(_SCRIPT_PATH))} {shlex.quote(subcommand)} "
        f"{joined_args}"
    )


# --- Launcher ----------------------------------------------------------------


class TensorcastLauncher:
    """Lifecycle for one global-store + one or more daemons.

    Phase 7 smoke uses one daemon (single worker); the launcher accepts
    multiple to keep the multi-worker case aligned with arch.
    """

    def __init__(self) -> None:
        self._global_runtime_home: Optional[str] = None
        self._daemon_runtime_homes: dict[str, str] = {}  # worker_id -> runtime_home

    async def launch_global_store(
        self,
        worker,
        spec: TensorcastLaunchSpec,
    ) -> Service:
        """Launch global store on `worker` (typically `service_placement.global_store_worker_id`)."""
        cfg_dir = Path(spec.config_dir)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(spec.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Service log inside log_dir (visible from driver via shared mount).
        service_log_path = str(log_dir / "tensorcast_global_store.log")
        # Tensorcast's own logging.file inside the daemon — keep separate so
        # the wrapper's stdout (start command output) doesn't compete.
        tc_log_file = str(log_dir / "tensorcast_global_store.tclog")

        cfg = render_global_store_config(
            spec, advertise_host=worker.address, log_path=tc_log_file
        )
        cfg_path = str(cfg_dir / "tensorcast_global_store.yaml")
        Path(cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=False))

        runtime_home = _runtime_home(spec, name="global_store")
        Path(runtime_home).mkdir(parents=True, exist_ok=True)
        self._global_runtime_home = runtime_home

        cmd = _service_cmd(
            spec, runtime_home=runtime_home, subcommand="start-global", args=[cfg_path]
        )
        pid_path = str(log_dir / "tensorcast_global_store.pid")
        # `start-global` is itself a long-running process (the global store
        # CLI blocks while the server is alive), so we run it as a
        # background process with a PID file.
        pid = await worker.start_background(
            cmd,
            name="tensorcast_global_store",
            log_path=service_log_path,
            pid_path=pid_path,
        )
        return Service(
            name="tensorcast_global_store",
            worker_id=worker.id,
            endpoints={
                "grpc": f"{worker.address}:{spec.global_store_port}",
                "advertise_host": worker.address,
            },
            pid=pid,
            pid_path=pid_path,
            log_path=service_log_path,
            metadata={
                "config_path": cfg_path,
                "tensorcast_home": runtime_home,
                "tc_log_file": tc_log_file,
            },
        )

    async def launch_daemon(
        self,
        worker,
        spec: TensorcastLaunchSpec,
        *,
        global_store_address: tuple[str, int],
        capability_token_secret: str,
    ) -> Service:
        """Launch a tensorcast daemon on `worker`.

        `global_store_address = (host, port)` of an already-running global store.
        """
        cfg_dir = Path(spec.config_dir)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(spec.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        service_log_path = str(log_dir / f"tensorcast_daemon_{worker.id}.log")
        tc_log_file = str(log_dir / f"tensorcast_daemon_{worker.id}.tclog")

        cfg = render_daemon_config(
            spec,
            advertise_host=worker.address,
            global_store_endpoint=global_store_address,
            log_path=tc_log_file,
            capability_token_secret=capability_token_secret,
        )
        cfg_path = str(cfg_dir / f"tensorcast_daemon_{worker.id}.yaml")
        Path(cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=False))

        runtime_home = _runtime_home(spec, name=f"daemon_{worker.id}")
        Path(runtime_home).mkdir(parents=True, exist_ok=True)
        self._daemon_runtime_homes[worker.id] = runtime_home

        gs_addr = f"{global_store_address[0]}:{global_store_address[1]}"
        cmd = _service_cmd(
            spec,
            runtime_home=runtime_home,
            subcommand="start-daemon",
            args=[cfg_path, gs_addr, spec.cuda_home, spec.nvidia_lib_dirs],
        )
        pid_path = str(log_dir / f"tensorcast_daemon_{worker.id}.pid")
        pid = await worker.start_background(
            cmd,
            name=f"tensorcast_daemon_{worker.id}",
            log_path=service_log_path,
            pid_path=pid_path,
        )
        return Service(
            name=f"tensorcast_daemon_{worker.id}",
            worker_id=worker.id,
            endpoints={
                "grpc": f"{worker.address}:{spec.daemon_port}",
                "p2p": f"{worker.address}:{spec.daemon_p2p_port}",
                "advertise_host": worker.address,
            },
            pid=pid,
            pid_path=pid_path,
            log_path=service_log_path,
            metadata={
                "config_path": cfg_path,
                "tensorcast_home": runtime_home,
                "tc_log_file": tc_log_file,
            },
        )

    async def wait_global_ready(
        self,
        worker,
        spec: TensorcastLaunchSpec,
        service: Service,
    ) -> None:
        """Wait until the global store reports `health : SERVING`.

        We invoke `tensorcast_service.sh status-global`, which runs the
        Tensorcast CLI and prints a small block including a `health :
        <state>` line. The store is ready when health is `SERVING`.
        """
        runtime_home = service.metadata["tensorcast_home"]
        deadline = time.monotonic() + spec.service_ready_timeout_s
        last_detail = "no probe yet"
        while time.monotonic() < deadline:
            cmd = _service_cmd(
                spec, runtime_home=runtime_home, subcommand="status-global", args=[]
            )
            proc = await worker.run(cmd, timeout_s=15.0, check=False, as_user=True)
            stdout = (proc.stdout or "") + (proc.stderr or "")
            if "SERVING" in stdout:
                return
            last_detail = stdout.strip().splitlines()[-1] if stdout.strip() else "(empty)"
            await asyncio.sleep(spec.service_poll_interval_s)
        raise TimeoutError(
            f"tensorcast global store at {service.endpoints['grpc']} "
            f"not ready (health=SERVING) within {spec.service_ready_timeout_s}s: "
            f"last status: {last_detail}"
        )

    async def wait_daemon_ready(
        self,
        worker,
        spec: TensorcastLaunchSpec,
        service: Service,
    ) -> None:
        """Wait via the script's `wait-daemon-ready` subcommand (uses the SDK probe)."""
        runtime_home = service.metadata["tensorcast_home"]
        cmd = _service_cmd(
            spec,
            runtime_home=runtime_home,
            subcommand="wait-daemon-ready",
            args=[
                service.endpoints["grpc"],
                str(spec.service_ready_timeout_s),
                str(spec.service_poll_interval_s),
            ],
        )
        proc = await worker.run(
            cmd,
            timeout_s=spec.service_ready_timeout_s + 30.0,
            check=False,
            as_user=True,
        )
        if proc.returncode != 0:
            tail = (proc.stdout or "") + (proc.stderr or "")
            raise TimeoutError(
                f"tensorcast daemon at {service.endpoints['grpc']} "
                f"not ready within {spec.service_ready_timeout_s}s:\n{tail[-2000:]}"
            )

    async def stop_daemon(
        self,
        worker,
        spec: TensorcastLaunchSpec,
        service: Service,
    ) -> None:
        runtime_home = service.metadata["tensorcast_home"]
        cmd = _service_cmd(
            spec, runtime_home=runtime_home, subcommand="stop-daemon", args=["45"]
        )
        try:
            await worker.run(cmd, timeout_s=90.0, check=False, as_user=True)
        finally:
            try:
                await worker.stop_background(pid_path=service.pid_path)
            except Exception:  # noqa: BLE001
                logger.exception("failed to stop daemon background %s", service.name)

    async def stop_global(
        self,
        worker,
        spec: TensorcastLaunchSpec,
        service: Service,
    ) -> None:
        runtime_home = service.metadata["tensorcast_home"]
        cmd = _service_cmd(
            spec, runtime_home=runtime_home, subcommand="stop-global", args=["45"]
        )
        try:
            await worker.run(cmd, timeout_s=90.0, check=False, as_user=True)
        finally:
            try:
                await worker.stop_background(pid_path=service.pid_path)
            except Exception:  # noqa: BLE001
                logger.exception("failed to stop global store background %s", service.name)
