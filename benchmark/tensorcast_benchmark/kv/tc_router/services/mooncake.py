"""Mooncake master + HTTP metadata service launcher."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import shlex
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

from tensorcast_benchmark.kv.tc_router.services.base import Service


logger = logging.getLogger(__name__)


def _default_workspace_root() -> str:
    """Infer the repo root from this file's location."""
    return str(Path(__file__).resolve().parents[7])


def _is_loopback_or_unspecified_host(host: str) -> bool:
    if host in {"localhost", ""}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_unspecified


def _detect_local_routable_ipv4() -> str:
    """Return the primary non-loopback IPv4 visible from this host."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("8.8.8.8", 80))
        candidate = sock.getsockname()[0]
    if _is_loopback_or_unspecified_host(candidate):
        raise RuntimeError(f"detected non-routable local IPv4 address: {candidate}")
    return candidate


def _mooncake_advertise_host(worker_address: str) -> str:
    if _is_loopback_or_unspecified_host(worker_address):
        return _detect_local_routable_ipv4()
    return worker_address


@dataclass(frozen=True)
class MooncakeLaunchSpec:
    """Inputs to `mooncake_master`."""

    http_metadata_server_port: int = 62300
    master_port: int = 62301
    eviction_high_watermark_ratio: float = 0.9
    workspace_root: str = field(default_factory=_default_workspace_root)
    service_ready_timeout_s: float = 120.0
    service_poll_interval_s: float = 1.0

    @property
    def mooncake_master_bin(self) -> str:
        return f"{self.workspace_root.rstrip('/')}/.venv/bin/mooncake_master"


def build_mooncake_master_command(spec: MooncakeLaunchSpec) -> str:
    workspace = spec.workspace_root.rstrip("/")
    venv_activate = f"{workspace}/.venv/bin/activate"
    return (
        f"cd {shlex.quote(workspace)}; "
        f"source {shlex.quote(venv_activate)}; "
        f"{shlex.quote(spec.mooncake_master_bin)} "
        "--enable_http_metadata_server=true "
        f"--http_metadata_server_port={spec.http_metadata_server_port} "
        f"--eviction_high_watermark_ratio={spec.eviction_high_watermark_ratio} "
        f"--port={spec.master_port}"
    )


class MooncakeLauncher:
    """Lifecycle for one Mooncake master + metadata-service process."""

    async def launch(self, worker, spec: MooncakeLaunchSpec) -> Service:
        log_dir = f"{worker.scratch_dir.rstrip('/')}/services/mooncake_master"
        log_path = f"{log_dir}/mooncake_master.log"
        pid_path = f"{log_dir}/mooncake_master.pid"
        advertise_host = _mooncake_advertise_host(worker.address)
        cmd = build_mooncake_master_command(spec)

        pid = await worker.start_background(
            cmd,
            name="mooncake_master",
            log_path=log_path,
            pid_path=pid_path,
        )
        return Service(
            name="mooncake_master",
            worker_id=worker.id,
            endpoints={
                "advertise_host": advertise_host,
                "master_server_address": f"{advertise_host}:{spec.master_port}",
                "metadata_server": (
                    f"http://{advertise_host}:{spec.http_metadata_server_port}/metadata"
                ),
                "health_http": (
                    f"http://{advertise_host}:{spec.http_metadata_server_port}/health"
                ),
            },
            pid=pid,
            pid_path=pid_path,
            log_path=log_path,
            metadata={
                "http_metadata_server_port": str(spec.http_metadata_server_port),
                "master_port": str(spec.master_port),
                "eviction_high_watermark_ratio": str(
                    spec.eviction_high_watermark_ratio
                ),
            },
        )

    async def wait_ready(
        self,
        service: Service,
        *,
        timeout_s: float = 120.0,
        poll_interval_s: float = 1.0,
    ) -> None:
        url = service.endpoints["health_http"]
        deadline = time.monotonic() + timeout_s
        last_detail = "no probe yet"
        async with aiohttp.ClientSession(trust_env=False) as session:
            while time.monotonic() < deadline:
                try:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=5.0),
                        proxy=None,
                    ) as resp:
                        if resp.status == 200:
                            return
                        body = (await resp.text()).strip()
                        last_detail = body[:500] if body else f"HTTP {resp.status}"
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    last_detail = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(poll_interval_s)
        raise TimeoutError(
            f"Mooncake master at {url} not ready within {timeout_s}s: {last_detail}"
        )

    async def stop(self, worker, service: Service) -> None:
        try:
            await worker.stop_background(pid_path=service.pid_path)
        except Exception:  # noqa: BLE001
            logger.exception("failed to stop Mooncake service %s", service.name)
            raise
