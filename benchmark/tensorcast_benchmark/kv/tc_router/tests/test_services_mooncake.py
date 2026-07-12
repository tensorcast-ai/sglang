"""Tests for services/mooncake.py command construction and readiness."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from aiohttp import web

from tensorcast_benchmark.kv.tc_router.services.mooncake import (
    MooncakeLaunchSpec,
    MooncakeLauncher,
    _mooncake_advertise_host,
    build_mooncake_master_command,
)


class _FakeWorker:
    id = "local_h800"
    address = "127.0.0.1"
    scratch_dir = "/mnt/data/tc_router_unit/worker"

    def __init__(self) -> None:
        self.started: dict[str, str] = {}
        self.stopped_pid_path = ""

    async def start_background(
        self,
        cmd: str,
        *,
        name: str,
        log_path: str,
        pid_path: str,
        env: dict[str, str] | None = None,
    ) -> int:
        del env
        self.started = {
            "cmd": cmd,
            "name": name,
            "log_path": log_path,
            "pid_path": pid_path,
        }
        return 123

    async def stop_background(self, *, pid_path: str) -> None:
        self.stopped_pid_path = pid_path


async def _start_app(app: web.Application) -> tuple[web.AppRunner, int]:
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    site = web.SockSite(runner, sock)
    await site.start()
    return runner, port


def test_mooncake_master_command_activates_workspace_venv() -> None:
    spec = MooncakeLaunchSpec(
        workspace_root="/home/u/tot",
        http_metadata_server_port=62300,
        master_port=62301,
        eviction_high_watermark_ratio=0.85,
    )

    cmd = build_mooncake_master_command(spec)

    assert "cd /home/u/tot;" in cmd
    assert "source /home/u/tot/.venv/bin/activate" in cmd
    assert "/home/u/tot/.venv/bin/mooncake_master" in cmd
    assert "--enable_http_metadata_server=true" in cmd
    assert "--http_metadata_server_port=62300" in cmd
    assert "--eviction_high_watermark_ratio=0.85" in cmd
    assert "--port=62301" in cmd


def test_mooncake_advertise_host_keeps_routable_address() -> None:
    assert _mooncake_advertise_host("10.0.10.58") == "10.0.10.58"


def test_mooncake_advertise_host_replaces_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tensorcast_benchmark.kv.tc_router.services.mooncake._detect_local_routable_ipv4",
        lambda: "10.0.10.49",
    )

    assert _mooncake_advertise_host("127.0.0.1") == "10.0.10.49"


@pytest.mark.asyncio
async def test_launch_returns_master_and_metadata_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tensorcast_benchmark.kv.tc_router.services.mooncake._detect_local_routable_ipv4",
        lambda: "10.0.10.49",
    )
    worker = _FakeWorker()
    spec = MooncakeLaunchSpec(
        workspace_root="/home/u/tot",
        http_metadata_server_port=62300,
        master_port=62301,
    )

    service = await MooncakeLauncher().launch(worker, spec)

    assert worker.started["name"] == "mooncake_master"
    assert service.endpoints["advertise_host"] == "10.0.10.49"
    assert service.endpoints["master_server_address"] == "10.0.10.49:62301"
    assert service.endpoints["metadata_server"] == "http://10.0.10.49:62300/metadata"
    assert service.endpoints["health_http"] == "http://10.0.10.49:62300/health"
    assert Path(service.pid_path).name == "mooncake_master.pid"


@pytest.mark.asyncio
async def test_wait_ready_accepts_health_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tensorcast_benchmark.kv.tc_router.services.mooncake._detect_local_routable_ipv4",
        lambda: "10.0.10.49",
    )
    app = web.Application()

    async def health_handler(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app.router.add_get("/health", health_handler)
    runner, port = await _start_app(app)
    worker = _FakeWorker()
    service = await MooncakeLauncher().launch(
        worker,
        MooncakeLaunchSpec(http_metadata_server_port=port, master_port=62301),
    )
    service.endpoints["health_http"] = f"http://127.0.0.1:{port}/health"
    try:
        await MooncakeLauncher().wait_ready(
            service, timeout_s=1.0, poll_interval_s=0.01
        )
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_stop_uses_worker_pid_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tensorcast_benchmark.kv.tc_router.services.mooncake._detect_local_routable_ipv4",
        lambda: "10.0.10.49",
    )
    worker = _FakeWorker()
    service = await MooncakeLauncher().launch(worker, MooncakeLaunchSpec())

    await MooncakeLauncher().stop(worker, service)

    assert worker.stopped_pid_path == service.pid_path
