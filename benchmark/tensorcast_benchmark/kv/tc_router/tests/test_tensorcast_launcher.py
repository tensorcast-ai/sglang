"""Tests for services/tensorcast.py command construction helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tensorcast_benchmark.kv.tc_router.services.tensorcast import (
    TensorcastLaunchSpec,
    TensorcastLauncher,
    _is_loopback_or_unspecified_host,
    _service_cmd,
    _tensorcast_advertise_host,
    render_daemon_config,
)


class _FakeWorker:
    id = "local_h800"
    address = "127.0.0.1"

    async def start_background(
        self,
        cmd: str,
        *,
        name: str,
        log_path: str,
        pid_path: str,
        env: dict[str, str] | None = None,
    ) -> int:
        del cmd, name, log_path, pid_path, env
        return 123


def test_tensorcast_service_command_activates_workspace_venv() -> None:
    spec = TensorcastLaunchSpec(
        namespace="unit",
        workspace_root="/home/u/tot",
        uv_bin="/opt/uv",
    )

    cmd = _service_cmd(
        spec,
        runtime_home="/tmp/tc",
        subcommand="status-global",
        args=[],
    )

    assert "cd /home/u/tot;" in cmd
    assert "source /home/u/tot/.venv/bin/activate" in cmd
    assert "export TENSORCAST_HOME=/tmp/tc" in cmd
    assert "export UV_BIN=/opt/uv" in cmd
    assert "tensorcast_service.sh status-global" in cmd


def test_tensorcast_launch_spec_uses_local_cuda_default() -> None:
    spec = TensorcastLaunchSpec(namespace="unit")

    assert spec.cuda_home == "/usr/local/cuda"
    assert spec.nvidia_lib_dirs == ""


def test_tensorcast_advertise_host_keeps_routable_address() -> None:
    assert _tensorcast_advertise_host("10.0.10.49") == "10.0.10.49"


def test_tensorcast_advertise_host_replaces_loopback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "tensorcast_benchmark.kv.tc_router.services.tensorcast._detect_local_routable_ipv4",
        lambda: "10.0.10.49",
    )

    assert _tensorcast_advertise_host("127.0.0.1") == "10.0.10.49"


@pytest.mark.asyncio
async def test_global_store_service_endpoint_uses_advertise_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tensorcast_benchmark.kv.tc_router.services.tensorcast._detect_local_routable_ipv4",
        lambda: "10.0.10.49",
    )
    spec = TensorcastLaunchSpec(
        namespace="unit",
        config_dir=str(tmp_path / "configs"),
        log_dir=str(tmp_path / "logs"),
        runtime_home_root=str(tmp_path / "runtime"),
    )

    service = await TensorcastLauncher().launch_global_store(_FakeWorker(), spec)

    assert service.endpoints["grpc"] == "10.0.10.49:61050"
    assert service.endpoints["advertise_host"] == "10.0.10.49"
    with Path(service.metadata["config_path"]).open("r", encoding="utf-8") as fh:
        rendered = yaml.safe_load(fh)
    assert rendered["server"]["advertise"]["host"] == "10.0.10.49"


def test_loopback_or_unspecified_host_detection() -> None:
    assert _is_loopback_or_unspecified_host("127.0.0.1")
    assert _is_loopback_or_unspecified_host("0.0.0.0")
    assert _is_loopback_or_unspecified_host("localhost")
    assert not _is_loopback_or_unspecified_host("10.0.10.49")


def test_render_daemon_config_enables_gateway_ingress() -> None:
    spec = TensorcastLaunchSpec(namespace="unit")

    cfg = render_daemon_config(
        spec,
        advertise_host="10.0.10.49",
        global_store_endpoint=("10.0.10.49", 61050),
        log_path="/tmp/tensorcast.log",
        capability_token_secret="secret",
    )

    assert cfg["capability_directory"]["enabled"] is True
    assert cfg["capability_directory"]["gateway_ingress_enabled"] is True
