"""Tests for services/gateway.py command construction (no subprocess)."""

from __future__ import annotations

import pytest

from tensorcast_benchmark.kv.tc_router.services.gateway import (
    GatewayLaunchSpec,
    build_gateway_command,
)


def make_spec(**overrides) -> GatewayLaunchSpec:
    defaults = dict(
        worker_urls=("http://10.0.0.1:30001", "http://10.0.0.1:30002"),
        policy="cache_aware",
        host="127.0.0.1",
        port=30100,
    )
    defaults.update(overrides)
    return GatewayLaunchSpec(**defaults)


def test_basic_command_has_required_args() -> None:
    cmd = build_gateway_command(make_spec())
    assert "sglang_router.launch_router" in cmd
    assert "--host 127.0.0.1" in cmd
    assert "--port 30100" in cmd
    assert "--policy cache_aware" in cmd
    assert "--worker-urls" in cmd
    assert "http://10.0.0.1:30001" in cmd
    assert "http://10.0.0.1:30002" in cmd


def test_command_activates_workspace_venv() -> None:
    cmd = build_gateway_command(make_spec(workspace_root="/home/u/tot"))
    assert "cd /home/u/tot/thirdparty/sglang;" in cmd
    assert "source /home/u/tot/.venv/bin/activate" in cmd
    assert "export PYTHONPATH=/home/u/tot/thirdparty/sglang/python" in cmd


def test_uses_uv_with_offline_no_project() -> None:
    cmd = build_gateway_command(make_spec())
    assert "uv " in cmd
    assert "run" in cmd
    assert "--active" in cmd
    assert "--no-project" in cmd
    assert "--offline" in cmd


def test_extra_args_appended() -> None:
    cmd = build_gateway_command(make_spec(extra_args=("--max-concurrent-requests", "256")))
    assert "--max-concurrent-requests 256" in cmd


def test_empty_worker_urls_rejected() -> None:
    with pytest.raises(ValueError, match="worker_urls"):
        build_gateway_command(make_spec(worker_urls=()))


def test_policy_serialized_correctly() -> None:
    cmd = build_gateway_command(make_spec(policy="power_of_two"))
    assert "--policy power_of_two" in cmd
