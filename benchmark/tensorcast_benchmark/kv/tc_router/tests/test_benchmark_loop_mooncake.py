"""Mooncake profile lifecycle tests for the benchmark driver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from tensorcast_benchmark.kv.tc_router.driver import benchmark_loop
from tensorcast_benchmark.kv.tc_router.driver.config import BenchmarkConfig
from tensorcast_benchmark.kv.tc_router.metrics.summary import RunSummary
from tensorcast_benchmark.kv.tc_router.services.base import Service


@dataclass(frozen=True)
class _Worker:
    id: str
    address: str
    gpu_indices: tuple[int, ...]
    scratch_dir: str


@dataclass(frozen=True)
class _ServicePlacement:
    mooncake_master_worker_id: str


@dataclass(frozen=True)
class _ClusterConfig:
    service_placement: _ServicePlacement


@dataclass(frozen=True)
class _Completed:
    stdout: str


class _Provider:
    def __init__(self, worker: _Worker, events: list[tuple[str, Any]]) -> None:
        self._worker = worker
        self._events = events
        self._config = _ClusterConfig(
            service_placement=_ServicePlacement(mooncake_master_worker_id=worker.id)
        )

    def workers(self) -> list[_Worker]:
        return [self._worker]

    async def health_check(self) -> None:
        self._events.append(("health", self._worker.id))


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _cluster_yaml(tmp_path: Path) -> dict:
    return {
        "provider": {"kind": "local"},
        "driver_host": {"scratch_dir": str(tmp_path / "old_driver")},
        "mount": {"path": str(tmp_path / "old_mount"), "spec": "local"},
        "workers": [
            {
                "id": "local_h800",
                "address": "127.0.0.1",
                "node": "local_h800",
                "process_handle": "local",
                "gpu_indices": [0],
                "scratch_dir": str(tmp_path / "old_worker"),
                "base_env": {},
            }
        ],
        "service_placement": {
            "global_store_worker_id": "local_h800",
            "mooncake_master_worker_id": "local_h800",
        },
    }


def _benchmark_yaml() -> dict:
    return {
        "run_id": "unit-mooncake-profile",
        "model": {"path": "/models/qwen", "tp_size": 1},
        "instances": {"count": 1, "base_port": 55001},
        "transport": {"use_rdma": False},
        "workload": {
            "dataset_path": "/data/workload",
            "pool_filter": {"min_turns": 1, "min_total_tokens": 1},
            "inter_turn_delay": {"preset": "agent_medium"},
            "max_new_tokens_clip": 16,
            "start_jitter_s": 0.0,
            "wall_seconds": 1,
            "warmup_seconds": 0,
            "warmup_counts": 0,
            "trials": 1,
            "c_target_sweep": [1],
        },
        "configs": [
            {"kind": "gw_load_aware"},
            {"kind": "gw_load_aware_mooncake"},
        ],
        "gateway": {"host": "127.0.0.1", "port": 55100},
        "mooncake": {
            "http_metadata_server_port": 62300,
            "master_port": 62301,
            "global_segment_size": "64gb",
            "eviction_high_watermark_ratio": 0.9,
            "device_name": "",
            "clear_storage_between_cells": True,
        },
    }


@pytest.mark.asyncio
async def test_mooncake_profile_has_separate_fleet_and_storage_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Any]] = []
    worker = _Worker(
        id="local_h800",
        address="127.0.0.1",
        gpu_indices=(0,),
        scratch_dir=str(tmp_path / "worker"),
    )
    provider = _Provider(worker, events)

    class FakeMooncakeLauncher:
        async def launch(self, worker_arg, spec) -> Service:
            events.append(
                (
                    "mooncake_launch",
                    worker_arg.id,
                    spec.http_metadata_server_port,
                    spec.master_port,
                )
            )
            return Service(
                name="mooncake_master",
                worker_id=worker_arg.id,
                endpoints={
                    "advertise_host": "10.0.10.49",
                    "master_server_address": "10.0.10.49:62301",
                    "metadata_server": "http://10.0.10.49:62300/metadata",
                    "health_http": "http://10.0.10.49:62300/health",
                },
                pid=1,
                pid_path=str(tmp_path / "mooncake.pid"),
                log_path=str(tmp_path / "mooncake.log"),
            )

        async def wait_ready(
            self,
            service: Service,
            *,
            timeout_s: float,
            poll_interval_s: float,
        ) -> None:
            del timeout_s, poll_interval_s
            events.append(("mooncake_wait", service.endpoints["health_http"]))

        async def stop(self, worker_arg, service: Service) -> None:
            events.append(("mooncake_stop", worker_arg.id, service.name))

    class FakeSGLangLauncher:
        async def launch(self, worker_arg, spec) -> Service:
            events.append(
                (
                    "sglang_launch",
                    spec.hicache_storage_backend,
                    spec.hicache_storage_backend_extra_config,
                )
            )
            return Service(
                name=f"sglang_{spec.port}",
                worker_id=worker_arg.id,
                endpoints={
                    "serving_http": f"http://{worker_arg.address}:{spec.port}",
                    "instance_id": f"{worker_arg.address}:{spec.port}",
                },
                pid=2,
                pid_path=str(tmp_path / f"sglang_{spec.port}.pid"),
                log_path=str(tmp_path / f"sglang_{spec.port}.log"),
            )

        async def wait_ready(self, service: Service, *, timeout_s: float) -> None:
            del timeout_s
            events.append(("sglang_wait", service.name))

        async def stop(self, worker_arg, service: Service) -> None:
            events.append(("sglang_stop", service.name))

        async def aclose(self) -> None:
            events.append(("sglang_aclose", None))

    class FakeGatewayLauncher:
        async def launch(self, spec) -> Service:
            events.append(("gateway_launch", spec.policy, tuple(spec.worker_urls)))
            return Service(
                name="gateway",
                worker_id="driver",
                endpoints={"openai_http": f"http://{spec.host}:{spec.port}"},
                pid=3,
                pid_path=str(Path(spec.log_dir) / "gateway.pid"),
                log_path=str(Path(spec.log_dir) / "gateway.log"),
            )

        async def wait_ready(self, service: Service, *, timeout_s: float) -> None:
            del timeout_s
            events.append(("gateway_wait", service.endpoints["openai_http"]))

        async def stop(self, service: Service) -> None:
            events.append(("gateway_stop", service.name))

    class FakeGatewayRouter:
        def __init__(self, base_url: str, *, default_model: str) -> None:
            events.append(("router_init", base_url, default_model))

        async def close(self) -> None:
            events.append(("router_close", None))

    async def fake_flush(instance_services: list, *, label: str) -> None:
        events.append(("flush", label, len(instance_services)))

    async def fake_clear(instance_services: list, *, label: str) -> None:
        events.append(("clear", label, len(instance_services)))

    async def fake_run_one_cell(**kwargs) -> tuple[RunSummary, dict]:
        events.append(("run", kwargs["cfg_kind"], kwargs["c_target"]))
        return (
            RunSummary(
                config=kwargs["cfg_kind"],
                c_target=kwargs["c_target"],
                trial=kwargs["trial"],
                inter_turn_delay_preset="agent_medium",
                transport_mode="tcp",
                ttft_p50_ms=1.0,
                ttft_p95_ms=1.0,
                ttft_p99_ms=1.0,
                ttft_mean_ms=1.0,
                cached_token_ratio_mean=0.0,
                total_turns_completed=1,
                total_requests_failed=0,
            ),
            {
                "total_turns": 1,
                "successful_turns": 1,
                "failed_turns": 0,
            },
        )

    monkeypatch.setattr(
        benchmark_loop.resource_factory,
        "from_cluster_config",
        lambda path: provider,
    )
    monkeypatch.setattr(benchmark_loop, "MooncakeLauncher", FakeMooncakeLauncher)
    monkeypatch.setattr(benchmark_loop, "SGLangLauncher", FakeSGLangLauncher)
    monkeypatch.setattr(benchmark_loop, "GatewayLauncher", FakeGatewayLauncher)
    monkeypatch.setattr(benchmark_loop, "GatewayRouter", FakeGatewayRouter)
    monkeypatch.setattr(benchmark_loop, "_flush_sglang_instances", fake_flush)
    monkeypatch.setattr(benchmark_loop, "_clear_sglang_hicache_storage", fake_clear)
    monkeypatch.setattr(benchmark_loop, "_run_one_cell", fake_run_one_cell)
    monkeypatch.setattr(benchmark_loop, "_now_stamp", lambda: "20260710-000000")
    monkeypatch.setattr(benchmark_loop, "load_pool", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        benchmark_loop,
        "_mooncake_advertise_host",
        lambda address: "10.0.10.49" if address == "127.0.0.1" else address,
    )

    cluster_yaml = _write_yaml(tmp_path / "cluster.yaml", _cluster_yaml(tmp_path))
    bench_yaml = _write_yaml(tmp_path / "benchmark.yaml", _benchmark_yaml())

    run_dir = await benchmark_loop.run_benchmark(
        cluster_yaml,
        bench_yaml,
        outputs_root=tmp_path / "outputs",
        sglang_ready_timeout_s=1.0,
    )

    assert run_dir.name == "20260710-000000_unit-mooncake-profile"
    launch_events = [event for event in events if event[0] == "sglang_launch"]
    assert len(launch_events) == 2
    assert launch_events[0][1] is None
    assert launch_events[0][2] is None
    assert launch_events[1][1] == "mooncake"
    mooncake_extra_config = launch_events[1][2]
    assert mooncake_extra_config == {
        "master_server_address": "10.0.10.49:62301",
        "metadata_server": "http://10.0.10.49:62300/metadata",
        "local_hostname": "10.0.10.49",
        "protocol": "tcp",
        "global_segment_size": "64gb",
    }
    assert "prefetch_threshold" not in mooncake_extra_config

    mooncake_launch_index = events.index(
        ("mooncake_launch", "local_h800", 62300, 62301)
    )
    mooncake_sglang_index = events.index(launch_events[1])
    assert mooncake_launch_index < mooncake_sglang_index
    assert events.count(("clear", "gw_load_aware_mooncake c=1 trial=0", 1)) == 1
    assert ("clear", "gw_load_aware c=1 trial=0", 1) not in events
    assert ("gateway_launch", "power_of_two", ("http://127.0.0.1:55001",)) in events
    assert ("mooncake_stop", "local_h800", "mooncake_master") in events


@pytest.mark.asyncio
async def test_mooncake_rdma_backend_uses_selected_hca_ipv4_for_advertise() -> None:
    class RdmaWorker:
        address = "10.0.10.58"

        async def run(
            self,
            cmd: str,
            *,
            timeout_s: float | None = None,
            check: bool = True,
        ) -> _Completed:
            del timeout_s, check
            if cmd == "rdma link show":
                return _Completed(
                    "link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev eth0\n"
                    "link mlx5_1/1 state ACTIVE physical_state LINK_UP netdev eth1\n"
                )
            if cmd == "ip -o -4 addr show dev eth0 scope global":
                return _Completed("2: eth0    inet 22.32.111.67/32 scope global eth0\n")
            raise AssertionError(f"unexpected command: {cmd}")

    bench_data = _benchmark_yaml()
    bench_data["transport"]["use_rdma"] = True
    bench_data["mooncake"]["device_name"] = "mlx5_0,mlx5_1"
    bench_cfg = BenchmarkConfig.model_validate(bench_data)
    mooncake_service = Service(
        name="mooncake_master",
        worker_id="local_h800",
        endpoints={
            "master_server_address": "10.0.10.59:62301",
            "metadata_server": "http://10.0.10.59:62300/metadata",
        },
        pid=1,
        pid_path="/tmp/mooncake.pid",
        log_path="/tmp/mooncake.log",
    )

    options = await benchmark_loop._mooncake_backend_options(
        bench_cfg=bench_cfg,
        mooncake_service=mooncake_service,
        worker=RdmaWorker(),
    )

    assert options.extra_config == {
        "master_server_address": "10.0.10.59:62301",
        "metadata_server": "http://10.0.10.59:62300/metadata",
        "local_hostname": "22.32.111.67",
        "protocol": "rdma",
        "global_segment_size": "64gb",
        "device_name": "mlx5_0,mlx5_1",
    }
    assert options.extra_env == {"MC_TCP_BIND_ADDRESS": "22.32.111.67"}
