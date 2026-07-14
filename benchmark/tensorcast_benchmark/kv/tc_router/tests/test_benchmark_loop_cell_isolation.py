"""Cell-isolation lifecycle tests for the benchmark driver."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any

import pytest

from tensorcast_benchmark.kv.tc_router.driver import benchmark_loop
from tensorcast_benchmark.kv.tc_router.driver.config import (
    BenchmarkConfig,
    ConfigSpec,
)
from tensorcast_benchmark.kv.tc_router.driver.placement import InstanceAssignment
from tensorcast_benchmark.kv.tc_router.metrics.summary import RunSummary
from tensorcast_benchmark.kv.tc_router.resource.base import ClusterConfig
from tensorcast_benchmark.kv.tc_router.services.base import Service


def _benchmark_config() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "run_id": "unit",
            "model": {"path": "/models/qwen", "tp_size": 1},
            "instances": {"count": 2, "base_port": 55001},
            "transport": {"use_rdma": False},
            "workload": {
                "dataset_path": "/data/workload",
                "pool_filter": {"min_turns": 1, "min_total_tokens": 1},
                "inter_turn_delay": {"preset": "agent_medium"},
                "max_new_tokens_clip": 16,
                "start_jitter_s": 0.0,
                "wall_seconds": 1,
                "warmup_seconds": 0,
                "trials": 1,
                "c_target_sweep": [2, 4],
            },
            "configs": [{"kind": "gw_load_aware"}],
            "gateway": {"host": "127.0.0.1", "port": 55100},
        }
    )


def _service(name: str, url: str) -> Service:
    return Service(
        name=name,
        worker_id="worker",
        endpoints={"serving_http": url, "instance_id": name},
        pid=1,
        pid_path=f"/tmp/{name}.pid",
        log_path=f"/tmp/{name}.log",
    )


@dataclass(frozen=True)
class _Worker:
    id: str
    address: str
    gpu_indices: tuple[int, ...]
    scratch_dir: str = "/mnt/data/tc_router_unit/worker"


class _ClusterProvider:
    def __init__(self, worker: _Worker) -> None:
        self._worker = worker
        self._config = ClusterConfig.model_validate(
            {
                "provider": {"kind": "static"},
                "driver_host": {"scratch_dir": "/mnt/data/tc_router_unit/driver"},
                "mount": {"path": "/mnt/data", "spec": "shared"},
                "workers": [
                    {
                        "id": worker.id,
                        "address": worker.address,
                        "node": worker.id,
                        "process_handle": "local",
                        "execution": "local",
                        "gpu_indices": list(worker.gpu_indices),
                        "scratch_dir": worker.scratch_dir,
                    }
                ],
                "service_placement": {
                    "global_store_worker_id": worker.id,
                    "mooncake_master_worker_id": worker.id,
                },
            }
        )

    def workers(self) -> list[_Worker]:
        return [self._worker]


@pytest.mark.asyncio
async def test_gateway_is_launched_and_stopped_per_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Any]] = []

    class FakeGatewayLauncher:
        async def launch(self, spec) -> Service:
            events.append(("launch", Path(spec.log_dir).relative_to(tmp_path)))
            return Service(
                name="gateway",
                worker_id="driver",
                endpoints={"openai_http": f"http://{spec.host}:{spec.port}"},
                pid=2,
                pid_path=str(Path(spec.log_dir) / "gateway.pid"),
                log_path=str(Path(spec.log_dir) / "gateway.log"),
            )

        async def wait_ready(self, service: Service, *, timeout_s: float) -> None:
            events.append(("wait", service.endpoints["openai_http"]))

        async def stop(self, service: Service) -> None:
            events.append(("stop", Path(service.pid_path).relative_to(tmp_path)))

    class FakeGatewayRouter:
        def __init__(self, base_url: str, *, default_model: str) -> None:
            events.append(("router_init", base_url, default_model))

        async def close(self) -> None:
            events.append(("router_close", None))

    async def fake_flush(instance_services: list, *, label: str) -> None:
        events.append(("flush", label, len(instance_services)))

    async def fake_run_one_cell(**kwargs) -> tuple[RunSummary, dict]:
        events.append(("run", kwargs["c_target"], kwargs["trial"]))
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

    monkeypatch.setattr(benchmark_loop, "GatewayLauncher", FakeGatewayLauncher)
    monkeypatch.setattr(benchmark_loop, "GatewayRouter", FakeGatewayRouter)
    monkeypatch.setattr(benchmark_loop, "_flush_sglang_instances", fake_flush)
    monkeypatch.setattr(benchmark_loop, "_run_one_cell", fake_run_one_cell)

    rows = await benchmark_loop._run_gateway_config(
        cfg_spec=ConfigSpec(kind="gw_load_aware"),
        bench_cfg=_benchmark_config(),
        instance_services=(
            _service("inst0", "http://127.0.0.1:55001"),
            _service("inst1", "http://127.0.0.1:55002"),
        ),
        pool=[],
        run_dir=tmp_path,
    )

    assert [row.c_target for row in rows] == [2, 4]
    assert events == [
        ("flush", "gw_load_aware c=2 trial=0", 2),
        ("launch", Path("gw_load_aware/c2/trial0/gateway_log")),
        ("wait", "http://127.0.0.1:55100"),
        ("router_init", "http://127.0.0.1:55100", "/models/qwen"),
        ("run", 2, 0),
        ("router_close", None),
        ("stop", Path("gw_load_aware/c2/trial0/gateway_log/gateway.pid")),
        ("flush", "gw_load_aware c=4 trial=0", 2),
        ("launch", Path("gw_load_aware/c4/trial0/gateway_log")),
        ("wait", "http://127.0.0.1:55100"),
        ("router_init", "http://127.0.0.1:55100", "/models/qwen"),
        ("run", 4, 0),
        ("router_close", None),
        ("stop", Path("gw_load_aware/c4/trial0/gateway_log/gateway.pid")),
    ]


@pytest.mark.asyncio
async def test_tc_router_is_constructed_per_cell_while_tensorcast_stays_per_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Any]] = []
    worker = _Worker(id="worker0", address="10.0.0.1", gpu_indices=(0, 1))

    class FakeTcRouter:
        def __init__(self, cfg, *, policy) -> None:
            events.append(
                (
                    "tc_init",
                    cfg.daemon_address,
                    Path(cfg.migrations_path).relative_to(tmp_path),
                    policy.name,
                )
            )

        async def start(self) -> None:
            events.append(("tc_start", None))

        async def finalize_migrations(self) -> None:
            events.append(("tc_finalize", None))

        async def close(self) -> None:
            events.append(("tc_close", None))

    async def fake_flush(instance_services: list, *, label: str) -> None:
        events.append(("flush", label, len(instance_services)))

    async def fake_clear(instance_services: list, *, label: str) -> None:
        events.append(("clear", label, len(instance_services)))

    async def fake_run_one_cell(**kwargs) -> tuple[RunSummary, dict]:
        assert kwargs["migrations_path"] == kwargs["cell_dir"] / "migrations.jsonl"
        events.append(("run", kwargs["c_target"], kwargs["trial"]))
        post_run_hook = kwargs["post_run_hook"]
        await post_run_hook()
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

    monkeypatch.setattr(benchmark_loop, "TcRouter", FakeTcRouter)
    monkeypatch.setattr(benchmark_loop, "_flush_sglang_instances", fake_flush)
    monkeypatch.setattr(benchmark_loop, "_clear_sglang_hicache_storage", fake_clear)
    monkeypatch.setattr(benchmark_loop, "_run_one_cell", fake_run_one_cell)
    tensorcast_profile = benchmark_loop.TensorcastProfileServices(
        spec=benchmark_loop.TensorcastLaunchSpec(
            namespace="unit",
            config_dir=str(tmp_path / "tc_cfg"),
            log_dir=str(tmp_path / "tc_log"),
            runtime_home_root=str(tmp_path / "tc_runtime"),
        ),
        launcher=object(),  # type: ignore[arg-type]
        global_worker=worker,
        global_store_service=Service(
            name="global",
            worker_id=worker.id,
            endpoints={"advertise_host": worker.address, "grpc": "10.0.0.1:61050"},
            pid=1,
            pid_path="/tmp/global.pid",
            log_path="/tmp/global.log",
        ),
        daemon_services=(
            (
                worker,
                Service(
                    name="daemon",
                    worker_id=worker.id,
                    endpoints={"grpc": "10.0.0.1:61053"},
                    pid=2,
                    pid_path="/tmp/daemon.pid",
                    log_path="/tmp/daemon.log",
                ),
            ),
        ),
        global_store_address=("10.0.0.1", 61050),
    )

    rows = await benchmark_loop._run_tc_router_config(
        cfg_spec=ConfigSpec(kind="tc_router"),
        bench_cfg=_benchmark_config().model_copy(
            update={"configs": (ConfigSpec(kind="tc_router"),)}
        ),
        instance_services=(
            _service("10.0.0.1:55001", "http://10.0.0.1:55001"),
            _service("10.0.0.1:55002", "http://10.0.0.1:55002"),
        ),
        tensorcast_profile=tensorcast_profile,
        pool=[],
        run_dir=tmp_path,
    )

    assert [row.c_target for row in rows] == [2, 4]
    assert events == [
        ("flush", "tc_router c=2 trial=0", 2),
        ("clear", "tc_router c=2 trial=0", 2),
        (
            "tc_init",
            "10.0.0.1:61053",
            Path("tc_router/c2/trial0/migrations.jsonl"),
            "NeverRebalance",
        ),
        ("tc_start", None),
        ("run", 2, 0),
        ("tc_finalize", None),
        ("tc_close", None),
        ("flush", "tc_router c=4 trial=0", 2),
        ("clear", "tc_router c=4 trial=0", 2),
        (
            "tc_init",
            "10.0.0.1:61053",
            Path("tc_router/c4/trial0/migrations.jsonl"),
            "NeverRebalance",
        ),
        ("tc_start", None),
        ("run", 4, 0),
        ("tc_finalize", None),
        ("tc_close", None),
    ]


def test_tensorcast_backend_extra_config_uses_unique_allocator_regions(
    tmp_path: Path,
) -> None:
    worker = _Worker(id="worker0", address="10.0.0.1", gpu_indices=(0, 1))
    bench_cfg = _benchmark_config().model_copy(
        update={"configs": (ConfigSpec(kind="tc_router"),)}
    )
    tensorcast_profile = benchmark_loop.TensorcastProfileServices(
        spec=benchmark_loop.TensorcastLaunchSpec(
            namespace="unit",
            config_dir=str(tmp_path / "tc_cfg"),
            log_dir=str(tmp_path / "tc_log"),
            runtime_home_root=str(tmp_path / "tc_runtime"),
        ),
        launcher=object(),  # type: ignore[arg-type]
        global_worker=worker,
        global_store_service=Service(
            name="global",
            worker_id=worker.id,
            endpoints={"advertise_host": worker.address, "grpc": "10.0.0.1:61050"},
            pid=1,
            pid_path="/tmp/global.pid",
            log_path="/tmp/global.log",
        ),
        daemon_services=(
            (
                worker,
                Service(
                    name="daemon",
                    worker_id=worker.id,
                    endpoints={"grpc": "10.0.0.1:61053"},
                    pid=2,
                    pid_path="/tmp/daemon.pid",
                    log_path="/tmp/daemon.log",
                ),
            ),
        ),
        global_store_address=("10.0.0.1", 61050),
    )

    first = benchmark_loop._tensorcast_backend_extra_config(
        bench_cfg=bench_cfg,
        tensorcast_profile=tensorcast_profile,
        placement=InstanceAssignment(worker=worker, port=55001, gpu_indices=(0,)),
        placement_index=0,
    )
    second = benchmark_loop._tensorcast_backend_extra_config(
        bench_cfg=bench_cfg,
        tensorcast_profile=tensorcast_profile,
        placement=InstanceAssignment(worker=worker, port=55002, gpu_indices=(1,)),
        placement_index=1,
    )

    assert first["tensorcast_kv_mode"] == "explicit_request_transfer"
    assert first["daemon_address"] == "127.0.0.1:61053"
    assert first["instance_directory_address"] == "10.0.0.1:61050"
    assert first["instance_agent_execution_endpoint"] == "10.0.0.1:61400"
    assert first["instance_agent_start_timeout_s"] == 180.0
    assert second["instance_agent_execution_endpoint"] == "10.0.0.1:61401"
    assert first["host_allocator_enabled"] is True
    assert first["host_allocator_region_name"] != second["host_allocator_region_name"]
