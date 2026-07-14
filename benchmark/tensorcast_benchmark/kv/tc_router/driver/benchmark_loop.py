"""Phase 5 sweep orchestrator.

Coordinates SGLang instance launch, gateway launch, workload execution,
metrics aggregation, and teardown. Handles `gw_load_aware` and
`gw_cache_aware` configs (Phase 5). Mooncake / tc_router come in
Phases 6 and 7.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import logging
import re
import shutil
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

import yaml

from ..metrics.per_turn import TurnRecordWriter
from ..metrics.prepared_bundle_signals import (
    verify_prepared_bundle_signals_for_migrations,
)
from ..metrics.summary import RunSummary, aggregate_cell, write_summary_csv
from ..resource import factory as resource_factory
from ..resource.base import ClusterConfig, Worker, load_cluster_config
from ..router.gateway_router import GatewayRouter
from ..router.interface import Router
from ..router.policy import make_policy
from ..router.tc_router import TcRouter, TcRouterConfig
from ..services.base import Service
from ..services.gateway import GatewayLaunchSpec, GatewayLauncher
from ..services.mooncake import (
    MooncakeLaunchSpec,
    MooncakeLauncher,
    _mooncake_advertise_host,
)
from ..services.sglang import (
    SGLangLaunchSpec,
    SGLangLauncher,
    clear_sglang_hicache_storage_backends,
    flush_sglang_caches,
)
from ..services.tensorcast import TensorcastLaunchSpec, TensorcastLauncher
from ..workload.generator import WorkloadDriver
from ..workload.inter_turn_delay import (
    DelayParams,
    LogNormalSampler,
    Preset,
)
from ..workload.trajectory_pool import load_pool

from .config import BenchmarkConfig, ConfigSpec, load_benchmark_yaml
from .placement import InstanceAssignment, plan_instance_placement


logger = logging.getLogger(__name__)


# Map our config kind to the gateway's `--policy` value.
_GATEWAY_POLICY = {
    "gw_load_aware": "power_of_two",
    "gw_cache_aware": "cache_aware",
    "gw_load_aware_mooncake": "power_of_two",
}

_PLAIN_CONFIG_KINDS = frozenset({"gw_load_aware", "gw_cache_aware"})
_MOONCAKE_CONFIG_KINDS = frozenset({"gw_load_aware_mooncake"})
_TENSORCAST_CONFIG_KINDS = frozenset({"tc_router"})


@dataclass(frozen=True)
class MooncakeBackendOptions:
    extra_config: dict[str, object]
    extra_env: dict[str, str]


@dataclass(frozen=True)
class TensorcastProfileServices:
    spec: TensorcastLaunchSpec
    launcher: TensorcastLauncher
    global_worker: Worker
    global_store_service: Service
    daemon_services: tuple[tuple[Worker, Service], ...]
    global_store_address: tuple[str, int]

    @property
    def primary_daemon_service(self) -> Service:
        return self.daemon_services[0][1]


def _gateway_extra_args(cfg_spec: ConfigSpec) -> tuple[str, ...]:
    """Translate gateway baseline policy knobs into sgl-model-gateway CLI args."""
    if not cfg_spec.policy:
        return ()
    if cfg_spec.kind != "gw_cache_aware":
        raise ValueError(f"{cfg_spec.kind} does not accept gateway policy knobs")

    allowed = {
        "cache_threshold": "--cache-threshold",
        "balance_abs_threshold": "--balance-abs-threshold",
        "balance_rel_threshold": "--balance-rel-threshold",
    }
    unknown = sorted(set(cfg_spec.policy) - set(allowed))
    if unknown:
        raise ValueError(f"unknown gw_cache_aware policy knob(s): {', '.join(unknown)}")

    args: list[str] = []
    for key, cli_flag in allowed.items():
        value = cfg_spec.policy.get(key)
        if value is None:
            continue
        args.extend((cli_flag, str(value)))
    return tuple(args)


def _serving_profile_for_config(kind: str) -> str:
    if kind in _PLAIN_CONFIG_KINDS:
        return "plain"
    if kind in _MOONCAKE_CONFIG_KINDS:
        return "mooncake"
    if kind in _TENSORCAST_CONFIG_KINDS:
        return "tensorcast"
    raise ValueError(f"unknown config kind for serving profile: {kind}")


def _serving_profiles_for_configs(configs: list[ConfigSpec]) -> list[str]:
    profiles: list[str] = []
    for cfg_spec in configs:
        profile = _serving_profile_for_config(cfg_spec.kind)
        if profile not in profiles:
            profiles.append(profile)
    return profiles


def _derive_nccl_port(serving_port: int) -> int:
    for candidate in (serving_port + 100, serving_port - 100):
        if 1024 <= candidate <= 65535:
            return candidate
    raise ValueError(
        f"cannot derive a valid nccl_port from serving_port={serving_port}"
    )


def _first_mooncake_device_name(device_name: str) -> str:
    devices = [part.strip() for part in device_name.split(",") if part.strip()]
    if not devices:
        raise ValueError("Mooncake RDMA device_name must contain at least one device")
    return devices[0]


def _parse_rdma_netdev(rdma_output: str, device_name: str) -> str:
    pattern = re.compile(
        rf"\blink\s+{re.escape(device_name)}/\S+.*\bnetdev\s+(?P<netdev>\S+)"
    )
    for line in rdma_output.splitlines():
        match = pattern.search(line)
        if match is not None:
            return match.group("netdev")
    raise RuntimeError(
        f"could not find netdev for RDMA device {device_name!r} in `rdma link show`"
    )


def _parse_ipv4_addr(ip_output: str, netdev: str) -> str:
    match = re.search(r"\binet\s+(?P<cidr>[0-9.]+/\d+)", ip_output)
    if match is None:
        raise RuntimeError(f"could not find IPv4 address for netdev {netdev!r}")
    return match.group("cidr").split("/", 1)[0]


async def _resolve_worker_rdma_ipv4(worker: Worker, device_name: str) -> str:
    rdma_proc = await worker.run(
        "rdma link show",
        timeout_s=30.0,
        check=True,
    )
    netdev = _parse_rdma_netdev(rdma_proc.stdout, device_name)
    ip_proc = await worker.run(
        f"ip -o -4 addr show dev {netdev} scope global",
        timeout_s=30.0,
        check=True,
    )
    return _parse_ipv4_addr(ip_proc.stdout, netdev)


async def _mooncake_backend_options(
    *,
    bench_cfg: BenchmarkConfig,
    mooncake_service: Service,
    worker: Worker,
) -> MooncakeBackendOptions:
    local_hostname = _mooncake_advertise_host(worker.address)
    extra_env: dict[str, str] = {}
    device_name = bench_cfg.mooncake.device_name.strip()
    if bench_cfg.transport.use_rdma and device_name:
        rpc_device_name = _first_mooncake_device_name(device_name)
        local_hostname = await _resolve_worker_rdma_ipv4(worker, rpc_device_name)
        extra_env["MC_TCP_BIND_ADDRESS"] = local_hostname

    payload: dict[str, object] = {
        "master_server_address": mooncake_service.endpoints["master_server_address"],
        "metadata_server": mooncake_service.endpoints["metadata_server"],
        "local_hostname": local_hostname,
        "protocol": "rdma" if bench_cfg.transport.use_rdma else "tcp",
        "global_segment_size": bench_cfg.mooncake.global_segment_size,
    }
    if device_name:
        payload["device_name"] = device_name
    return MooncakeBackendOptions(extra_config=payload, extra_env=extra_env)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def _model_version(model_path: str) -> str:
    return hashlib.sha256(str(model_path).encode("utf-8")).hexdigest()[:16]


def _daemon_workers_for_placements(
    placements: list[InstanceAssignment],
    *,
    global_store_worker: Worker,
) -> list[Worker]:
    daemon_workers: list[Worker] = []
    seen: set[str] = set()
    for placement in placements:
        worker_id = placement.worker.id
        if worker_id in seen:
            continue
        seen.add(worker_id)
        daemon_workers.append(placement.worker)
    if global_store_worker.id not in {worker.id for worker in daemon_workers}:
        daemon_workers.insert(0, global_store_worker)
    return daemon_workers


async def _launch_tensorcast_profile_services(
    *,
    cfg_kind: str,
    bench_cfg: BenchmarkConfig,
    cluster_provider,
    placements: list[InstanceAssignment],
    profile_dir: Path,
) -> TensorcastProfileServices:
    cluster_cfg = cluster_provider._config  # type: ignore[attr-defined]
    workers_by_id = {worker.id: worker for worker in cluster_provider.workers()}
    global_worker = workers_by_id[cluster_cfg.service_placement.global_store_worker_id]
    tc_spec = TensorcastLaunchSpec(
        namespace=bench_cfg.run_id,
        global_store_port=bench_cfg.tensorcast.global_store_port,
        daemon_port=bench_cfg.tensorcast.daemon_port,
        daemon_p2p_port=bench_cfg.tensorcast.daemon_p2p_port,
        daemon_stable_bytes=bench_cfg.tensorcast.daemon_stable_bytes,
        config_dir=str(profile_dir / "tensorcast_configs"),
        log_dir=str(profile_dir / "tensorcast_log"),
        runtime_home_root=str(profile_dir / "tensorcast_runtime"),
        enable_rdma=bench_cfg.transport.use_rdma,
    )
    launcher = TensorcastLauncher()
    global_store_service = await launcher.launch_global_store(global_worker, tc_spec)
    await launcher.wait_global_ready(global_worker, tc_spec, global_store_service)
    global_store_address = (
        str(global_store_service.endpoints["advertise_host"]),
        tc_spec.global_store_port,
    )
    daemon_services: list[tuple[Worker, Service]] = []
    capability_secret = f"tc_router-{bench_cfg.run_id}"
    try:
        for worker in _daemon_workers_for_placements(
            placements,
            global_store_worker=global_worker,
        ):
            logger.info(
                "[%s] launching tensorcast daemon on %s...", cfg_kind, worker.id
            )
            service = await launcher.launch_daemon(
                worker,
                tc_spec,
                global_store_address=global_store_address,
                capability_token_secret=capability_secret,
            )
            await launcher.wait_daemon_ready(worker, tc_spec, service)
            daemon_services.append((worker, service))
            logger.info("[%s] daemon ready at %s", cfg_kind, service.endpoints["grpc"])
    except BaseException:
        for worker, service in reversed(daemon_services):
            with suppress(Exception):
                await launcher.stop_daemon(worker, tc_spec, service)
        with suppress(Exception):
            await launcher.stop_global(global_worker, tc_spec, global_store_service)
        raise

    return TensorcastProfileServices(
        spec=tc_spec,
        launcher=launcher,
        global_worker=global_worker,
        global_store_service=global_store_service,
        daemon_services=tuple(daemon_services),
        global_store_address=global_store_address,
    )


async def _stop_tensorcast_profile_services(
    services: TensorcastProfileServices | None,
) -> None:
    if services is None:
        return
    for worker, service in reversed(services.daemon_services):
        with suppress(Exception):
            await services.launcher.stop_daemon(worker, services.spec, service)
    with suppress(Exception):
        await services.launcher.stop_global(
            services.global_worker,
            services.spec,
            services.global_store_service,
        )


def _tensorcast_backend_extra_config(
    *,
    bench_cfg: BenchmarkConfig,
    tensorcast_profile: TensorcastProfileServices,
    placement: InstanceAssignment,
    placement_index: int,
) -> dict[str, object]:
    safe_run_id = _safe_name(bench_cfg.run_id)
    payload: dict[str, object] = {
        "daemon_address": f"127.0.0.1:{tensorcast_profile.spec.daemon_port}",
        "namespace": bench_cfg.run_id,
        "engine": "sglang",
        "model_id": Path(bench_cfg.model.path).name,
        "model_version": _model_version(bench_cfg.model.path),
        "policy_profile": "durable",
        "instance_directory_address": (
            f"{tensorcast_profile.global_store_address[0]}:"
            f"{tensorcast_profile.global_store_address[1]}"
        ),
        "instance_agent_execution_endpoint": (
            f"{placement.worker.address}:"
            f"{bench_cfg.tensorcast.instance_agent_base_port + placement_index}"
        ),
        "instance_agent_start_timeout_s": (
            bench_cfg.tensorcast.instance_agent_start_timeout_s
        ),
        "tensorcast_kv_mode": "explicit_request_transfer",
        "background_page_publish": False,
        "ordinary_storage_prefetch": False,
        "record_host_residency_for_publish": True,
        "logical_session_id_source": "routing_key",
    }
    if bench_cfg.tensorcast.host_allocator_enabled:
        payload["host_allocator_enabled"] = True
        payload["host_allocator_region_ttl_ms"] = (
            bench_cfg.tensorcast.host_allocator_region_ttl_ms
        )
        payload["host_allocator_region_name"] = (
            f"{bench_cfg.tensorcast.host_allocator_region_name_prefix}-"
            f"{safe_run_id}-{placement_index}"
        )
    return payload


def _wait_tensorcast_instance_routes_sync(
    *,
    daemon_address: str,
    expected_routes: dict[str, str],
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    import tensorcast as tc

    runtime = tc.connect(daemon_address=daemon_address)
    deadline = time.monotonic() + timeout_s
    last_error = "no probe yet"
    try:
        while time.monotonic() < deadline:
            missing: list[str] = []
            mismatched: list[str] = []
            for instance_id, expected_endpoint in expected_routes.items():
                try:
                    route = (
                        runtime.directory()
                        .resolve_instance_execution(instance_id)
                        .value
                    )
                except Exception as exc:  # noqa: BLE001
                    missing.append(instance_id)
                    last_error = f"{instance_id}: {type(exc).__name__}: {exc}"
                    continue
                actual_endpoint = str(route.execution_endpoint or "")
                if expected_endpoint and actual_endpoint != expected_endpoint:
                    mismatched.append(
                        f"{instance_id} expected={expected_endpoint} "
                        f"actual={actual_endpoint}"
                    )
            if not missing and not mismatched:
                return
            detail = []
            if missing:
                detail.append(f"missing={missing}")
            if mismatched:
                detail.append(f"mismatched={mismatched}")
            last_error = "; ".join(detail)
            time.sleep(poll_interval_s)
    finally:
        with suppress(Exception):
            runtime.close()
    raise TimeoutError(
        "Tensorcast instance directory routes not ready within "
        f"{timeout_s}s: {last_error}"
    )


async def _wait_tensorcast_instance_routes(
    *,
    bench_cfg: BenchmarkConfig,
    tensorcast_profile: TensorcastProfileServices,
    placements: list[InstanceAssignment],
    timeout_s: float,
) -> None:
    expected_routes = {
        placement.instance_id: (
            f"{placement.worker.address}:"
            f"{bench_cfg.tensorcast.instance_agent_base_port + placement_index}"
        )
        for placement_index, placement in enumerate(placements)
    }
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: _wait_tensorcast_instance_routes_sync(
            daemon_address=tensorcast_profile.primary_daemon_service.endpoints["grpc"],
            expected_routes=expected_routes,
            timeout_s=timeout_s,
            poll_interval_s=2.0,
        ),
    )


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _resolve_run_dir(bench_cfg: BenchmarkConfig, *, root: Path) -> Path:
    run_dir = root / f"{_now_stamp()}_{bench_cfg.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _cluster_config_for_run(cluster_yaml: Path, run_dir: Path) -> ClusterConfig:
    """Return the effective cluster config for this run.

    Local and static workers should write all transient state under the run
    directory so a completed or failed experiment is self-contained. Static
    workers share /mnt/data across hosts, so the run directory is visible to
    both local and SSH workers. Other remote providers keep their configured
    scratch paths because those paths may be pre-mounted shared storage
    outside the driver filesystem.
    """
    cfg = load_cluster_config(cluster_yaml)
    if cfg.provider.kind not in {"local", "static"}:
        return cfg

    data = cfg.model_dump(mode="json")
    scratch_root = run_dir / "scratch"
    data["driver_host"]["scratch_dir"] = str(scratch_root / "driver")
    if cfg.provider.kind == "local":
        data["mount"]["path"] = str(run_dir / "mount")
    for worker in data["workers"]:
        worker["scratch_dir"] = str(scratch_root / "workers" / worker["id"])
    return ClusterConfig.model_validate(data)


def _save_resolved_configs(
    run_dir: Path,
    *,
    cluster_yaml: Path,
    bench_yaml: Path,
    bench_cfg: BenchmarkConfig,
    cluster_cfg: ClusterConfig,
) -> None:
    shutil.copy2(cluster_yaml, run_dir / "cluster_input.yaml")
    shutil.copy2(bench_yaml, run_dir / "benchmark.yaml")
    (run_dir / "cluster.yaml").write_text(
        yaml.safe_dump(
            cluster_cfg.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
        )
    )
    # Resolved (post-validation) form.
    (run_dir / "benchmark_resolved.yaml").write_text(
        yaml.safe_dump(
            bench_cfg.model_dump(), sort_keys=False, default_flow_style=False
        )
    )


def _append_top_level_csv(
    rows: list[RunSummary], top_csv: Path, *, run_id: str
) -> None:
    """Append summary rows to a rolling `outputs/benchmark_results.csv`.

    Adds a `run_id` column up front so multiple runs can coexist.
    """
    if not rows:
        return
    fieldnames = ["run_id"] + list(RunSummary.model_fields.keys())
    write_header = not top_csv.exists()
    top_csv.parent.mkdir(parents=True, exist_ok=True)
    with top_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            d = {"run_id": run_id, **row.model_dump()}
            writer.writerow(d)


async def _launch_sglang_fleet(
    *,
    placements: list[InstanceAssignment],
    bench_cfg: BenchmarkConfig,
    sglang_launcher: SGLangLauncher,
    ready_timeout_s: float,
    bind_host: str | None = None,
    mooncake_service: Service | None = None,
    tensorcast_profile: TensorcastProfileServices | None = None,
) -> list:
    """Launch every SGLang instance in parallel, then wait for readiness."""
    services: list[object | None] = [None] * len(placements)
    extra_args: tuple[str, ...] = ()
    if bench_cfg.instances.sglang_log_level is not None:
        extra_args = ("--log-level", bench_cfg.instances.sglang_log_level)

    async def launch_one(idx: int, p: InstanceAssignment) -> None:
        storage_backend = None
        storage_extra_config = None
        storage_extra_env: dict[str, str] = {}
        hicache_mem_layout = "page_first_direct"
        hicache_io_backend = "direct"
        if mooncake_service is not None:
            mooncake_options = await _mooncake_backend_options(
                bench_cfg=bench_cfg,
                mooncake_service=mooncake_service,
                worker=p.worker,
            )
            storage_backend = "mooncake"
            storage_extra_config = mooncake_options.extra_config
            storage_extra_env = mooncake_options.extra_env
        if tensorcast_profile is not None:
            storage_backend = "tensorcast"
            storage_extra_config = _tensorcast_backend_extra_config(
                bench_cfg=bench_cfg,
                tensorcast_profile=tensorcast_profile,
                placement=p,
                placement_index=idx,
            )
            hicache_mem_layout = bench_cfg.tensorcast.hicache_mem_layout
            hicache_io_backend = bench_cfg.tensorcast.hicache_io_backend
        spec = SGLangLaunchSpec(
            model_path=bench_cfg.model.path,
            host=p.worker.address,
            bind_host=bind_host,
            port=p.port,
            tp_size=bench_cfg.model.tp_size,
            nccl_port=_derive_nccl_port(p.port),
            gpu_indices=p.gpu_indices,
            mem_fraction_static=bench_cfg.instances.mem_fraction_static,
            page_size=bench_cfg.instances.page_size,
            enable_hierarchical_cache=(
                mooncake_service is not None or tensorcast_profile is not None
            ),
            hicache_mem_layout=hicache_mem_layout,
            hicache_io_backend=hicache_io_backend,
            hicache_storage_backend=storage_backend,
            hicache_storage_backend_extra_config=storage_extra_config,
            extra_args=extra_args,
            extra_env=storage_extra_env,
        )
        service = await sglang_launcher.launch(p.worker, spec)
        services[idx] = service

    try:
        await asyncio.gather(*(launch_one(idx, p) for idx, p in enumerate(placements)))
        ready_services = [service for service in services if service is not None]
        logger.info(
            "launched %d SGLang services; waiting ready...", len(ready_services)
        )
        await asyncio.gather(
            *[
                sglang_launcher.wait_ready(s, timeout_s=ready_timeout_s)
                for s in ready_services
            ]
        )
    except BaseException:
        for p, svc in reversed(list(zip(placements, services))):
            if svc is not None:
                with suppress(Exception):
                    await sglang_launcher.stop(p.worker, svc)
        raise
    logger.info("all %d SGLang services are healthy", len(services))
    return [service for service in services if service is not None]


async def _stop_sglang_fleet(
    *,
    placements: list[InstanceAssignment],
    services: list,
    sglang_launcher: SGLangLauncher,
) -> None:
    for p, svc in zip(placements, services):
        try:
            await sglang_launcher.stop(p.worker, svc)
        except Exception:  # noqa: BLE001
            logger.exception("failed to stop SGLang service %s", svc.name)


def _sglang_serving_urls(instance_services: list) -> tuple[str, ...]:
    return tuple(
        str(svc.endpoints["serving_http"]).rstrip("/") for svc in instance_services
    )


async def _flush_sglang_instances(instance_services: list, *, label: str) -> None:
    urls = _sglang_serving_urls(instance_services)
    logger.info("[%s] flushing SGLang caches on %d endpoint(s)...", label, len(urls))
    result = await flush_sglang_caches(urls, timeout_s=180.0, poll_interval_s=1.0)
    logger.info(
        "[%s] SGLang cache flush complete on %d endpoint(s) in %.2fs",
        label,
        len(result.attempts),
        result.elapsed_s,
    )


async def _clear_sglang_hicache_storage(instance_services: list, *, label: str) -> None:
    urls = _sglang_serving_urls(instance_services)
    logger.info(
        "[%s] clearing SGLang HiCache storage backends on %d endpoint(s)...",
        label,
        len(urls),
    )
    result = await clear_sglang_hicache_storage_backends(
        urls, timeout_s=180.0, poll_interval_s=1.0
    )
    logger.info(
        "[%s] SGLang HiCache storage clear complete on %d endpoint(s) in %.2fs",
        label,
        len(result.attempts),
        result.elapsed_s,
    )


async def _run_one_cell(
    *,
    cell_dir: Path,
    cfg_kind: str,
    c_target: int,
    trial: int,
    bench_cfg: BenchmarkConfig,
    pool,
    sampler: LogNormalSampler,
    router: Router,
    migrations_path: Path | None = None,
    post_run_hook: Callable[[], Awaitable[None]] | None = None,
) -> tuple[RunSummary, dict]:
    cell_dir.mkdir(parents=True, exist_ok=True)
    turns_path = cell_dir / "turns.jsonl"

    with TurnRecordWriter(turns_path) as writer:
        wd = WorkloadDriver(
            router=router,
            pool=pool,
            inter_turn_sampler=sampler,
            c_target=c_target,
            wall_seconds=bench_cfg.workload.wall_seconds,
            warmup_seconds=bench_cfg.workload.warmup_seconds,
            warmup_counts=bench_cfg.workload.warmup_counts,
            start_jitter_s=bench_cfg.workload.start_jitter_s,
            max_new_tokens_clip=bench_cfg.workload.max_new_tokens_clip,
            record_sink=writer.write,
            rng_seed=trial,
            supervisor_tick_s=0.1,
        )
        outcome = await wd.run()

    if post_run_hook is not None:
        await post_run_hook()

    summary = aggregate_cell(
        turns_path=turns_path,
        migrations_path=migrations_path,
        config=cfg_kind,
        c_target=c_target,
        trial=trial,
        inter_turn_delay_preset=bench_cfg.workload.inter_turn_delay.preset,
        transport_mode="rdma" if bench_cfg.transport.use_rdma else "tcp",
    )

    info = {
        "total_turns": outcome.total_turns,
        "successful_turns": outcome.successful_turns,
        "failed_turns": outcome.failed_turns,
        "wall_seconds_actual": outcome.wall_seconds_actual,
        "distinct_sessions_started": outcome.distinct_sessions_started,
    }
    return summary, info


async def _run_gateway_config(
    *,
    cfg_spec: ConfigSpec,
    bench_cfg: BenchmarkConfig,
    instance_services: list,
    pool,
    run_dir: Path,
    clear_hicache_storage_between_cells: bool = False,
) -> list[RunSummary]:
    """Run gateway cells with a fresh gateway process per cell."""
    if cfg_spec.kind not in _GATEWAY_POLICY:
        return []
    policy = _GATEWAY_POLICY[cfg_spec.kind]

    cfg_dir = run_dir / cfg_spec.kind
    cfg_dir.mkdir(parents=True, exist_ok=True)

    gateway_launcher = GatewayLauncher()
    summary_rows: list[RunSummary] = []

    # Inter-turn sampler — same params for all (c_target, trial) cells of this config.
    delay_params = DelayParams.from_preset(
        Preset(bench_cfg.workload.inter_turn_delay.preset),
        custom_mu=bench_cfg.workload.inter_turn_delay.custom_mu,
        custom_sigma=bench_cfg.workload.inter_turn_delay.custom_sigma,
    )

    for c_target in bench_cfg.workload.c_target_sweep:
        for trial in range(bench_cfg.workload.trials):
            cell_dir = cfg_dir / f"c{c_target}" / f"trial{trial}"
            sampler = LogNormalSampler(
                delay_params,
                seed=hash((cfg_spec.kind, c_target, trial)) & 0xFFFFFFFF,
            )
            print(f"[run] {cfg_spec.kind} c={c_target} trial={trial} -> {cell_dir}")

            await _flush_sglang_instances(
                instance_services,
                label=f"{cfg_spec.kind} c={c_target} trial={trial}",
            )
            if clear_hicache_storage_between_cells:
                await _clear_sglang_hicache_storage(
                    instance_services,
                    label=f"{cfg_spec.kind} c={c_target} trial={trial}",
                )

            gateway_spec = GatewayLaunchSpec(
                worker_urls=tuple(
                    svc.endpoints["serving_http"] for svc in instance_services
                ),
                policy=policy,  # type: ignore[arg-type]
                host=bench_cfg.gateway.host,
                port=bench_cfg.gateway.port,
                log_dir=str(cell_dir / "gateway_log"),
                extra_args=_gateway_extra_args(cfg_spec),
            )
            logger.info("[%s] launching gateway (policy=%s)...", cfg_spec.kind, policy)
            gateway_svc = await gateway_launcher.launch(gateway_spec)
            router: Router | None = None
            try:
                await gateway_launcher.wait_ready(gateway_svc, timeout_s=180.0)
                logger.info(
                    "[%s] gateway healthy at %s",
                    cfg_spec.kind,
                    gateway_svc.endpoints["openai_http"],
                )
                router = GatewayRouter(
                    gateway_svc.endpoints["openai_http"],
                    default_model=bench_cfg.model.path,
                )
                summary, info = await _run_one_cell(
                    cell_dir=cell_dir,
                    cfg_kind=cfg_spec.kind,
                    c_target=c_target,
                    trial=trial,
                    bench_cfg=bench_cfg,
                    pool=pool,
                    sampler=sampler,
                    router=router,
                )
                summary_rows.append(summary)
                print(
                    f"  -> turns={info['total_turns']} "
                    f"(success={info['successful_turns']}, fail={info['failed_turns']}); "
                    f"ttft p50={summary.ttft_p50_ms} p95={summary.ttft_p95_ms} "
                    f"cached_ratio={summary.cached_token_ratio_mean}"
                )
            finally:
                if router is not None:
                    with suppress(Exception):
                        await router.close()
                try:
                    await gateway_launcher.stop(gateway_svc)
                except Exception:  # noqa: BLE001
                    logger.exception("failed to stop gateway")
    return summary_rows


async def _run_tc_router_config(
    *,
    cfg_spec: ConfigSpec,
    bench_cfg: BenchmarkConfig,
    instance_services: list,
    tensorcast_profile: TensorcastProfileServices,
    pool,
    run_dir: Path,
) -> list[RunSummary]:
    """Run tc_router cells against an already-launched Tensorcast serving profile."""
    cfg_dir = run_dir / cfg_spec.kind
    cfg_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[RunSummary] = []
    primary_daemon = tensorcast_profile.primary_daemon_service
    instance_endpoints = {
        svc.endpoints["instance_id"]: svc.endpoints["serving_http"]
        for svc in instance_services
    }
    instance_log_paths = {
        str(svc.endpoints["instance_id"]): Path(svc.log_path)
        for svc in instance_services
    }

    delay_params = DelayParams.from_preset(
        Preset(bench_cfg.workload.inter_turn_delay.preset),
        custom_mu=bench_cfg.workload.inter_turn_delay.custom_mu,
        custom_sigma=bench_cfg.workload.inter_turn_delay.custom_sigma,
    )
    for c_target in bench_cfg.workload.c_target_sweep:
        for trial in range(bench_cfg.workload.trials):
            cell_dir = cfg_dir / f"c{c_target}" / f"trial{trial}"
            migrations_path = cell_dir / "migrations.jsonl"
            sampler = LogNormalSampler(
                delay_params,
                seed=hash((cfg_spec.kind, c_target, trial)) & 0xFFFFFFFF,
            )
            print(f"[run] {cfg_spec.kind} c={c_target} trial={trial} -> {cell_dir}")
            await _flush_sglang_instances(
                instance_services,
                label=f"{cfg_spec.kind} c={c_target} trial={trial}",
            )
            if bench_cfg.tensorcast.clear_storage_between_cells:
                await _clear_sglang_hicache_storage(
                    instance_services,
                    label=f"{cfg_spec.kind} c={c_target} trial={trial}",
                )

            policy = make_policy(cfg_spec.policy)
            tc_router_cfg = TcRouterConfig(
                instance_endpoints=instance_endpoints,
                default_model=bench_cfg.model.path,
                daemon_address=primary_daemon.endpoints["grpc"],
                request_timeout_s=600.0,
                load_polling_period_ms=bench_cfg.load_polling.period_ms,
                migrations_path=str(migrations_path),
            )
            tc_router = TcRouter(tc_router_cfg, policy=policy)
            try:
                await tc_router.start()
                logger.info(
                    "[%s] TcRouter ready (policy=%s, daemon=%s, %d instances)",
                    cfg_spec.kind,
                    policy.name,
                    primary_daemon.endpoints["grpc"],
                    len(instance_endpoints),
                )

                async def finalize_and_verify_migrations() -> None:
                    await tc_router.finalize_migrations()
                    await verify_prepared_bundle_signals_for_migrations(
                        migrations_path=migrations_path,
                        instance_log_paths=instance_log_paths,
                    )

                summary, info = await _run_one_cell(
                    cell_dir=cell_dir,
                    cfg_kind=cfg_spec.kind,
                    c_target=c_target,
                    trial=trial,
                    bench_cfg=bench_cfg,
                    pool=pool,
                    sampler=sampler,
                    router=tc_router,
                    migrations_path=migrations_path,
                    post_run_hook=finalize_and_verify_migrations,
                )
                summary_rows.append(summary)
                print(
                    f"  -> turns={info['total_turns']} "
                    f"(success={info['successful_turns']}, fail={info['failed_turns']}); "
                    f"ttft p50={summary.ttft_p50_ms} p95={summary.ttft_p95_ms} "
                    f"cached_ratio={summary.cached_token_ratio_mean}"
                )
            finally:
                with suppress(Exception):
                    await tc_router.close()
    return summary_rows


async def _launch_mooncake_service(
    *,
    bench_cfg: BenchmarkConfig,
    cluster_provider,
) -> tuple[Worker, MooncakeLauncher, Service]:
    cluster_cfg = cluster_provider._config  # type: ignore[attr-defined]
    workers_by_id = {w.id: w for w in cluster_provider.workers()}
    worker_id = cluster_cfg.service_placement.mooncake_master_worker_id
    worker = workers_by_id[worker_id]
    spec = MooncakeLaunchSpec(
        http_metadata_server_port=bench_cfg.mooncake.http_metadata_server_port,
        master_port=bench_cfg.mooncake.master_port,
        eviction_high_watermark_ratio=(
            bench_cfg.mooncake.eviction_high_watermark_ratio
        ),
    )
    launcher = MooncakeLauncher()
    service = await launcher.launch(worker, spec)
    try:
        await launcher.wait_ready(
            service,
            timeout_s=spec.service_ready_timeout_s,
            poll_interval_s=spec.service_poll_interval_s,
        )
    except BaseException:
        with suppress(Exception):
            await launcher.stop(worker, service)
        raise
    return worker, launcher, service


async def run_benchmark(
    cluster_yaml: Path,
    bench_yaml: Path,
    *,
    outputs_root: Optional[Path] = None,
    config_filter: Optional[set[str]] = None,
    sglang_ready_timeout_s: float = 1500.0,
) -> Path:
    """Top-level entry. Returns the run output directory."""
    bench_cfg = load_benchmark_yaml(bench_yaml)
    outputs_root = outputs_root or Path(__file__).resolve().parents[1] / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    run_dir = _resolve_run_dir(bench_cfg, root=outputs_root)
    print(f"[run_benchmark] run_dir = {run_dir}")
    cluster_cfg = _cluster_config_for_run(cluster_yaml, run_dir)
    _save_resolved_configs(
        run_dir,
        cluster_yaml=cluster_yaml,
        bench_yaml=bench_yaml,
        bench_cfg=bench_cfg,
        cluster_cfg=cluster_cfg,
    )

    effective_cluster_yaml = run_dir / "cluster.yaml"
    provider = resource_factory.from_cluster_config(effective_cluster_yaml)

    workers = provider.workers()
    placements = plan_instance_placement(
        workers,
        instances_count=bench_cfg.instances.count,
        tp_size=bench_cfg.model.tp_size,
        base_port=bench_cfg.instances.base_port,
    )

    # Save placement plan for postmortem / debugging.
    (run_dir / "placement.txt").write_text(
        "\n".join(
            f"{i}: worker={p.worker.id} address={p.worker.address} "
            f"port={p.port} gpus={list(p.gpu_indices)}"
            for i, p in enumerate(placements)
        )
    )

    print("[run_benchmark] cluster health check...")
    await provider.health_check()

    sglang_launcher = SGLangLauncher()
    summary_rows: list[RunSummary] = []
    try:
        # Load workload pool ONCE, shared across configs/cells.
        print(
            f"[run_benchmark] loading trajectory pool from {bench_cfg.workload.dataset_path}..."
        )
        pool = load_pool(
            bench_cfg.workload.dataset_path,
            min_turns=bench_cfg.workload.pool_filter.min_turns,
            min_total_tokens=bench_cfg.workload.pool_filter.min_total_tokens,
            seed=0,
        )
        print(f"[run_benchmark] pool size: {len(pool)} trajectories")

        selected_configs = [
            cfg_spec
            for cfg_spec in bench_cfg.configs
            if config_filter is None or cfg_spec.kind in config_filter
        ]

        for profile in _serving_profiles_for_configs(selected_configs):
            profile_configs = [
                cfg_spec
                for cfg_spec in selected_configs
                if _serving_profile_for_config(cfg_spec.kind) == profile
            ]
            mooncake_worker: Worker | None = None
            mooncake_launcher: MooncakeLauncher | None = None
            mooncake_service: Service | None = None
            tensorcast_profile: TensorcastProfileServices | None = None
            instance_services: list = []
            try:
                if profile == "mooncake":
                    print("[run_benchmark] launching Mooncake master...")
                    (
                        mooncake_worker,
                        mooncake_launcher,
                        mooncake_service,
                    ) = await _launch_mooncake_service(
                        bench_cfg=bench_cfg,
                        cluster_provider=provider,
                    )
                    print(
                        "[run_benchmark] Mooncake master ready at "
                        f"{mooncake_service.endpoints['master_server_address']}"
                    )
                if profile == "tensorcast":
                    print("[run_benchmark] launching Tensorcast profile services...")
                    tensorcast_profile = await _launch_tensorcast_profile_services(
                        cfg_kind="tc_router",
                        bench_cfg=bench_cfg,
                        cluster_provider=provider,
                        placements=placements,
                        profile_dir=run_dir / "tc_router",
                    )
                    print(
                        "[run_benchmark] Tensorcast services ready; primary daemon "
                        f"at {tensorcast_profile.primary_daemon_service.endpoints['grpc']}"
                    )

                print(
                    f"[run_benchmark] launching {len(placements)} SGLang instances "
                    f"({bench_cfg.model.path}, tp={bench_cfg.model.tp_size}, "
                    f"profile={profile})..."
                )
                t0 = time.monotonic()
                instance_services = await _launch_sglang_fleet(
                    placements=placements,
                    bench_cfg=bench_cfg,
                    sglang_launcher=sglang_launcher,
                    ready_timeout_s=sglang_ready_timeout_s,
                    bind_host=(
                        "0.0.0.0"
                        if cluster_cfg.provider.kind == "static"
                        and profile != "tensorcast"
                        else None
                    ),
                    mooncake_service=mooncake_service,
                    tensorcast_profile=tensorcast_profile,
                )
                print(
                    "[run_benchmark] SGLang instances ready in "
                    f"{time.monotonic() - t0:.1f}s (profile={profile})"
                )
                if tensorcast_profile is not None:
                    print("[run_benchmark] waiting for Tensorcast instance routes...")
                    await _wait_tensorcast_instance_routes(
                        bench_cfg=bench_cfg,
                        tensorcast_profile=tensorcast_profile,
                        placements=placements,
                        timeout_s=180.0,
                    )
                    print("[run_benchmark] Tensorcast instance routes ready")

                for cfg_spec in profile_configs:
                    try:
                        if cfg_spec.kind in _GATEWAY_POLICY:
                            rows = await _run_gateway_config(
                                cfg_spec=cfg_spec,
                                bench_cfg=bench_cfg,
                                instance_services=instance_services,
                                pool=pool,
                                run_dir=run_dir,
                                clear_hicache_storage_between_cells=(
                                    profile == "mooncake"
                                    and bench_cfg.mooncake.clear_storage_between_cells
                                ),
                            )
                        elif cfg_spec.kind == "tc_router":
                            if tensorcast_profile is None:
                                raise RuntimeError(
                                    "tc_router requires tensorcast profile services"
                                )
                            rows = await _run_tc_router_config(
                                cfg_spec=cfg_spec,
                                bench_cfg=bench_cfg,
                                instance_services=instance_services,
                                tensorcast_profile=tensorcast_profile,
                                pool=pool,
                                run_dir=run_dir,
                            )
                        else:
                            logger.info(
                                "skipping unsupported config %s in this driver",
                                cfg_spec.kind,
                            )
                            rows = []
                        summary_rows.extend(rows)
                    except Exception:  # noqa: BLE001
                        logger.exception("config %s failed; continuing", cfg_spec.kind)
                        traceback.print_exc()
            finally:
                if instance_services:
                    print(
                        f"[run_benchmark] tearing down SGLang instances "
                        f"(profile={profile})..."
                    )
                    with suppress(Exception):
                        await _stop_sglang_fleet(
                            placements=placements,
                            services=instance_services,
                            sglang_launcher=sglang_launcher,
                        )
                if (
                    mooncake_worker is not None
                    and mooncake_launcher is not None
                    and mooncake_service is not None
                ):
                    print("[run_benchmark] tearing down Mooncake master...")
                    with suppress(Exception):
                        await mooncake_launcher.stop(mooncake_worker, mooncake_service)
                if tensorcast_profile is not None:
                    print("[run_benchmark] tearing down Tensorcast services...")
                    with suppress(Exception):
                        await _stop_tensorcast_profile_services(tensorcast_profile)

        # Per-run summary
        write_summary_csv(summary_rows, run_dir / "summary.csv")
        # Top-level rolling CSV
        _append_top_level_csv(
            summary_rows,
            outputs_root / "benchmark_results.csv",
            run_id=run_dir.name,
        )
        print(f"[run_benchmark] wrote summary.csv with {len(summary_rows)} rows")
        return run_dir
    finally:
        with suppress(Exception):
            await sglang_launcher.aclose()
