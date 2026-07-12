"""Phase 5 sweep orchestrator.

Coordinates SGLang instance launch, gateway launch, workload execution,
metrics aggregation, and teardown. Handles `gw_load_aware` and
`gw_cache_aware` configs (Phase 5). Mooncake / tc_router come in
Phases 6 and 7.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import re
import shutil
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from ..metrics.per_turn import TurnRecordWriter
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

_PLAIN_CONFIG_KINDS = frozenset({"gw_load_aware", "gw_cache_aware", "tc_router"})
_MOONCAKE_CONFIG_KINDS = frozenset({"gw_load_aware_mooncake"})


@dataclass(frozen=True)
class MooncakeBackendOptions:
    extra_config: dict[str, object]
    extra_env: dict[str, str]


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
) -> list:
    """Launch every SGLang instance in parallel, then wait for readiness."""
    services: list[object | None] = [None] * len(placements)
    extra_args: tuple[str, ...] = ()
    if bench_cfg.instances.sglang_log_level is not None:
        extra_args = ("--log-level", bench_cfg.instances.sglang_log_level)

    async def launch_one(idx: int, p: InstanceAssignment) -> None:
        mooncake_extra_config = None
        mooncake_extra_env: dict[str, str] = {}
        if mooncake_service is not None:
            mooncake_options = await _mooncake_backend_options(
                bench_cfg=bench_cfg,
                mooncake_service=mooncake_service,
                worker=p.worker,
            )
            mooncake_extra_config = mooncake_options.extra_config
            mooncake_extra_env = mooncake_options.extra_env
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
            enable_hierarchical_cache=mooncake_service is not None,
            hicache_storage_backend=(
                "mooncake" if mooncake_service is not None else None
            ),
            hicache_storage_backend_extra_config=mooncake_extra_config,
            extra_args=extra_args,
            extra_env=mooncake_extra_env,
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

    summary = aggregate_cell(
        turns_path=turns_path,
        migrations_path=None,
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
    cluster_provider,
    instance_services: list,
    placements: list[InstanceAssignment],
    pool,
    run_dir: Path,
) -> list[RunSummary]:
    """Phase 7 stub: launch Tensorcast global store + daemon (on the placement's
    primary worker), connect a TcRouter via the daemon, run the C × trials sweep.

    Migration policy is `_NeverRebalance`: no plans issued, no migrations
    recorded, but the wiring (poller / runtime / session-state map) is live.
    """
    cfg_dir = run_dir / cfg_spec.kind
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Pick the worker that hosts the global store. The cluster YAML's
    # `service_placement.global_store_worker_id` is the canonical source of
    # truth; we hit `cluster_provider` to resolve that worker.
    cluster_cfg = cluster_provider._config  # type: ignore[attr-defined]
    gs_worker_id = cluster_cfg.service_placement.global_store_worker_id
    workers_by_id = {w.id: w for w in cluster_provider.workers()}
    gs_worker = workers_by_id[gs_worker_id]
    # Daemon-bearing workers are unique by worker_id present in placements.
    daemon_workers = []
    seen: set[str] = set()
    for p in placements:
        wid = p.worker.id
        if wid in seen:
            continue
        seen.add(wid)
        daemon_workers.append(p.worker)
    if gs_worker_id not in {w.id for w in daemon_workers}:
        # Global store is also where SGLang lives in the smoke setup.
        daemon_workers.insert(0, gs_worker)

    tc_spec = TensorcastLaunchSpec(
        namespace=bench_cfg.run_id,
        config_dir=str(cfg_dir / "tensorcast_configs"),
        log_dir=str(cfg_dir / "tensorcast_log"),
        runtime_home_root=str(cfg_dir / "tensorcast_runtime"),
        enable_rdma=bench_cfg.transport.use_rdma,
    )

    tc_launcher = TensorcastLauncher()
    global_store_svc: Optional[object] = None
    daemon_svcs: list = []
    summary_rows: list[RunSummary] = []
    try:
        # 1. Global store
        logger.info(
            "[%s] launching tensorcast global store on %s...",
            cfg_spec.kind,
            gs_worker.id,
        )
        global_store_svc = await tc_launcher.launch_global_store(gs_worker, tc_spec)
        await tc_launcher.wait_global_ready(gs_worker, tc_spec, global_store_svc)
        gs_host_port = (
            str(global_store_svc.endpoints["advertise_host"]),
            tc_spec.global_store_port,
        )
        logger.info("[%s] global store ready at %s:%d", cfg_spec.kind, *gs_host_port)

        # 2. Daemons (one per unique worker)
        capability_secret = f"tc_router-{bench_cfg.run_id}"
        for w in daemon_workers:
            logger.info(
                "[%s] launching tensorcast daemon on %s...", cfg_spec.kind, w.id
            )
            svc = await tc_launcher.launch_daemon(
                w,
                tc_spec,
                global_store_address=gs_host_port,
                capability_token_secret=capability_secret,
            )
            await tc_launcher.wait_daemon_ready(w, tc_spec, svc)
            daemon_svcs.append((w, svc))
            logger.info("[%s] daemon ready at %s", cfg_spec.kind, svc.endpoints["grpc"])

        # 3. TcRouter connection target. For multi-worker setups we pick the
        # first daemon; its directory connects to the global store, which sees
        # all daemons.
        primary_daemon = daemon_svcs[0][1]
        instance_endpoints = {
            svc.endpoints["instance_id"]: svc.endpoints["serving_http"]
            for svc in instance_services
        }
        tc_router_cfg = TcRouterConfig(
            instance_endpoints=instance_endpoints,
            default_model=bench_cfg.model.path,
            daemon_address=primary_daemon.endpoints["grpc"],
            request_timeout_s=600.0,
            load_polling_period_ms=bench_cfg.load_polling.period_ms,
        )

        # 4. Sweep cells
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

                policy = make_policy(cfg_spec.policy)
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
                    summary, info = await _run_one_cell(
                        cell_dir=cell_dir,
                        cfg_kind=cfg_spec.kind,
                        c_target=c_target,
                        trial=trial,
                        bench_cfg=bench_cfg,
                        pool=pool,
                        sampler=sampler,
                        router=tc_router,
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
    finally:
        # Stop daemons first, then global store.
        for w, svc in reversed(daemon_svcs):
            with suppress(Exception):
                await tc_launcher.stop_daemon(w, tc_spec, svc)
        if global_store_svc is not None:
            with suppress(Exception):
                await tc_launcher.stop_global(gs_worker, tc_spec, global_store_svc)
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
                        "0.0.0.0" if cluster_cfg.provider.kind == "static" else None
                    ),
                    mooncake_service=mooncake_service,
                )
                print(
                    "[run_benchmark] SGLang instances ready in "
                    f"{time.monotonic() - t0:.1f}s (profile={profile})"
                )

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
                            rows = await _run_tc_router_config(
                                cfg_spec=cfg_spec,
                                bench_cfg=bench_cfg,
                                cluster_provider=provider,
                                instance_services=instance_services,
                                placements=placements,
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
