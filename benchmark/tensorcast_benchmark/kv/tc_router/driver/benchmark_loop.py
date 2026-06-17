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
import shutil
import time
import traceback
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from ..metrics.per_turn import TurnRecordWriter
from ..metrics.summary import RunSummary, aggregate_cell, write_summary_csv
from ..resource import factory as resource_factory
from ..router.gateway_router import GatewayRouter
from ..router.policy import make_policy
from ..router.tc_router import TcRouter, TcRouterConfig
from ..services.gateway import GatewayLaunchSpec, GatewayLauncher
from ..services.sglang import SGLangLaunchSpec, SGLangLauncher
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
}


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _resolve_run_dir(bench_cfg: BenchmarkConfig, *, root: Path) -> Path:
    run_dir = root / f"{_now_stamp()}_{bench_cfg.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_resolved_configs(
    run_dir: Path,
    *,
    cluster_yaml: Path,
    bench_yaml: Path,
    bench_cfg: BenchmarkConfig,
) -> None:
    shutil.copy2(cluster_yaml, run_dir / "cluster.yaml")
    shutil.copy2(bench_yaml, run_dir / "benchmark.yaml")
    # Resolved (post-validation) form.
    (run_dir / "benchmark_resolved.yaml").write_text(
        yaml.safe_dump(bench_cfg.model_dump(), sort_keys=False, default_flow_style=False)
    )


def _append_top_level_csv(rows: list[RunSummary], top_csv: Path, *, run_id: str) -> None:
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
) -> list:
    """Launch every SGLang instance in parallel, then wait for all to become healthy."""
    services: list = []
    launch_tasks = []
    for p in placements:
        spec = SGLangLaunchSpec(
            model_path=bench_cfg.model.path,
            host=p.worker.address,
            port=p.port,
            tp_size=bench_cfg.model.tp_size,
            gpu_indices=p.gpu_indices,
            mem_fraction_static=bench_cfg.instances.mem_fraction_static,
            page_size=bench_cfg.instances.page_size,
        )
        launch_tasks.append(sglang_launcher.launch(p.worker, spec))
    services = await asyncio.gather(*launch_tasks)
    logger.info("launched %d SGLang services; waiting ready...", len(services))
    await asyncio.gather(
        *[sglang_launcher.wait_ready(s, timeout_s=ready_timeout_s) for s in services]
    )
    logger.info("all %d SGLang services are healthy", len(services))
    return services


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


async def _run_one_cell(
    *,
    cell_dir: Path,
    cfg_kind: str,
    c_target: int,
    trial: int,
    bench_cfg: BenchmarkConfig,
    pool,
    sampler: LogNormalSampler,
    router: GatewayRouter,
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
) -> list[RunSummary]:
    """Launch the gateway with this config's policy, run the c_target × trials sweep, tear down."""
    if cfg_spec.kind not in _GATEWAY_POLICY:
        return []
    policy = _GATEWAY_POLICY[cfg_spec.kind]

    cfg_dir = run_dir / cfg_spec.kind
    cfg_dir.mkdir(parents=True, exist_ok=True)

    gateway_launcher = GatewayLauncher()
    gateway_spec = GatewayLaunchSpec(
        worker_urls=tuple(s.endpoints["serving_http"] for s in instance_services),
        policy=policy,  # type: ignore[arg-type]
        host=bench_cfg.gateway.host,
        port=bench_cfg.gateway.port,
        log_dir=str(cfg_dir / "gateway_log"),
    )
    logger.info("[%s] launching gateway (policy=%s)...", cfg_spec.kind, policy)
    gateway_svc = await gateway_launcher.launch(gateway_spec)
    summary_rows: list[RunSummary] = []
    try:
        await gateway_launcher.wait_ready(gateway_svc, timeout_s=180.0)
        logger.info("[%s] gateway healthy at %s", cfg_spec.kind, gateway_svc.endpoints["openai_http"])

        # Inter-turn sampler — same params for all (c_target, trial) cells of this config.
        delay_params = DelayParams.from_preset(
            Preset(bench_cfg.workload.inter_turn_delay.preset),
            custom_mu=bench_cfg.workload.inter_turn_delay.custom_mu,
            custom_sigma=bench_cfg.workload.inter_turn_delay.custom_sigma,
        )

        router = GatewayRouter(
            gateway_svc.endpoints["openai_http"],
            default_model=bench_cfg.model.path,
        )
        try:
            for c_target in bench_cfg.workload.c_target_sweep:
                for trial in range(bench_cfg.workload.trials):
                    cell_dir = cfg_dir / f"c{c_target}" / f"trial{trial}"
                    sampler = LogNormalSampler(delay_params, seed=hash((cfg_spec.kind, c_target, trial)) & 0xFFFFFFFF)
                    print(f"[run] {cfg_spec.kind} c={c_target} trial={trial} -> {cell_dir}")
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
            await router.close()
    finally:
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
    tc_router: Optional[TcRouter] = None
    try:
        # 1. Global store
        logger.info("[%s] launching tensorcast global store on %s...", cfg_spec.kind, gs_worker.id)
        global_store_svc = await tc_launcher.launch_global_store(gs_worker, tc_spec)
        await tc_launcher.wait_global_ready(gs_worker, tc_spec, global_store_svc)
        gs_host_port = (gs_worker.address, tc_spec.global_store_port)
        logger.info("[%s] global store ready at %s:%d", cfg_spec.kind, *gs_host_port)

        # 2. Daemons (one per unique worker)
        capability_secret = f"tc_router-{bench_cfg.run_id}"
        for w in daemon_workers:
            logger.info("[%s] launching tensorcast daemon on %s...", cfg_spec.kind, w.id)
            svc = await tc_launcher.launch_daemon(
                w,
                tc_spec,
                global_store_address=gs_host_port,
                capability_token_secret=capability_secret,
            )
            await tc_launcher.wait_daemon_ready(w, tc_spec, svc)
            daemon_svcs.append((w, svc))
            logger.info("[%s] daemon ready at %s", cfg_spec.kind, svc.endpoints["grpc"])

        # 3. TcRouter — connect to the daemon co-located with SGLang. For
        # the single-worker smoke that's the same worker; for multi-worker
        # setups we pick the first daemon (its directory connects to the
        # global store, which sees all daemons).
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
        policy = make_policy(cfg_spec.policy)
        tc_router = TcRouter(tc_router_cfg, policy=policy)
        await tc_router.start()
        logger.info(
            "[%s] TcRouter ready (policy=%s, daemon=%s, %d instances)",
            cfg_spec.kind,
            policy.name,
            primary_daemon.endpoints["grpc"],
            len(instance_endpoints),
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
                summary, info = await _run_one_cell(
                    cell_dir=cell_dir,
                    cfg_kind=cfg_spec.kind,
                    c_target=c_target,
                    trial=trial,
                    bench_cfg=bench_cfg,
                    pool=pool,
                    sampler=sampler,
                    router=tc_router,  # type: ignore[arg-type]
                )
                summary_rows.append(summary)
                print(
                    f"  -> turns={info['total_turns']} "
                    f"(success={info['successful_turns']}, fail={info['failed_turns']}); "
                    f"ttft p50={summary.ttft_p50_ms} p95={summary.ttft_p95_ms} "
                    f"cached_ratio={summary.cached_token_ratio_mean}"
                )
    finally:
        if tc_router is not None:
            with suppress(Exception):
                await tc_router.close()
        # Stop daemons first, then global store.
        for (w, svc) in reversed(daemon_svcs):
            with suppress(Exception):
                await tc_launcher.stop_daemon(w, tc_spec, svc)
        if global_store_svc is not None:
            with suppress(Exception):
                await tc_launcher.stop_global(gs_worker, tc_spec, global_store_svc)
    return summary_rows


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
    provider = resource_factory.from_cluster_config(cluster_yaml)

    workers = provider.workers()
    placements = plan_instance_placement(
        workers,
        instances_count=bench_cfg.instances.count,
        tp_size=bench_cfg.model.tp_size,
        base_port=bench_cfg.instances.base_port,
    )

    outputs_root = outputs_root or Path(__file__).resolve().parents[1] / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    run_dir = _resolve_run_dir(bench_cfg, root=outputs_root)
    print(f"[run_benchmark] run_dir = {run_dir}")
    _save_resolved_configs(
        run_dir, cluster_yaml=cluster_yaml, bench_yaml=bench_yaml, bench_cfg=bench_cfg
    )

    # Save placement plan for postmortem / debugging.
    (run_dir / "placement.txt").write_text(
        "\n".join(
            f"{i}: worker={p.worker.id} address={p.worker.address} "
            f"port={p.port} gpus={list(p.gpu_indices)}"
            for i, p in enumerate(placements)
        )
    )

    print(f"[run_benchmark] cluster health check...")
    await provider.health_check()

    sglang_launcher = SGLangLauncher()
    summary_rows: list[RunSummary] = []
    instance_services: list = []
    try:
        print(f"[run_benchmark] launching {len(placements)} SGLang instances "
              f"({bench_cfg.model.path}, tp={bench_cfg.model.tp_size})...")
        t0 = time.monotonic()
        instance_services = await _launch_sglang_fleet(
            placements=placements,
            bench_cfg=bench_cfg,
            sglang_launcher=sglang_launcher,
            ready_timeout_s=sglang_ready_timeout_s,
        )
        print(f"[run_benchmark] SGLang instances ready in {time.monotonic() - t0:.1f}s")

        # Load workload pool ONCE, shared across configs/cells.
        print(f"[run_benchmark] loading trajectory pool from {bench_cfg.workload.dataset_path}...")
        pool = load_pool(
            bench_cfg.workload.dataset_path,
            min_turns=bench_cfg.workload.pool_filter.min_turns,
            min_total_tokens=bench_cfg.workload.pool_filter.min_total_tokens,
            seed=0,
        )
        print(f"[run_benchmark] pool size: {len(pool)} trajectories")

        for cfg_spec in bench_cfg.configs:
            if config_filter is not None and cfg_spec.kind not in config_filter:
                continue
            try:
                if cfg_spec.kind in _GATEWAY_POLICY:
                    rows = await _run_gateway_config(
                        cfg_spec=cfg_spec,
                        bench_cfg=bench_cfg,
                        instance_services=instance_services,
                        pool=pool,
                        run_dir=run_dir,
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
                    logger.info("skipping unsupported config %s in this driver", cfg_spec.kind)
                    rows = []
                summary_rows.extend(rows)
            except Exception:  # noqa: BLE001
                logger.exception("config %s failed; continuing", cfg_spec.kind)
                traceback.print_exc()

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
        if instance_services:
            print(f"[run_benchmark] tearing down SGLang instances...")
            with suppress(Exception):
                await _stop_sglang_fleet(
                    placements=placements,
                    services=instance_services,
                    sglang_launcher=sglang_launcher,
                )
        with suppress(Exception):
            await sglang_launcher.aclose()
