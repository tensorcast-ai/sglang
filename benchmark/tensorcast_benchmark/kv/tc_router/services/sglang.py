"""SGLang serving instance launcher.

Builds the launch command and starts it as a background process via
`Worker.start_background`. The exact command shape is patterned after
`kv/share_remote/run_benchmark.build_sglang_command_for_instance` but
strips Mooncake/Tensorcast-specific bits unless explicitly enabled.

Per arch § 5.2.3, the launch command MUST NOT pass `--tool-call-parser`.
Tests assert this guarantee.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import shlex
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import aiohttp

from .base import Service


logger = logging.getLogger(__name__)


def _default_workspace_root() -> str:
    """Infer the repo root from this file's location."""
    return str(Path(__file__).resolve().parents[7])


def _default_uv_bin() -> str:
    return shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")


@dataclass(frozen=True)
class SGLangLaunchSpec:
    """Inputs to SGLang `launch_server`.

    Matches the knobs needed by Phase 5 and 6 baselines. Storage-backend
    flags are optional; v1 single-worker smoke runs leave them off.
    """

    # Required.
    model_path: str
    # Host advertised to the driver/gateway for HTTP traffic.
    host: str
    port: int
    # Host/interface SGLang binds inside the worker. If omitted, binds to
    # `host` for backward compatibility with existing local runs.
    bind_host: Optional[str] = None

    # TP topology.
    tp_size: int = 1
    nccl_port: Optional[int] = None

    # Memory / pagination.
    mem_fraction_static: float = 0.85
    page_size: int = 32

    # HiCache (host-DRAM L2).
    enable_hierarchical_cache: bool = False
    hicache_mem_layout: str = "page_first_direct"
    hicache_io_backend: str = "direct"
    hicache_ratio: float = 2.0
    hicache_size_gb: int = 0
    hicache_storage_prefetch_policy: str = "wait_complete"

    # Storage backend (off by default; Phase 6 adds Mooncake, Phase 7 adds Tensorcast).
    hicache_storage_backend: Optional[Literal["mooncake", "tensorcast"]] = None
    hicache_storage_backend_extra_config: Optional[dict] = None

    # Misc.
    trust_remote_code: bool = False

    # Caller-provided extra flags appended verbatim. Used for things like
    # `--log-level debug`. CALLERS MUST NOT smuggle in `--tool-call-parser`;
    # tests guard against it on the constructed command line.
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    # Override the GPU subset to pin via CUDA_VISIBLE_DEVICES. If None, the
    # launcher uses `worker.gpu_indices[:tp_size]` (one instance per worker).
    # When multiple instances share a worker (e.g. the Phase 5 smoke that
    # packs N=3 TP=2 instances on one 8-GPU host), the orchestrator passes
    # disjoint windows here. `len(gpu_indices)` MUST equal `tp_size`.
    gpu_indices: Optional[tuple[int, ...]] = None

    # Extra service environment, used by storage backends that need per-worker
    # bind/advertise controls outside the SGLang CLI surface.
    extra_env: dict[str, str] = field(default_factory=dict)

    # Workspace location used to find sglang sources / .venv.
    workspace_root: str = field(default_factory=_default_workspace_root)

    # Path to the `uv` binary on the worker.
    uv_bin: str = field(default_factory=_default_uv_bin)


# Args we must never produce in the launch command, per arch § 5.2.3.
FORBIDDEN_ARGS: tuple[str, ...] = ("--tool-call-parser",)


@dataclass(frozen=True)
class FlushAttemptResult:
    """One `/flush_cache` probe result for a SGLang endpoint."""

    endpoint: str
    ok: bool
    status: int | None = None
    detail: str = ""


@dataclass(frozen=True)
class FlushCachesResult:
    """Final state after flushing all requested SGLang endpoints."""

    attempts: tuple[FlushAttemptResult, ...]
    elapsed_s: float


def build_launch_command(spec: SGLangLaunchSpec) -> str:
    """Build the shell command (single string) that launches one SGLang instance.

    Returned string is suitable for `Worker.start_background(cmd, ...)`.

    The command activates the workspace venv first (so `uv run --active` picks
    up the correct Python with sglang installed) and sets PYTHONPATH /
    PATH the same way share_remote does. See
    `kv/share_remote.run_benchmark.build_remote_python_prefix`.
    """
    workspace = spec.workspace_root.rstrip("/")
    sglang_root = f"{workspace}/thirdparty/sglang"
    venv_activate = f"{workspace}/.venv/bin/activate"
    uv_bin_dir = spec.uv_bin.rsplit("/", 1)[0] if "/" in spec.uv_bin else "."
    pythonpath = f"{sglang_root}/python"

    prefix = (
        f"cd {shlex.quote(sglang_root)}; "
        f"source {shlex.quote(venv_activate)}; "
        f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}; "
        f"export PATH={shlex.quote(uv_bin_dir)}:$PATH; "
    )

    parts: list[str] = [
        shlex.quote(spec.uv_bin),
        "run",
        "--active",
        "--no-project",
        "--offline",
        "python",
        "-m",
        "sglang.launch_server",
        "--host",
        spec.bind_host or spec.host,
        "--port",
        str(spec.port),
        "--model-path",
        shlex.quote(spec.model_path),
        "--tp",
        str(spec.tp_size),
        "--page-size",
        str(spec.page_size),
        "--mem-fraction-static",
        str(spec.mem_fraction_static),
    ]
    if spec.nccl_port is not None:
        parts.extend(["--nccl-port", str(spec.nccl_port)])
    if spec.trust_remote_code:
        parts.append("--trust-remote-code")
    # `--enable-cache-report` makes SGLang populate
    # `usage.prompt_tokens_details.cached_tokens` in /v1/chat/completions
    # responses (default off). Without this our `cached_token_ratio_mean`
    # metric is always 0; arch § 10.3 + plan §13 risk register flagged
    # this as a Phase-5-blocking question.
    parts.append("--enable-cache-report")
    if spec.enable_hierarchical_cache:
        parts.extend(
            [
                "--enable-hierarchical-cache",
                "--hicache-mem-layout",
                shlex.quote(spec.hicache_mem_layout),
                "--hicache-io-backend",
                shlex.quote(spec.hicache_io_backend),
                "--hicache-ratio",
                str(spec.hicache_ratio),
                "--hicache-size",
                str(spec.hicache_size_gb),
                "--hicache-storage-prefetch-policy",
                shlex.quote(spec.hicache_storage_prefetch_policy),
            ]
        )
    if spec.hicache_storage_backend is not None:
        parts.extend(["--hicache-storage-backend", spec.hicache_storage_backend])
        if spec.hicache_storage_backend_extra_config is not None:
            cfg_json = json.dumps(
                spec.hicache_storage_backend_extra_config,
                separators=(",", ":"),
                sort_keys=True,
            )
            parts.extend(
                [
                    "--hicache-storage-backend-extra-config",
                    shlex.quote(cfg_json),
                ]
            )

    # Caller-provided extras, validated against the forbidden list.
    for arg in spec.extra_args:
        for forbidden in FORBIDDEN_ARGS:
            if forbidden in arg:
                raise ValueError(
                    f"extra_args contains forbidden flag {forbidden!r} (arch § 5.2.3)"
                )
    parts.extend(spec.extra_args)

    cmd = prefix + " ".join(parts)

    # Defense-in-depth: also assert on the final concatenated command.
    for forbidden in FORBIDDEN_ARGS:
        if forbidden in cmd:
            raise AssertionError(
                f"build_launch_command produced forbidden flag {forbidden!r}; "
                "this should be impossible — please report (arch § 5.2.3)."
            )
    return cmd


def _flush_cache_url(endpoint: str) -> str:
    return f"{endpoint.rstrip('/')}/flush_cache"


def _clear_hicache_storage_url(endpoint: str) -> str:
    return f"{endpoint.rstrip('/')}/clear_hicache_storage_backend"


async def _post_sglang_admin_endpoint(
    session: aiohttp.ClientSession,
    endpoint: str,
    *,
    path_builder: Callable[[str], str],
    request_timeout_s: float,
) -> FlushAttemptResult:
    try:
        async with session.post(
            path_builder(endpoint),
            timeout=aiohttp.ClientTimeout(total=request_timeout_s),
            proxy=None,
        ) as resp:
            body = (await resp.text()).strip()
            detail = body[:500] if body else f"HTTP {resp.status}"
            return FlushAttemptResult(
                endpoint=endpoint,
                ok=resp.status == 200,
                status=resp.status,
                detail=detail,
            )
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return FlushAttemptResult(
            endpoint=endpoint,
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
        )


async def _retry_sglang_admin_endpoint(
    endpoints: Sequence[str],
    *,
    operation_name: str,
    path_builder: Callable[[str], str],
    timeout_s: float = 180.0,
    poll_interval_s: float = 1.0,
    request_timeout_s: float = 5.0,
    session: aiohttp.ClientSession | None = None,
) -> FlushCachesResult:
    unique_endpoints = tuple(
        dict.fromkeys(endpoint.rstrip("/") for endpoint in endpoints)
    )
    if not unique_endpoints:
        raise ValueError(f"{operation_name} requires at least one endpoint")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be > 0")
    if request_timeout_s <= 0:
        raise ValueError("request_timeout_s must be > 0")

    own_session = session is None
    http = session or aiohttp.ClientSession(trust_env=False)
    start = time.monotonic()
    deadline = start + timeout_s
    pending = set(unique_endpoints)
    last_results: dict[str, FlushAttemptResult] = {}

    try:
        while pending:
            results = await asyncio.gather(
                *(
                    _post_sglang_admin_endpoint(
                        http,
                        endpoint,
                        path_builder=path_builder,
                        request_timeout_s=request_timeout_s,
                    )
                    for endpoint in sorted(pending)
                )
            )
            for result in results:
                last_results[result.endpoint] = result
                if result.ok:
                    pending.discard(result.endpoint)
                    continue
                logger.warning(
                    "SGLang %s not ready endpoint=%s status=%s detail=%s",
                    operation_name,
                    result.endpoint,
                    result.status,
                    result.detail,
                )

            if not pending:
                break
            now = time.monotonic()
            if now >= deadline:
                failures = [
                    last_results.get(
                        endpoint,
                        FlushAttemptResult(
                            endpoint=endpoint,
                            ok=False,
                            detail="no attempt completed",
                        ),
                    )
                    for endpoint in sorted(pending)
                ]
                detail = "; ".join(
                    f"{failure.endpoint} status={failure.status} detail={failure.detail}"
                    for failure in failures
                )
                raise TimeoutError(
                    f"SGLang {operation_name} did not succeed for "
                    f"{len(failures)} endpoint(s) within {timeout_s}s: {detail}"
                )
            await asyncio.sleep(min(poll_interval_s, deadline - now))

        return FlushCachesResult(
            attempts=tuple(last_results[endpoint] for endpoint in unique_endpoints),
            elapsed_s=time.monotonic() - start,
        )
    finally:
        if own_session:
            await http.close()


async def flush_sglang_caches(
    endpoints: Sequence[str],
    *,
    timeout_s: float = 180.0,
    poll_interval_s: float = 1.0,
    request_timeout_s: float = 5.0,
    session: aiohttp.ClientSession | None = None,
) -> FlushCachesResult:
    """POST `/flush_cache` to every SGLang endpoint until all return HTTP 200.

    SGLang may reject a flush while it still observes running or waiting
    requests from the previous cell, so this helper retries every
    non-successful endpoint until the whole set is clean or the timeout
    expires. It never routes through `sgl-model-gateway`.
    """
    return await _retry_sglang_admin_endpoint(
        endpoints,
        operation_name="flush_cache",
        path_builder=_flush_cache_url,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        request_timeout_s=request_timeout_s,
        session=session,
    )


async def clear_sglang_hicache_storage_backends(
    endpoints: Sequence[str],
    *,
    timeout_s: float = 180.0,
    poll_interval_s: float = 1.0,
    request_timeout_s: float = 5.0,
    session: aiohttp.ClientSession | None = None,
) -> FlushCachesResult:
    """POST `/clear_hicache_storage_backend` until every endpoint returns HTTP 200."""
    return await _retry_sglang_admin_endpoint(
        endpoints,
        operation_name="clear_hicache_storage_backend",
        path_builder=_clear_hicache_storage_url,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        request_timeout_s=request_timeout_s,
        session=session,
    )


class SGLangLauncher:
    """ServiceLauncher for a SGLang serving instance."""

    def __init__(self) -> None:
        # Reuse a single aiohttp session across waits / requests during a run.
        # Driver-host traffic to the worker IP MUST NOT go through corporate
        # proxies, so disable env-driven proxies on this session.
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(trust_env=False)
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def launch(self, worker, spec: SGLangLaunchSpec) -> Service:
        log_dir = f"{worker.scratch_dir.rstrip('/')}/services/sglang_{spec.port}"
        log_path = f"{log_dir}/sglang.log"
        pid_path = f"{log_dir}/sglang.pid"

        # Pin the instance to specific GPU indices.
        # By default: first `tp_size` GPUs on the worker (one-instance-per-worker).
        # Override via spec.gpu_indices when packing multiple instances on a worker.
        if spec.gpu_indices is not None:
            if len(spec.gpu_indices) != spec.tp_size:
                raise ValueError(
                    f"spec.gpu_indices length {len(spec.gpu_indices)} "
                    f"does not match tp_size {spec.tp_size}"
                )
            chosen_gpus = spec.gpu_indices
        else:
            if spec.tp_size > len(worker.gpu_indices):
                raise ValueError(
                    f"worker {worker.id!r} has {len(worker.gpu_indices)} GPUs; "
                    f"spec.tp_size={spec.tp_size} cannot fit"
                )
            chosen_gpus = tuple(worker.gpu_indices[: spec.tp_size])
        cuda_visible = ",".join(str(g) for g in chosen_gpus)

        cmd = build_launch_command(spec)

        launch_env = {"CUDA_VISIBLE_DEVICES": cuda_visible}
        launch_env.update(spec.extra_env)

        pid = await worker.start_background(
            cmd,
            name=f"sglang_{spec.port}",
            log_path=log_path,
            pid_path=pid_path,
            env=launch_env,
        )
        return Service(
            name=f"sglang_{spec.port}",
            worker_id=worker.id,
            endpoints={
                "serving_http": f"http://{spec.host}:{spec.port}",
                "instance_id": f"{spec.host}:{spec.port}",
            },
            pid=pid,
            pid_path=pid_path,
            log_path=log_path,
            metadata={
                "model_path": spec.model_path,
                "tp_size": str(spec.tp_size),
                "bind_host": spec.bind_host or spec.host,
                "advertise_host": spec.host,
                "nccl_port": "" if spec.nccl_port is None else str(spec.nccl_port),
                "cuda_visible_devices": cuda_visible,
            },
        )

    async def wait_ready(
        self,
        service: Service,
        *,
        timeout_s: float = 1800.0,
        poll_interval_s: float = 2.0,
    ) -> None:
        """Poll SGLang until its OpenAI-compatible API is usable.

        Some SGLang builds can keep `/health` at 503 after model loading while
        `/v1/models` is already serving. The gateway only needs the
        OpenAI-compatible API, so either probe succeeding is enough.
        """
        base_url = service.endpoints["serving_http"].rstrip("/")
        health_url = f"{base_url}/health"
        models_url = f"{base_url}/v1/models"
        deadline = time.monotonic() + timeout_s
        last_detail = "no probe yet"
        session = await self._ensure_session()
        while time.monotonic() < deadline:
            health_ready, health_detail = await self._probe_ready_url(
                session, health_url
            )
            if health_ready:
                return
            models_ready, models_detail = await self._probe_ready_url(
                session, models_url
            )
            if models_ready:
                return
            last_detail = f"health={health_detail}; models={models_detail}"
            await asyncio.sleep(poll_interval_s)
        raise TimeoutError(
            f"SGLang service {service.name} at {base_url} not ready within "
            f"{timeout_s}s: {last_detail}"
        )

    async def _probe_ready_url(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> tuple[bool, str]:
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=5.0),
                proxy=None,
            ) as resp:
                if resp.status == 200:
                    return True, "HTTP 200"
                return False, f"HTTP {resp.status}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return False, f"{type(exc).__name__}: {exc}"

    async def stop(self, worker, service: Service) -> None:
        """Stop a service launched via `launch`."""
        await worker.stop_background(pid_path=service.pid_path)
