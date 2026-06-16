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
import shlex
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import aiohttp

from .base import Service


@dataclass(frozen=True)
class SGLangLaunchSpec:
    """Inputs to SGLang `launch_server`.

    Matches the knobs needed by Phase 5 and 6 baselines. Storage-backend
    flags are optional; v1 single-worker smoke runs leave them off.
    """

    # Required.
    model_path: str
    host: str
    port: int

    # TP topology.
    tp_size: int = 1

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

    # Workspace location used to find sglang sources / .venv (master pod
    # convention for this cluster).
    workspace_root: str = "/home/i-zhouyuhan/tot"

    # Path to the `uv` binary on the worker. share_remote first checks
    # `<workspace_root>/.venv/bin/uv` and falls back to `~/.local/bin/uv`;
    # on this cluster the fallback is what actually exists, so we make it
    # the default.
    uv_bin: str = "/home/i-zhouyuhan/.local/bin/uv"


# Args we must never produce in the launch command, per arch § 5.2.3.
FORBIDDEN_ARGS: tuple[str, ...] = ("--tool-call-parser",)


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
        spec.host,
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

        pid = await worker.start_background(
            cmd,
            name=f"sglang_{spec.port}",
            log_path=log_path,
            pid_path=pid_path,
            env={"CUDA_VISIBLE_DEVICES": cuda_visible},
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
        """Poll the SGLang `/health` endpoint until it returns 200 or timeout."""
        url = f"{service.endpoints['serving_http']}/health"
        deadline = time.monotonic() + timeout_s
        last_detail = "no probe yet"
        session = await self._ensure_session()
        while time.monotonic() < deadline:
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                    proxy=None,
                ) as resp:
                    if resp.status == 200:
                        return
                    last_detail = f"HTTP {resp.status}"
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_detail = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(poll_interval_s)
        raise TimeoutError(
            f"SGLang service {service.name} at {url} not ready within "
            f"{timeout_s}s: {last_detail}"
        )

    async def stop(self, worker, service: Service) -> None:
        """Stop a service launched via `launch`."""
        await worker.stop_background(pid_path=service.pid_path)
