#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import selectors
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib import error as urllib_error
from urllib import request as urllib_request

from pydantic import BaseModel, ConfigDict, Field, model_validator

TP_RANK_PATTERN = re.compile(r"\bTP(\d+)\b")
LOG_TIMESTAMP_PATTERN = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)"
)


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_path: str
    model_name: str = ""
    load_format: Literal["tensorcast", "default"] = "tensorcast"
    trials: int = Field(default=3, ge=1)
    tp_size: int = Field(default=4, ge=1)
    weight_version: int = Field(default=1, ge=0)
    port: int = Field(default=30000, ge=1, le=65535)
    mem_fraction_static: float = Field(default=0.7, gt=0.0, lt=1.0)
    log_level: str = "debug"
    enable_metrics: bool = True
    launch_vvv: bool = True
    extra_server_args: str = ""
    health_path: str = "/health"
    health_poll_interval_s: float = Field(default=0.5, gt=0.0)
    trial_timeout_s: float = Field(default=1800.0, gt=0.0)

    tensorcast_global_store_config: str = ""
    tensorcast_daemon_config: str = ""
    tensorcast_global_store_address: str = "127.0.0.1:50051"
    tensorcast_daemon_address: str = "127.0.0.1:50052"
    tensorcast_init_mode: Literal["connect", "auto", "create"] = "connect"
    tensorcast_show_daemon_logs: bool = False
    tensorcast_key_template: str = "model:{model_name}:v{weight_version}"
    tensorcast_allow_disk_fallback: bool = True
    tensorcast_fallback_prefer: str = "auto"
    tensorcast_get_prefer: str = "auto"
    tensorcast_export_policy: str = "auto"
    tensorcast_disk_fallback_auto_put: bool = False
    tensorcast_trace_tp_slices: bool = False
    tensorcast_tp_slice_plan_cache_dir: str = "/tmp/sglang_tensorcast_trace_cache"
    tensorcast_start_timeout_s: float = Field(default=60.0, gt=0.0)
    tensorcast_stop_timeout_s: float = Field(default=60.0, gt=0.0)
    tensorcast_status_poll_interval_s: float = Field(default=2.0, gt=0.0)
    tensorcast_cuda_home: str = "/usr/local/cuda-12.4"
    tensorcast_nvidia_lib_dirs: str = "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
    tensorcast_append_nvidia_lib_dir: bool = True

    @model_validator(mode="after")
    def _validate_tensorcast(self) -> BenchmarkConfig:
        if self.load_format != "tensorcast":
            return self

        if not self.model_name.strip():
            raise ValueError("--model-name is required when --load-format=tensorcast")
        if not self.tensorcast_global_store_config.strip():
            raise ValueError("tensorcast_global_store_config must be non-empty")
        if not self.tensorcast_daemon_config.strip():
            raise ValueError("tensorcast_daemon_config must be non-empty")
        return self


@dataclass(frozen=True)
class BenchmarkPaths:
    uv_project_root: Path
    benchmark_root: Path
    configs_dir: Path
    outputs_dir: Path
    logs_dir: Path
    csv_path: Path


@dataclass(frozen=True)
class TrialResult:
    timestamp: str
    run_id: str
    trial_id: int
    model_path: str
    model_name: str
    tp_size: int
    weight_version: int
    load_format: str
    port: int
    load_time_s: float | None
    ready_time_s: float | None
    status: str
    error_message: str
    log_path: str
    artifact_id: str


def _log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def _parse_log_timestamp(line: str) -> datetime | None:
    match = LOG_TIMESTAMP_PATTERN.match(line)
    if match is None:
        return None
    raw = match.group(1)
    with contextlib.suppress(ValueError):
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f")
    with contextlib.suppress(ValueError):
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    return None


def _discover_uv_project_root(benchmark_root: Path) -> Path:
    search_roots = [Path.cwd().resolve(), benchmark_root]
    for root in search_roots:
        for candidate in (root, *root.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
    return Path.cwd().resolve()


def _load_env_overrides(*, env_file: str, env_items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}

    def _parse_line(raw_line: str) -> None:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            return
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if sep != "=":
            raise ValueError(f"Invalid env override, expected KEY=VALUE: {raw_line!r}")
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid env override with empty key: {raw_line!r}")
        overrides[key] = value

    if env_file:
        env_path = Path(env_file).expanduser().resolve()
        if not env_path.is_file():
            raise RuntimeError(f"env-file does not exist: {env_path}")
        for line in env_path.read_text(encoding="utf-8").splitlines():
            _parse_line(line)

    for item in env_items:
        _parse_line(item)
    return overrides


def _build_paths() -> BenchmarkPaths:
    benchmark_root = Path(__file__).resolve().parent
    uv_project_root = _discover_uv_project_root(benchmark_root)
    configs_dir = benchmark_root / "configs"
    outputs_dir = benchmark_root / "outputs"
    logs_dir = outputs_dir / "logs"
    csv_path = outputs_dir / "benchmark_results.csv"
    return BenchmarkPaths(
        uv_project_root=uv_project_root,
        benchmark_root=benchmark_root,
        configs_dir=configs_dir,
        outputs_dir=outputs_dir,
        logs_dir=logs_dir,
        csv_path=csv_path,
    )


def _run_cmd(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout_s: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    if check and completed.returncode != 0:
        output = (completed.stdout + completed.stderr).strip()
        raise RuntimeError(
            f"Command failed: {shlex.join(cmd)}\n"
            f"returncode={completed.returncode}\n"
            f"{output}"
        )
    return completed


def _apply_cuda_runtime_env(config: BenchmarkConfig) -> None:
    cuda_home = Path(config.tensorcast_cuda_home)
    nvrtc_lib_dir = cuda_home / "targets/x86_64-linux/lib"
    if not cuda_home.is_dir():
        raise RuntimeError(f"CUDA home does not exist: {cuda_home}")
    if not nvrtc_lib_dir.is_dir():
        raise RuntimeError(f"CUDA NVRTC lib dir does not exist: {nvrtc_lib_dir}")

    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["PATH"] = f"{cuda_home / 'bin'}:{os.environ.get('PATH', '')}"

    lib_dirs = [str(nvrtc_lib_dir)]
    if config.tensorcast_append_nvidia_lib_dir:
        for raw_dir in config.tensorcast_nvidia_lib_dirs.split(":"):
            lib_dir = raw_dir.strip()
            if not lib_dir:
                continue
            if Path(lib_dir).is_dir():
                lib_dirs.append(lib_dir)
            else:
                _log(
                    "Skipping non-existent tensorcast nvidia lib dir: "
                    f"{lib_dir}"
                )
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs)

    _log(f"CUDA runtime configured: CUDA_HOME={os.environ['CUDA_HOME']}")
    _log(f"CUDA runtime configured: LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}")


def _global_status(env: dict[str, str], paths: BenchmarkPaths) -> str:
    completed = _run_cmd(
        ["uv", "run", "tensorcast-cli", "global", "status"],
        env=env,
        cwd=paths.uv_project_root,
        check=False,
    )
    return (completed.stdout + completed.stderr).strip()


def _daemon_status(env: dict[str, str], paths: BenchmarkPaths) -> str:
    completed = _run_cmd(
        ["uv", "run", "tensorcast-cli", "daemon", "status"],
        env=env,
        cwd=paths.uv_project_root,
        check=False,
    )
    return (completed.stdout + completed.stderr).strip()


def _global_is_serving(status: str) -> bool:
    return "Global Store session" in status and "health : SERVING" in status


def _global_is_stopped(status: str) -> bool:
    lowered = status.lower()
    return (
        "global store session: unknown" in lowered
        or "stopped global store session" in lowered
        or "no global store session found" in lowered
    )


def _daemon_is_running(status: str) -> bool:
    return bool(re.search(r"^\s*status\s*:\s*running", status, re.IGNORECASE | re.MULTILINE))


def _daemon_is_stopped(status: str) -> bool:
    if "No local daemon session found" in status:
        return True
    return not _daemon_is_running(status)


def _wait_until(
    *,
    timeout_s: float,
    poll_interval_s: float,
    description: str,
    check_fn,
) -> str:
    deadline = time.monotonic() + timeout_s
    last_status = ""
    while time.monotonic() < deadline:
        ok, status = check_fn()
        last_status = status
        if ok:
            return status
        time.sleep(poll_interval_s)
    raise RuntimeError(
        f"Timed out waiting for {description} in {timeout_s:.1f}s.\n"
        f"Last status output:\n{last_status}"
    )


def _start_tensorcast_services(
    config: BenchmarkConfig,
    env: dict[str, str],
    paths: BenchmarkPaths,
) -> None:
    _log("Starting Tensorcast Global Store")
    _run_cmd(
        [
            "uv",
            "run",
            "tensorcast-cli",
            "global",
            "start",
            "--config",
            config.tensorcast_global_store_config,
        ],
        env=env,
        cwd=paths.uv_project_root,
    )

    _wait_until(
        timeout_s=config.tensorcast_start_timeout_s,
        poll_interval_s=config.tensorcast_status_poll_interval_s,
        description="Global Store health=SERVING",
        check_fn=lambda: (
            _global_is_serving(status := _global_status(env, paths)),
            status,
        ),
    )

    _log("Starting Tensorcast Store Daemon")
    _run_cmd(
        [
            "uv",
            "run",
            "tensorcast-cli",
            "daemon",
            "start",
            "--config",
            config.tensorcast_daemon_config,
            "--global-store-mode",
            "connect",
            "--global-store-address",
            config.tensorcast_global_store_address,
        ],
        env=env,
        cwd=paths.uv_project_root,
    )

    _wait_until(
        timeout_s=config.tensorcast_start_timeout_s,
        poll_interval_s=config.tensorcast_status_poll_interval_s,
        description="Store Daemon running",
        check_fn=lambda: (
            _daemon_is_running(status := _daemon_status(env, paths)),
            status,
        ),
    )
    _log("Tensorcast services are ready")


def _stop_tensorcast_services(
    config: BenchmarkConfig,
    env: dict[str, str],
    paths: BenchmarkPaths,
) -> None:
    _log("Stopping Tensorcast Store Daemon")
    _run_cmd(
        ["uv", "run", "tensorcast-cli", "daemon", "stop"],
        env=env,
        cwd=paths.uv_project_root,
        check=False,
    )
    _wait_until(
        timeout_s=config.tensorcast_stop_timeout_s,
        poll_interval_s=config.tensorcast_status_poll_interval_s,
        description="Store Daemon stopped",
        check_fn=lambda: (
            _daemon_is_stopped(status := _daemon_status(env, paths)),
            status,
        ),
    )

    _log("Stopping Tensorcast Global Store")
    _run_cmd(
        ["uv", "run", "tensorcast-cli", "global", "stop"],
        env=env,
        cwd=paths.uv_project_root,
        check=False,
    )
    _wait_until(
        timeout_s=config.tensorcast_stop_timeout_s,
        poll_interval_s=config.tensorcast_status_poll_interval_s,
        description="Global Store stopped",
        check_fn=lambda: (
            _global_is_stopped(status := _global_status(env, paths)),
            status,
        ),
    )
    _log("Tensorcast services are stopped")


def _publish_tensorcast_artifact_once(config: BenchmarkConfig) -> str:
    import tensorcast as tc
    from safetensors.torch import load_file
    from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig

    publisher_config = WeightPublisherConfig(
        model_name=config.model_name,
        key_template=config.tensorcast_key_template,
        trigger_reload=False,
        wait_persistence=True,
        keep_last=2,
        policy="pinned",
        overflow_policy="reject",
    )

    tc.init(mode="connect", address=config.tensorcast_daemon_address)
    try:
        publisher = WeightPublisher(publisher_config)
        model_dir = Path(config.model_path)
        if not model_dir.is_dir():
            raise RuntimeError(f"model_path is not a directory: {model_dir}")

        safetensor_files = sorted(model_dir.glob("*.safetensors"))
        if not safetensor_files:
            raise RuntimeError(f"no .safetensors files found in: {model_dir}")

        tensors: dict[str, object] = {}
        for safetensor_file in safetensor_files:
            tensors.update(load_file(str(safetensor_file), device="cpu"))

        artifact_id = publisher.publish(
            tensors=tensors,
            version=config.weight_version,
        )
    finally:
        with contextlib.suppress(Exception):
            tc.shutdown()
    return str(artifact_id)


def _build_model_loader_extra_config(config: BenchmarkConfig) -> str:
    extra_config = {
        "tensorcast_init_mode": config.tensorcast_init_mode,
        "tensorcast_daemon_address": config.tensorcast_daemon_address,
        "tensorcast_show_daemon_logs": config.tensorcast_show_daemon_logs,
        "tensorcast_key_template": config.tensorcast_key_template,
        "tensorcast_model_name": config.model_name,
        "tensorcast_allow_disk_fallback": config.tensorcast_allow_disk_fallback,
        "tensorcast_fallback_prefer": config.tensorcast_fallback_prefer,
        "tensorcast_get_prefer": config.tensorcast_get_prefer,
        "tensorcast_export_policy": config.tensorcast_export_policy,
        "tensorcast_disk_fallback_auto_put": config.tensorcast_disk_fallback_auto_put,
        "tensorcast_trace_tp_slices": config.tensorcast_trace_tp_slices,
        "tensorcast_tp_slice_plan_cache_dir": config.tensorcast_tp_slice_plan_cache_dir,
    }
    return json.dumps(extra_config)


def _build_server_cmd(config: BenchmarkConfig) -> list[str]:
    cmd = ["uv", "run"]
    if config.launch_vvv:
        cmd.append("-vvv")
    cmd.extend(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            config.model_path,
            "--port",
            str(config.port),
            "--tp-size",
            str(config.tp_size),
            "--mem-fraction-static",
            str(config.mem_fraction_static),
            "--log-level",
            config.log_level,
            "--weight-version",
            str(config.weight_version),
        ]
    )
    if config.enable_metrics:
        cmd.append("--enable-metrics")

    if config.load_format == "tensorcast":
        cmd.extend(
            [
                "--load-format",
                "tensorcast",
                "--model-loader-extra-config",
                _build_model_loader_extra_config(config),
            ]
        )

    if config.extra_server_args.strip():
        cmd.extend(shlex.split(config.extra_server_args))
    return cmd


def _is_server_healthy(port: int, health_path: str) -> bool:
    url = f"http://127.0.0.1:{port}{health_path}"
    request = urllib_request.Request(url=url, method="GET")
    try:
        with urllib_request.urlopen(request, timeout=1.0) as response:
            return response.status == 200
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError):
        return False

def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


def _monitor_server_until_ready(
    *,
    config: BenchmarkConfig,
    env: dict[str, str],
    paths: BenchmarkPaths,
    trial_log_path: Path,
) -> tuple[float | None, float | None, str, str]:
    cmd = _build_server_cmd(config)
    load_marker = "Load weight end"
    load_begin_marker = "Load weight begin"
    start_mono = time.monotonic()
    load_begin_mono: float | None = None
    load_done_mono: float | None = None
    load_begin_wall: datetime | None = None
    load_done_wall: datetime | None = None
    ready_mono: float | None = None
    marker_ranks: set[int] = set()
    marker_rank_times: dict[int, datetime] = {}
    marker_count_without_rank = 0
    last_no_rank_marker_time: datetime | None = None
    status = "failed"
    error_message = ""

    with trial_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"command: {shlex.join(cmd)}\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(paths.uv_project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None

        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        try:
            while True:
                if time.monotonic() - start_mono > config.trial_timeout_s:
                    error_message = (
                        "trial timeout exceeded: "
                        f"trial_timeout_s={config.trial_timeout_s}"
                    )
                    break

                for key, _ in selector.select(timeout=0.2):
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    log_file.write(line)
                    line_ts = _parse_log_timestamp(line)
                    if load_begin_mono is None and load_begin_marker in line:
                        load_begin_mono = time.monotonic()
                        if line_ts is not None:
                            load_begin_wall = line_ts
                    if load_marker in line and load_done_mono is None:
                        rank_match = TP_RANK_PATTERN.search(line)
                        if rank_match:
                            rank = int(rank_match.group(1))
                            marker_ranks.add(rank)
                            if line_ts is not None:
                                previous_ts = marker_rank_times.get(rank)
                                if previous_ts is None or line_ts > previous_ts:
                                    marker_rank_times[rank] = line_ts
                        else:
                            marker_count_without_rank += 1
                            if line_ts is not None:
                                last_no_rank_marker_time = line_ts
                        if (
                            len(marker_ranks) >= config.tp_size
                            or marker_count_without_rank >= config.tp_size
                        ):
                            load_done_mono = time.monotonic()
                            if len(marker_ranks) >= config.tp_size and marker_rank_times:
                                load_done_wall = max(marker_rank_times.values())
                            elif last_no_rank_marker_time is not None:
                                load_done_wall = last_no_rank_marker_time
                            elif line_ts is not None:
                                load_done_wall = line_ts
                    log_file.flush()

                if ready_mono is None and _is_server_healthy(config.port, config.health_path):
                    ready_mono = time.monotonic()

                if load_done_mono is not None and ready_mono is not None:
                    if load_begin_mono is None:
                        status = "failed"
                        error_message = (
                            "load completion marker found, but no "
                            "'Load weight begin' marker was observed"
                        )
                    else:
                        status = "success"
                    break

                returncode = proc.poll()
                if returncode is not None:
                    for line in proc.stdout:
                        log_file.write(line)
                    log_file.flush()
                    if load_done_mono is not None and ready_mono is not None:
                        status = "success"
                    else:
                        error_message = (
                            "server exited before completion, "
                            f"returncode={returncode}"
                        )
                    break
        finally:
            selector.close()
            _terminate_process_group(proc)

    load_time_s = None
    if load_begin_wall is not None and load_done_wall is not None:
        wall_delta = (load_done_wall - load_begin_wall).total_seconds()
        if wall_delta >= 0:
            load_time_s = wall_delta
    if (
        load_time_s is None
        and load_done_mono is not None
        and load_begin_mono is not None
    ):
        load_time_s = load_done_mono - load_begin_mono
    ready_time_s = None
    if ready_mono is not None:
        ready_time_s = ready_mono - start_mono
    return load_time_s, ready_time_s, status, error_message


def _append_result(csv_path: Path, result: TrialResult) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp",
        "run_id",
        "trial_id",
        "model_path",
        "model_name",
        "tp_size",
        "weight_version",
        "load_format",
        "port",
        "load_time_s",
        "ready_time_s",
        "status",
        "error_message",
        "artifact_id",
        "log_path",
    ]
    row = {
        "timestamp": result.timestamp,
        "run_id": result.run_id,
        "trial_id": result.trial_id,
        "model_path": result.model_path,
        "model_name": result.model_name,
        "tp_size": result.tp_size,
        "weight_version": result.weight_version,
        "load_format": result.load_format,
        "port": result.port,
        "load_time_s": (
            f"{result.load_time_s:.6f}" if result.load_time_s is not None else ""
        ),
        "ready_time_s": (
            f"{result.ready_time_s:.6f}" if result.ready_time_s is not None else ""
        ),
        "status": result.status,
        "error_message": result.error_message,
        "artifact_id": result.artifact_id,
        "log_path": result.log_path,
    }
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _build_parser(default_global_cfg: Path, default_daemon_cfg: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SGLang model load latency for --load-format={tensorcast,default}. "
            "Results are appended to outputs/benchmark_results.csv."
        )
    )
    parser.add_argument("--load-format", choices=("tensorcast", "default"), default="tensorcast")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--weight-version", type=int, default=1)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--log-level", default="debug")
    parser.add_argument("--disable-metrics", action="store_true")
    parser.add_argument("--no-launch-vvv", action="store_true")
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--health-poll-interval-s", type=float, default=0.5)
    parser.add_argument("--trial-timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--uv-project-root",
        default="",
        help="Directory used as cwd for all `uv run` commands (must contain pyproject.toml).",
    )
    parser.add_argument(
        "--env-file",
        default="",
        help="Optional env file with KEY=VALUE lines to inject into child processes.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra env override KEY=VALUE for child processes. Repeatable.",
    )

    parser.add_argument(
        "--tensorcast-global-store-config",
        default=str(default_global_cfg),
    )
    parser.add_argument(
        "--tensorcast-daemon-config",
        default=str(default_daemon_cfg),
    )
    parser.add_argument("--tensorcast-global-store-address", default="127.0.0.1:50051")
    parser.add_argument("--tensorcast-daemon-address", default="127.0.0.1:50052")
    parser.add_argument("--tensorcast-init-mode", choices=("connect", "auto", "create"), default="connect")
    parser.add_argument("--tensorcast-show-daemon-logs", action="store_true")
    parser.add_argument("--tensorcast-key-template", default="model:{model_name}:v{weight_version}")
    parser.add_argument("--tensorcast-no-allow-disk-fallback", action="store_true")
    parser.add_argument("--tensorcast-fallback-prefer", default="auto")
    parser.add_argument("--tensorcast-get-prefer", default="auto")
    parser.add_argument("--tensorcast-export-policy", default="auto")
    parser.add_argument("--tensorcast-disk-fallback-auto-put", default=False)
    parser.add_argument("--tensorcast-trace-tp-slices", action="store_true")
    parser.add_argument("--tensorcast-tp-slice-plan-cache-dir", default="/tmp/sglang_tensorcast_trace_cache")
    parser.add_argument("--tensorcast-start-timeout-s", type=float, default=60.0)
    parser.add_argument("--tensorcast-stop-timeout-s", type=float, default=60.0)
    parser.add_argument("--tensorcast-status-poll-interval-s", type=float, default=2.0)
    parser.add_argument("--tensorcast-cuda-home", default="/usr/local/cuda-12.4")
    parser.add_argument(
        "--tensorcast-nvidia-lib-dirs",
        default="/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64",
        help="Additional library dirs appended to LD_LIBRARY_PATH (':' separated).",
    )
    parser.add_argument(
        "--tensorcast-nvidia-lib-dir",
        dest="tensorcast_nvidia_lib_dirs",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--tensorcast-no-append-nvidia-lib-dir", action="store_true")
    return parser


def main() -> int:
    paths = _build_paths()
    default_global_cfg = paths.configs_dir / "global_store_config.yaml"
    default_daemon_cfg = paths.configs_dir / "store_daemon_config.yaml"
    parser = _build_parser(default_global_cfg, default_daemon_cfg)
    args = parser.parse_args()
    if args.uv_project_root:
        uv_project_root = Path(args.uv_project_root).expanduser().resolve()
        if not uv_project_root.is_dir():
            raise RuntimeError(f"uv-project-root is not a directory: {uv_project_root}")
        if not (uv_project_root / "pyproject.toml").is_file():
            raise RuntimeError(
                f"uv-project-root has no pyproject.toml: {uv_project_root}"
            )
        paths = replace(paths, uv_project_root=uv_project_root)

    env_overrides = _load_env_overrides(
        env_file=args.env_file,
        env_items=list(args.env),
    )

    config = BenchmarkConfig(
        model_path=args.model_path,
        model_name=args.model_name,
        load_format=args.load_format,
        trials=args.trials,
        tp_size=args.tp_size,
        weight_version=args.weight_version,
        port=args.port,
        mem_fraction_static=args.mem_fraction_static,
        log_level=args.log_level,
        enable_metrics=not args.disable_metrics,
        launch_vvv=not args.no_launch_vvv,
        extra_server_args=args.extra_server_args,
        health_path=args.health_path,
        health_poll_interval_s=args.health_poll_interval_s,
        trial_timeout_s=args.trial_timeout_s,
        tensorcast_global_store_config=args.tensorcast_global_store_config,
        tensorcast_daemon_config=args.tensorcast_daemon_config,
        tensorcast_global_store_address=args.tensorcast_global_store_address,
        tensorcast_daemon_address=args.tensorcast_daemon_address,
        tensorcast_init_mode=args.tensorcast_init_mode,
        tensorcast_show_daemon_logs=args.tensorcast_show_daemon_logs,
        tensorcast_key_template=args.tensorcast_key_template,
        tensorcast_allow_disk_fallback=not args.tensorcast_no_allow_disk_fallback,
        tensorcast_fallback_prefer=args.tensorcast_fallback_prefer,
        tensorcast_get_prefer=args.tensorcast_get_prefer,
        tensorcast_export_policy=args.tensorcast_export_policy,
        tensorcast_disk_fallback_auto_put=args.tensorcast_disk_fallback_auto_put,
        tensorcast_trace_tp_slices=args.tensorcast_trace_tp_slices,
        tensorcast_tp_slice_plan_cache_dir=args.tensorcast_tp_slice_plan_cache_dir,
        tensorcast_start_timeout_s=args.tensorcast_start_timeout_s,
        tensorcast_stop_timeout_s=args.tensorcast_stop_timeout_s,
        tensorcast_status_poll_interval_s=args.tensorcast_status_poll_interval_s,
        tensorcast_cuda_home=args.tensorcast_cuda_home,
        tensorcast_nvidia_lib_dirs=args.tensorcast_nvidia_lib_dirs,
        tensorcast_append_nvidia_lib_dir=not args.tensorcast_no_append_nvidia_lib_dir,
    )

    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    _log(f"Run id: {run_id}")
    _log(
        f"Benchmark start: load_format={config.load_format}, tp_size={config.tp_size}, "
        f"weight_version={config.weight_version}, trials={config.trials}, port={config.port}"
    )

    artifact_id = ""
    setup_error = ""
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)
        _log(f"Applied env overrides for child processes: {sorted(env_overrides)}")

    if config.load_format == "tensorcast":
        try:
            _apply_cuda_runtime_env(config)
            env = dict(os.environ)
            if env_overrides:
                env.update(env_overrides)
            _start_tensorcast_services(config, env, paths)
            artifact_id = _publish_tensorcast_artifact_once(config)
            _log(f"Published artifact once for this config: artifact_id={artifact_id}")
        except Exception as exc:  # noqa: BLE001
            setup_error = f"tensorcast setup failed: {exc}"
            _log(setup_error)

    for trial_id in range(1, config.trials + 1):
        trial_log_path = paths.logs_dir / (
            f"{run_id}_{config.load_format}_tp{config.tp_size}_trial{trial_id:03d}.log"
        )
        timestamp = datetime.now().isoformat(timespec="seconds")
        _log(f"Trial {trial_id}/{config.trials} start")

        if setup_error:
            with trial_log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(setup_error + "\n")
            result = TrialResult(
                timestamp=timestamp,
                run_id=run_id,
                trial_id=trial_id,
                model_path=config.model_path,
                model_name=config.model_name,
                tp_size=config.tp_size,
                weight_version=config.weight_version,
                load_format=config.load_format,
                port=config.port,
                load_time_s=None,
                ready_time_s=None,
                status="setup_failed",
                error_message=setup_error,
                log_path=str(trial_log_path),
                artifact_id=artifact_id,
            )
        else:
            try:
                load_time_s, ready_time_s, status, error_message = _monitor_server_until_ready(
                    config=config,
                    env=env,
                    paths=paths,
                    trial_log_path=trial_log_path,
                )
            except Exception as exc:  # noqa: BLE001
                load_time_s = None
                ready_time_s = None
                status = "failed"
                error_message = f"trial runner exception: {exc}"

            result = TrialResult(
                timestamp=timestamp,
                run_id=run_id,
                trial_id=trial_id,
                model_path=config.model_path,
                model_name=config.model_name,
                tp_size=config.tp_size,
                weight_version=config.weight_version,
                load_format=config.load_format,
                port=config.port,
                load_time_s=load_time_s,
                ready_time_s=ready_time_s,
                status=status,
                error_message=error_message,
                log_path=str(trial_log_path),
                artifact_id=artifact_id,
            )

        _append_result(paths.csv_path, result)
        _log(
            f"Trial {trial_id} end: status={result.status}, "
            f"load_time_s={result.load_time_s}, ready_time_s={result.ready_time_s}"
        )
        if result.error_message:
            _log(f"Trial {trial_id} error: {result.error_message}")

    if config.load_format == "tensorcast":
        try:
            _stop_tensorcast_services(config, env, paths)
        except Exception as exc:  # noqa: BLE001
            _log(f"Tensorcast cleanup failed: {exc}")

    _log(f"Benchmark finished. CSV appended: {paths.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
