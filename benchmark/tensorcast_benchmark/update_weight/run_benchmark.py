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
import threading
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
    weight_version_start: int = Field(default=0, ge=0)
    port: int = Field(default=30000, ge=1, le=65535)
    mem_fraction_static: float = Field(default=0.7, gt=0.0, lt=1.0)
    log_level: str = "debug"
    enable_metrics: bool = True
    launch_vvv: bool = True
    extra_server_args: str = ""
    health_path: str = "/health"
    server_ready_timeout_s: float = Field(default=1800.0, gt=0.0)
    health_poll_interval_s: float = Field(default=0.5, gt=0.0)
    request_timeout_s: float = Field(default=1800.0, gt=0.0)

    flush_cache: bool = True
    abort_all_requests: bool = True
    recapture_cuda_graph: bool = True

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
    tensorcast_trace_tp_slices: bool = False
    tensorcast_tp_slice_plan_cache_dir: str = "/tmp/sglang_tensorcast_trace_cache"
    tensorcast_start_timeout_s: float = Field(default=60.0, gt=0.0)
    tensorcast_stop_timeout_s: float = Field(default=60.0, gt=0.0)
    tensorcast_status_poll_interval_s: float = Field(default=2.0, gt=0.0)
    tensorcast_cuda_home: str = "/usr/local/cuda-12.4"
    tensorcast_nvidia_lib_dirs: str = "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
    tensorcast_append_nvidia_lib_dir: bool = True

    @model_validator(mode="after")
    def _validate_config(self) -> BenchmarkConfig:
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


@dataclass
class HttpRequestOutcome:
    done: threading.Event
    status_code: int | None = None
    body: str = ""
    error_message: str = ""
    response_end_mono: float | None = None


@dataclass(frozen=True)
class TrialSuccessRecord:
    timestamp: str
    run_id: str
    trial_id: int
    target_weight_version: int
    model_path: str
    model_name: str
    tp_size: int
    load_format: str
    endpoint: str
    load_time_s: float
    ready_time_s: float
    status: str
    log_path: str


class ServerLogMonitor:
    def __init__(self, process: subprocess.Popen[str], log_path: Path, command: list[str]):
        self._process = process
        self._selector = selectors.DefaultSelector()
        assert process.stdout is not None
        self._selector.register(process.stdout, selectors.EVENT_READ)
        self._log_file = log_path.open("w", encoding="utf-8")
        self._events: list[tuple[float, str]] = []
        self._log_file.write(f"command: {shlex.join(command)}\n")
        self._log_file.flush()

    @property
    def process(self) -> subprocess.Popen[str]:
        return self._process

    @property
    def event_count(self) -> int:
        return len(self._events)

    def poll(self, timeout_s: float) -> int:
        lines = 0
        first = True
        while True:
            timeout = timeout_s if first else 0.0
            first = False
            events = self._selector.select(timeout=timeout)
            if not events:
                break
            for key, _ in events:
                line = key.fileobj.readline()
                if not line:
                    continue
                ts = time.monotonic()
                self._events.append((ts, line))
                self._log_file.write(line)
                self._log_file.flush()
                lines += 1
        return lines

    def get_events_since(self, start_index: int) -> list[tuple[float, str]]:
        return self._events[start_index:]

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._selector.close()
        with contextlib.suppress(Exception):
            self._log_file.close()


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
            f"returncode={completed.returncode}\n{output}"
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
    return bool(
        re.search(r"^\s*status\s*:\s*running", status, re.IGNORECASE | re.MULTILINE)
    )


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


def _load_safetensors_tensor_dict(model_path: str) -> dict[str, object]:
    from safetensors.torch import load_file

    model_dir = Path(model_path)
    if not model_dir.is_dir():
        raise RuntimeError(f"model_path is not a directory: {model_dir}")
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise RuntimeError(f"no .safetensors files found in: {model_dir}")

    tensors: dict[str, object] = {}
    for safetensor_file in safetensor_files:
        tensors.update(load_file(str(safetensor_file), device="cpu"))
    return tensors


def _publish_tensorcast_versions(
    config: BenchmarkConfig,
    versions: list[int],
) -> dict[int, str]:
    import tensorcast as tc
    from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig

    publisher_config = WeightPublisherConfig(
        model_name=config.model_name,
        key_template=config.tensorcast_key_template,
        trigger_reload=False,
        wait_persistence=True,
        keep_last=max(len(versions) + 1, 2),
        policy="pinned",
        overflow_policy="reject",
    )

    tc.init(mode="connect", address=config.tensorcast_daemon_address)
    try:
        publisher = WeightPublisher(publisher_config)
        tensors = _load_safetensors_tensor_dict(config.model_path)
        artifact_ids: dict[int, str] = {}
        for version in versions:
            artifact_id = publisher.publish(tensors=tensors, version=version)
            artifact_ids[version] = str(artifact_id)
            _log(f"Published tensorcast version={version}, artifact_id={artifact_id}")
        return artifact_ids
    finally:
        with contextlib.suppress(Exception):
            tc.shutdown()


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
            str(config.weight_version_start),
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
    req = urllib_request.Request(url=url, method="GET")
    try:
        with urllib_request.urlopen(req, timeout=1.0) as response:
            return response.status == 200
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError):
        return False


def _wait_for_server_ready(config: BenchmarkConfig, monitor: ServerLogMonitor) -> None:
    deadline = time.monotonic() + config.server_ready_timeout_s
    while time.monotonic() < deadline:
        monitor.poll(timeout_s=config.health_poll_interval_s)
        if _is_server_healthy(config.port, config.health_path):
            return
        if monitor.process.poll() is not None:
            raise RuntimeError(
                f"Server exited before ready, returncode={monitor.process.returncode}"
            )
    raise RuntimeError(
        f"Timed out waiting for server health in {config.server_ready_timeout_s:.1f}s"
    )


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


def _build_update_request(
    config: BenchmarkConfig,
    target_version: int,
) -> tuple[str, dict[str, object]]:
    if config.load_format == "tensorcast":
        artifact_key = config.tensorcast_key_template.format(
            model_name=config.model_name,
            version=target_version,
            weight_version=target_version,
        )
        endpoint = "/update_weights_from_tensorcast"
        payload: dict[str, object] = {
            "weight_version": target_version,
            "artifact_key": artifact_key,
            "flush_cache": config.flush_cache,
            "abort_all_requests": config.abort_all_requests,
            "recapture_cuda_graph": config.recapture_cuda_graph,
        }
        return endpoint, payload

    endpoint = "/update_weights_from_disk"
    payload = {
        "model_path": config.model_path,
        "weight_version": str(target_version),
        "flush_cache": config.flush_cache,
        "abort_all_requests": config.abort_all_requests,
        "recapture_cuda_graph": config.recapture_cuda_graph,
    }
    return endpoint, payload


def _start_http_post_request(
    *,
    url: str,
    payload: dict[str, object],
    timeout_s: float,
) -> tuple[threading.Thread, HttpRequestOutcome]:
    outcome = HttpRequestOutcome(done=threading.Event())

    def _worker() -> None:
        try:
            req = urllib_request.Request(
                url=url,
                method="POST",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload).encode("utf-8"),
            )
            with urllib_request.urlopen(req, timeout=timeout_s) as response:
                outcome.status_code = int(response.status)
                outcome.body = response.read().decode("utf-8", errors="replace")
        except urllib_error.HTTPError as exc:
            outcome.status_code = int(exc.code)
            body = exc.read()
            outcome.body = body.decode("utf-8", errors="replace")
            outcome.error_message = f"HTTPError: {exc}"
        except Exception as exc:  # noqa: BLE001
            outcome.error_message = str(exc)
        finally:
            outcome.response_end_mono = time.monotonic()
            outcome.done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread, outcome


def _update_begin_marker(load_format: str) -> str:
    if load_format == "tensorcast":
        return "Update engine weights online from tensorcast begin."
    return "Update engine weights online from disk begin."

def _update_end_marker(load_format: str) -> str:
    if load_format == "tensorcast":
        return "store.tensor_dict.materialized"
    return "Capture cuda graph begin."

def _compute_load_time_from_events(
    *,
    events: list[tuple[float, str]],
    load_format: str,
    tp_size: int,
) -> float:
    begin_marker = _update_begin_marker(load_format)
    end_marker = _update_end_marker(load_format)

    begin_mono: float | None = None
    end_mono: float | None = None
    begin_wall: datetime | None = None
    end_wall: datetime | None = None
    end_ranks: set[int] = set()
    end_rank_walls: dict[int, datetime] = {}
    end_without_rank = 0
    last_no_rank_end_wall: datetime | None = None

    for ts, line in events:
        line_wall = _parse_log_timestamp(line)

        if begin_mono is None and begin_marker in line:
            begin_mono = ts
            if line_wall is not None:
                begin_wall = line_wall
        if end_marker not in line:
            continue
        rank_match = TP_RANK_PATTERN.search(line)
        if rank_match:
            rank = int(rank_match.group(1))
            end_ranks.add(rank)
            if line_wall is not None:
                prev_wall = end_rank_walls.get(rank)
                if prev_wall is None or line_wall > prev_wall:
                    end_rank_walls[rank] = line_wall
        else:
            end_without_rank += 1
            if line_wall is not None:
                last_no_rank_end_wall = line_wall
        if len(end_ranks) >= tp_size or end_without_rank >= tp_size:
            end_mono = ts
            if len(end_ranks) >= tp_size and end_rank_walls:
                end_wall = max(end_rank_walls.values())
            elif last_no_rank_end_wall is not None:
                end_wall = last_no_rank_end_wall
            elif line_wall is not None:
                end_wall = line_wall
            break

    if begin_mono is None:
        raise RuntimeError(f"Missing begin marker: {begin_marker!r}")
    if end_mono is None:
        raise RuntimeError(f"Missing complete end marker set: {end_marker!r}, tp_size={tp_size}")

    if begin_wall is not None and end_wall is not None:
        wall_delta_s = (end_wall - begin_wall).total_seconds()
        if wall_delta_s >= 0:
            return wall_delta_s
    return end_mono - begin_mono


def _wait_for_trial_markers(
    *,
    monitor: ServerLogMonitor,
    start_event_idx: int,
    load_format: str,
    tp_size: int,
    timeout_s: float,
) -> list[tuple[float, str]]:
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        trial_events = monitor.get_events_since(start_event_idx)
        try:
            _compute_load_time_from_events(
                events=trial_events,
                load_format=load_format,
                tp_size=tp_size,
            )
            return trial_events
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        if monitor.process.poll() is not None:
            break
        monitor.poll(timeout_s=0.2)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unknown marker parsing error")


def _write_trial_log(
    *,
    path: Path,
    endpoint: str,
    payload: dict[str, object],
    status_code: int | None,
    response_body: str,
    response_error: str,
    log_lines: list[str],
) -> None:
    with path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"endpoint: {endpoint}\n")
        log_file.write(f"payload: {json.dumps(payload, ensure_ascii=True)}\n")
        log_file.write(f"status_code: {status_code}\n")
        if response_error:
            log_file.write(f"response_error: {response_error}\n")
        log_file.write(f"response_body: {response_body}\n")
        log_file.write("--- server log lines captured for this trial ---\n")
        for line in log_lines:
            log_file.write(line)


def _append_success_record(csv_path: Path, record: TrialSuccessRecord) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp",
        "run_id",
        "trial_id",
        "target_weight_version",
        "model_path",
        "model_name",
        "tp_size",
        "load_format",
        "endpoint",
        "load_time_s",
        "ready_time_s",
        "status",
        "log_path",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": record.timestamp,
                "run_id": record.run_id,
                "trial_id": record.trial_id,
                "target_weight_version": record.target_weight_version,
                "model_path": record.model_path,
                "model_name": record.model_name,
                "tp_size": record.tp_size,
                "load_format": record.load_format,
                "endpoint": record.endpoint,
                "load_time_s": f"{record.load_time_s:.6f}",
                "ready_time_s": f"{record.ready_time_s:.6f}",
                "status": record.status,
                "log_path": record.log_path,
            }
        )


def _build_parser(default_global_cfg: Path, default_daemon_cfg: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark in-place model weight update latency for Tensorcast and baseline "
            "disk update paths."
        )
    )
    parser.add_argument("--load-format", choices=("tensorcast", "default"), default="tensorcast")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--weight-version-start", type=int, default=0)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--log-level", default="debug")
    parser.add_argument("--disable-metrics", action="store_true")
    parser.add_argument("--no-launch-vvv", action="store_true")
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--server-ready-timeout-s", type=float, default=1800.0)
    parser.add_argument("--health-poll-interval-s", type=float, default=0.5)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
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

    parser.add_argument("--flush-cache", action="store_true", default=True)
    parser.add_argument("--no-flush-cache", action="store_true")
    parser.add_argument("--abort-all-requests", action="store_true", default=True)
    parser.add_argument("--no-abort-all-requests", action="store_true")
    parser.add_argument("--recapture-cuda-graph", action="store_true", default=True)
    parser.add_argument("--no-recapture-cuda-graph", action="store_true")

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

    flush_cache = not args.no_flush_cache
    abort_all_requests = not args.no_abort_all_requests
    recapture_cuda_graph = not args.no_recapture_cuda_graph

    config = BenchmarkConfig(
        model_path=args.model_path,
        model_name=args.model_name,
        load_format=args.load_format,
        trials=args.trials,
        tp_size=args.tp_size,
        weight_version_start=args.weight_version_start,
        port=args.port,
        mem_fraction_static=args.mem_fraction_static,
        log_level=args.log_level,
        enable_metrics=not args.disable_metrics,
        launch_vvv=not args.no_launch_vvv,
        extra_server_args=args.extra_server_args,
        health_path=args.health_path,
        server_ready_timeout_s=args.server_ready_timeout_s,
        health_poll_interval_s=args.health_poll_interval_s,
        request_timeout_s=args.request_timeout_s,
        flush_cache=flush_cache,
        abort_all_requests=abort_all_requests,
        recapture_cuda_graph=recapture_cuda_graph,
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
    _log(
        f"Run id={run_id}, load_format={config.load_format}, tp_size={config.tp_size}, "
        f"trials={config.trials}, weight_version_start={config.weight_version_start}"
    )

    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)
        _log(f"Applied env overrides for child processes: {sorted(env_overrides)}")
    monitor: ServerLogMonitor | None = None
    server_proc: subprocess.Popen[str] | None = None
    trial_failure_reason = ""
    successful_trials = 0

    try:
        if config.load_format == "tensorcast":
            _apply_cuda_runtime_env(config)
            env = dict(os.environ)
            if env_overrides:
                env.update(env_overrides)
            _start_tensorcast_services(config, env, paths)
            versions = list(
                range(
                    config.weight_version_start,
                    config.weight_version_start + config.trials + 1,
                )
            )
            _publish_tensorcast_versions(config, versions)

        server_cmd = _build_server_cmd(config)
        server_log_path = (
            paths.logs_dir
            / f"{run_id}_{config.load_format}_tp{config.tp_size}_server.log"
        )
        server_proc = subprocess.Popen(
            server_cmd,
            cwd=str(paths.uv_project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        monitor = ServerLogMonitor(server_proc, server_log_path, server_cmd)
        _wait_for_server_ready(config, monitor)
        _log("Server is ready, starting update trials")

        for trial_id in range(1, config.trials + 1):
            target_version = config.weight_version_start + trial_id
            endpoint, payload = _build_update_request(config, target_version)
            update_url = f"http://127.0.0.1:{config.port}{endpoint}"
            request_start_mono = time.monotonic()
            start_event_idx = monitor.event_count
            thread, outcome = _start_http_post_request(
                url=update_url,
                payload=payload,
                timeout_s=config.request_timeout_s,
            )

            while not outcome.done.is_set():
                monitor.poll(timeout_s=0.2)
                if monitor.process.poll() is not None:
                    break
            thread.join(timeout=1.0)

            response_end_mono = outcome.response_end_mono or time.monotonic()
            ready_time_s = response_end_mono - request_start_mono
            try:
                trial_events = _wait_for_trial_markers(
                    monitor=monitor,
                    start_event_idx=start_event_idx,
                    load_format=config.load_format,
                    tp_size=config.tp_size,
                    timeout_s=30.0,
                )
            except Exception:
                trial_events = monitor.get_events_since(start_event_idx)
            trial_log_lines = [line for _, line in trial_events]

            trial_log_path = (
                paths.logs_dir
                / f"{run_id}_{config.load_format}_tp{config.tp_size}_trial{trial_id:03d}.log"
            )
            _write_trial_log(
                path=trial_log_path,
                endpoint=endpoint,
                payload=payload,
                status_code=outcome.status_code,
                response_body=outcome.body,
                response_error=outcome.error_message,
                log_lines=trial_log_lines,
            )

            if monitor.process.poll() is not None:
                trial_failure_reason = (
                    f"trial {trial_id} failed: server exited, "
                    f"returncode={monitor.process.returncode}"
                )
                _log(trial_failure_reason)
                break

            if outcome.status_code != 200:
                trial_failure_reason = (
                    f"trial {trial_id} failed: HTTP {outcome.status_code}, "
                    f"body={outcome.body}, error={outcome.error_message}"
                )
                _log(trial_failure_reason)
                break

            try:
                load_time_s = _compute_load_time_from_events(
                    events=trial_events,
                    load_format=config.load_format,
                    tp_size=config.tp_size,
                )
            except Exception as exc:  # noqa: BLE001
                trial_failure_reason = f"trial {trial_id} failed to parse load_time: {exc}"
                _log(trial_failure_reason)
                break

            record = TrialSuccessRecord(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                run_id=run_id,
                trial_id=trial_id,
                target_weight_version=target_version,
                model_path=config.model_path,
                model_name=config.model_name,
                tp_size=config.tp_size,
                load_format=config.load_format,
                endpoint=endpoint,
                load_time_s=load_time_s,
                ready_time_s=ready_time_s,
                status="success",
                log_path=str(trial_log_path),
            )
            _append_success_record(paths.csv_path, record)
            successful_trials += 1
            _log(
                f"trial {trial_id} success: target_version={target_version}, "
                f"load_time_s={load_time_s:.3f}, ready_time_s={ready_time_s:.3f}"
            )

    finally:
        if monitor is not None:
            monitor.close()
        if server_proc is not None:
            _terminate_process_group(server_proc)
        if config.load_format == "tensorcast":
            with contextlib.suppress(Exception):
                _stop_tensorcast_services(config, env, paths)

    if trial_failure_reason:
        _log(
            f"Benchmark aborted after {successful_trials} successful trials. "
            f"reason={trial_failure_reason}"
        )
        return 1

    _log(
        f"Benchmark finished with {successful_trials}/{config.trials} successful trials. "
        f"CSV appended: {paths.csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
