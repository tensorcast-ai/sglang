#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import csv
import getpass
import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

BRAINCTL_PROXY_ENV_KEYS = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "https_proxy",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)
ORCHESTRATOR_LOG_PATH: Path | None = None


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_path: str
    model_name: str = ""
    load_format: Literal["tensorcast", "default"] = "tensorcast"
    topology_mode: Literal["direct", "relay"] = "relay"
    trials: int = Field(default=3, ge=1)
    tp_size: int = Field(default=4, ge=1)
    weight_version_start: int = Field(default=0, ge=0)
    port: int = Field(default=30000, ge=1, le=65535)
    mem_fraction_static: float = Field(default=0.85, gt=0.0, lt=1.0)
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

    uv_project_root: str = ""
    keep_workers: bool = False

    brainctl_namespace: str = "shai-core"
    brainctl_charged_group: str = "codesign"
    brainctl_private_machine: str = "group"
    brainctl_mount: str = (
        "juicefs+s3://oss.i.shaipower.com/step2-alignment-jfs:"
        "/mnt/step2-alignment-jfs"
    )
    brainctl_max_wait_duration: str = "10m"
    worker_ready_timeout_s: float = Field(default=900.0, gt=0.0)
    worker_poll_interval_s: float = Field(default=5.0, gt=0.0)
    different_node_max_attempts: int = Field(default=5, ge=1)
    existing_worker_a_process: str = ""
    existing_worker_b_process: str = ""

    worker_a_gpu: int = Field(default=1, ge=0)
    worker_a_cpu: int = Field(default=64, ge=1)
    worker_a_memory: int = Field(default=500000, ge=1)
    worker_a_positive_tags: str = "H800"
    worker_a_negative_tags: str = ""

    worker_b_gpu: int = Field(default=4, ge=1)
    worker_b_cpu: int = Field(default=64, ge=1)
    worker_b_memory: int = Field(default=500000, ge=1)
    worker_b_positive_tags: str = "H800"

    tensorcast_global_store_port: int = Field(default=50051, ge=1, le=65535)
    tensorcast_daemon_port: int = Field(default=50052, ge=1, le=65535)
    tensorcast_daemon_p2p_port: int = Field(default=65090, ge=1, le=65535)
    tensorcast_global_store_config: str = ""
    tensorcast_daemon_config: str = ""
    tensorcast_service_ready_timeout_s: float = Field(default=120.0, gt=0.0)
    tensorcast_service_poll_interval_s: float = Field(default=2.0, gt=0.0)
    tensorcast_cuda_home: str = "/usr/local/cuda-12.4"
    tensorcast_nvidia_lib_dirs: str = "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
    tensorcast_init_mode: Literal["connect", "auto", "create"] = "connect"
    tensorcast_show_daemon_logs: bool = False
    tensorcast_key_template: str = "model:{model_name}:v{weight_version}"
    tensorcast_allow_disk_fallback: bool = True
    tensorcast_fallback_prefer: str = "auto"
    tensorcast_get_prefer: str = "auto"
    tensorcast_export_policy: str = "auto"
    tensorcast_trace_tp_slices: bool = False
    tensorcast_tp_slice_plan_cache_dir: str = "/tmp/sglang_tensorcast_trace_cache"

    cache_dirs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_config(self) -> "BenchmarkConfig":
        if self.load_format == "tensorcast" and not self.model_name.strip():
            raise ValueError("--model-name is required when --load-format=tensorcast")
        if not self.brainctl_charged_group.strip() and not self.existing_worker_b_process.strip():
            raise ValueError("--brainctl-charged-group is required when launching workers")
        return self


@dataclass(frozen=True)
class BenchmarkPaths:
    benchmark_root: Path
    configs_dir: Path
    scripts_dir: Path
    outputs_dir: Path
    run_dir: Path
    logs_dir: Path
    generated_configs_dir: Path
    csv_path: Path
    uv_project_root: Path


@dataclass(frozen=True)
class WorkerInfo:
    process_name: str
    hostname: str
    creator: str
    ready: str
    status: str
    ip: str
    node: str


@dataclass(frozen=True)
class ServiceLogs:
    global_store_data_log: Path | None
    daemon_a_data_log: Path | None
    daemon_b_data_log: Path | None


@dataclass(frozen=True)
class TrialResult:
    timestamp: str
    run_id: str
    trial_id: int
    target_weight_version: int
    model_path: str
    model_name: str
    tp_size: int
    load_format: str
    topology_mode: str
    endpoint: str
    load_time_s: float | None
    ready_time_s: float | None
    status: str
    error_message: str
    artifact_id: str
    worker_a_process: str
    worker_a_hostname: str
    worker_a_ip: str
    worker_a_node: str
    worker_b_process: str
    worker_b_hostname: str
    worker_b_ip: str
    worker_b_node: str
    global_store_address: str
    daemon_a_address: str
    daemon_b_address: str
    log_path: str
    server_log_path: str
    run_dir: str


def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {message}"
    print(line, flush=True)
    if ORCHESTRATOR_LOG_PATH is not None:
        ORCHESTRATOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ORCHESTRATOR_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(line + "\n")


def discover_uv_project_root(benchmark_root: Path) -> Path:
    for candidate in (Path.cwd().resolve(), benchmark_root, *benchmark_root.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def build_paths(run_id: str, uv_project_root: str) -> BenchmarkPaths:
    benchmark_root = Path(__file__).resolve().parent
    chosen_uv_root = Path(uv_project_root).expanduser().resolve() if uv_project_root else discover_uv_project_root(benchmark_root)
    outputs_dir = benchmark_root / "outputs"
    run_dir = outputs_dir / run_id
    logs_dir = run_dir / "logs"
    generated_configs_dir = run_dir / "generated_configs"
    return BenchmarkPaths(
        benchmark_root=benchmark_root,
        configs_dir=benchmark_root / "configs",
        scripts_dir=benchmark_root / "scripts",
        outputs_dir=outputs_dir,
        run_dir=run_dir,
        logs_dir=logs_dir,
        generated_configs_dir=generated_configs_dir,
        csv_path=outputs_dir / "benchmark_results.csv",
        uv_project_root=chosen_uv_root,
    )


def prepare_paths(paths: BenchmarkPaths) -> None:
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.generated_configs_dir.mkdir(parents=True, exist_ok=True)


def local_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in BRAINCTL_PROXY_ENV_KEYS:
        env.pop(key, None)
    return env


def run_local(cmd: list[str], *, timeout_s: float | None = None, check: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd or Path.cwd()),
        env=local_env(),
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {shlex.join(cmd)}\nreturncode={completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def brainctl_base(config: BenchmarkConfig) -> list[str]:
    return ["brainctl"]


def parse_worker_info(process_name: str, output: str) -> WorkerInfo:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    data_line = ""
    for line in reversed(lines):
        if not line.startswith("ID ") and not line.startswith("ID\t"):
            data_line = line
            break
    if not data_line:
        raise RuntimeError(f"could not parse worker info for {process_name}:\n{output}")
    tokens = data_line.split()
    if len(tokens) < 9:
        raise RuntimeError(f"unexpected worker info format for {process_name}:\n{output}")
    return WorkerInfo(
        process_name=tokens[0],
        hostname=tokens[1],
        creator=tokens[2],
        ready=tokens[3],
        status=tokens[4],
        ip=tokens[7],
        node=tokens[8],
    )


def get_worker_info(config: BenchmarkConfig, process_name: str) -> WorkerInfo:
    completed = run_local([
        *brainctl_base(config),
        "get",
        f"process/{process_name}",
        "-n",
        config.brainctl_namespace,
        "-o",
        "wide",
    ])
    return parse_worker_info(process_name, completed.stdout)


def describe_worker_tail(config: BenchmarkConfig, process_name: str, lines: int = 40) -> str:
    completed = run_local([
        *brainctl_base(config),
        "describe",
        f"process/{process_name}",
        "-n",
        config.brainctl_namespace,
    ], check=False)
    text = (completed.stdout + completed.stderr).strip().splitlines()
    return "\n".join(text[-lines:])


def wait_for_worker_running(config: BenchmarkConfig, process_name: str) -> WorkerInfo:
    deadline = time.monotonic() + config.worker_ready_timeout_s
    while time.monotonic() < deadline:
        info = get_worker_info(config, process_name)
        if info.status == "Running" and info.ready == "1/1":
            return info
        time.sleep(config.worker_poll_interval_s)
    raise RuntimeError(
        f"worker did not reach Running within {config.worker_ready_timeout_s}s: {process_name}\n{describe_worker_tail(config, process_name)}"
    )


def launch_worker(config: BenchmarkConfig, *, run_id: str, role: str, gpu: int, cpu: int, memory: int, positive_tags: str, negative_tags: str) -> str:
    keepalive_log = f"/data/{run_id}_{role}_keepalive.log"
    remote_cmd = (
        "set -euo pipefail; "
        f"LOG_FILE={shlex.quote(keepalive_log)}; "
        "echo KEEPALIVE_START $(date -Is) HOST=$(hostname) USER=$(id -un) | tee -a \"$LOG_FILE\"; "
        "while true; do echo KEEPALIVE_HEARTBEAT $(date -Is) | tee -a \"$LOG_FILE\"; sleep 30; done"
    )
    cmd = [
        *brainctl_base(config),
        "launch",
        "-d",
        "--i-know-i-am-wasting-resource",
        f"--charged-group={config.brainctl_charged_group}",
        f"--gpu={gpu}",
        f"--cpu={cpu}",
        f"--memory={memory}",
        f"--mount={config.brainctl_mount}",
        f"--private-machine={config.brainctl_private_machine}",
        f"--max-wait-duration={config.brainctl_max_wait_duration}",
        f"--comment={run_id}-{role}",
    ]
    if positive_tags.strip():
        cmd.append(f"--positive-tags={positive_tags}")
    if negative_tags.strip():
        cmd.append(f"--negative-tags={negative_tags}")
    cmd.extend(["--", "bash", "-lc", remote_cmd])
    completed = run_local(cmd)
    process_name = completed.stdout.strip().splitlines()[-1].strip()
    if not process_name:
        raise RuntimeError(f"brainctl launch returned empty process name:\n{completed.stdout}\n{completed.stderr}")
    return process_name


def stop_delete_worker(config: BenchmarkConfig, process_name: str) -> None:
    log(f"Cleaning up worker: {process_name}")
    run_local([*brainctl_base(config), "stop", f"process/{process_name}", "-n", config.brainctl_namespace], check=False)
    run_local([*brainctl_base(config), "delete", f"process/{process_name}", "-n", config.brainctl_namespace], check=False)


def exec_root(config: BenchmarkConfig, process_name: str, remote_cmd: str, *, timeout_s: float | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run_local([
        *brainctl_base(config),
        "exec",
        f"process/{process_name}",
        "-n",
        config.brainctl_namespace,
        "--",
        "bash",
        "-lc",
        remote_cmd,
    ], timeout_s=timeout_s, check=check)


def exec_user(config: BenchmarkConfig, process_name: str, remote_cmd: str, *, timeout_s: float | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    user = getpass.getuser()
    wrapped = (
        "set -euo pipefail; "
        f"if ! id -u {user} >/dev/null 2>&1; then echo missing user {user} >&2; exit 1; fi; "
        f"su - {user} -s /bin/bash -c {shlex.quote(remote_cmd)}"
    )
    return exec_root(config, process_name, wrapped, timeout_s=timeout_s, check=check)


def wait_for_condition(*, timeout_s: float, poll_interval_s: float, description: str, check_fn) -> str:
    deadline = time.monotonic() + timeout_s
    last_detail = ""
    while time.monotonic() < deadline:
        ok, detail = check_fn()
        last_detail = detail
        if ok:
            return detail
        time.sleep(poll_interval_s)
    raise RuntimeError(f"Timed out waiting for {description}:\n{last_detail}")


def global_is_ready(status_text: str) -> bool:
    return "Global Store session" in status_text and "health : SERVING" in status_text


def daemon_is_running(status_text: str) -> bool:
    return "running" in status_text.lower()


def daemon_is_stopped(status_text: str) -> bool:
    lowered = status_text.lower()
    return "no local daemon session found" in lowered or "not found" in lowered


def build_service_remote_cmd(paths: BenchmarkPaths, subcommand: str, *args: str) -> str:
    service_script = paths.scripts_dir / "tensorcast_service.sh"
    joined_args = " ".join(shlex.quote(arg) for arg in args)
    return (
        f"cd {shlex.quote(str(paths.uv_project_root))}; "
        "source .venv/bin/activate; "
        f"bash {shlex.quote(str(service_script))} {shlex.quote(subcommand)}"
        + (f" {joined_args}" if joined_args else "")
    )


def reset_runtime_state(config: BenchmarkConfig, paths: BenchmarkPaths, worker: WorkerInfo) -> None:
    exec_user(config, worker.process_name, build_service_remote_cmd(paths, "reset-runtime-state"))


def start_global_store(config: BenchmarkConfig, paths: BenchmarkPaths, worker: WorkerInfo, config_path: Path) -> None:
    stop_global_store(config, paths, worker)
    reset_runtime_state(config, paths, worker)
    exec_user(config, worker.process_name, build_service_remote_cmd(paths, "start-global", str(config_path)))
    wait_for_condition(
        timeout_s=config.tensorcast_service_ready_timeout_s,
        poll_interval_s=config.tensorcast_service_poll_interval_s,
        description="global store ready",
        check_fn=lambda: (
            global_is_ready(status := exec_user(config, worker.process_name, build_service_remote_cmd(paths, "status-global"), check=False).stdout),
            status,
        ),
    )


def stop_global_store(config: BenchmarkConfig, paths: BenchmarkPaths, worker: WorkerInfo) -> None:
    exec_user(config, worker.process_name, build_service_remote_cmd(paths, "stop-global"), check=False)


def start_daemon(config: BenchmarkConfig, paths: BenchmarkPaths, worker: WorkerInfo, config_path: Path, global_store_address: str) -> None:
    stop_daemon(config, paths, worker)
    exec_user(
        config,
        worker.process_name,
        build_service_remote_cmd(paths, "start-daemon", str(config_path), global_store_address, config.tensorcast_cuda_home, config.tensorcast_nvidia_lib_dirs),
    )
    exec_user(
        config,
        worker.process_name,
        build_service_remote_cmd(
            paths,
            "wait-daemon-ready",
            build_local_daemon_address(config),
            str(config.tensorcast_service_ready_timeout_s),
            str(config.tensorcast_service_poll_interval_s),
        ),
        timeout_s=config.tensorcast_service_ready_timeout_s + 30.0,
    )


def stop_daemon(config: BenchmarkConfig, paths: BenchmarkPaths, worker: WorkerInfo) -> None:
    exec_user(config, worker.process_name, build_service_remote_cmd(paths, "stop-daemon"), check=False)
    with contextlib.suppress(Exception):
        wait_for_condition(
            timeout_s=config.tensorcast_service_ready_timeout_s,
            poll_interval_s=config.tensorcast_service_poll_interval_s,
            description="daemon stopped",
            check_fn=lambda: (
                daemon_is_stopped(status := exec_user(config, worker.process_name, build_service_remote_cmd(paths, "status-daemon"), check=False).stdout),
                status,
            ),
        )


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def build_tensorcast_configs(config: BenchmarkConfig, paths: BenchmarkPaths, run_id: str, worker_a: WorkerInfo, worker_b: WorkerInfo | None) -> ServiceLogs:
    global_cfg = load_yaml(Path(config.tensorcast_global_store_config))
    daemon_cfg = load_yaml(Path(config.tensorcast_daemon_config))
    global_store_log = Path(f"/data/{run_id}_worker_a_global_store.log")
    daemon_a_log = Path(f"/data/{run_id}_worker_a_daemon_a.log")
    global_cfg["server"]["listen"]["port"] = config.tensorcast_global_store_port
    global_cfg["server"]["advertise"]["host"] = worker_a.ip
    global_cfg["server"]["advertise"]["port"] = config.tensorcast_global_store_port
    global_cfg["observability"]["logging"]["file"] = str(global_store_log)
    daemon_a_cfg = json.loads(json.dumps(daemon_cfg))
    daemon_a_cfg["server"]["listen"]["port"] = config.tensorcast_daemon_port
    daemon_a_cfg["server"]["advertise"]["host"] = worker_a.ip
    daemon_a_cfg["server"]["p2p_listen"]["port"] = config.tensorcast_daemon_p2p_port
    daemon_a_cfg["high_availability"]["global_store_endpoints"][0]["host"] = worker_a.ip
    daemon_a_cfg["high_availability"]["global_store_endpoints"][0]["port"] = config.tensorcast_global_store_port
    daemon_a_cfg["observability"]["logging"]["file"] = str(daemon_a_log)
    dump_yaml(paths.generated_configs_dir / "global_store.yaml", global_cfg)
    dump_yaml(paths.generated_configs_dir / "daemon_a.yaml", daemon_a_cfg)
    daemon_b_log = None
    if worker_b is not None:
        daemon_b_log = Path(f"/data/{run_id}_worker_b_daemon_b.log")
        daemon_b_cfg = json.loads(json.dumps(daemon_cfg))
        daemon_b_cfg["server"]["listen"]["port"] = config.tensorcast_daemon_port
        daemon_b_cfg["server"]["advertise"]["host"] = worker_b.ip
        daemon_b_cfg["server"]["p2p_listen"]["port"] = config.tensorcast_daemon_p2p_port
        daemon_b_cfg["high_availability"]["global_store_endpoints"][0]["host"] = worker_a.ip
        daemon_b_cfg["high_availability"]["global_store_endpoints"][0]["port"] = config.tensorcast_global_store_port
        daemon_b_cfg["observability"]["logging"]["file"] = str(daemon_b_log)
        dump_yaml(paths.generated_configs_dir / "daemon_b.yaml", daemon_b_cfg)
    return ServiceLogs(global_store_data_log=global_store_log, daemon_a_data_log=daemon_a_log, daemon_b_data_log=daemon_b_log)


def build_local_daemon_address(config: BenchmarkConfig) -> str:
    return f"127.0.0.1:{config.tensorcast_daemon_port}"


def copy_if_exists(src: Path | None, dst: Path) -> None:
    if src is None or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def publish_artifact_versions(config: BenchmarkConfig, paths: BenchmarkPaths, worker_a: WorkerInfo, daemon_address: str, run_id: str) -> dict[int, str]:
    publish_script = paths.scripts_dir / "publish_weight_versions.py"
    publish_log = paths.logs_dir / "worker_a_publish_versions.log"
    publish_json = paths.logs_dir / "worker_a_publish_versions.json"
    history_path = f"/data/{run_id}_weights_history.json"
    remote_cmd = (
        "set -euo pipefail; "
        f"cd {shlex.quote(str(paths.uv_project_root))}; "
        "source .venv/bin/activate; "
        f"/home/i-zhouyuhan/.local/bin/uv run --active --no-project --offline python {shlex.quote(str(publish_script))} "
        f"--model-path {shlex.quote(config.model_path)} "
        f"--model-name {shlex.quote(config.model_name)} "
        f"--weight-version-start {config.weight_version_start} "
        f"--num-versions {config.trials + 1} "
        f"--daemon-address {shlex.quote(daemon_address)} "
        f"--key-template {shlex.quote(config.tensorcast_key_template)} "
        f"--history-path {shlex.quote(history_path)} "
        f"--output-json {shlex.quote(str(publish_json))} "
        f"2>&1 | tee {shlex.quote(str(publish_log))}"
    )
    exec_user(config, worker_a.process_name, remote_cmd, timeout_s=7200.0)
    payload = json.loads(publish_json.read_text(encoding="utf-8"))
    return {int(item["weight_version"]): str(item["artifact_id"]) for item in payload["versions"]}


def run_remote_benchmark(config: BenchmarkConfig, paths: BenchmarkPaths, worker_b: WorkerInfo, *, run_id: str, daemon_address: str) -> dict[str, object]:
    remote_script = paths.scripts_dir / "run_sglang_update_trials.py"
    server_log_path = paths.logs_dir / f"{run_id}_{config.load_format}_{config.topology_mode}_tp{config.tp_size}_server.log"
    result_json = paths.logs_dir / "worker_b_result.json"
    trial_prefix = f"{run_id}_{config.load_format}_{config.topology_mode}_tp{config.tp_size}"
    remote_cmd = (
        "set -euo pipefail; "
        f"cd {shlex.quote(str(paths.uv_project_root))}; "
        "source .venv/bin/activate; "
        f"/home/i-zhouyuhan/.local/bin/uv run --active --no-project --offline python {shlex.quote(str(remote_script))} "
        f"--load-format {shlex.quote(config.load_format)} "
        f"--model-path {shlex.quote(config.model_path)} "
        f"--model-name {shlex.quote(config.model_name)} "
        f"--weight-version-start {config.weight_version_start} "
        f"--trials {config.trials} "
        f"--tp-size {config.tp_size} "
        f"--port {config.port} "
        f"--mem-fraction-static {config.mem_fraction_static} "
        f"--log-level {shlex.quote(config.log_level)} "
        f"--health-path {shlex.quote(config.health_path)} "
        f"--server-ready-timeout-s {config.server_ready_timeout_s} "
        f"--health-poll-interval-s {config.health_poll_interval_s} "
        f"--request-timeout-s {config.request_timeout_s} "
        f"--log-path {shlex.quote(str(server_log_path))} "
        f"--output-json {shlex.quote(str(result_json))} "
        f"--trial-log-dir {shlex.quote(str(paths.logs_dir))} "
        f"--trial-log-prefix {shlex.quote(trial_prefix)} "
        f"--extra-server-args {shlex.quote(config.extra_server_args)} "
        + ("--launch-vvv " if config.launch_vvv else "")
        + ("--enable-metrics " if config.enable_metrics else "")
        + ("--flush-cache " if config.flush_cache else "")
        + ("--abort-all-requests " if config.abort_all_requests else "")
        + ("--recapture-cuda-graph " if config.recapture_cuda_graph else "")
    )
    if config.load_format == "default":
        remote_cmd += f"--clear-cache-script {shlex.quote(str(paths.scripts_dir / 'clear_remote_fs_cache.sh'))} "
        for cache_dir in config.cache_dirs:
            remote_cmd += f"--cache-dir {shlex.quote(cache_dir)} "
    else:
        remote_cmd += (
            f"--tensorcast-init-mode {shlex.quote(config.tensorcast_init_mode)} "
            f"--tensorcast-daemon-address {shlex.quote(daemon_address)} "
            f"--tensorcast-key-template {shlex.quote(config.tensorcast_key_template)} "
            f"--tensorcast-fallback-prefer {shlex.quote(config.tensorcast_fallback_prefer)} "
            f"--tensorcast-get-prefer {shlex.quote(config.tensorcast_get_prefer)} "
            f"--tensorcast-export-policy {shlex.quote(config.tensorcast_export_policy)} "
            f"--tensorcast-tp-slice-plan-cache-dir {shlex.quote(config.tensorcast_tp_slice_plan_cache_dir)} "
        )
        if config.tensorcast_show_daemon_logs:
            remote_cmd += "--tensorcast-show-daemon-logs "
        if config.tensorcast_allow_disk_fallback:
            remote_cmd += "--tensorcast-allow-disk-fallback "
        if config.tensorcast_trace_tp_slices:
            remote_cmd += "--tensorcast-trace-tp-slices "
    completed = exec_user(config, worker_b.process_name, remote_cmd, timeout_s=config.server_ready_timeout_s + config.request_timeout_s * max(config.trials, 1) + 600.0, check=False)
    if not result_json.exists():
        raise RuntimeError(
            f"remote benchmark result json missing; returncode={completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    if completed.returncode != 0 and payload.get("status") == "success":
        payload["status"] = "failed"
        payload["error_message"] = f"remote benchmark command returned non-zero: returncode={completed.returncode}"
    return payload


def append_result(csv_path: Path, result: TrialResult) -> None:
    header = [
        "timestamp",
        "run_id",
        "trial_id",
        "target_weight_version",
        "model_path",
        "model_name",
        "tp_size",
        "load_format",
        "topology_mode",
        "endpoint",
        "load_time_s",
        "ready_time_s",
        "status",
        "error_message",
        "artifact_id",
        "worker_a_process",
        "worker_a_hostname",
        "worker_a_ip",
        "worker_a_node",
        "worker_b_process",
        "worker_b_hostname",
        "worker_b_ip",
        "worker_b_node",
        "global_store_address",
        "daemon_a_address",
        "daemon_b_address",
        "log_path",
        "server_log_path",
        "run_dir",
    ]
    row = {
        "timestamp": result.timestamp,
        "run_id": result.run_id,
        "trial_id": result.trial_id,
        "target_weight_version": result.target_weight_version,
        "model_path": result.model_path,
        "model_name": result.model_name,
        "tp_size": result.tp_size,
        "load_format": result.load_format,
        "topology_mode": result.topology_mode,
        "endpoint": result.endpoint,
        "load_time_s": f"{result.load_time_s:.6f}" if result.load_time_s is not None else "",
        "ready_time_s": f"{result.ready_time_s:.6f}" if result.ready_time_s is not None else "",
        "status": result.status,
        "error_message": result.error_message,
        "artifact_id": result.artifact_id,
        "worker_a_process": result.worker_a_process,
        "worker_a_hostname": result.worker_a_hostname,
        "worker_a_ip": result.worker_a_ip,
        "worker_a_node": result.worker_a_node,
        "worker_b_process": result.worker_b_process,
        "worker_b_hostname": result.worker_b_hostname,
        "worker_b_ip": result.worker_b_ip,
        "worker_b_node": result.worker_b_node,
        "global_store_address": result.global_store_address,
        "daemon_a_address": result.daemon_a_address,
        "daemon_b_address": result.daemon_b_address,
        "log_path": result.log_path,
        "server_log_path": result.server_log_path,
        "run_dir": result.run_dir,
    }
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_parser(default_global_cfg: Path, default_daemon_cfg: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark remote SGLang update-weight latency on brainctl workers.")
    parser.add_argument("--load-format", choices=("tensorcast", "default"), default="tensorcast")
    parser.add_argument("--topology-mode", choices=("direct", "relay"), default="relay")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--weight-version-start", type=int, default=0)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--log-level", default="debug")
    parser.add_argument("--disable-metrics", action="store_true")
    parser.add_argument("--no-launch-vvv", action="store_true")
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--server-ready-timeout-s", type=float, default=1800.0)
    parser.add_argument("--health-poll-interval-s", type=float, default=0.5)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--uv-project-root", default="")
    parser.add_argument("--keep-workers", action="store_true")
    parser.add_argument("--worker-ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--worker-poll-interval-s", type=float, default=5.0)
    parser.add_argument("--different-node-max-attempts", type=int, default=5)
    parser.add_argument("--existing-worker-a-process", default="")
    parser.add_argument("--existing-worker-b-process", default="")
    parser.add_argument("--worker-a-gpu", type=int, default=1)
    parser.add_argument("--worker-a-cpu", type=int, default=64)
    parser.add_argument("--worker-a-memory", type=int, default=500000)
    parser.add_argument("--worker-a-positive-tags", default="H800")
    parser.add_argument("--worker-a-negative-tags", default="")
    parser.add_argument("--worker-b-gpu", type=int, default=4)
    parser.add_argument("--worker-b-cpu", type=int, default=64)
    parser.add_argument("--worker-b-memory", type=int, default=500000)
    parser.add_argument("--worker-b-positive-tags", default="H800")
    parser.add_argument("--brainctl-namespace", default="shai-core")
    parser.add_argument("--brainctl-charged-group", default="codesign")
    parser.add_argument("--brainctl-private-machine", default="group")
    parser.add_argument("--brainctl-mount", default="juicefs+s3://oss.i.shaipower.com/step2-alignment-jfs:/mnt/step2-alignment-jfs")
    parser.add_argument("--brainctl-max-wait-duration", default="10m")
    parser.add_argument("--flush-cache", action="store_true", default=True)
    parser.add_argument("--no-flush-cache", action="store_true")
    parser.add_argument("--abort-all-requests", action="store_true", default=True)
    parser.add_argument("--no-abort-all-requests", action="store_true")
    parser.add_argument("--recapture-cuda-graph", action="store_true", default=True)
    parser.add_argument("--no-recapture-cuda-graph", action="store_true")
    parser.add_argument("--cache-dir", action="append", default=[])
    parser.add_argument("--tensorcast-global-store-config", default=str(default_global_cfg))
    parser.add_argument("--tensorcast-daemon-config", default=str(default_daemon_cfg))
    parser.add_argument("--tensorcast-service-ready-timeout-s", type=float, default=120.0)
    parser.add_argument("--tensorcast-service-poll-interval-s", type=float, default=2.0)
    parser.add_argument("--tensorcast-cuda-home", default="/usr/local/cuda-12.4")
    parser.add_argument("--tensorcast-nvidia-lib-dirs", default="/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64")
    parser.add_argument("--tensorcast-init-mode", choices=("connect", "auto", "create"), default="connect")
    parser.add_argument("--tensorcast-show-daemon-logs", action="store_true")
    parser.add_argument("--tensorcast-key-template", default="model:{model_name}:v{weight_version}")
    parser.add_argument("--tensorcast-no-allow-disk-fallback", action="store_true")
    parser.add_argument("--tensorcast-fallback-prefer", default="auto")
    parser.add_argument("--tensorcast-get-prefer", default="auto")
    parser.add_argument("--tensorcast-export-policy", default="auto")
    parser.add_argument("--tensorcast-trace-tp-slices", action="store_true")
    parser.add_argument("--tensorcast-tp-slice-plan-cache-dir", default="/tmp/sglang_tensorcast_trace_cache")
    return parser


def parse_config(paths: BenchmarkPaths) -> BenchmarkConfig:
    parser = build_parser(paths.configs_dir / "global_store_config.yaml", paths.configs_dir / "store_daemon_config.yaml")
    args = parser.parse_args()
    return BenchmarkConfig(
        model_path=args.model_path,
        model_name=args.model_name,
        load_format=args.load_format,
        topology_mode=args.topology_mode,
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
        flush_cache=not args.no_flush_cache,
        abort_all_requests=not args.no_abort_all_requests,
        recapture_cuda_graph=not args.no_recapture_cuda_graph,
        uv_project_root=args.uv_project_root,
        keep_workers=args.keep_workers,
        brainctl_namespace=args.brainctl_namespace,
        brainctl_charged_group=args.brainctl_charged_group,
        brainctl_private_machine=args.brainctl_private_machine,
        brainctl_mount=args.brainctl_mount,
        brainctl_max_wait_duration=args.brainctl_max_wait_duration,
        worker_ready_timeout_s=args.worker_ready_timeout_s,
        worker_poll_interval_s=args.worker_poll_interval_s,
        different_node_max_attempts=args.different_node_max_attempts,
        existing_worker_a_process=args.existing_worker_a_process,
        existing_worker_b_process=args.existing_worker_b_process,
        worker_a_gpu=args.worker_a_gpu,
        worker_a_cpu=args.worker_a_cpu,
        worker_a_memory=args.worker_a_memory,
        worker_a_positive_tags=args.worker_a_positive_tags,
        worker_a_negative_tags=args.worker_a_negative_tags,
        worker_b_gpu=args.worker_b_gpu,
        worker_b_cpu=args.worker_b_cpu,
        worker_b_memory=args.worker_b_memory,
        worker_b_positive_tags=args.worker_b_positive_tags,
        cache_dirs=args.cache_dir,
        tensorcast_global_store_config=args.tensorcast_global_store_config,
        tensorcast_daemon_config=args.tensorcast_daemon_config,
        tensorcast_service_ready_timeout_s=args.tensorcast_service_ready_timeout_s,
        tensorcast_service_poll_interval_s=args.tensorcast_service_poll_interval_s,
        tensorcast_cuda_home=args.tensorcast_cuda_home,
        tensorcast_nvidia_lib_dirs=args.tensorcast_nvidia_lib_dirs,
        tensorcast_init_mode=args.tensorcast_init_mode,
        tensorcast_show_daemon_logs=args.tensorcast_show_daemon_logs,
        tensorcast_key_template=args.tensorcast_key_template,
        tensorcast_allow_disk_fallback=not args.tensorcast_no_allow_disk_fallback,
        tensorcast_fallback_prefer=args.tensorcast_fallback_prefer,
        tensorcast_get_prefer=args.tensorcast_get_prefer,
        tensorcast_export_policy=args.tensorcast_export_policy,
        tensorcast_trace_tp_slices=args.tensorcast_trace_tp_slices,
        tensorcast_tp_slice_plan_cache_dir=args.tensorcast_tp_slice_plan_cache_dir,
    )


def relaunch_worker_a_until_distinct(config: BenchmarkConfig, run_id: str, worker_b: WorkerInfo) -> tuple[str, WorkerInfo]:
    attempts = 0
    while attempts < config.different_node_max_attempts:
        attempts += 1
        process_name = launch_worker(
            config,
            run_id=run_id,
            role=f"worker-a-attempt{attempts}",
            gpu=config.worker_a_gpu,
            cpu=config.worker_a_cpu,
            memory=config.worker_a_memory,
            positive_tags=config.worker_a_positive_tags,
            negative_tags=",".join(item for item in (config.worker_a_negative_tags.strip(), f"node/{worker_b.node}") if item),
        )
        info = wait_for_worker_running(config, process_name)
        if info.node != worker_b.node:
            return process_name, info
        log(f"worker A landed on same node as worker B ({info.node}); relaunching worker A")
        stop_delete_worker(config, process_name)
    raise RuntimeError("failed to place worker A on a different node from worker B")


def main() -> None:
    global ORCHESTRATOR_LOG_PATH
    provisional_run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    provisional_paths = build_paths(provisional_run_id, "")
    config = parse_config(provisional_paths)
    topology_label = config.topology_mode if config.load_format == "tensorcast" else "default"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{config.load_format}_{topology_label}_tp{config.tp_size}"
    paths = build_paths(run_id, config.uv_project_root)
    prepare_paths(paths)
    ORCHESTRATOR_LOG_PATH = paths.logs_dir / "orchestrator.log"

    log(f"Run id: {run_id}")
    log(f"Benchmark start: load_format={config.load_format}, topology_mode={config.topology_mode}, tp_size={config.tp_size}, trials={config.trials}")

    worker_a: WorkerInfo | None = None
    worker_b: WorkerInfo | None = None
    launched_worker_a = False
    launched_worker_b = False
    service_logs = ServiceLogs(None, None, None)
    artifact_ids: dict[int, str] = {}
    global_store_address = ""
    daemon_a_address = ""
    daemon_b_address = ""

    try:
        if config.existing_worker_b_process.strip():
            worker_b = get_worker_info(config, config.existing_worker_b_process)
            log(f"Reusing worker B: process={worker_b.process_name}, node={worker_b.node}, ip={worker_b.ip}")
        else:
            log("Launching worker B")
            worker_b_name = launch_worker(
                config,
                run_id=run_id,
                role="worker-b",
                gpu=config.worker_b_gpu,
                cpu=config.worker_b_cpu,
                memory=config.worker_b_memory,
                positive_tags=config.worker_b_positive_tags,
                negative_tags="",
            )
            worker_b = wait_for_worker_running(config, worker_b_name)
            launched_worker_b = True
        log(f"worker B ready: process={worker_b.process_name}, node={worker_b.node}, ip={worker_b.ip}")

        if config.load_format == "tensorcast":
            if config.existing_worker_a_process.strip():
                worker_a = get_worker_info(config, config.existing_worker_a_process)
                log(f"Reusing worker A: process={worker_a.process_name}, node={worker_a.node}, ip={worker_a.ip}")
                if config.topology_mode == "relay" and worker_a.node == worker_b.node:
                    raise RuntimeError("existing worker A and worker B are on the same node; relay mode requires different nodes")
            else:
                log("Launching worker A")
                worker_a_name, worker_a = relaunch_worker_a_until_distinct(config, run_id, worker_b)
                launched_worker_a = True
                log(f"worker A ready: process={worker_a_name}, node={worker_a.node}, ip={worker_a.ip}")

            service_logs = build_tensorcast_configs(
                config,
                paths,
                run_id,
                worker_a,
                worker_b if config.topology_mode == "relay" else None,
            )
            global_store_address = f"{worker_a.ip}:{config.tensorcast_global_store_port}"
            daemon_a_address = f"{worker_a.ip}:{config.tensorcast_daemon_port}"
            log(f"Starting Global Store on worker A at {global_store_address}")
            start_global_store(config, paths, worker_a, paths.generated_configs_dir / "global_store.yaml")
            log(f"Starting daemon A on worker A at {daemon_a_address}")
            start_daemon(config, paths, worker_a, paths.generated_configs_dir / "daemon_a.yaml", global_store_address)
            log("Publishing weight versions on worker A")
            artifact_ids = publish_artifact_versions(config, paths, worker_a, daemon_a_address, run_id)
            log(f"Published versions: {sorted(artifact_ids)}")
            if config.topology_mode == "relay":
                log("Starting daemon B on worker B")
                reset_runtime_state(config, paths, worker_b)
                start_daemon(config, paths, worker_b, paths.generated_configs_dir / "daemon_b.yaml", global_store_address)
                daemon_b_address = build_local_daemon_address(config)
            else:
                daemon_b_address = daemon_a_address

        payload = run_remote_benchmark(
            config,
            paths,
            worker_b,
            run_id=run_id,
            daemon_address=daemon_b_address,
        )

        server_log_path = str(payload.get("server_log_path", ""))
        for trial_payload in payload.get("trial_results", []):
            if str(trial_payload.get("status", "failed")) != "success":
                break
            version = int(trial_payload["target_weight_version"])
            result = TrialResult(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                run_id=run_id,
                trial_id=int(trial_payload["trial_id"]),
                target_weight_version=version,
                model_path=config.model_path,
                model_name=config.model_name,
                tp_size=config.tp_size,
                load_format=config.load_format,
                topology_mode=(config.topology_mode if config.load_format == "tensorcast" else "default"),
                endpoint=str(trial_payload["endpoint"]),
                load_time_s=(float(trial_payload["load_time_s"]) if trial_payload.get("load_time_s") is not None else None),
                ready_time_s=(float(trial_payload["ready_time_s"]) if trial_payload.get("ready_time_s") is not None else None),
                status="success",
                error_message="",
                artifact_id=artifact_ids.get(version, ""),
                worker_a_process="" if worker_a is None else worker_a.process_name,
                worker_a_hostname="" if worker_a is None else worker_a.hostname,
                worker_a_ip="" if worker_a is None else worker_a.ip,
                worker_a_node="" if worker_a is None else worker_a.node,
                worker_b_process=worker_b.process_name,
                worker_b_hostname=worker_b.hostname,
                worker_b_ip=worker_b.ip,
                worker_b_node=worker_b.node,
                global_store_address=global_store_address,
                daemon_a_address=daemon_a_address,
                daemon_b_address=daemon_b_address,
                log_path=str(trial_payload["log_path"]),
                server_log_path=server_log_path,
                run_dir=str(paths.run_dir),
            )
            append_result(paths.csv_path, result)
            log(f"Trial {result.trial_id} end: status=success, load_time_s={result.load_time_s}, ready_time_s={result.ready_time_s}")

        if payload.get("status") != "success":
            raise RuntimeError(str(payload.get("error_message", "remote benchmark failed")))

    finally:
        if config.load_format == "tensorcast":
            if config.topology_mode == "relay":
                with contextlib.suppress(Exception):
                    stop_daemon(config, paths, worker_b)
            if worker_a is not None and (launched_worker_a or config.keep_workers):
                copy_if_exists(service_logs.global_store_data_log, paths.logs_dir / "worker_a_global_store.log")
                copy_if_exists(service_logs.daemon_a_data_log, paths.logs_dir / "worker_a_daemon_a.log")
            if config.topology_mode == "relay":
                copy_if_exists(service_logs.daemon_b_data_log, paths.logs_dir / "worker_b_daemon_b.log")
            if worker_a is not None and launched_worker_a and not config.keep_workers:
                with contextlib.suppress(Exception):
                    stop_daemon(config, paths, worker_a)
                with contextlib.suppress(Exception):
                    stop_global_store(config, paths, worker_a)
        if not config.keep_workers:
            if worker_a is not None and launched_worker_a:
                stop_delete_worker(config, worker_a.process_name)
            if worker_b is not None and launched_worker_b:
                stop_delete_worker(config, worker_b.process_name)
        else:
            if worker_a is not None:
                suffix = " (reused)" if not launched_worker_a else ""
                log(f"Keeping worker A for debug: {worker_a.process_name}{suffix}")
            if worker_b is not None:
                suffix = " (reused)" if not launched_worker_b else ""
                log(f"Keeping worker B for debug: {worker_b.process_name}{suffix}")

    log(f"Benchmark finished. CSV appended: {paths.csv_path}")


if __name__ == "__main__":
    main()
