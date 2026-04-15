#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shlex
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from tensorcast_benchmark.kv.models import BenchmarkConfig, RunSummary
from tensorcast_benchmark.kv.outputs import (
    append_csv_row,
    build_paths,
    create_run_id,
    prepare_paths,
    write_json,
)
from tensorcast_benchmark.kv.remote import (
    exec_user,
    launch_worker,
    stop_delete_worker,
    wait_for_condition,
    wait_for_worker_running,
)

ORCHESTRATOR_LOG_PATH: Path | None = None


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    if ORCHESTRATOR_LOG_PATH is not None:
        ORCHESTRATOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ORCHESTRATOR_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(line + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the share-local KV benchmark.")
    parser.add_argument(
        "--hicache-storage-backend",
        choices=["mooncake", "tensorcast"],
        default="mooncake",
    )
    parser.add_argument(
        "--tensorcast-daemon-mode", choices=["share", "separate"], default="share"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--prompt-count", type=int, default=10)
    parser.add_argument("--pair-rps", type=float, default=1.0)
    parser.add_argument("--settle-ms", type=int, default=20000)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--min-prompt-chars", type=int, default=0)
    parser.add_argument("--max-prompt-chars", type=int, default=0)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--hicache-mem-layout", default="page_first")
    parser.add_argument("--hicache-io-backend", default="kernel")
    parser.add_argument("--hicache-ratio", type=float, default=2.0)
    parser.add_argument("--hicache-size-gb", type=int, default=0)
    parser.add_argument(
        "--hicache-storage-prefetch-policy",
        choices=["best_effort", "wait_complete", "timeout"],
        default="best_effort",
    )
    parser.add_argument("--port-a", type=int, default=31000)
    parser.add_argument("--port-b", type=int, default=31001)
    parser.add_argument("--instance-a-cuda-visible-devices", default="0,1")
    parser.add_argument("--instance-b-cuda-visible-devices", default="4,5")
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--instance-ready-timeout-s", type=float, default=1800.0)
    parser.add_argument("--instance-health-poll-interval-s", type=float, default=1.0)
    parser.add_argument("--wait-for-source-publication-drain", action="store_true")
    parser.add_argument(
        "--source-publication-drain-timeout-s", type=float, default=120.0
    )
    parser.add_argument("--source-publication-drain-idle-s", type=float, default=10.0)
    parser.add_argument("--source-publication-drain-poll-s", type=float, default=0.25)
    parser.add_argument("--require-positive-ttft-improvement", action="store_true")
    parser.add_argument("--data-path", default="/home/i-zhouyuhan/tot/data/test.jsonl")
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-worker", action="store_true")
    parser.add_argument("--existing-worker-process", default="")
    parser.add_argument("--brainctl-charged-group", default="")
    parser.add_argument("--worker-gpu", type=int, default=8)
    parser.add_argument("--worker-cpu", type=int, default=128)
    parser.add_argument("--worker-memory", type=int, default=1000000)
    parser.add_argument("--worker-positive-tags", default="H800")
    parser.add_argument("--worker-negative-tags", default="")
    parser.add_argument("--tensorcast-global-store-port", type=int, default=50051)
    parser.add_argument("--tensorcast-daemon-port-a", type=int, default=50052)
    parser.add_argument("--tensorcast-daemon-port-b", type=int, default=50053)
    parser.add_argument(
        "--tensorcast-instance-agent-port-a",
        type=int,
        default=31110,
    )
    parser.add_argument(
        "--tensorcast-instance-agent-port-b",
        type=int,
        default=31111,
    )
    parser.add_argument("--tensorcast-daemon-p2p-port-a", type=int, default=65090)
    parser.add_argument("--tensorcast-daemon-p2p-port-b", type=int, default=65091)
    parser.add_argument("--tensorcast-prefetch-threshold", type=int, default=1)
    parser.add_argument(
        "--tensorcast-service-ready-timeout-s", type=float, default=120.0
    )
    parser.add_argument("--tensorcast-service-poll-interval-s", type=float, default=2.0)
    parser.add_argument("--tensorcast-daemon-stable-bytes", default="64GB")
    parser.add_argument("--tensorcast-byte-artifact-shard-count", type=int, default=8)
    parser.add_argument(
        "--tensorcast-byte-artifact-lease-ttl-s", type=float, default=30.0
    )
    parser.add_argument(
        "--tensorcast-byte-artifact-keepalive-interval-s", type=float, default=5.0
    )
    parser.add_argument(
        "--tensorcast-payload-max-chunk-bytes", type=int, default=(1 << 20)
    )
    parser.add_argument(
        "--tensorcast-max-batch-payload-bytes", type=int, default=(16 << 20)
    )
    parser.add_argument(
        "--tensorcast-host-allocator-enabled",
        action="store_true",
    )
    parser.add_argument(
        "--tensorcast-host-allocator-region-ttl-ms",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--tensorcast-host-allocator-region-name",
        default="sglang_tensorcast_host_pool",
    )
    parser.add_argument("--tensorcast-cuda-home", default="/usr/local/cuda-12.4")
    parser.add_argument(
        "--tensorcast-nvidia-lib-dirs",
        default="/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64",
    )
    parser.add_argument("--mooncake-http-metadata-server-port", type=int, default=8080)
    parser.add_argument("--mooncake-master-port", type=int, default=60051)
    parser.add_argument("--mooncake-global-segment-size", default="4gb")
    parser.add_argument("--mooncake-local-buffer-size", type=int, default=0)
    parser.add_argument(
        "--mooncake-eviction-high-watermark-ratio", type=float, default=0.9
    )
    return parser


def build_remote_prefix(paths) -> str:
    return build_remote_python_prefix(paths, include_benchmark=False)


def build_remote_python_prefix(paths, *, include_benchmark: bool) -> str:
    python_paths = [paths.sglang_root / "python"]
    if include_benchmark:
        python_paths.insert(0, paths.sglang_root / "benchmark")
    python_path = ":".join(str(path) for path in python_paths)
    return (
        f"cd {shlex.quote(str(paths.sglang_root))}; "
        f"source {shlex.quote(str(paths.workspace_root / '.venv' / 'bin' / 'activate'))}; "
        f"export PYTHONPATH={shlex.quote(python_path)}:${{PYTHONPATH:-}}; "
        f"export PATH={shlex.quote(str(paths.venv_python.parent))}:$PATH; "
    )


def build_remote_python(paths) -> str:
    return shlex.quote(str(paths.venv_python))


def build_remote_uv(paths) -> str:
    return shlex.quote(str(paths.uv_bin))


def remote_file_log(run_id: str, name: str) -> Path:
    return Path(f"/data/{run_id}_{name}.log")


def remote_pid_file(paths, name: str) -> Path:
    return paths.run_dir / f"{name}.pid"


def stop_remote_process(
    config: BenchmarkConfig,
    worker_process: str,
    pid_path: Path,
) -> None:
    remote_cmd = (
        "set -euo pipefail; "
        f"PID_PATH={shlex.quote(str(pid_path))}; "
        'if [[ ! -f "$PID_PATH" ]]; then exit 0; fi; '
        'PID=$(cat "$PID_PATH"); '
        'if kill -0 "$PID" >/dev/null 2>&1; then '
        '  kill "$PID" >/dev/null 2>&1 || true; '
        "  sleep 2; "
        '  if kill -0 "$PID" >/dev/null 2>&1; then kill -9 "$PID" >/dev/null 2>&1 || true; fi; '
        "fi; "
        'rm -f "$PID_PATH"'
    )
    exec_user(config, worker_process, remote_cmd, check=False)


def start_remote_process(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    paths,
    name: str,
    command: str,
    log_path: Path,
    pid_path: Path,
    exports: dict[str, str] | None = None,
) -> None:
    export_cmd = ""
    for key, value in (exports or {}).items():
        export_cmd += f"export {key}={shlex.quote(value)}; "
    remote_cmd = (
        "set -euo pipefail; "
        f"{build_remote_prefix(paths)}"
        f"{export_cmd}"
        f"LOG_PATH={shlex.quote(str(log_path))}; "
        f"PID_PATH={shlex.quote(str(pid_path))}; "
        'mkdir -p "$(dirname "$LOG_PATH")"; '
        'mkdir -p "$(dirname "$PID_PATH")"; '
        'if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" >/dev/null 2>&1; then '
        "  echo already-running; exit 0; "
        "fi; "
        f'nohup bash -lc {shlex.quote(command)} > "$LOG_PATH" 2>&1 < /dev/null & '
        'echo $! > "$PID_PATH"; '
        "echo STARTED " + shlex.quote(name) + ' PID=$(cat "$PID_PATH")'
    )
    exec_user(config, worker_process, remote_cmd)


def wait_remote_health(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    paths,
    url: str,
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    python_snippet = (
        "import sys, urllib.request; "
        f"response = urllib.request.urlopen({url!r}, timeout=2); "
        "print(response.status); "
        "raise SystemExit(0 if response.status == 200 else 1)"
    )

    def check_fn() -> tuple[bool, str]:
        completed = exec_user(
            config,
            worker_process,
            f"{build_remote_prefix(paths)}"
            f"{build_remote_uv(paths)} run --active --no-project --offline "
            f"python -c {shlex.quote(python_snippet)}",
            check=False,
        )
        detail = (completed.stdout + completed.stderr).strip()
        return completed.returncode == 0, detail

    wait_for_condition(
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        description=f"HTTP ready {url}",
        check_fn=check_fn,
    )


def capture_remote_snapshots(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    paths,
    run_id: str,
) -> None:
    gpu_log = remote_file_log(run_id, "gpu_snapshot")
    ports_log = remote_file_log(run_id, "port_snapshot")
    gpu_cmd = (
        "set -euo pipefail; "
        f"LOG_PATH={shlex.quote(str(gpu_log))}; "
        'nvidia-smi > "$LOG_PATH" 2>&1'
    )
    ports_cmd = (
        "set -euo pipefail; "
        f"LOG_PATH={shlex.quote(str(ports_log))}; "
        'ss -ltnp > "$LOG_PATH" 2>&1 || netstat -ltnp > "$LOG_PATH" 2>&1 || true'
    )
    exec_user(config, worker_process, gpu_cmd, check=False)
    exec_user(config, worker_process, ports_cmd, check=False)


def run_remote_smoke_checks(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    paths,
    run_id: str,
) -> None:
    log("Running remote environment smoke checks")
    smoke_file = Path(f"/data/{run_id}_smoke_check.txt")
    remote_cmd = (
        "set -euo pipefail; "
        f"{build_remote_prefix(paths)}"
        f"test -d {shlex.quote(str(paths.sglang_root))}; "
        f"test -x {shlex.quote(str(paths.venv_python))}; "
        f"test -x {shlex.quote(str(paths.uv_bin))}; "
        "nvidia-smi -L >/dev/null; "
        f"touch {shlex.quote(str(smoke_file))}; "
        f"echo SMOKE_OK > {shlex.quote(str(smoke_file))}; "
        f"test -s {shlex.quote(str(smoke_file))}; "
        f"rm -f {shlex.quote(str(smoke_file))}"
    )
    exec_user(config, worker_process, remote_cmd)


def write_mooncake_config(paths, config: BenchmarkConfig) -> Path:
    config_path = paths.generated_configs_dir / "mooncake_config.json"
    write_json(
        config_path,
        {
            "local_hostname": "localhost",
            "metadata_server": (
                f"http://127.0.0.1:{config.mooncake_http_metadata_server_port}/metadata"
            ),
            "master_server_address": f"127.0.0.1:{config.mooncake_master_port}",
            "protocol": config.mooncake_protocol,
            "device_name": config.mooncake_device_name,
            "global_segment_size": config.mooncake_global_segment_size,
            "local_buffer_size": config.mooncake_local_buffer_size,
        },
    )
    return config_path


def start_mooncake_service(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    paths,
    run_id: str,
) -> None:
    log("Starting Mooncake service")
    stop_mooncake_service(config, worker_process, paths=paths)
    log_path = remote_file_log(run_id, "mooncake_master")
    pid_path = remote_pid_file(paths, "mooncake_master")
    command = (
        f"{shlex.quote(str(paths.mooncake_master_bin))} "
        "--enable_http_metadata_server=true "
        f"--http_metadata_server_port={config.mooncake_http_metadata_server_port} "
        f"--eviction_high_watermark_ratio={config.mooncake_eviction_high_watermark_ratio} "
        f"--port={config.mooncake_master_port}"
    )
    start_remote_process(
        config,
        worker_process,
        paths=paths,
        name="mooncake_master",
        command=command,
        log_path=log_path,
        pid_path=pid_path,
    )
    wait_remote_health(
        config,
        worker_process,
        paths=paths,
        url=f"http://127.0.0.1:{config.mooncake_http_metadata_server_port}/health",
        timeout_s=120.0,
        poll_interval_s=1.0,
    )


def stop_mooncake_service(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    paths,
) -> None:
    stop_remote_process(
        config,
        worker_process,
        remote_pid_file(paths, "mooncake_master"),
    )
    remote_cmd = (
        "set -euo pipefail; "
        "if command -v pkill >/dev/null 2>&1; then "
        "  pkill -f '[m]ooncake_master' >/dev/null 2>&1 || true; "
        "fi; "
        f"for port in {config.mooncake_http_metadata_server_port} {config.mooncake_master_port}; do "
        "  if command -v fuser >/dev/null 2>&1; then "
        "    fuser -k ${port}/tcp >/dev/null 2>&1 || true; "
        "  fi; "
        "done"
    )
    exec_user(config, worker_process, remote_cmd, check=False)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def tensorcast_runtime_home(paths, worker_process: str, name: str) -> Path:
    return paths.outputs_dir / "_tensorcast_runtime" / worker_process / name


def build_tensorcast_service_remote_cmd(
    paths,
    *,
    tensorcast_home: Path,
    uv_bin: Path,
    subcommand: str,
    args: list[str] | None = None,
) -> str:
    joined_args = " ".join(shlex.quote(arg) for arg in (args or []))
    service_script = paths.scripts_dir / "tensorcast_service.sh"
    return (
        f"cd {shlex.quote(str(paths.workspace_root))}; "
        f"source {shlex.quote(str(paths.workspace_root / '.venv' / 'bin' / 'activate'))}; "
        f"export TENSORCAST_HOME={shlex.quote(str(tensorcast_home))}; "
        f"export UV_BIN={shlex.quote(str(uv_bin))}; "
        f"bash {shlex.quote(str(service_script))} {shlex.quote(subcommand)}"
        + (f" {joined_args}" if joined_args else "")
    )


def global_is_ready(status_text: str) -> bool:
    return "Global Store session" in status_text and "health : SERVING" in status_text


def daemon_is_stopped(status_text: str) -> bool:
    lowered = status_text.lower()
    return "no local daemon session found" in lowered or "not found" in lowered


def build_local_daemon_address(port: int) -> str:
    return f"127.0.0.1:{port}"


def daemon_handle_socket_path(paths, port: int) -> str:
    return str(paths.workspace_root.parent / ".tensorcast-kv-sockets" / f"{port}.sock")


def cleanup_remote_tcp_ports(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    ports: list[int],
) -> None:
    joined_ports = " ".join(str(port) for port in ports)
    remote_cmd = (
        "set -euo pipefail; "
        f"for port in {joined_ports}; do "
        "  if command -v fuser >/dev/null 2>&1; then "
        "    fuser -k ${port}/tcp >/dev/null 2>&1 || true; "
        "  fi; "
        "done"
    )
    exec_user(config, worker_process, remote_cmd, check=False)


def cleanup_remote_path(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    path: str,
) -> None:
    remote_cmd = (
        "set -euo pipefail; "
        f"mkdir -p {shlex.quote(str(Path(path).parent))}; "
        f"rm -f {shlex.quote(path)}"
    )
    exec_user(config, worker_process, remote_cmd, check=False)


def start_global_store(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    paths,
    global_store_config_path: Path,
    runtime_home: Path,
) -> None:
    cleanup_remote_tcp_ports(
        config,
        worker_process,
        ports=[config.tensorcast_global_store_port],
    )
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
            uv_bin=paths.uv_bin,
            subcommand="stop-global",
        ),
        check=False,
    )
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
            uv_bin=paths.uv_bin,
            subcommand="reset-runtime-state",
        ),
    )
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
            uv_bin=paths.uv_bin,
            subcommand="start-global",
            args=[str(global_store_config_path)],
        ),
    )
    wait_for_condition(
        timeout_s=config.tensorcast_service_ready_timeout_s,
        poll_interval_s=config.tensorcast_service_poll_interval_s,
        description="global store ready",
        check_fn=lambda: (
            global_is_ready(
                status := exec_user(
                    config,
                    worker_process,
                    build_tensorcast_service_remote_cmd(
                        paths,
                        tensorcast_home=runtime_home,
                        uv_bin=paths.uv_bin,
                        subcommand="status-global",
                    ),
                    check=False,
                ).stdout
            ),
            status,
        ),
    )


def stop_global_store(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    paths,
    runtime_home: Path,
) -> None:
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
            uv_bin=paths.uv_bin,
            subcommand="stop-global",
        ),
        check=False,
    )


def start_tensorcast_daemon(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    paths,
    daemon_config_path: Path,
    runtime_home: Path,
    global_store_address: str,
    daemon_port: int,
) -> None:
    cleanup_remote_tcp_ports(
        config,
        worker_process,
        ports=[daemon_port],
    )
    cleanup_remote_path(
        config,
        worker_process,
        path=daemon_handle_socket_path(paths, daemon_port),
    )
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
            uv_bin=paths.uv_bin,
            subcommand="stop-daemon",
        ),
        check=False,
    )
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
            uv_bin=paths.uv_bin,
            subcommand="reset-runtime-state",
        ),
    )
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
            uv_bin=paths.uv_bin,
            subcommand="start-daemon",
            args=[
                str(daemon_config_path),
                global_store_address,
                config.tensorcast_cuda_home,
                config.tensorcast_nvidia_lib_dirs,
            ],
        ),
    )
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
            uv_bin=paths.uv_bin,
            subcommand="wait-daemon-ready",
            args=[
                build_local_daemon_address(daemon_port),
                str(config.tensorcast_service_ready_timeout_s),
                str(config.tensorcast_service_poll_interval_s),
            ],
        ),
        timeout_s=config.tensorcast_service_ready_timeout_s + 30.0,
    )


def stop_tensorcast_daemon(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    paths,
    runtime_home: Path,
) -> None:
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
            uv_bin=paths.uv_bin,
            subcommand="stop-daemon",
        ),
        check=False,
    )
    with contextlib.suppress(Exception):
        wait_for_condition(
            timeout_s=config.tensorcast_service_ready_timeout_s,
            poll_interval_s=config.tensorcast_service_poll_interval_s,
            description="daemon stopped",
            check_fn=lambda: (
                daemon_is_stopped(
                    status := exec_user(
                        config,
                        worker_process,
                        build_tensorcast_service_remote_cmd(
                            paths,
                            tensorcast_home=runtime_home,
                            uv_bin=paths.uv_bin,
                            subcommand="status-daemon",
                        ),
                        check=False,
                    ).stdout
                ),
                status,
            ),
        )


def build_tensorcast_configs(
    config: BenchmarkConfig,
    *,
    paths,
    run_id: str,
    worker_ip: str,
) -> dict[str, Path]:
    global_cfg = load_yaml(
        paths.benchmark_root / "configs" / "global_store_config.yaml"
    )
    daemon_cfg = load_yaml(
        paths.benchmark_root / "configs" / "store_daemon_config.yaml"
    )
    global_cfg["server"]["listen"]["port"] = config.tensorcast_global_store_port
    global_cfg["server"]["advertise"]["host"] = worker_ip
    global_cfg["server"]["advertise"]["port"] = config.tensorcast_global_store_port
    global_cfg["observability"]["logging"]["file"] = str(
        remote_file_log(run_id, "tensorcast_global_store")
    )

    generated: dict[str, Path] = {
        "global": paths.generated_configs_dir / "tensorcast_global_store.yaml",
    }
    dump_yaml(generated["global"], global_cfg)
    capability_token_secret = f"tensorcast-benchmark-{run_id}"

    def build_daemon_config(port: int, p2p_port: int, log_name: str) -> dict:
        payload = json.loads(json.dumps(daemon_cfg))
        payload["server"]["listen"]["port"] = port
        payload["server"]["advertise"]["host"] = worker_ip
        payload["server"]["p2p_listen"]["port"] = p2p_port
        payload["engine"]["memory_tiers"]["stable_bytes"] = (
            config.tensorcast_daemon_stable_bytes
        )
        payload["high_availability"]["global_store_endpoints"][0]["host"] = worker_ip
        payload["high_availability"]["global_store_endpoints"][0]["port"] = (
            config.tensorcast_global_store_port
        )
        payload["high_availability"]["heartbeat_interval"] = "10s"
        payload["observability"]["logging"]["file"] = str(
            remote_file_log(run_id, log_name)
        )
        payload["lifecycle"]["handle_leases"]["local_handle_socket_path"] = (
            daemon_handle_socket_path(paths, port)
        )
        byte_artifact_routing = payload.setdefault("byte_artifact_routing", {})
        byte_artifact_routing["shard_count"] = (
            config.tensorcast_byte_artifact_shard_count
        )
        byte_artifact_routing["lease_ttl"] = (
            f"{config.tensorcast_byte_artifact_lease_ttl_s:g}s"
        )
        byte_artifact_routing["keepalive_interval"] = (
            f"{config.tensorcast_byte_artifact_keepalive_interval_s:g}s"
        )
        payload_transport = byte_artifact_routing.setdefault("payload_transport", {})
        payload_transport["max_chunk_bytes"] = config.tensorcast_payload_max_chunk_bytes
        payload_transport["batch_transport_protocol_version"] = 2
        payload_transport["communicator_source_enabled"] = True
        payload_transport["host_memory_export_enabled"] = True
        payload_transport["max_batch_payload_bytes"] = (
            config.tensorcast_max_batch_payload_bytes
        )
        payload_transport.setdefault("max_batch_items", 256)
        capability_tokens = payload.setdefault("capability_tokens", {})
        active_tokens = capability_tokens.setdefault("active", {})
        active_tokens["version"] = int(active_tokens.get("version", 1) or 1)
        if not str(active_tokens.get("secret", "")).strip():
            active_tokens["secret"] = capability_token_secret
        return payload

    if config.tensorcast_daemon_mode == "share":
        generated["daemon_shared"] = (
            paths.generated_configs_dir / "tensorcast_daemon_shared.yaml"
        )
        dump_yaml(
            generated["daemon_shared"],
            build_daemon_config(
                config.tensorcast_daemon_port_a,
                config.tensorcast_daemon_p2p_port_a,
                "tensorcast_daemon_shared",
            ),
        )
        return generated

    generated["daemon_a"] = paths.generated_configs_dir / "tensorcast_daemon_a.yaml"
    generated["daemon_b"] = paths.generated_configs_dir / "tensorcast_daemon_b.yaml"
    dump_yaml(
        generated["daemon_a"],
        build_daemon_config(
            config.tensorcast_daemon_port_a,
            config.tensorcast_daemon_p2p_port_a,
            "tensorcast_daemon_a",
        ),
    )
    dump_yaml(
        generated["daemon_b"],
        build_daemon_config(
            config.tensorcast_daemon_port_b,
            config.tensorcast_daemon_p2p_port_b,
            "tensorcast_daemon_b",
        ),
    )
    return generated


def build_tensorcast_backend_extra_config(
    *,
    config: BenchmarkConfig,
    daemon_port: int,
    global_store_address: str | None = None,
    execution_endpoint: str | None = None,
) -> dict[str, object]:
    model_id = config.model_name or Path(config.model_path).name
    model_version = hashlib.sha256(str(config.model_path).encode("utf-8")).hexdigest()[:16]
    payload: dict[str, object] = {
        "daemon_address": build_local_daemon_address(daemon_port),
        # Namespace is part of the byte-artifact CGID. Keep it stable across
        # runs so identical KV pages map to identical artifact IDs.
        "namespace": "share_local",
        "engine": "sglang",
        "model_id": model_id,
        "model_version": model_version,
        "policy_profile": "durable",
        "prefetch_threshold": config.tensorcast_prefetch_threshold,
    }
    if config.tensorcast_host_allocator_enabled:
        payload["host_allocator_enabled"] = True
        payload["host_allocator_region_ttl_ms"] = (
            config.tensorcast_host_allocator_region_ttl_ms
        )
        payload["host_allocator_region_name"] = (
            config.tensorcast_host_allocator_region_name
        )
    if global_store_address is not None:
        payload["instance_directory_address"] = global_store_address
    if execution_endpoint is not None:
        payload["instance_agent_execution_endpoint"] = execution_endpoint
    return payload


def build_sglang_command_for_instance(
    *,
    paths,
    config: BenchmarkConfig,
    port: int,
    mooncake_config_path: Path | None,
    tensorcast_backend_extra_config: dict[str, object] | None,
) -> tuple[str, dict[str, str]]:
    cmd = [
        shlex.quote(str(paths.uv_bin)),
        "run",
        "--active",
        "--no-project",
        "--offline",
        "python",
        "-m",
        "sglang.launch_server",
        "--host",
        config.host,
        "--port",
        str(port),
        "--model-path",
        shlex.quote(config.model_path),
        "--tp",
        str(config.tp_size),
        "--mem-fraction-static",
        str(config.mem_fraction_static),
    ]
    if config.trust_remote_code:
        cmd.append("--trust-remote-code")
    if config.enable_hierarchical_cache:
        cmd.append("--enable-hierarchical-cache")
        cmd.extend(["--hicache-mem-layout", shlex.quote(config.hicache_mem_layout)])
        cmd.extend(["--hicache-io-backend", shlex.quote(config.hicache_io_backend)])
        cmd.extend(["--hicache-ratio", str(config.hicache_ratio)])
        cmd.extend(["--hicache-size", str(config.hicache_size_gb)])
        cmd.extend(
            [
                "--hicache-storage-prefetch-policy",
                shlex.quote(config.hicache_storage_prefetch_policy),
            ]
        )
    env: dict[str, str] = {}
    if config.hicache_storage_backend == "mooncake":
        cmd.extend(["--hicache-storage-backend", "mooncake"])
        if mooncake_config_path is None:
            raise RuntimeError("mooncake_config_path is required for mooncake backend")
        env["SGLANG_HICACHE_MOONCAKE_CONFIG_PATH"] = str(mooncake_config_path)
    elif config.hicache_storage_backend == "tensorcast":
        if tensorcast_backend_extra_config is None:
            raise RuntimeError(
                "tensorcast_backend_extra_config is required for tensorcast backend"
            )
        cmd.extend(["--hicache-storage-backend", "tensorcast"])
        cmd.extend(
            [
                "--hicache-storage-backend-extra-config",
                shlex.quote(
                    json.dumps(
                        tensorcast_backend_extra_config,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ),
            ]
        )
    if config.extra_server_args.strip():
        cmd.append(config.extra_server_args.strip())
    return " ".join(cmd), env


def start_sglang_instance(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    paths,
    run_id: str,
    instance_name: str,
    cuda_visible_devices: str,
    port: int,
    mooncake_config_path: Path | None,
    tensorcast_backend_extra_config: dict[str, object] | None,
) -> None:
    log(
        f"Starting SGLang instance {instance_name} on port {port} with CUDA_VISIBLE_DEVICES={cuda_visible_devices}"
    )
    stop_sglang_instance(
        config,
        worker_process,
        paths=paths,
        instance_name=instance_name,
        port=port,
    )
    command, env = build_sglang_command_for_instance(
        paths=paths,
        config=config,
        port=port,
        mooncake_config_path=mooncake_config_path,
        tensorcast_backend_extra_config=tensorcast_backend_extra_config,
    )
    env = {"CUDA_VISIBLE_DEVICES": cuda_visible_devices, **env}
    start_remote_process(
        config,
        worker_process,
        paths=paths,
        name=f"sglang_{instance_name}",
        command=command,
        log_path=remote_file_log(run_id, f"sglang_{instance_name}"),
        pid_path=remote_pid_file(paths, f"sglang_{instance_name}"),
        exports=env,
    )
    wait_remote_health(
        config,
        worker_process,
        paths=paths,
        url=f"http://127.0.0.1:{port}/health",
        timeout_s=config.instance_ready_timeout_s,
        poll_interval_s=config.instance_health_poll_interval_s,
    )


def stop_sglang_instance(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    paths,
    instance_name: str,
    port: int | None = None,
) -> None:
    stop_remote_process(
        config,
        worker_process,
        remote_pid_file(paths, f"sglang_{instance_name}"),
    )
    if port is None:
        return
    remote_cmd = (
        "set -euo pipefail; "
        f"if command -v fuser >/dev/null 2>&1; then fuser -k {port}/tcp >/dev/null 2>&1 || true; fi"
    )
    exec_user(config, worker_process, remote_cmd, check=False)


def run_request_driver(
    config: BenchmarkConfig,
    worker_process: str,
    *,
    paths,
    run_id: str,
    worker_info,
) -> RunSummary:
    log("Running request-pair driver")
    remote_cmd = (
        f"{build_remote_python_prefix(paths, include_benchmark=True)}"
        f"{build_remote_uv(paths)} run --active --no-project --offline python -m "
        "tensorcast_benchmark.kv.share_local.request_driver "
        f"--run-id {shlex.quote(run_id)} "
        f"--backend {shlex.quote(config.hicache_storage_backend)} "
        f"--tensorcast-daemon-mode {shlex.quote(config.tensorcast_daemon_mode)} "
        f"--instance-a-url {shlex.quote(f'http://127.0.0.1:{config.port_a}')} "
        f"--instance-b-url {shlex.quote(f'http://127.0.0.1:{config.port_b}')} "
        f"--dataset-path {shlex.quote(config.data_path)} "
        f"--prompt-count {config.prompt_count} "
        f"--min-prompt-chars {config.min_prompt_chars} "
        f"--max-prompt-chars {config.max_prompt_chars} "
        f"--pair-rps {config.pair_rps} "
        f"--settle-ms {config.settle_ms} "
        f"--max-new-tokens {config.max_new_tokens} "
        f"--temperature {config.temperature} "
        f"--request-timeout-s {config.request_timeout_s} "
        f"--results-json-path {shlex.quote(str(paths.results_json_path))} "
        f"--summary-json-path {shlex.quote(str(paths.summary_json_path))} "
        f"--source-instance-log-path {shlex.quote(str(remote_file_log(run_id, 'sglang_instance_a')))} "
        f"--worker-process {shlex.quote(worker_info.process_name)} "
        f"--worker-host {shlex.quote(worker_info.hostname)} "
        f"--worker-ip {shlex.quote(worker_info.ip)} "
        f"--worker-node {shlex.quote(worker_info.node)} "
        f"--model-path {shlex.quote(config.model_path)} "
        f"--tp-size {config.tp_size}"
    )
    if config.wait_for_source_publication_drain:
        remote_cmd += (
            " --wait-for-source-publication-drain"
            f" --source-publication-drain-timeout-s {config.source_publication_drain_timeout_s}"
            f" --source-publication-drain-idle-s {config.source_publication_drain_idle_s}"
            f" --source-publication-drain-poll-s {config.source_publication_drain_poll_s}"
        )
    driver_log_path = remote_file_log(run_id, "request_driver")
    wrapped_cmd = (
        "set -euo pipefail; "
        f"LOG_PATH={shlex.quote(str(driver_log_path))}; "
        f'{remote_cmd} 2>&1 | tee "$LOG_PATH"'
    )
    exec_user(config, worker_process, wrapped_cmd)
    with paths.summary_json_path.open("r", encoding="utf-8") as file:
        return RunSummary.model_validate(json.load(file))


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def latest_tensorcast_session_dir(runtime_home: Path) -> Path | None:
    hosts_root = runtime_home / "hosts"
    if not hosts_root.exists():
        return None
    session_dirs = sorted(hosts_root.glob("*/sessions/*"))
    return session_dirs[-1] if session_dirs else None


def copy_tensorcast_runtime_stdio_logs(
    *,
    runtime_home: Path,
    logs_dir: Path,
    name: str,
) -> None:
    session_dir = latest_tensorcast_session_dir(runtime_home)
    if session_dir is None:
        return
    copy_if_exists(
        session_dir / "logs" / "daemon.err",
        logs_dir / f"{name}.stderr.log",
    )
    copy_if_exists(
        session_dir / "logs" / "daemon.out",
        logs_dir / f"{name}.stdout.log",
    )
    copy_if_exists(
        session_dir / "session" / "session_state.json",
        logs_dir / f"{name}.session_state.json",
    )


def collect_logs(
    paths,
    run_id: str,
    config: BenchmarkConfig,
    *,
    worker_process: str | None,
) -> None:
    log("Collecting remote logs from shared /data")
    remote_names = [
        "main_keepalive",
        "gpu_snapshot",
        "port_snapshot",
        "sglang_instance_a",
        "sglang_instance_b",
        "request_driver",
    ]
    if config.hicache_storage_backend == "mooncake":
        remote_names.append("mooncake_master")
    else:
        remote_names.append("tensorcast_global_store")
        if config.tensorcast_daemon_mode == "share":
            remote_names.append("tensorcast_daemon_shared")
        else:
            remote_names.extend(["tensorcast_daemon_a", "tensorcast_daemon_b"])
    for name in remote_names:
        src = remote_file_log(run_id, name)
        dst = paths.logs_dir / f"{name}.log"
        copy_if_exists(src, dst)
    if config.hicache_storage_backend != "tensorcast" or not worker_process:
        return
    copy_tensorcast_runtime_stdio_logs(
        runtime_home=tensorcast_runtime_home(paths, worker_process, "global_store"),
        logs_dir=paths.logs_dir,
        name="tensorcast_global_store",
    )
    if config.tensorcast_daemon_mode == "share":
        copy_tensorcast_runtime_stdio_logs(
            runtime_home=tensorcast_runtime_home(
                paths, worker_process, "daemon_shared"
            ),
            logs_dir=paths.logs_dir,
            name="tensorcast_daemon_shared",
        )
        return
    copy_tensorcast_runtime_stdio_logs(
        runtime_home=tensorcast_runtime_home(paths, worker_process, "daemon_a"),
        logs_dir=paths.logs_dir,
        name="tensorcast_daemon_a",
    )
    copy_tensorcast_runtime_stdio_logs(
        runtime_home=tensorcast_runtime_home(paths, worker_process, "daemon_b"),
        logs_dir=paths.logs_dir,
        name="tensorcast_daemon_b",
    )


def parse_args() -> BenchmarkConfig:
    parser = build_parser()
    args = parser.parse_args()
    payload = vars(args)
    if payload["brainctl_charged_group"].strip():
        payload["brainctl_charged_group"] = payload["brainctl_charged_group"].strip()
    data_path = Path(payload["data_path"]).expanduser()
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()
    payload["data_path"] = str(data_path)
    return BenchmarkConfig.model_validate(payload)


def main() -> None:
    config = parse_args()
    benchmark_root = Path(__file__).resolve().parent
    run_id = create_run_id(
        config.hicache_storage_backend,
        config.tp_size,
        config.prompt_count,
    )
    paths = build_paths(benchmark_root, run_id)
    prepare_paths(paths)

    global ORCHESTRATOR_LOG_PATH
    ORCHESTRATOR_LOG_PATH = paths.orchestrator_log_path

    worker_info = None
    launched_worker = False
    mooncake_config_path: Path | None = None
    tensorcast_configs: dict[str, Path] = {}
    log(f"Run directory: {paths.run_dir}")
    log(f"Benchmark backend: {config.hicache_storage_backend}")

    try:
        if config.existing_worker_process.strip():
            worker_process = config.existing_worker_process.strip()
            log(f"Reusing worker: {worker_process}")
        else:
            worker_process = launch_worker(config, run_id, role="main")
            launched_worker = True
            log(f"Launched worker: {worker_process}")
        worker_info = wait_for_worker_running(config, worker_process)
        log(
            f"Worker running: process={worker_info.process_name} host={worker_info.hostname} "
            f"ip={worker_info.ip} node={worker_info.node}"
        )
        run_remote_smoke_checks(config, worker_process, paths=paths, run_id=run_id)
        capture_remote_snapshots(config, worker_process, paths=paths, run_id=run_id)

        if config.hicache_storage_backend == "mooncake":
            mooncake_config_path = write_mooncake_config(paths, config)
            start_mooncake_service(config, worker_process, paths=paths, run_id=run_id)
        else:
            tensorcast_configs = build_tensorcast_configs(
                config,
                paths=paths,
                run_id=run_id,
                worker_ip=worker_info.ip,
            )
            global_store_home = tensorcast_runtime_home(
                paths, worker_process, "global_store"
            )
            start_global_store(
                config,
                worker_process,
                paths=paths,
                global_store_config_path=tensorcast_configs["global"],
                runtime_home=global_store_home,
            )
            global_store_address = (
                f"{worker_info.ip}:{config.tensorcast_global_store_port}"
            )
            if config.tensorcast_daemon_mode == "share":
                start_tensorcast_daemon(
                    config,
                    worker_process,
                    paths=paths,
                    daemon_config_path=tensorcast_configs["daemon_shared"],
                    runtime_home=tensorcast_runtime_home(
                        paths, worker_process, "daemon_shared"
                    ),
                    global_store_address=global_store_address,
                    daemon_port=config.tensorcast_daemon_port_a,
                )
            else:
                start_tensorcast_daemon(
                    config,
                    worker_process,
                    paths=paths,
                    daemon_config_path=tensorcast_configs["daemon_a"],
                    runtime_home=tensorcast_runtime_home(
                        paths, worker_process, "daemon_a"
                    ),
                    global_store_address=global_store_address,
                    daemon_port=config.tensorcast_daemon_port_a,
                )
                start_tensorcast_daemon(
                    config,
                    worker_process,
                    paths=paths,
                    daemon_config_path=tensorcast_configs["daemon_b"],
                    runtime_home=tensorcast_runtime_home(
                        paths, worker_process, "daemon_b"
                    ),
                    global_store_address=global_store_address,
                    daemon_port=config.tensorcast_daemon_port_b,
                )

        start_sglang_instance(
            config,
            worker_process,
            paths=paths,
            run_id=run_id,
            instance_name="instance_a",
            cuda_visible_devices=config.instance_a_cuda_visible_devices,
            port=config.port_a,
            mooncake_config_path=mooncake_config_path,
            tensorcast_backend_extra_config=(
                build_tensorcast_backend_extra_config(
                    config=config,
                    daemon_port=config.tensorcast_daemon_port_a,
                    global_store_address=global_store_address,
                    execution_endpoint=(
                        f"{worker_info.ip}:{config.tensorcast_instance_agent_port_a}"
                    ),
                )
                if config.hicache_storage_backend == "tensorcast"
                else None
            ),
        )
        start_sglang_instance(
            config,
            worker_process,
            paths=paths,
            run_id=run_id,
            instance_name="instance_b",
            cuda_visible_devices=config.instance_b_cuda_visible_devices,
            port=config.port_b,
            mooncake_config_path=mooncake_config_path,
            tensorcast_backend_extra_config=(
                build_tensorcast_backend_extra_config(
                    config=config,
                    daemon_port=(
                        config.tensorcast_daemon_port_a
                        if config.tensorcast_daemon_mode == "share"
                        else config.tensorcast_daemon_port_b
                    ),
                    global_store_address=global_store_address,
                    execution_endpoint=(
                        f"{worker_info.ip}:{config.tensorcast_instance_agent_port_b}"
                    ),
                )
                if config.hicache_storage_backend == "tensorcast"
                else None
            ),
        )

        summary = run_request_driver(
            config,
            worker_process,
            paths=paths,
            run_id=run_id,
            worker_info=worker_info,
        )
        append_csv_row(paths.csv_path, summary.model_dump(mode="json"))
        log(summary.observation)
        if summary.mean_ttft_improvement_ms is None:
            raise RuntimeError("No TTFT improvement samples were recorded")
        if (
            config.require_positive_ttft_improvement
            and summary.mean_ttft_improvement_ms <= 0
        ):
            raise RuntimeError(
                f"Mean TTFT improvement was not positive: {summary.mean_ttft_improvement_ms:.2f} ms"
            )
        direction = (
            "improvement" if summary.mean_ttft_improvement_ms >= 0 else "regression"
        )
        log(
            f"Completed run: mean TTFT {direction} {summary.mean_ttft_improvement_ms:.2f} ms "
            f"over {summary.success_pairs} successful pairs"
        )

    finally:
        if worker_info is not None:
            with contextlib.suppress(Exception):
                stop_sglang_instance(
                    config,
                    worker_info.process_name,
                    paths=paths,
                    instance_name="instance_a",
                    port=config.port_a,
                )
            with contextlib.suppress(Exception):
                stop_sglang_instance(
                    config,
                    worker_info.process_name,
                    paths=paths,
                    instance_name="instance_b",
                    port=config.port_b,
                )
            if config.hicache_storage_backend == "mooncake":
                with contextlib.suppress(Exception):
                    stop_mooncake_service(config, worker_info.process_name, paths=paths)
            else:
                if config.tensorcast_daemon_mode == "share":
                    with contextlib.suppress(Exception):
                        stop_tensorcast_daemon(
                            config,
                            worker_info.process_name,
                            paths=paths,
                            runtime_home=tensorcast_runtime_home(
                                paths, worker_info.process_name, "daemon_shared"
                            ),
                        )
                else:
                    with contextlib.suppress(Exception):
                        stop_tensorcast_daemon(
                            config,
                            worker_info.process_name,
                            paths=paths,
                            runtime_home=tensorcast_runtime_home(
                                paths, worker_info.process_name, "daemon_a"
                            ),
                        )
                    with contextlib.suppress(Exception):
                        stop_tensorcast_daemon(
                            config,
                            worker_info.process_name,
                            paths=paths,
                            runtime_home=tensorcast_runtime_home(
                                paths, worker_info.process_name, "daemon_b"
                            ),
                        )
                with contextlib.suppress(Exception):
                    stop_global_store(
                        config,
                        worker_info.process_name,
                        paths=paths,
                        runtime_home=tensorcast_runtime_home(
                            paths, worker_info.process_name, "global_store"
                        ),
                    )
        collect_logs(
            paths,
            run_id,
            config,
            worker_process=worker_info.process_name
            if worker_info is not None
            else None,
        )
        if worker_info is not None and launched_worker and not config.keep_worker:
            log(f"Cleaning up worker: {worker_info.process_name}")
            stop_delete_worker(config, worker_info.process_name)
        elif worker_info is not None:
            log(f"Keeping worker for debug: {worker_info.process_name}")


if __name__ == "__main__":
    main()
