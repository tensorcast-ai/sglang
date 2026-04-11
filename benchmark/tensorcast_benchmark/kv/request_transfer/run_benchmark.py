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

from tensorcast_benchmark.kv.outputs import build_paths, create_run_id, prepare_paths
from tensorcast_benchmark.kv.remote import (
    exec_user,
    launch_worker,
    stop_delete_worker,
    wait_for_condition,
    wait_for_worker_running,
)
from tensorcast_benchmark.kv.request_transfer.models import (
    RequestTransferBenchmarkConfig,
    RequestTransferRunSummary,
)
from tensorcast_benchmark.kv.request_transfer.outputs import append_csv_row

ORCHESTRATOR_LOG_PATH: Path | None = None


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    if ORCHESTRATOR_LOG_PATH is None:
        return
    ORCHESTRATOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ORCHESTRATOR_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Tensorcast request-transfer benchmark."
    )
    parser.add_argument("--topology-mode", choices=["local", "remote"], default="local")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--prompt-count", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--min-prompt-chars", type=int, default=0)
    parser.add_argument("--max-prompt-chars", type=int, default=0)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--hicache-mem-layout", default="page_blob_direct")
    parser.add_argument("--hicache-io-backend", default="direct")
    parser.add_argument("--hicache-ratio", type=float, default=2.0)
    parser.add_argument("--hicache-size-gb", type=int, default=0)
    parser.add_argument(
        "--hicache-storage-prefetch-policy",
        choices=["best_effort", "wait_complete", "timeout"],
        default="best_effort",
    )
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--instance-ready-timeout-s", type=float, default=1800.0)
    parser.add_argument("--instance-health-poll-interval-s", type=float, default=1.0)
    parser.add_argument("--plan-deadline-ms", type=int, default=15_000)
    parser.add_argument("--publish-ttl-ms", type=int, default=60_000)
    parser.add_argument(
        "--enable-target-worker-warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--verify-log-timeout-s", type=float, default=15.0)
    parser.add_argument("--verify-log-poll-interval-s", type=float, default=0.25)
    parser.add_argument("--post-target-generate-settle-s", type=float, default=1.0)
    parser.add_argument(
        "--evict-after-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--port-a", type=int, default=34000)
    parser.add_argument("--port-b", type=int, default=34001)
    parser.add_argument("--instance-a-cuda-visible-devices", default="0,1")
    parser.add_argument("--instance-b-cuda-visible-devices", default="2,3")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--extra-server-args", default="--log-level debug")
    parser.add_argument(
        "--data-path",
        default=(
            "/home/i-zhouyuhan/tot/thirdparty/sglang/benchmark/"
            "tensorcast_benchmark/kv/dataset/LongBench/hotpotqa.jsonl"
        ),
    )
    parser.add_argument(
        "--keep-worker",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--existing-worker-process-a", default="")
    parser.add_argument("--existing-worker-process-b", default="")
    parser.add_argument("--brainctl-namespace", default="shai-core")
    parser.add_argument("--brainctl-charged-group", default="")
    parser.add_argument("--brainctl-private-machine", default="group")
    parser.add_argument(
        "--brainctl-mount",
        default=(
            "juicefs+s3://oss.i.shaipower.com/step2-alignment-jfs:"
            "/mnt/step2-alignment-jfs"
        ),
    )
    parser.add_argument("--brainctl-max-wait-duration", default="10m")
    parser.add_argument("--worker-ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--worker-poll-interval-s", type=float, default=5.0)
    parser.add_argument("--worker-gpu", type=int, default=4)
    parser.add_argument("--worker-cpu", type=int, default=64)
    parser.add_argument("--worker-memory", type=int, default=500_000)
    parser.add_argument("--worker-positive-tags", default="H800")
    parser.add_argument("--worker-negative-tags", default="")
    parser.add_argument("--tensorcast-namespace", default="request_transfer")
    parser.add_argument("--tensorcast-global-store-port", type=int, default=50051)
    parser.add_argument("--tensorcast-daemon-port-a", type=int, default=50052)
    parser.add_argument("--tensorcast-daemon-port-b", type=int, default=50053)
    parser.add_argument("--tensorcast-instance-agent-port-a", type=int, default=34110)
    parser.add_argument("--tensorcast-instance-agent-port-b", type=int, default=34111)
    parser.add_argument("--tensorcast-daemon-p2p-port-a", type=int, default=65090)
    parser.add_argument("--tensorcast-daemon-p2p-port-b", type=int, default=65091)
    parser.add_argument("--tensorcast-source-prefetch-threshold", type=int, default=1)
    parser.add_argument(
        "--tensorcast-target-prefetch-threshold", type=int, default=1_000_000
    )
    parser.add_argument(
        "--tensorcast-service-ready-timeout-s", type=float, default=120.0
    )
    parser.add_argument("--tensorcast-service-poll-interval-s", type=float, default=2.0)
    parser.add_argument("--tensorcast-cuda-home", default="/usr/local/cuda-12.4")
    parser.add_argument(
        "--tensorcast-nvidia-lib-dirs",
        default="/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64",
    )
    parser.add_argument("--tensorcast-daemon-stable-bytes", default="64GB")
    parser.add_argument("--tensorcast-byte-artifact-shard-count", type=int, default=8)
    parser.add_argument(
        "--tensorcast-byte-artifact-lease-ttl-s", type=float, default=30.0
    )
    parser.add_argument(
        "--tensorcast-byte-artifact-keepalive-interval-s", type=float, default=10.0
    )
    parser.add_argument(
        "--tensorcast-payload-max-chunk-bytes", type=int, default=(1 << 20)
    )
    parser.add_argument(
        "--tensorcast-max-batch-payload-bytes", type=int, default=(16 << 20)
    )
    parser.add_argument(
        "--tensorcast-host-allocator-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--tensorcast-host-allocator-region-ttl-ms", type=int, default=0
    )
    parser.add_argument(
        "--tensorcast-host-allocator-region-name",
        default="sglang_tensorcast_host_pool",
    )
    return parser


def parse_args() -> RequestTransferBenchmarkConfig:
    args = build_parser().parse_args()
    payload = vars(args)
    for key in ("data_path", "model_path"):
        value = Path(payload[key]).expanduser()
        if not value.is_absolute():
            value = (Path.cwd() / value).resolve()
        payload[key] = str(value)
    if payload["brainctl_charged_group"].strip():
        payload["brainctl_charged_group"] = payload["brainctl_charged_group"].strip()
    return RequestTransferBenchmarkConfig.model_validate(payload)


def shared_log_path(paths, name: str) -> Path:
    return paths.logs_dir / f"{name}.log"


def shared_pid_path(paths, name: str) -> Path:
    return paths.run_dir / "pids" / f"{name}.pid"


def build_remote_python_prefix(
    paths,
    *,
    include_benchmark: bool,
    include_tensorcast: bool,
) -> str:
    python_paths = [paths.sglang_root / "python"]
    if include_benchmark:
        python_paths.insert(0, paths.sglang_root / "benchmark")
    if include_tensorcast:
        python_paths.append(paths.workspace_root / "thirdparty" / "tensorcast")
    python_path = ":".join(str(path) for path in python_paths)
    return (
        f"cd {shlex.quote(str(paths.sglang_root))}; "
        f"source {shlex.quote(str(paths.workspace_root / '.venv' / 'bin' / 'activate'))}; "
        f"export PYTHONPATH={shlex.quote(python_path)}:${{PYTHONPATH:-}}; "
        f"export PATH={shlex.quote(str(paths.uv_bin.parent))}:$PATH; "
    )


def build_remote_uv(paths) -> str:
    return shlex.quote(str(paths.uv_bin))


def stop_remote_process(
    config: RequestTransferBenchmarkConfig,
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
        '  if kill -0 "$PID" >/dev/null 2>&1; then '
        '    kill -9 "$PID" >/dev/null 2>&1 || true; '
        "  fi; "
        "fi; "
        'rm -f "$PID_PATH"'
    )
    exec_user(config, worker_process, remote_cmd, check=False)


def start_remote_process(
    config: RequestTransferBenchmarkConfig,
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
        f"{build_remote_python_prefix(paths, include_benchmark=False, include_tensorcast=False)}"
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
    config: RequestTransferBenchmarkConfig,
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
            f"{build_remote_python_prefix(paths, include_benchmark=False, include_tensorcast=False)}"
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


def run_remote_smoke_checks(
    config: RequestTransferBenchmarkConfig,
    worker_process: str,
    *,
    paths,
    role: str,
) -> None:
    tensorcast_root = paths.workspace_root / "thirdparty" / "tensorcast"
    smoke_file = paths.logs_dir / f"{role}_smoke_check.txt"
    remote_cmd = (
        "set -euo pipefail; "
        f"{build_remote_python_prefix(paths, include_benchmark=False, include_tensorcast=True)}"
        f"test -d {shlex.quote(str(paths.sglang_root))}; "
        f"test -d {shlex.quote(str(tensorcast_root))}; "
        f"test -x {shlex.quote(str(paths.venv_python))}; "
        f"test -x {shlex.quote(str(paths.uv_bin))}; "
        "nvidia-smi -L >/dev/null; "
        f"touch {shlex.quote(str(smoke_file))}; "
        f"echo SMOKE_OK > {shlex.quote(str(smoke_file))}; "
        f"test -s {shlex.quote(str(smoke_file))}; "
        f"rm -f {shlex.quote(str(smoke_file))}"
    )
    exec_user(config, worker_process, remote_cmd)


def capture_remote_snapshots(
    config: RequestTransferBenchmarkConfig,
    worker_process: str,
    *,
    paths,
    role: str,
) -> None:
    gpu_log = shared_log_path(paths, f"{role}_gpu_snapshot")
    ports_log = shared_log_path(paths, f"{role}_port_snapshot")
    gpu_cmd = (
        "set -euo pipefail; "
        f"LOG_PATH={shlex.quote(str(gpu_log))}; "
        'mkdir -p "$(dirname "$LOG_PATH")"; '
        'nvidia-smi > "$LOG_PATH" 2>&1'
    )
    ports_cmd = (
        "set -euo pipefail; "
        f"LOG_PATH={shlex.quote(str(ports_log))}; "
        'mkdir -p "$(dirname "$LOG_PATH")"; '
        'ss -ltnp > "$LOG_PATH" 2>&1 || netstat -ltnp > "$LOG_PATH" 2>&1 || true'
    )
    exec_user(config, worker_process, gpu_cmd, check=False)
    exec_user(config, worker_process, ports_cmd, check=False)


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


def cleanup_remote_tcp_ports(
    config: RequestTransferBenchmarkConfig,
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
    config: RequestTransferBenchmarkConfig,
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


def build_local_daemon_address(port: int) -> str:
    return f"127.0.0.1:{port}"


def build_daemon_address(host: str, port: int) -> str:
    return f"{host}:{port}"


def build_daemon_client_address(port: int) -> str:
    return build_local_daemon_address(port)


def daemon_handle_socket_path(paths, port: int) -> str:
    return str(paths.workspace_root.parent / ".tensorcast-kv-sockets" / f"{port}.sock")


def start_global_store(
    config: RequestTransferBenchmarkConfig,
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
    config: RequestTransferBenchmarkConfig,
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
    config: RequestTransferBenchmarkConfig,
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
    config: RequestTransferBenchmarkConfig,
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
    config: RequestTransferBenchmarkConfig,
    *,
    paths,
    run_id: str,
    global_store_host: str,
    daemon_a_host: str,
    daemon_b_host: str,
) -> dict[str, Path]:
    global_cfg = load_yaml(
        paths.benchmark_root / "configs" / "global_store_config.yaml"
    )
    daemon_cfg = load_yaml(
        paths.benchmark_root / "configs" / "store_daemon_config.yaml"
    )
    global_cfg["server"]["listen"]["port"] = config.tensorcast_global_store_port
    global_cfg["server"]["advertise"]["host"] = global_store_host
    global_cfg["server"]["advertise"]["port"] = config.tensorcast_global_store_port
    global_cfg["observability"]["logging"]["file"] = str(
        shared_log_path(paths, "tensorcast_global_store")
    )
    generated = {
        "global": paths.generated_configs_dir / "tensorcast_global_store.yaml",
        "daemon_a": paths.generated_configs_dir / "tensorcast_daemon_a.yaml",
        "daemon_b": paths.generated_configs_dir / "tensorcast_daemon_b.yaml",
    }
    dump_yaml(generated["global"], global_cfg)
    capability_token_secret = f"tensorcast-benchmark-{run_id}"

    def build_daemon_config(
        *,
        advertise_host: str,
        daemon_port: int,
        p2p_port: int,
        log_name: str,
    ) -> dict:
        payload = json.loads(json.dumps(daemon_cfg))
        payload["server"]["listen"]["port"] = daemon_port
        payload["server"]["advertise"]["host"] = advertise_host
        payload["server"]["p2p_listen"]["port"] = p2p_port
        payload["engine"]["memory_tiers"]["stable_bytes"] = (
            config.tensorcast_daemon_stable_bytes
        )
        payload["high_availability"]["global_store_endpoints"][0]["host"] = (
            global_store_host
        )
        payload["high_availability"]["global_store_endpoints"][0]["port"] = (
            config.tensorcast_global_store_port
        )
        payload["high_availability"]["heartbeat_interval"] = "10s"
        payload["observability"]["logging"]["file"] = str(
            shared_log_path(paths, log_name)
        )
        payload["lifecycle"]["handle_leases"]["local_handle_socket_path"] = (
            daemon_handle_socket_path(paths, daemon_port)
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
        capability_directory = payload.setdefault("capability_directory", {})
        capability_directory["enabled"] = True
        capability_directory["gateway_ingress_enabled"] = True
        return payload

    dump_yaml(
        generated["daemon_a"],
        build_daemon_config(
            advertise_host=daemon_a_host,
            daemon_port=config.tensorcast_daemon_port_a,
            p2p_port=config.tensorcast_daemon_p2p_port_a,
            log_name="tensorcast_daemon_a",
        ),
    )
    dump_yaml(
        generated["daemon_b"],
        build_daemon_config(
            advertise_host=daemon_b_host,
            daemon_port=config.tensorcast_daemon_port_b,
            p2p_port=config.tensorcast_daemon_p2p_port_b,
            log_name="tensorcast_daemon_b",
        ),
    )
    return generated


def build_tensorcast_backend_extra_config(
    *,
    config: RequestTransferBenchmarkConfig,
    daemon_address: str,
    global_store_address: str,
    execution_endpoint: str,
    prefetch_threshold: int,
) -> dict[str, object]:
    model_id = config.model_name or Path(config.model_path).name
    model_version = hashlib.sha256(str(config.model_path).encode("utf-8")).hexdigest()[
        :16
    ]
    payload: dict[str, object] = {
        "daemon_address": daemon_address,
        "namespace": config.tensorcast_namespace,
        "engine": "sglang",
        "model_id": model_id,
        "model_version": model_version,
        "policy_profile": "durable",
        "prefetch_threshold": prefetch_threshold,
        "instance_directory_address": global_store_address,
        "instance_agent_execution_endpoint": execution_endpoint,
    }
    if config.tensorcast_host_allocator_enabled:
        payload["host_allocator_enabled"] = True
        payload["host_allocator_region_ttl_ms"] = (
            config.tensorcast_host_allocator_region_ttl_ms
        )
        payload["host_allocator_region_name"] = (
            config.tensorcast_host_allocator_region_name
        )
    return payload


def build_sglang_command_for_instance(
    *,
    paths,
    config: RequestTransferBenchmarkConfig,
    host: str,
    port: int,
    tensorcast_backend_extra_config: dict[str, object],
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
        host,
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
    return " ".join(cmd), {}


def start_sglang_instance(
    config: RequestTransferBenchmarkConfig,
    worker_process: str,
    *,
    paths,
    instance_name: str,
    host: str,
    cuda_visible_devices: str,
    port: int,
    tensorcast_backend_extra_config: dict[str, object],
) -> None:
    log(
        f"Starting SGLang instance {instance_name} on {host}:{port} "
        f"with CUDA_VISIBLE_DEVICES={cuda_visible_devices}"
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
        host=host,
        port=port,
        tensorcast_backend_extra_config=tensorcast_backend_extra_config,
    )
    env = {"CUDA_VISIBLE_DEVICES": cuda_visible_devices, **env}
    start_remote_process(
        config,
        worker_process,
        paths=paths,
        name=f"sglang_{instance_name}",
        command=command,
        log_path=shared_log_path(paths, f"sglang_{instance_name}"),
        pid_path=shared_pid_path(paths, f"sglang_{instance_name}"),
        exports=env,
    )
    wait_remote_health(
        config,
        worker_process,
        paths=paths,
        url=f"http://{host}:{port}/health",
        timeout_s=config.instance_ready_timeout_s,
        poll_interval_s=config.instance_health_poll_interval_s,
    )


def stop_sglang_instance(
    config: RequestTransferBenchmarkConfig,
    worker_process: str,
    *,
    paths,
    instance_name: str,
    port: int | None = None,
) -> None:
    stop_remote_process(
        config,
        worker_process,
        shared_pid_path(paths, f"sglang_{instance_name}"),
    )
    if port is None:
        return
    remote_cmd = (
        "set -euo pipefail; "
        f"if command -v fuser >/dev/null 2>&1; then fuser -k {port}/tcp >/dev/null 2>&1 || true; fi"
    )
    exec_user(config, worker_process, remote_cmd, check=False)


def run_caller_driver(
    config: RequestTransferBenchmarkConfig,
    worker_process: str,
    *,
    paths,
    results_json_path: Path,
    summary_json_path: Path,
    gateway_daemon_address: str,
    source_instance_id: str,
    target_instance_id: str,
    source_instance_url: str,
    target_instance_url: str,
    worker_process_a: str,
    worker_process_b: str,
    worker_host_a: str,
    worker_host_b: str,
    worker_ip_a: str,
    worker_ip_b: str,
    worker_node_a: str,
    worker_node_b: str,
) -> RequestTransferRunSummary:
    log("Running request-transfer caller driver")
    remote_cmd = (
        f"{build_remote_python_prefix(paths, include_benchmark=True, include_tensorcast=True)}"
        f"{build_remote_uv(paths)} run --active --no-project --offline python -m "
        "tensorcast_benchmark.kv.request_transfer.caller_driver "
        f"--run-id {shlex.quote(summary_json_path.parent.name)} "
        f"--topology-mode {shlex.quote(config.topology_mode)} "
        f"--gateway-daemon-address {shlex.quote(gateway_daemon_address)} "
        f"--source-instance-id {shlex.quote(source_instance_id)} "
        f"--target-instance-id {shlex.quote(target_instance_id)} "
        f"--source-instance-url {shlex.quote(source_instance_url)} "
        f"--target-instance-url {shlex.quote(target_instance_url)} "
        f"--target-instance-log-path {shlex.quote(str(shared_log_path(paths, 'sglang_instance_b')))} "
        f"--dataset-path {shlex.quote(config.data_path)} "
        f"--prompt-count {config.prompt_count} "
        f"--min-prompt-chars {config.min_prompt_chars} "
        f"--max-prompt-chars {config.max_prompt_chars} "
        f"--max-new-tokens {config.max_new_tokens} "
        f"--temperature {config.temperature} "
        f"--request-timeout-s {config.request_timeout_s} "
        f"--plan-deadline-ms {config.plan_deadline_ms} "
        f"--publish-ttl-ms {config.publish_ttl_ms} "
        f"--verify-log-timeout-s {config.verify_log_timeout_s} "
        f"--verify-log-poll-interval-s {config.verify_log_poll_interval_s} "
        f"--post-target-generate-settle-s {config.post_target_generate_settle_s} "
        f"--results-json-path {shlex.quote(str(results_json_path))} "
        f"--summary-json-path {shlex.quote(str(summary_json_path))} "
        f"--worker-process-a {shlex.quote(worker_process_a)} "
        f"--worker-process-b {shlex.quote(worker_process_b)} "
        f"--worker-host-a {shlex.quote(worker_host_a)} "
        f"--worker-host-b {shlex.quote(worker_host_b)} "
        f"--worker-ip-a {shlex.quote(worker_ip_a)} "
        f"--worker-ip-b {shlex.quote(worker_ip_b)} "
        f"--worker-node-a {shlex.quote(worker_node_a)} "
        f"--worker-node-b {shlex.quote(worker_node_b)} "
        f"--model-path {shlex.quote(config.model_path)}"
    )
    if config.enable_target_worker_warmup:
        remote_cmd += " --enable-target-worker-warmup"
    if config.evict_after_prompt:
        remote_cmd += " --evict-after-prompt"
    else:
        remote_cmd += " --no-evict-after-prompt"
    driver_log_path = shared_log_path(paths, "caller_driver")
    wrapped_cmd = (
        "set -euo pipefail; "
        f"LOG_PATH={shlex.quote(str(driver_log_path))}; "
        f'{remote_cmd} 2>&1 | tee "$LOG_PATH"'
    )
    exec_user(config, worker_process, wrapped_cmd)
    with summary_json_path.open("r", encoding="utf-8") as file:
        return RequestTransferRunSummary.model_validate(json.load(file))


def latest_tensorcast_session_dir(runtime_home: Path) -> Path | None:
    hosts_root = runtime_home / "hosts"
    if not hosts_root.exists():
        return None
    session_dirs = sorted(hosts_root.glob("*/sessions/*"))
    return session_dirs[-1] if session_dirs else None


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


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


def collect_runtime_logs(
    paths,
    *,
    worker_process_a: str,
    worker_process_b: str,
    topology_mode: str,
) -> None:
    copy_tensorcast_runtime_stdio_logs(
        runtime_home=tensorcast_runtime_home(paths, worker_process_a, "global_store"),
        logs_dir=paths.logs_dir,
        name="tensorcast_global_store",
    )
    copy_tensorcast_runtime_stdio_logs(
        runtime_home=tensorcast_runtime_home(paths, worker_process_a, "daemon_a"),
        logs_dir=paths.logs_dir,
        name="tensorcast_daemon_a",
    )
    daemon_b_worker = worker_process_a if topology_mode == "local" else worker_process_b
    copy_tensorcast_runtime_stdio_logs(
        runtime_home=tensorcast_runtime_home(paths, daemon_b_worker, "daemon_b"),
        logs_dir=paths.logs_dir,
        name="tensorcast_daemon_b",
    )


def launch_worker_for_role(
    config: RequestTransferBenchmarkConfig,
    *,
    run_id: str,
    role: str,
    extra_negative_tag: str = "",
) -> str:
    if not extra_negative_tag:
        return launch_worker(config, run_id, role=role)
    negative_tags = ",".join(
        value
        for value in (config.worker_negative_tags.strip(), extra_negative_tag.strip())
        if value
    )
    launch_config = config.model_copy(update={"worker_negative_tags": negative_tags})
    return launch_worker(launch_config, run_id, role=role)


def main() -> None:
    config = parse_args()
    benchmark_root = Path(__file__).resolve().parent
    run_id = create_run_id("request_transfer", config.tp_size, config.prompt_count)
    paths = build_paths(benchmark_root, run_id)
    prepare_paths(paths)
    results_json_path = paths.run_dir / "prompt_results.jsonl"
    summary_json_path = paths.summary_json_path

    global ORCHESTRATOR_LOG_PATH
    ORCHESTRATOR_LOG_PATH = paths.orchestrator_log_path

    worker_info_a = None
    worker_info_b = None
    launched_worker_a = False
    launched_worker_b = False
    log(f"Run directory: {paths.run_dir}")
    log(f"Topology mode: {config.topology_mode}")

    try:
        if config.existing_worker_process_a.strip():
            worker_process_a = config.existing_worker_process_a.strip()
            log(f"Reusing worker A: {worker_process_a}")
        else:
            worker_process_a = launch_worker_for_role(
                config,
                run_id=run_id,
                role="worker-a",
            )
            launched_worker_a = True
            log(f"Launched worker A: {worker_process_a}")
        worker_info_a = wait_for_worker_running(config, worker_process_a)
        log(
            f"Worker A running: process={worker_info_a.process_name} "
            f"host={worker_info_a.hostname} ip={worker_info_a.ip} node={worker_info_a.node}"
        )
        run_remote_smoke_checks(config, worker_process_a, paths=paths, role="worker_a")
        capture_remote_snapshots(config, worker_process_a, paths=paths, role="worker_a")

        if config.topology_mode == "local":
            worker_info_b = worker_info_a
            worker_process_b = worker_process_a
        else:
            if config.existing_worker_process_b.strip():
                worker_process_b = config.existing_worker_process_b.strip()
                log(f"Reusing worker B: {worker_process_b}")
            else:
                negative_node_tag = (
                    f"node/{worker_info_a.node}" if worker_info_a.node.strip() else ""
                )
                worker_process_b = launch_worker_for_role(
                    config,
                    run_id=run_id,
                    role="worker-b",
                    extra_negative_tag=negative_node_tag,
                )
                launched_worker_b = True
                log(f"Launched worker B: {worker_process_b}")
            worker_info_b = wait_for_worker_running(config, worker_process_b)
            log(
                f"Worker B running: process={worker_info_b.process_name} "
                f"host={worker_info_b.hostname} ip={worker_info_b.ip} node={worker_info_b.node}"
            )
            run_remote_smoke_checks(
                config,
                worker_process_b,
                paths=paths,
                role="worker_b",
            )
            capture_remote_snapshots(
                config,
                worker_process_b,
                paths=paths,
                role="worker_b",
            )

        tensorcast_configs = build_tensorcast_configs(
            config,
            paths=paths,
            run_id=run_id,
            global_store_host=worker_info_a.ip,
            daemon_a_host=worker_info_a.ip,
            daemon_b_host=worker_info_b.ip,
        )
        start_global_store(
            config,
            worker_info_a.process_name,
            paths=paths,
            global_store_config_path=tensorcast_configs["global"],
            runtime_home=tensorcast_runtime_home(
                paths, worker_info_a.process_name, "global_store"
            ),
        )
        global_store_address = build_daemon_address(
            worker_info_a.ip,
            config.tensorcast_global_store_port,
        )
        start_tensorcast_daemon(
            config,
            worker_info_a.process_name,
            paths=paths,
            daemon_config_path=tensorcast_configs["daemon_a"],
            runtime_home=tensorcast_runtime_home(
                paths, worker_info_a.process_name, "daemon_a"
            ),
            global_store_address=global_store_address,
            daemon_port=config.tensorcast_daemon_port_a,
        )
        daemon_b_worker_process = (
            worker_info_a.process_name
            if config.topology_mode == "local"
            else worker_info_b.process_name
        )
        start_tensorcast_daemon(
            config,
            daemon_b_worker_process,
            paths=paths,
            daemon_config_path=tensorcast_configs["daemon_b"],
            runtime_home=tensorcast_runtime_home(
                paths, daemon_b_worker_process, "daemon_b"
            ),
            global_store_address=global_store_address,
            daemon_port=config.tensorcast_daemon_port_b,
        )

        start_sglang_instance(
            config,
            worker_info_a.process_name,
            paths=paths,
            instance_name="instance_a",
            host=worker_info_a.ip,
            cuda_visible_devices=config.instance_a_cuda_visible_devices,
            port=config.port_a,
            tensorcast_backend_extra_config=build_tensorcast_backend_extra_config(
                config=config,
                daemon_address=build_daemon_client_address(
                    config.tensorcast_daemon_port_a
                ),
                global_store_address=global_store_address,
                execution_endpoint=build_daemon_address(
                    worker_info_a.ip,
                    config.tensorcast_instance_agent_port_a,
                ),
                prefetch_threshold=config.tensorcast_source_prefetch_threshold,
            ),
        )
        start_sglang_instance(
            config,
            worker_info_b.process_name,
            paths=paths,
            instance_name="instance_b",
            host=worker_info_b.ip,
            cuda_visible_devices=config.instance_b_cuda_visible_devices,
            port=config.port_b,
            tensorcast_backend_extra_config=build_tensorcast_backend_extra_config(
                config=config,
                daemon_address=build_daemon_client_address(
                    config.tensorcast_daemon_port_b
                ),
                global_store_address=global_store_address,
                execution_endpoint=build_daemon_address(
                    worker_info_b.ip,
                    config.tensorcast_instance_agent_port_b,
                ),
                prefetch_threshold=config.tensorcast_target_prefetch_threshold,
            ),
        )

        summary = run_caller_driver(
            config,
            worker_info_a.process_name,
            paths=paths,
            results_json_path=results_json_path,
            summary_json_path=summary_json_path,
            gateway_daemon_address=build_daemon_client_address(
                config.tensorcast_daemon_port_a
            ),
            source_instance_id=build_daemon_address(worker_info_a.ip, config.port_a),
            target_instance_id=build_daemon_address(worker_info_b.ip, config.port_b),
            source_instance_url=f"http://{worker_info_a.ip}:{config.port_a}",
            target_instance_url=f"http://{worker_info_b.ip}:{config.port_b}",
            worker_process_a=worker_info_a.process_name,
            worker_process_b=worker_info_b.process_name,
            worker_host_a=worker_info_a.hostname,
            worker_host_b=worker_info_b.hostname,
            worker_ip_a=worker_info_a.ip,
            worker_ip_b=worker_info_b.ip,
            worker_node_a=worker_info_a.node,
            worker_node_b=worker_info_b.node,
        )
        append_csv_row(paths.csv_path, summary.model_dump(mode="json"))
        log(summary.observation)
    finally:
        if worker_info_a is not None:
            with contextlib.suppress(Exception):
                stop_sglang_instance(
                    config,
                    worker_info_a.process_name,
                    paths=paths,
                    instance_name="instance_a",
                    port=config.port_a,
                )
        if worker_info_b is not None:
            with contextlib.suppress(Exception):
                stop_sglang_instance(
                    config,
                    worker_info_b.process_name,
                    paths=paths,
                    instance_name="instance_b",
                    port=config.port_b,
                )
        if worker_info_a is not None:
            with contextlib.suppress(Exception):
                stop_tensorcast_daemon(
                    config,
                    worker_info_a.process_name,
                    paths=paths,
                    runtime_home=tensorcast_runtime_home(
                        paths, worker_info_a.process_name, "daemon_a"
                    ),
                )
        if worker_info_b is not None:
            daemon_b_worker_process = (
                worker_info_a.process_name
                if config.topology_mode == "local" and worker_info_a is not None
                else worker_info_b.process_name
            )
            with contextlib.suppress(Exception):
                stop_tensorcast_daemon(
                    config,
                    daemon_b_worker_process,
                    paths=paths,
                    runtime_home=tensorcast_runtime_home(
                        paths, daemon_b_worker_process, "daemon_b"
                    ),
                )
        if worker_info_a is not None:
            with contextlib.suppress(Exception):
                stop_global_store(
                    config,
                    worker_info_a.process_name,
                    paths=paths,
                    runtime_home=tensorcast_runtime_home(
                        paths, worker_info_a.process_name, "global_store"
                    ),
                )
        if worker_info_a is not None and worker_info_b is not None:
            collect_runtime_logs(
                paths,
                worker_process_a=worker_info_a.process_name,
                worker_process_b=worker_info_b.process_name,
                topology_mode=config.topology_mode,
            )
        if worker_info_b is not None and launched_worker_b and not config.keep_worker:
            log(f"Cleaning up worker B: {worker_info_b.process_name}")
            stop_delete_worker(config, worker_info_b.process_name)
        elif worker_info_b is not None and launched_worker_b:
            log(f"Keeping worker B for debug: {worker_info_b.process_name}")
        if worker_info_a is not None and launched_worker_a and not config.keep_worker:
            log(f"Cleaning up worker A: {worker_info_a.process_name}")
            stop_delete_worker(config, worker_info_a.process_name)
        elif worker_info_a is not None and launched_worker_a:
            log(f"Keeping worker A for debug: {worker_info_a.process_name}")


if __name__ == "__main__":
    main()
