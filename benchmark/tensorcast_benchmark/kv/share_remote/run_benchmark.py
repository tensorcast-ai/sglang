#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from tensorcast_benchmark.kv.share_remote.models import (
    DriverInstanceTarget,
    ResolvedWorkerSpec,
    ShareRemoteBenchmarkConfig,
    ShareRemoteDriverConfig,
    ShareRemotePaths,
    ShareRemoteRunSummary,
    WorkerDirectoryInfo,
    WorkerInventoryRecord,
    WorkerRDMAInfo,
)
from tensorcast_benchmark.kv.share_remote.outputs import (
    append_csv_row,
    build_paths,
    create_run_id,
    dump_yaml,
    load_yaml,
    prepare_paths,
    worker_log_dir,
    worker_log_path,
    write_json,
)


BRAINCTL_PROXY_ENV_KEYS = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "https_proxy",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)
ORCHESTRATOR_LOG_PATH: Path | None = None
RDMA_HELPER_ROOT = Path("/home/i-zhouyuhan/.codex/skills/brainctl-launch-remote-gpu")
RDMA_DERIVE_SCRIPT = RDMA_HELPER_ROOT / "scripts" / "derive_nccl_ib_hca.py"


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    if ORCHESTRATOR_LOG_PATH is None:
        return
    ORCHESTRATOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ORCHESTRATOR_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the share-remote multi-worker KV benchmark."
    )
    parser.add_argument("--config", required=True)
    return parser


def local_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in BRAINCTL_PROXY_ENV_KEYS:
        env.pop(key, None)
    return env


def run_local(
    cmd: list[str],
    *,
    timeout_s: float | None = None,
    check: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
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
            f"Command failed: {shlex.join(cmd)}\n"
            f"returncode={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def parse_config(config_path: Path) -> ShareRemoteBenchmarkConfig:
    raw_payload = load_yaml(config_path)
    workload_payload = raw_payload.get("workload")
    if isinstance(workload_payload, dict):
        data_path = workload_payload.get("data_path")
        if isinstance(data_path, str) and data_path and not Path(data_path).is_absolute():
            workload_payload["data_path"] = str((config_path.parent / data_path).resolve())
        model_path = workload_payload.get("model_path")
        if isinstance(model_path, str) and model_path and not Path(model_path).is_absolute():
            candidate = (config_path.parent / model_path).resolve()
            if candidate.exists():
                workload_payload["model_path"] = str(candidate)
    return ShareRemoteBenchmarkConfig.model_validate(raw_payload)


def parse_worker_info(process_name: str, output: str) -> WorkerDirectoryInfo:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    data_line = ""
    for line in reversed(lines):
        if not line.startswith("ID ") and not line.startswith("ID\t"):
            data_line = line
            break
    if not data_line:
        raise RuntimeError(f"Could not parse worker info for {process_name}:\n{output}")
    tokens = data_line.split()
    if len(tokens) < 9:
        raise RuntimeError(
            f"Unexpected worker info format for {process_name}:\n{output}"
        )
    return WorkerDirectoryInfo(
        process_name=tokens[0],
        hostname=tokens[1],
        creator=tokens[2],
        ready=tokens[3],
        status=tokens[4],
        ip=tokens[7],
        node=tokens[8],
    )


def get_worker_info(
    config: ShareRemoteBenchmarkConfig,
    process_name: str,
) -> WorkerDirectoryInfo:
    completed = run_local(
        [
            "brainctl",
            "get",
            f"process/{process_name}",
            "-n",
            config.brainctl.namespace,
            "-o",
            "wide",
        ]
    )
    return parse_worker_info(process_name, completed.stdout)


def describe_worker_tail(
    config: ShareRemoteBenchmarkConfig,
    process_name: str,
    lines: int = 40,
) -> str:
    completed = run_local(
        [
            "brainctl",
            "describe",
            f"process/{process_name}",
            "-n",
            config.brainctl.namespace,
        ],
        check=False,
    )
    text = (completed.stdout + completed.stderr).strip().splitlines()
    return "\n".join(text[-lines:])


def wait_for_worker_running(
    config: ShareRemoteBenchmarkConfig,
    process_name: str,
) -> WorkerDirectoryInfo:
    deadline = time.monotonic() + config.brainctl.worker_ready_timeout_s
    while time.monotonic() < deadline:
        info = get_worker_info(config, process_name)
        if info.status == "Running" and info.ready == "1/1":
            return info
        time.sleep(config.brainctl.worker_poll_interval_s)
    tail = describe_worker_tail(config, process_name)
    raise RuntimeError(
        f"Worker did not reach Running within {config.brainctl.worker_ready_timeout_s}s: "
        f"{process_name}\n{tail}"
    )


def launch_worker(
    config: ShareRemoteBenchmarkConfig,
    *,
    run_id: str,
    spec: ResolvedWorkerSpec,
    extra_negative_tags: tuple[str, ...] = (),
) -> str:
    keepalive_log = f"/data/{run_id}_worker_{spec.index:02d}_keepalive.log"
    remote_cmd = (
        "set -euo pipefail; "
        f"LOG_FILE={shlex.quote(keepalive_log)}; "
        'echo KEEPALIVE_START $(date -Is) HOST=$(hostname) USER=$(id -un) | tee -a "$LOG_FILE"; '
        'while true; do echo KEEPALIVE_HEARTBEAT $(date -Is) | tee -a "$LOG_FILE"; sleep 30; done'
    )
    cmd = [
        "brainctl",
        "launch",
        "-d",
        "--i-know-i-am-wasting-resource",
        f"--charged-group={config.brainctl.charged_group}",
        f"--gpu={spec.gpu}",
        f"--cpu={spec.cpu}",
        f"--memory={spec.memory}",
        f"--mount={config.brainctl.mount}",
        f"--private-machine={config.brainctl.private_machine}",
        f"--max-wait-duration={config.brainctl.max_wait_duration}",
        f"--comment={run_id}-worker-{spec.index:02d}",
    ]
    if spec.positive_tags.strip():
        cmd.append(f"--positive-tags={spec.positive_tags}")
    negative_tags = [value for value in (spec.negative_tags.strip(), *extra_negative_tags) if value]
    if negative_tags:
        cmd.append(f"--negative-tags={','.join(negative_tags)}")
    cmd.extend(["--", "bash", "-lc", remote_cmd])
    completed = run_local(cmd)
    process_name = completed.stdout.strip().splitlines()[-1].strip()
    if not process_name:
        raise RuntimeError(
            f"brainctl launch returned empty process name:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return process_name


def stop_delete_worker(config: ShareRemoteBenchmarkConfig, process_name: str) -> None:
    run_local(
        [
            "brainctl",
            "stop",
            f"process/{process_name}",
            "-n",
            config.brainctl.namespace,
        ],
        check=False,
    )
    run_local(
        [
            "brainctl",
            "delete",
            f"process/{process_name}",
            "-n",
            config.brainctl.namespace,
        ],
        check=False,
    )


def exec_root(
    config: ShareRemoteBenchmarkConfig,
    process_name: str,
    remote_cmd: str,
    *,
    timeout_s: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return run_local(
        [
            "brainctl",
            "exec",
            f"process/{process_name}",
            "-n",
            config.brainctl.namespace,
            "--",
            "bash",
            "-lc",
            remote_cmd,
        ],
        timeout_s=timeout_s,
        check=check,
    )


def exec_user(
    config: ShareRemoteBenchmarkConfig,
    process_name: str,
    remote_cmd: str,
    *,
    timeout_s: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    user = os.environ.get("USER", "").strip() or subprocess.check_output(
        ["id", "-un"], text=True
    ).strip()
    wrapped = (
        "set -euo pipefail; "
        f"if ! id -u {user} >/dev/null 2>&1; then echo missing user {user} >&2; exit 1; fi; "
        f"su - {user} -s /bin/bash -c {shlex.quote(remote_cmd)}"
    )
    return exec_root(
        config,
        process_name,
        wrapped,
        timeout_s=timeout_s,
        check=check,
    )


def wait_for_condition(
    *,
    timeout_s: float,
    poll_interval_s: float,
    description: str,
    check_fn,
) -> str:
    deadline = time.monotonic() + timeout_s
    last_detail = ""
    while time.monotonic() < deadline:
        ok, detail = check_fn()
        last_detail = detail
        if ok:
            return detail
        time.sleep(poll_interval_s)
    raise RuntimeError(f"Timed out waiting for {description}:\n{last_detail}")


def build_remote_python_prefix(
    paths: ShareRemotePaths,
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


def build_remote_uv(paths: ShareRemotePaths) -> str:
    return shlex.quote(str(paths.uv_bin))


def worker_pid_path(paths: ShareRemotePaths, worker_index: int, name: str) -> Path:
    return paths.run_dir / "pids" / f"worker_{worker_index:02d}" / f"{name}.pid"


def stop_remote_process(
    config: ShareRemoteBenchmarkConfig,
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
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    worker_index: int,
    name: str,
    command: str,
    log_path: Path,
    pid_path: Path,
    exports: dict[str, str] | None = None,
    include_benchmark: bool = False,
    include_tensorcast: bool = False,
) -> None:
    export_cmd = ""
    for key, value in (exports or {}).items():
        export_cmd += f"export {key}={shlex.quote(value)}; "
    remote_cmd = (
        "set -euo pipefail; "
        f"{build_remote_python_prefix(paths, include_benchmark=include_benchmark, include_tensorcast=include_tensorcast)}"
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
        f"echo STARTED worker={worker_index} name={shlex.quote(name)} PID=$(cat \"$PID_PATH\")"
    )
    exec_user(config, worker_process, remote_cmd)


def wait_remote_health(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    url: str,
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    python_snippet = (
        "import urllib.request; "
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


def run_remote_environment_smoke_checks(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    worker_index: int,
) -> None:
    tensorcast_root = paths.workspace_root / "thirdparty" / "tensorcast"
    smoke_file = worker_log_path(paths, worker_index, "environment_smoke")
    remote_cmd = (
        "set -euo pipefail; "
        f"{build_remote_python_prefix(paths, include_benchmark=False, include_tensorcast=True)}"
        f"test -d {shlex.quote(str(paths.sglang_root))}; "
        f"test -d {shlex.quote(str(tensorcast_root))}; "
        f"test -x {shlex.quote(str(paths.venv_python))}; "
        f"test -x {shlex.quote(str(paths.uv_bin))}; "
        "nvidia-smi -L >/dev/null; "
        f'mkdir -p {shlex.quote(str(smoke_file.parent))}; '
        f"touch {shlex.quote(str(smoke_file))}; "
        f"echo SMOKE_OK > {shlex.quote(str(smoke_file))}; "
        f"test -s {shlex.quote(str(smoke_file))}; "
        f"rm -f {shlex.quote(str(smoke_file))}"
    )
    exec_user(config, worker_process, remote_cmd)


def capture_remote_snapshots(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    worker_index: int,
) -> None:
    gpu_log = worker_log_path(paths, worker_index, "gpu_snapshot")
    ports_log = worker_log_path(paths, worker_index, "port_snapshot")
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


def discover_remote_socket_ifname(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
) -> str:
    remote_cmd = (
        "set -euo pipefail; "
        "iface=$(ip -o -4 route show to default 2>/dev/null | awk '{print $5; exit}'); "
        'if [[ -z "${iface:-}" ]]; then '
        "  iface=$(ip route get 8.8.8.8 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i==\"dev\") {print $(i+1); exit}}'); "
        "fi; "
        'printf "%s" "${iface:-}"'
    )
    completed = exec_user(config, worker_process, remote_cmd)
    socket_ifname = completed.stdout.strip()
    if not socket_ifname:
        raise RuntimeError(f"Failed to discover NCCL_SOCKET_IFNAME on {worker_process}")
    return socket_ifname


def _to_exact_match_hca(raw_value: str) -> str:
    names = [name.strip() for name in raw_value.split(",") if name.strip()]
    return ",".join(f"={name}" for name in names)


def derive_remote_rdma_info(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    socket_ifname: str,
) -> WorkerRDMAInfo:
    remote_cmd = (
        "set -euo pipefail; "
        f"{build_remote_python_prefix(paths, include_benchmark=False, include_tensorcast=False)}"
        f"export NCCL_SOCKET_IFNAME={shlex.quote(socket_ifname)}; "
        f"{shlex.quote(str(paths.venv_python))} "
        f"{shlex.quote(str(RDMA_DERIVE_SCRIPT))} --json"
    )
    completed = exec_user(config, worker_process, remote_cmd)
    payload = json.loads(completed.stdout)
    raw_hca = str(payload.get("nccl_ib_hca", "")).strip()
    exact_hca = _to_exact_match_hca(raw_hca)
    candidates = payload.get("gpu_candidates", [])
    preferred_ib_device = ""
    if candidates:
        preferred_ib_device = str(candidates[0].get("hca_name", "")).strip()
    return WorkerRDMAInfo.model_validate(
        {
            "socket_ifname": str(payload.get("socket_ifname", socket_ifname)).strip()
            or socket_ifname,
            "nccl_ib_hca_raw": raw_hca,
            "nccl_ib_hca_exact": exact_hca,
            "preferred_ib_device": preferred_ib_device,
            "gpu_candidates": candidates,
        }
    )


def capture_remote_rdma_inventory(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    worker_index: int,
) -> None:
    for command_name, command in (
        ("ibv_devices", "ibv_devices"),
        ("ibstat", "ibstat"),
        ("rdma_link", "rdma link"),
    ):
        log_path = worker_log_path(paths, worker_index, command_name)
        remote_cmd = (
            "set -euo pipefail; "
            f"LOG_PATH={shlex.quote(str(log_path))}; "
            'mkdir -p "$(dirname "$LOG_PATH")"; '
            f"{command} > \"$LOG_PATH\" 2>&1"
        )
        exec_user(config, worker_process, remote_cmd, check=False)


def run_ib_write_bw_smoke(
    config: ShareRemoteBenchmarkConfig,
    *,
    paths: ShareRemotePaths,
    client_process: str,
    client_worker_index: int,
    client_rdma: WorkerRDMAInfo,
    server_process: str,
    server_worker_index: int,
    server_ip: str,
    server_rdma: WorkerRDMAInfo,
) -> None:
    if not client_rdma.preferred_ib_device or not server_rdma.preferred_ib_device:
        raise RuntimeError("RDMA smoke requires preferred_ib_device on both workers")
    capture_remote_rdma_inventory(
        config,
        client_process,
        paths=paths,
        worker_index=client_worker_index,
    )
    capture_remote_rdma_inventory(
        config,
        server_process,
        paths=paths,
        worker_index=server_worker_index,
    )
    for smoke_name, extra_args, port in (
        ("fixed", "", config.transport.ib_write_bw_fixed_port),
        ("sweep", "-a", config.transport.ib_write_bw_sweep_port),
    ):
        server_log = worker_log_path(
            paths,
            server_worker_index,
            f"ib_write_bw_{smoke_name}_server_from_worker{client_worker_index:02d}",
        )
        client_log = worker_log_path(
            paths,
            client_worker_index,
            f"ib_write_bw_{smoke_name}_client_to_worker{server_worker_index:02d}",
        )
        server_pid = worker_pid_path(
            paths,
            server_worker_index,
            f"ib_write_bw_{smoke_name}_server_from_worker{client_worker_index:02d}",
        )
        server_command = (
            f"ib_write_bw {extra_args} -d {shlex.quote(server_rdma.preferred_ib_device)} "
            f"-x {config.transport.rdma_gid_index} -F --report_gbits -p {port}"
        ).strip()
        start_remote_process(
            config,
            server_process,
            paths=paths,
            worker_index=server_worker_index,
            name=f"ib_write_bw_{smoke_name}_server",
            command=server_command,
            log_path=server_log,
            pid_path=server_pid,
        )
        time.sleep(config.transport.ib_write_bw_server_ready_sleep_s)
        client_command = (
            "set -euo pipefail; "
            f"LOG_PATH={shlex.quote(str(client_log))}; "
            'mkdir -p "$(dirname "$LOG_PATH")"; '
            f"ib_write_bw {extra_args} -d {shlex.quote(client_rdma.preferred_ib_device)} "
            f"-x {config.transport.rdma_gid_index} -F --report_gbits -p {port} "
            f"{shlex.quote(server_ip)} > \"$LOG_PATH\" 2>&1"
        ).strip()
        try:
            exec_user(
                config,
                client_process,
                client_command,
                timeout_s=120.0,
            )
        finally:
            stop_remote_process(config, server_process, server_pid)


def build_tensorcast_runtime_home(
    paths: ShareRemotePaths,
    worker_process: str,
    name: str,
) -> Path:
    return paths.outputs_dir / "_tensorcast_runtime" / worker_process / name


def build_tensorcast_service_remote_cmd(
    paths: ShareRemotePaths,
    *,
    tensorcast_home: Path,
    subcommand: str,
    args: list[str] | None = None,
) -> str:
    joined_args = " ".join(shlex.quote(arg) for arg in (args or []))
    service_script = paths.scripts_dir / "tensorcast_service.sh"
    return (
        f"cd {shlex.quote(str(paths.workspace_root))}; "
        f"source {shlex.quote(str(paths.workspace_root / '.venv' / 'bin' / 'activate'))}; "
        f"export TENSORCAST_HOME={shlex.quote(str(tensorcast_home))}; "
        f"export UV_BIN={shlex.quote(str(paths.uv_bin))}; "
        f"bash {shlex.quote(str(service_script))} {shlex.quote(subcommand)}"
        + (f" {joined_args}" if joined_args else "")
    )


def prepend_exports(command: str, exports: dict[str, str] | None) -> str:
    if not exports:
        return command
    export_prefix = "".join(
        f"export {key}={shlex.quote(value)}; " for key, value in exports.items()
    )
    return f"{export_prefix}{command}"


def global_is_ready(status_text: str) -> bool:
    return "Global Store session" in status_text and "health : SERVING" in status_text


def daemon_is_stopped(status_text: str) -> bool:
    lowered = status_text.lower()
    return "no local daemon session found" in lowered or "not found" in lowered


def cleanup_remote_tcp_ports(
    config: ShareRemoteBenchmarkConfig,
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
    config: ShareRemoteBenchmarkConfig,
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


def daemon_handle_socket_path(
    paths: ShareRemotePaths,
    *,
    run_id: str,
    worker_index: int,
    port: int,
) -> str:
    digest = hashlib.sha1(
        f"{run_id}:worker{worker_index:02d}:port{port}".encode("utf-8")
    ).hexdigest()[:12]
    return str(paths.workspace_root.parent / ".tcsr" / f"{digest}.sock")


def build_local_daemon_address(port: int) -> str:
    return f"127.0.0.1:{port}"


def build_tensorcast_configs(
    config: ShareRemoteBenchmarkConfig,
    *,
    paths: ShareRemotePaths,
    run_id: str,
    service_host_ip: str,
    worker_ips: tuple[str, ...],
) -> dict[str, Path]:
    global_cfg = load_yaml(paths.benchmark_root / "configs" / "global_store_config.yaml")
    daemon_cfg = load_yaml(paths.benchmark_root / "configs" / "store_daemon_config.yaml")
    tensorcast_cfg = config.backend_config.tensorcast

    global_cfg["server"]["listen"]["port"] = tensorcast_cfg.global_store_port
    global_cfg["server"]["advertise"]["host"] = service_host_ip
    global_cfg["server"]["advertise"]["port"] = tensorcast_cfg.global_store_port
    global_cfg["observability"]["logging"]["file"] = str(
        worker_log_path(paths, config.workers.service_host_worker_index, "tensorcast_global_store")
    )

    generated: dict[str, Path] = {
        "global": paths.generated_configs_dir / "tensorcast_global_store.yaml",
    }
    dump_yaml(generated["global"], global_cfg)
    capability_token_secret = f"tensorcast-benchmark-{run_id}"

    for worker_index, worker_ip in enumerate(worker_ips):
        payload = json.loads(json.dumps(daemon_cfg))
        payload["server"]["listen"]["port"] = tensorcast_cfg.daemon_port
        payload["server"]["advertise"]["host"] = worker_ip
        payload["server"]["p2p_listen"]["host"] = worker_ip
        payload["server"]["p2p_listen"]["port"] = tensorcast_cfg.daemon_p2p_port
        payload["engine"]["memory_tiers"]["stable_bytes"] = tensorcast_cfg.daemon_stable_bytes
        payload["high_availability"]["global_store_endpoints"][0]["host"] = service_host_ip
        payload["high_availability"]["global_store_endpoints"][0]["port"] = (
            tensorcast_cfg.global_store_port
        )
        payload["high_availability"]["heartbeat_interval"] = "10s"
        payload["observability"]["logging"]["file"] = str(
            worker_log_path(paths, worker_index, "tensorcast_daemon")
        )
        payload["lifecycle"]["handle_leases"]["local_handle_socket_path"] = (
            daemon_handle_socket_path(
                paths,
                run_id=run_id,
                worker_index=worker_index,
                port=tensorcast_cfg.daemon_port,
            )
        )
        payload["communicator"]["enable_rdma"] = config.transport.use_rdma
        byte_artifact_routing = payload.setdefault("byte_artifact_routing", {})
        byte_artifact_routing["shard_count"] = tensorcast_cfg.byte_artifact_shard_count
        byte_artifact_routing["lease_ttl"] = (
            f"{tensorcast_cfg.byte_artifact_lease_ttl_s:g}s"
        )
        byte_artifact_routing["keepalive_interval"] = (
            f"{tensorcast_cfg.byte_artifact_keepalive_interval_s:g}s"
        )
        payload_transport = byte_artifact_routing.setdefault("payload_transport", {})
        payload_transport["max_chunk_bytes"] = tensorcast_cfg.payload_max_chunk_bytes
        payload_transport["batch_transport_protocol_version"] = 2
        payload_transport["communicator_source_enabled"] = True
        payload_transport["host_memory_export_enabled"] = True
        payload_transport["max_batch_payload_bytes"] = (
            tensorcast_cfg.max_batch_payload_bytes
        )
        payload_transport.setdefault("max_batch_items", 256)
        capability_tokens = payload.setdefault("capability_tokens", {})
        active_tokens = capability_tokens.setdefault("active", {})
        active_tokens["version"] = int(active_tokens.get("version", 1) or 1)
        if not str(active_tokens.get("secret", "")).strip():
            active_tokens["secret"] = capability_token_secret
        daemon_path = (
            paths.generated_configs_dir / f"tensorcast_daemon_worker_{worker_index:02d}.yaml"
        )
        generated[f"daemon_{worker_index}"] = daemon_path
        dump_yaml(daemon_path, payload)
    return generated


def start_global_store(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    global_store_config_path: Path,
    runtime_home: Path,
) -> None:
    tensorcast_cfg = config.backend_config.tensorcast
    cleanup_remote_tcp_ports(
        config,
        worker_process,
        ports=[tensorcast_cfg.global_store_port],
    )
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
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
            subcommand="reset-runtime-state",
        ),
    )
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
            subcommand="start-global",
            args=[str(global_store_config_path)],
        ),
    )
    wait_for_condition(
        timeout_s=tensorcast_cfg.service_ready_timeout_s,
        poll_interval_s=tensorcast_cfg.service_poll_interval_s,
        description="global store ready",
        check_fn=lambda: (
            global_is_ready(
                status := exec_user(
                    config,
                    worker_process,
                    build_tensorcast_service_remote_cmd(
                        paths,
                        tensorcast_home=runtime_home,
                        subcommand="status-global",
                    ),
                    check=False,
                ).stdout
            ),
            status,
        ),
    )


def stop_global_store(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    runtime_home: Path,
) -> None:
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
            subcommand="stop-global",
        ),
        check=False,
    )


def start_tensorcast_daemon(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    daemon_config_path: Path,
    runtime_home: Path,
    global_store_address: str,
    rdma_env: dict[str, str],
) -> None:
    tensorcast_cfg = config.backend_config.tensorcast
    cleanup_remote_tcp_ports(
        config,
        worker_process,
        ports=[tensorcast_cfg.daemon_port],
    )
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
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
            subcommand="reset-runtime-state",
        ),
    )
    exec_user(
        config,
        worker_process,
        prepend_exports(
            build_tensorcast_service_remote_cmd(
                paths,
                tensorcast_home=runtime_home,
                subcommand="start-daemon",
                args=[
                    str(daemon_config_path),
                    global_store_address,
                    tensorcast_cfg.cuda_home,
                    tensorcast_cfg.nvidia_lib_dirs,
                ],
            ),
            rdma_env,
        ),
    )
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
            subcommand="wait-daemon-ready",
            args=[
                build_local_daemon_address(tensorcast_cfg.daemon_port),
                str(tensorcast_cfg.service_ready_timeout_s),
                str(tensorcast_cfg.service_poll_interval_s),
            ],
        ),
        timeout_s=tensorcast_cfg.service_ready_timeout_s + 30.0,
    )


def stop_tensorcast_daemon(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    runtime_home: Path,
) -> None:
    exec_user(
        config,
        worker_process,
        build_tensorcast_service_remote_cmd(
            paths,
            tensorcast_home=runtime_home,
            subcommand="stop-daemon",
        ),
        check=False,
    )
    with contextlib.suppress(Exception):
        wait_for_condition(
            timeout_s=config.backend_config.tensorcast.service_ready_timeout_s,
            poll_interval_s=config.backend_config.tensorcast.service_poll_interval_s,
            description="daemon stopped",
            check_fn=lambda: (
                daemon_is_stopped(
                    status := exec_user(
                        config,
                        worker_process,
                        build_tensorcast_service_remote_cmd(
                            paths,
                            tensorcast_home=runtime_home,
                            subcommand="status-daemon",
                        ),
                        check=False,
                    ).stdout
                ),
                status,
            ),
        )


def write_mooncake_config(
    config: ShareRemoteBenchmarkConfig,
    *,
    paths: ShareRemotePaths,
    worker_index: int,
    worker_ip: str,
    service_host_ip: str,
    device_name: str,
) -> Path:
    mooncake_cfg = config.backend_config.mooncake
    config_path = paths.generated_configs_dir / f"mooncake_worker_{worker_index:02d}.json"
    write_json(
        config_path,
        {
            "local_hostname": worker_ip,
            "metadata_server": (
                f"http://{service_host_ip}:{mooncake_cfg.http_metadata_server_port}/metadata"
            ),
            "master_server_address": f"{service_host_ip}:{mooncake_cfg.master_port}",
            "protocol": "rdma" if config.transport.use_rdma else "tcp",
            "device_name": device_name,
            "global_segment_size": mooncake_cfg.global_segment_size,
            "local_buffer_size": mooncake_cfg.local_buffer_size,
        },
    )
    return config_path


def start_mooncake_service(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    worker_index: int,
    health_host: str,
    rdma_env: dict[str, str],
) -> None:
    mooncake_cfg = config.backend_config.mooncake
    stop_mooncake_service(
        config,
        worker_process,
        paths=paths,
        worker_index=worker_index,
    )
    command = (
        f"{shlex.quote(str(paths.mooncake_master_bin))} "
        "--enable_http_metadata_server=true "
        f"--http_metadata_server_port={mooncake_cfg.http_metadata_server_port} "
        f"--eviction_high_watermark_ratio={mooncake_cfg.eviction_high_watermark_ratio} "
        f"--port={mooncake_cfg.master_port}"
    )
    start_remote_process(
        config,
        worker_process,
        paths=paths,
        worker_index=worker_index,
        name="mooncake_master",
        command=command,
        log_path=worker_log_path(paths, worker_index, "mooncake_master"),
        pid_path=worker_pid_path(paths, worker_index, "mooncake_master"),
        exports=rdma_env,
    )
    wait_remote_health(
        config,
        worker_process,
        paths=paths,
        url=f"http://{health_host}:{mooncake_cfg.http_metadata_server_port}/health",
        timeout_s=120.0,
        poll_interval_s=1.0,
    )


def stop_mooncake_service(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    worker_index: int,
) -> None:
    mooncake_cfg = config.backend_config.mooncake
    stop_remote_process(
        config,
        worker_process,
        worker_pid_path(paths, worker_index, "mooncake_master"),
    )
    remote_cmd = (
        "set -euo pipefail; "
        "if command -v pkill >/dev/null 2>&1; then "
        "  pkill -f '[m]ooncake_master' >/dev/null 2>&1 || true; "
        "fi; "
        f"for port in {mooncake_cfg.http_metadata_server_port} {mooncake_cfg.master_port}; do "
        "  if command -v fuser >/dev/null 2>&1; then "
        "    fuser -k ${port}/tcp >/dev/null 2>&1 || true; "
        "  fi; "
        "done"
    )
    exec_user(config, worker_process, remote_cmd, check=False)


def build_tensorcast_backend_extra_config(
    *,
    config: ShareRemoteBenchmarkConfig,
) -> dict[str, object]:
    tensorcast_cfg = config.backend_config.tensorcast
    model_id = config.workload.model_name or Path(config.workload.model_path).name
    model_version = hashlib.sha256(
        str(config.workload.model_path).encode("utf-8")
    ).hexdigest()[:16]
    payload: dict[str, object] = {
        "daemon_address": build_local_daemon_address(tensorcast_cfg.daemon_port),
        "namespace": tensorcast_cfg.namespace,
        "engine": "sglang",
        "model_id": model_id,
        "model_version": model_version,
        "policy_profile": "durable",
        "prefetch_threshold": tensorcast_cfg.prefetch_threshold,
    }
    if tensorcast_cfg.host_allocator_enabled:
        payload["host_allocator_enabled"] = True
        payload["host_allocator_region_ttl_ms"] = (
            tensorcast_cfg.host_allocator_region_ttl_ms
        )
        payload["host_allocator_region_name"] = tensorcast_cfg.host_allocator_region_name
    return payload


def build_mooncake_backend_extra_config(
    *,
    config: ShareRemoteBenchmarkConfig,
) -> dict[str, object]:
    return {
        "prefetch_threshold": config.backend_config.mooncake.prefetch_threshold,
    }


def build_sglang_command_for_instance(
    *,
    paths: ShareRemotePaths,
    config: ShareRemoteBenchmarkConfig,
    host: str,
    port: int,
    mooncake_config_path: Path | None,
    mooncake_backend_extra_config: dict[str, object] | None,
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
        host,
        "--port",
        str(port),
        "--model-path",
        shlex.quote(config.workload.model_path),
        "--tp",
        str(config.workload.tp_size),
        "--page-size",
        str(config.workload.page_size),
        "--mem-fraction-static",
        str(config.workload.mem_fraction_static),
    ]
    if config.workload.trust_remote_code:
        cmd.append("--trust-remote-code")
    if config.workload.enable_hierarchical_cache:
        cmd.append("--enable-hierarchical-cache")
        cmd.extend(["--hicache-mem-layout", shlex.quote(config.workload.hicache_mem_layout)])
        cmd.extend(["--hicache-io-backend", shlex.quote(config.workload.hicache_io_backend)])
        cmd.extend(["--hicache-ratio", str(config.workload.hicache_ratio)])
        cmd.extend(["--hicache-size", str(config.workload.hicache_size_gb)])
        cmd.extend(
            [
                "--hicache-storage-prefetch-policy",
                shlex.quote(config.workload.hicache_storage_prefetch_policy),
            ]
        )
    env: dict[str, str] = {}
    if config.backend == "mooncake":
        if mooncake_config_path is None:
            raise RuntimeError("mooncake_config_path is required for Mooncake backend")
        cmd.extend(["--hicache-storage-backend", "mooncake"])
        if mooncake_backend_extra_config is not None:
            cmd.extend(
                [
                    "--hicache-storage-backend-extra-config",
                    shlex.quote(
                        json.dumps(
                            mooncake_backend_extra_config,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    ),
                ]
            )
        env["SGLANG_HICACHE_MOONCAKE_CONFIG_PATH"] = str(mooncake_config_path)
    else:
        if tensorcast_backend_extra_config is None:
            raise RuntimeError(
                "tensorcast_backend_extra_config is required for Tensorcast backend"
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
    if config.workload.extra_server_args.strip():
        cmd.append(config.workload.extra_server_args.strip())
    return " ".join(cmd), env


def launch_sglang_instance(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    worker_index: int,
    host: str,
    mooncake_config_path: Path | None,
    mooncake_backend_extra_config: dict[str, object] | None,
    tensorcast_backend_extra_config: dict[str, object] | None,
    rdma_env: dict[str, str],
) -> None:
    cuda_visible_devices = ",".join(str(device) for device in range(config.workload.tp_size))
    stop_sglang_instance(
        config,
        worker_process,
        paths=paths,
        worker_index=worker_index,
    )
    command, env = build_sglang_command_for_instance(
        paths=paths,
        config=config,
        host=host,
        port=config.workload.instance_port,
        mooncake_config_path=mooncake_config_path,
        mooncake_backend_extra_config=mooncake_backend_extra_config,
        tensorcast_backend_extra_config=tensorcast_backend_extra_config,
    )
    exports = {"CUDA_VISIBLE_DEVICES": cuda_visible_devices, **env, **rdma_env}
    start_remote_process(
        config,
        worker_process,
        paths=paths,
        worker_index=worker_index,
        name="sglang_instance",
        command=command,
        log_path=worker_log_path(paths, worker_index, "sglang_instance"),
        pid_path=worker_pid_path(paths, worker_index, "sglang_instance"),
        exports=exports,
    )


def wait_sglang_instance_ready(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    host: str,
) -> None:
    wait_remote_health(
        config,
        worker_process,
        paths=paths,
        url=f"http://{host}:{config.workload.instance_port}/health",
        timeout_s=config.workload.instance_ready_timeout_s,
        poll_interval_s=config.workload.instance_health_poll_interval_s,
    )


def stop_sglang_instance(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
    worker_index: int,
) -> None:
    stop_remote_process(
        config,
        worker_process,
        worker_pid_path(paths, worker_index, "sglang_instance"),
    )
    remote_cmd = (
        "set -euo pipefail; "
        f"if command -v fuser >/dev/null 2>&1; then "
        f"fuser -k {config.workload.instance_port}/tcp >/dev/null 2>&1 || true; "
        "fi; "
        "PIDS=$(ps -ef | awk "
        f"'/[s]glang\\.launch_server/ && /--port {config.workload.instance_port}/ {{print $2}}'"
        "); "
        'if [[ -n "${PIDS:-}" ]]; then '
        '  kill ${PIDS} >/dev/null 2>&1 || true; '
        "  sleep 2; "
        '  for pid in ${PIDS}; do '
        '    if kill -0 "${pid}" >/dev/null 2>&1; then '
        '      kill -9 "${pid}" >/dev/null 2>&1 || true; '
        "    fi; "
        "  done; "
        "fi"
    )
    exec_user(config, worker_process, remote_cmd, check=False)


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
    log_dir: Path,
    name: str,
) -> None:
    session_dir = latest_tensorcast_session_dir(runtime_home)
    if session_dir is None:
        return
    copy_if_exists(session_dir / "logs" / "daemon.err", log_dir / f"{name}.stderr.log")
    copy_if_exists(session_dir / "logs" / "daemon.out", log_dir / f"{name}.stdout.log")
    copy_if_exists(
        session_dir / "session" / "session_state.json",
        log_dir / f"{name}.session_state.json",
    )


def build_rdma_env(
    *,
    worker_rdma: WorkerRDMAInfo | None,
    master_addr: str,
    use_rdma: bool,
    rdma_gid_index: int,
) -> dict[str, str]:
    if not use_rdma or worker_rdma is None:
        return {}
    return {
        "NCCL_SOCKET_IFNAME": worker_rdma.socket_ifname,
        "NCCL_IB_HCA": worker_rdma.nccl_ib_hca_exact,
        "NCCL_IB_GID_INDEX": str(rdma_gid_index),
        "NCCL_SOCKET_FAMILY": "AF_INET",
        "MASTER_ADDR": master_addr,
    }


def run_request_driver(
    config: ShareRemoteBenchmarkConfig,
    worker_process: str,
    *,
    paths: ShareRemotePaths,
) -> ShareRemoteRunSummary:
    remote_cmd = (
        f"{build_remote_python_prefix(paths, include_benchmark=True, include_tensorcast=False)}"
        f"{build_remote_uv(paths)} run --active --no-project --offline python -m "
        "tensorcast_benchmark.kv.share_remote.request_driver "
        f"--config {shlex.quote(str(paths.driver_config_path))}"
    )
    driver_log_path = worker_log_path(
        paths,
        config.workers.service_host_worker_index,
        "request_driver",
    )
    wrapped_cmd = (
        "set -euo pipefail; "
        f"LOG_PATH={shlex.quote(str(driver_log_path))}; "
        f'{remote_cmd} 2>&1 | tee "$LOG_PATH"'
    )
    exec_user(config, worker_process, wrapped_cmd)
    with paths.summary_json_path.open("r", encoding="utf-8") as file:
        return ShareRemoteRunSummary.model_validate(json.load(file))


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = parse_config(config_path)
    run_id = create_run_id(
        config.backend,
        config.workload.tp_size,
        config.workers.resolved_count(),
        config.workload.prompt_count,
    )
    benchmark_root = Path(__file__).resolve().parent
    paths = build_paths(benchmark_root, run_id)
    prepare_paths(paths)
    shutil.copyfile(config_path, paths.config_copy_path)
    dump_yaml(paths.resolved_config_path, config.model_dump(mode="json"))

    global ORCHESTRATOR_LOG_PATH
    ORCHESTRATOR_LOG_PATH = paths.orchestrator_log_path

    log(f"Run directory: {paths.run_dir}")
    log(f"Backend: {config.backend}")

    worker_specs = config.resolved_worker_specs()
    worker_infos: list[WorkerDirectoryInfo] = []
    worker_processes: list[str] = []
    launched_workers: list[bool] = []
    worker_rdma: list[WorkerRDMAInfo | None] = [None] * len(worker_specs)
    tensorcast_runtime_homes: list[Path | None] = [None] * len(worker_specs)
    global_runtime_home: Path | None = None
    summary: ShareRemoteRunSummary | None = None

    try:
        used_nodes: list[str] = []
        for spec in worker_specs:
            if spec.process_name.strip():
                process_name = spec.process_name.strip()
                launched = False
                log(f"Reusing worker {spec.index}: {process_name}")
            else:
                negative_tags = tuple(
                    f"node/{node}" for node in used_nodes if node.strip()
                )
                process_name = launch_worker(
                    config,
                    run_id=run_id,
                    spec=spec,
                    extra_negative_tags=negative_tags,
                )
                launched = True
                log(f"Launched worker {spec.index}: {process_name}")
            info = wait_for_worker_running(config, process_name)
            run_remote_environment_smoke_checks(
                config,
                process_name,
                paths=paths,
                worker_index=spec.index,
            )
            capture_remote_snapshots(
                config,
                process_name,
                paths=paths,
                worker_index=spec.index,
            )
            worker_processes.append(process_name)
            launched_workers.append(launched)
            worker_infos.append(info)
            used_nodes.append(info.node)
            log(
                f"Worker {spec.index} running: process={info.process_name} "
                f"host={info.hostname} ip={info.ip} node={info.node}"
            )

        if config.workers.require_distinct_nodes:
            nodes = [info.node for info in worker_infos]
            if len(set(nodes)) != len(nodes):
                raise RuntimeError(f"Workers are not on distinct nodes: {nodes}")

        for spec, process_name, _info in zip(
            worker_specs,
            worker_processes,
            worker_infos,
            strict=True,
        ):
            socket_ifname = discover_remote_socket_ifname(config, process_name)
            worker_rdma[spec.index] = derive_remote_rdma_info(
                config,
                process_name,
                paths=paths,
                socket_ifname=socket_ifname,
            )
            log(
                f"Worker {spec.index} RDMA: socket_ifname={worker_rdma[spec.index].socket_ifname} "
                f"nccl_ib_hca={worker_rdma[spec.index].nccl_ib_hca_exact}"
            )

        if config.workers.rdma_required and config.transport.rdma_smoke_mode == "star":
            for target_index in range(1, len(worker_specs)):
                should_smoke = (
                    config.workers.verify_rdma_on_reuse
                    or launched_workers[0]
                    or launched_workers[target_index]
                )
                if not should_smoke:
                    continue
                log(f"Running RDMA smoke: worker0 -> worker{target_index}")
                run_ib_write_bw_smoke(
                    config,
                    paths=paths,
                    client_process=worker_processes[0],
                    client_worker_index=0,
                    client_rdma=worker_rdma[0],
                    server_process=worker_processes[target_index],
                    server_worker_index=target_index,
                    server_ip=worker_infos[target_index].ip,
                    server_rdma=worker_rdma[target_index],
                )

        inventory = [
            WorkerInventoryRecord(
                index=spec.index,
                process_name=info.process_name,
                hostname=info.hostname,
                ip=info.ip,
                node=info.node,
                launched_in_run=launched,
                gpu=spec.gpu,
                cpu=spec.cpu,
                memory=spec.memory,
                positive_tags=spec.positive_tags,
                negative_tags=spec.negative_tags,
                rdma=worker_rdma[spec.index],
            )
            for spec, info, launched in zip(worker_specs, worker_infos, launched_workers, strict=True)
        ]
        write_json(
            paths.worker_inventory_path,
            [record.model_dump(mode="json") for record in inventory],
        )

        service_host_index = config.workers.service_host_worker_index
        service_host_info = worker_infos[service_host_index]
        service_host_process = worker_processes[service_host_index]
        service_host_ip = service_host_info.ip
        worker_ips = tuple(info.ip for info in worker_infos)
        rdma_env_by_worker = {
            index: build_rdma_env(
                worker_rdma=worker_rdma[index],
                master_addr=service_host_ip,
                use_rdma=config.transport.use_rdma,
                rdma_gid_index=config.transport.rdma_gid_index,
            )
            for index in range(len(worker_specs))
        }

        mooncake_config_paths: list[Path | None] = [None] * len(worker_specs)
        mooncake_backend_extra_config: dict[str, object] | None = None
        tensorcast_backend_extra_config: dict[str, object] | None = None
        global_store_address = ""

        if config.backend == "tensorcast":
            generated_tensorcast_configs = build_tensorcast_configs(
                config,
                paths=paths,
                run_id=run_id,
                service_host_ip=service_host_ip,
                worker_ips=worker_ips,
            )
            global_runtime_home = build_tensorcast_runtime_home(
                paths,
                worker_processes[service_host_index],
                "global_store",
            )
            start_global_store(
                config,
                service_host_process,
                paths=paths,
                global_store_config_path=generated_tensorcast_configs["global"],
                runtime_home=global_runtime_home,
            )
            global_store_address = (
                f"{service_host_ip}:{config.backend_config.tensorcast.global_store_port}"
            )
            for index, process_name in enumerate(worker_processes):
                runtime_home = build_tensorcast_runtime_home(
                    paths,
                    process_name,
                    f"daemon_worker_{index:02d}",
                )
                tensorcast_runtime_homes[index] = runtime_home
                start_tensorcast_daemon(
                    config,
                    process_name,
                    paths=paths,
                    daemon_config_path=generated_tensorcast_configs[f"daemon_{index}"],
                    runtime_home=runtime_home,
                    global_store_address=global_store_address,
                    rdma_env=rdma_env_by_worker[index],
                )
            tensorcast_backend_extra_config = build_tensorcast_backend_extra_config(
                config=config
            )
        else:
            mooncake_backend_extra_config = build_mooncake_backend_extra_config(
                config=config
            )
            start_mooncake_service(
                config,
                service_host_process,
                paths=paths,
                worker_index=service_host_index,
                health_host=service_host_ip,
                rdma_env=rdma_env_by_worker[service_host_index],
            )
            for index, info in enumerate(worker_infos):
                explicit_device = config.backend_config.mooncake.device_name.strip()
                if explicit_device:
                    device_name = explicit_device
                elif config.transport.use_rdma and worker_rdma[index] is not None:
                    device_name = worker_rdma[index].preferred_ib_device
                else:
                    device_name = ""
                mooncake_config_paths[index] = write_mooncake_config(
                    config,
                    paths=paths,
                    worker_index=index,
                    worker_ip=info.ip,
                    service_host_ip=service_host_ip,
                    device_name=device_name,
                )

        for index, (process_name, info) in enumerate(
            zip(worker_processes, worker_infos, strict=True)
        ):
            launch_sglang_instance(
                config,
                process_name,
                paths=paths,
                worker_index=index,
                host=info.ip,
                mooncake_config_path=mooncake_config_paths[index],
                mooncake_backend_extra_config=mooncake_backend_extra_config,
                tensorcast_backend_extra_config=tensorcast_backend_extra_config,
                rdma_env=rdma_env_by_worker[index],
            )
        for index, (process_name, info) in enumerate(
            zip(worker_processes, worker_infos, strict=True)
        ):
            wait_sglang_instance_ready(
                config,
                process_name,
                paths=paths,
                host=info.ip,
            )

        driver_config = ShareRemoteDriverConfig(
            run_id=run_id,
            backend=config.backend,
            dataset_path=config.workload.data_path,
            prompt_count=config.workload.prompt_count,
            min_prompt_chars=config.workload.min_prompt_chars,
            max_prompt_chars=config.workload.max_prompt_chars,
            rps=config.workload.rps,
            settle_ms=config.workload.settle_ms,
            max_new_tokens=config.workload.max_new_tokens,
            temperature=config.workload.temperature,
            request_timeout_s=config.workload.request_timeout_s,
            results_json_path=str(paths.results_json_path),
            summary_json_path=str(paths.summary_json_path),
            model_path=config.workload.model_path,
            tp_size=config.workload.tp_size,
            transport_use_rdma=config.transport.use_rdma,
            service_host_worker_index=service_host_index,
            instance_targets=tuple(
                DriverInstanceTarget(
                    index=index,
                    worker_process=process_name,
                    worker_host=info.hostname,
                    worker_ip=info.ip,
                    worker_node=info.node,
                    instance_url=f"http://{info.ip}:{config.workload.instance_port}",
                    instance_log_path=str(worker_log_path(paths, index, "sglang_instance")),
                )
                for index, (process_name, info) in enumerate(
                    zip(worker_processes, worker_infos, strict=True)
                )
            ),
        )
        dump_yaml(paths.driver_config_path, driver_config.model_dump(mode="json"))

        summary = run_request_driver(
            config,
            service_host_process,
            paths=paths,
        )
        append_csv_row(paths.csv_path, summary)

    finally:
        for index, process_name in reversed(list(enumerate(worker_processes))):
            with contextlib.suppress(Exception):
                stop_sglang_instance(
                    config,
                    process_name,
                    paths=paths,
                    worker_index=index,
                )
            if config.backend == "tensorcast" and tensorcast_runtime_homes[index] is not None:
                with contextlib.suppress(Exception):
                    stop_tensorcast_daemon(
                        config,
                        process_name,
                        paths=paths,
                        runtime_home=tensorcast_runtime_homes[index],
                    )
        if config.backend == "tensorcast" and global_runtime_home is not None and worker_processes:
            with contextlib.suppress(Exception):
                stop_global_store(
                    config,
                    worker_processes[config.workers.service_host_worker_index],
                    paths=paths,
                    runtime_home=global_runtime_home,
                )
        if config.backend == "mooncake" and worker_processes:
            with contextlib.suppress(Exception):
                stop_mooncake_service(
                    config,
                    worker_processes[config.workers.service_host_worker_index],
                    paths=paths,
                    worker_index=config.workers.service_host_worker_index,
                )
        if config.backend == "tensorcast":
            for index, process_name in enumerate(worker_processes):
                runtime_home = tensorcast_runtime_homes[index]
                if runtime_home is None:
                    continue
                copy_tensorcast_runtime_stdio_logs(
                    runtime_home=runtime_home,
                    log_dir=worker_log_dir(paths, index),
                    name="tensorcast_daemon",
                )
            if global_runtime_home is not None:
                copy_tensorcast_runtime_stdio_logs(
                    runtime_home=global_runtime_home,
                    log_dir=worker_log_dir(paths, config.workers.service_host_worker_index),
                    name="tensorcast_global_store",
                )
        if not config.workers.keep_workers:
            for process_name, launched in zip(worker_processes, launched_workers, strict=True):
                if not launched:
                    continue
                with contextlib.suppress(Exception):
                    stop_delete_worker(config, process_name)

    if summary is not None:
        log(f"Completed run: {summary.run_id}")
        log(f"Observation: {summary.observation}")


if __name__ == "__main__":
    main()
