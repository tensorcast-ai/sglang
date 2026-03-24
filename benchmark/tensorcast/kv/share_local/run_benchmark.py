#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import shlex
import shutil
from datetime import datetime
from pathlib import Path

from tensorcast.kv.models import BenchmarkConfig, RunSummary
from tensorcast.kv.outputs import (
    append_csv_row,
    build_paths,
    create_run_id,
    prepare_paths,
    write_json,
)
from tensorcast.kv.remote import (
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
    parser.add_argument("--settle-ms", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--min-prompt-chars", type=int, default=0)
    parser.add_argument("--max-prompt-chars", type=int, default=0)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--hicache-mem-layout", default="page_first")
    parser.add_argument("--hicache-ratio", type=float, default=2.0)
    parser.add_argument("--hicache-size-gb", type=int, default=0)
    parser.add_argument("--port-a", type=int, default=31000)
    parser.add_argument("--port-b", type=int, default=31001)
    parser.add_argument("--instance-a-cuda-visible-devices", default="0,1")
    parser.add_argument("--instance-b-cuda-visible-devices", default="4,5")
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--instance-ready-timeout-s", type=float, default=1800.0)
    parser.add_argument("--instance-health-poll-interval-s", type=float, default=1.0)
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
    parser.add_argument("--mooncake-http-metadata-server-port", type=int, default=8080)
    parser.add_argument("--mooncake-master-port", type=int, default=60051)
    parser.add_argument("--mooncake-global-segment-size", default="4gb")
    parser.add_argument("--mooncake-local-buffer-size", type=int, default=0)
    parser.add_argument(
        "--mooncake-eviction-high-watermark-ratio", type=float, default=0.9
    )
    return parser


def build_remote_prefix(paths) -> str:
    benchmark_package_root = paths.sglang_root / "benchmark"
    return (
        f"cd {shlex.quote(str(paths.sglang_root))}; "
        f"source {shlex.quote(str(paths.workspace_root / '.venv' / 'bin' / 'activate'))}; "
        f"export PYTHONPATH={shlex.quote(str(benchmark_package_root))}:{shlex.quote(str(paths.sglang_root))}:${{PYTHONPATH:-}}; "
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


def build_sglang_command_for_instance(
    *,
    paths,
    config: BenchmarkConfig,
    port: int,
    mooncake_config_path: Path | None,
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
        f"{build_remote_prefix(paths)}"
        f"{build_remote_uv(paths)} run --active --no-project --offline python -m "
        "tensorcast.kv.share_local.request_driver "
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
        f"--worker-process {shlex.quote(worker_info.process_name)} "
        f"--worker-host {shlex.quote(worker_info.hostname)} "
        f"--worker-ip {shlex.quote(worker_info.ip)} "
        f"--worker-node {shlex.quote(worker_info.node)} "
        f"--model-path {shlex.quote(config.model_path)} "
        f"--tp-size {config.tp_size}"
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


def collect_logs(paths, run_id: str) -> None:
    log("Collecting remote logs from shared /data")
    remote_names = [
        "main_keepalive",
        "gpu_snapshot",
        "port_snapshot",
        "mooncake_master",
        "sglang_instance_a",
        "sglang_instance_b",
        "request_driver",
    ]
    for name in remote_names:
        src = remote_file_log(run_id, name)
        dst = paths.logs_dir / f"{name}.log"
        copy_if_exists(src, dst)


def parse_args() -> BenchmarkConfig:
    parser = build_parser()
    args = parser.parse_args()
    payload = vars(args)
    if payload["brainctl_charged_group"].strip():
        payload["brainctl_charged_group"] = payload["brainctl_charged_group"].strip()
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
            raise RuntimeError(
                "tensorcast backend implementation is not wired yet; use mooncake for the first end-to-end run"
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
            "improvement"
            if summary.mean_ttft_improvement_ms >= 0
            else "regression"
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
        collect_logs(paths, run_id)
        if worker_info is not None and launched_worker and not config.keep_worker:
            log(f"Cleaning up worker: {worker_info.process_name}")
            stop_delete_worker(config, worker_info.process_name)
        elif worker_info is not None:
            log(f"Keeping worker for debug: {worker_info.process_name}")


if __name__ == "__main__":
    main()
