from __future__ import annotations

import getpass
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from tensorcast.kv.models import BenchmarkConfig, WorkerInfo

BRAINCTL_PROXY_ENV_KEYS = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "https_proxy",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def local_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in BRAINCTL_PROXY_ENV_KEYS:
        env.pop(key, None)
    return env


def run_local(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout_s: float | None = None,
    check: bool = True,
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


def parse_worker_info(process_name: str, output: str) -> WorkerInfo:
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
    completed = run_local(
        [
            "brainctl",
            "get",
            f"process/{process_name}",
            "-n",
            config.brainctl_namespace,
            "-o",
            "wide",
        ]
    )
    return parse_worker_info(process_name, completed.stdout)


def describe_worker_tail(
    config: BenchmarkConfig, process_name: str, lines: int = 40
) -> str:
    completed = run_local(
        [
            "brainctl",
            "describe",
            f"process/{process_name}",
            "-n",
            config.brainctl_namespace,
        ],
        check=False,
    )
    text = (completed.stdout + completed.stderr).strip().splitlines()
    return "\n".join(text[-lines:])


def wait_for_worker_running(config: BenchmarkConfig, process_name: str) -> WorkerInfo:
    deadline = time.monotonic() + config.worker_ready_timeout_s
    while time.monotonic() < deadline:
        info = get_worker_info(config, process_name)
        if info.status == "Running" and info.ready == "1/1":
            return info
        time.sleep(config.worker_poll_interval_s)
    tail = describe_worker_tail(config, process_name)
    raise RuntimeError(
        f"Worker did not reach Running within {config.worker_ready_timeout_s}s: "
        f"{process_name}\n{tail}"
    )


def launch_worker(config: BenchmarkConfig, run_id: str, role: str = "main") -> str:
    keepalive_log = f"/data/{run_id}_{role}_keepalive.log"
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
        f"--charged-group={config.brainctl_charged_group}",
        f"--gpu={config.worker_gpu}",
        f"--cpu={config.worker_cpu}",
        f"--memory={config.worker_memory}",
        f"--mount={config.brainctl_mount}",
        f"--private-machine={config.brainctl_private_machine}",
        f"--max-wait-duration={config.brainctl_max_wait_duration}",
        f"--comment={run_id}-{role}",
    ]
    if config.worker_positive_tags.strip():
        cmd.append(f"--positive-tags={config.worker_positive_tags}")
    if config.worker_negative_tags.strip():
        cmd.append(f"--negative-tags={config.worker_negative_tags}")
    cmd.extend(["--", "bash", "-lc", remote_cmd])
    completed = run_local(cmd)
    process_name = completed.stdout.strip().splitlines()[-1].strip()
    if not process_name:
        raise RuntimeError(
            f"brainctl launch returned empty process name:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return process_name


def stop_delete_worker(config: BenchmarkConfig, process_name: str) -> None:
    run_local(
        [
            "brainctl",
            "stop",
            f"process/{process_name}",
            "-n",
            config.brainctl_namespace,
        ],
        check=False,
    )
    run_local(
        [
            "brainctl",
            "delete",
            f"process/{process_name}",
            "-n",
            config.brainctl_namespace,
        ],
        check=False,
    )


def exec_root(
    config: BenchmarkConfig,
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
            config.brainctl_namespace,
            "--",
            "bash",
            "-lc",
            remote_cmd,
        ],
        timeout_s=timeout_s,
        check=check,
    )


def exec_user(
    config: BenchmarkConfig,
    process_name: str,
    remote_cmd: str,
    *,
    timeout_s: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    user = getpass.getuser()
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
    check_fn: callable,
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


def build_remote_python_env(
    *,
    workdir: Path,
    python_bin: Path,
    exports: dict[str, str] | None = None,
) -> str:
    export_lines = ""
    for key, value in (exports or {}).items():
        export_lines += f"export {key}={shlex.quote(value)}; "
    return (
        f"cd {shlex.quote(str(workdir))}; "
        f"{export_lines}"
        f"export PATH={shlex.quote(str(python_bin.parent))}:$PATH; "
    )


def start_remote_background_process(
    config: BenchmarkConfig,
    process_name: str,
    *,
    workdir: Path,
    command: str,
    log_path: Path | str,
    pid_path: Path | str,
    env: dict[str, str] | None = None,
) -> None:
    remote_env = build_remote_python_env(
        workdir=workdir,
        python_bin=Path(shutil.which("python3") or "/usr/bin/python3"),
        exports=env,
    )
    remote_cmd = (
        "set -euo pipefail; "
        f"{remote_env}"
        f"LOG_PATH={shlex.quote(str(log_path))}; "
        f"PID_PATH={shlex.quote(str(pid_path))}; "
        'mkdir -p "$(dirname "$LOG_PATH")"; '
        'mkdir -p "$(dirname "$PID_PATH")"; '
        'if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" >/dev/null 2>&1; then '
        "  echo already-running; exit 0; "
        "fi; "
        f'nohup bash -lc {shlex.quote(command)} > "$LOG_PATH" 2>&1 < /dev/null & '
        'echo $! > "$PID_PATH"; '
        'cat "$PID_PATH"'
    )
    exec_user(config, process_name, remote_cmd)


def stop_remote_background_process(
    config: BenchmarkConfig,
    process_name: str,
    *,
    pid_path: Path | str,
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
    exec_user(config, process_name, remote_cmd, check=False)


def wait_remote_http_ready(
    config: BenchmarkConfig,
    process_name: str,
    *,
    workdir: Path,
    python_bin: Path,
    url: str,
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    remote_env = build_remote_python_env(workdir=workdir, python_bin=python_bin)

    def check_fn() -> tuple[bool, str]:
        python_snippet = (
            "import sys, urllib.request; "
            f"url={url!r}; "
            "try:\n"
            "    with urllib.request.urlopen(url, timeout=2) as resp:\n"
            "        print(resp.status)\n"
            "        raise SystemExit(0 if resp.status == 200 else 1)\n"
            "except Exception as exc:\n"
            "    print(exc)\n"
            "    raise SystemExit(1)\n"
        )
        completed = exec_user(
            config,
            process_name,
            f"{remote_env}{shlex.quote(str(python_bin))} -c {shlex.quote(python_snippet)}",
            check=False,
        )
        detail = (completed.stdout + completed.stderr).strip()
        return completed.returncode == 0, detail

    wait_for_condition(
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
        description=f"remote HTTP ready: {url}",
        check_fn=check_fn,
    )
