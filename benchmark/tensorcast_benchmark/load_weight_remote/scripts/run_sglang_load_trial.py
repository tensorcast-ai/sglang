#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import selectors
import shlex
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

TP_RANK_PATTERN = re.compile(r"\bTP(\d+)\b")
LOG_TIMESTAMP_PATTERN = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)"
)
UV_RUN = ["/home/i-zhouyuhan/.local/bin/uv", "run", "--active", "--no-project", "--offline"]


@dataclass(frozen=True)
class TrialPayload:
    status: str
    error_message: str
    load_time_s: float | None
    ready_time_s: float | None
    log_path: str
    command: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SGLang load-weight trial.")
    parser.add_argument("--load-format", choices=("tensorcast", "default"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--weight-version", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mem-fraction-static", type=float, required=True)
    parser.add_argument("--log-level", required=True)
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--trial-timeout-s", type=float, default=1800.0)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument("--launch-vvv", action="store_true")
    parser.add_argument("--enable-metrics", action="store_true")
    parser.add_argument("--tensorcast-init-mode", default="connect")
    parser.add_argument("--tensorcast-daemon-address", default="")
    parser.add_argument("--tensorcast-show-daemon-logs", action="store_true")
    parser.add_argument("--tensorcast-key-template", default="model:{model_name}:v{weight_version}")
    parser.add_argument("--tensorcast-allow-disk-fallback", action="store_true")
    parser.add_argument("--tensorcast-fallback-prefer", default="auto")
    parser.add_argument("--tensorcast-get-prefer", default="auto")
    parser.add_argument("--tensorcast-export-policy", default="auto")
    parser.add_argument("--tensorcast-disk-fallback-auto-put", action="store_true")
    parser.add_argument("--tensorcast-trace-tp-slices", action="store_true")
    parser.add_argument("--tensorcast-tp-slice-plan-cache-dir", default="/tmp/sglang_tensorcast_trace_cache")
    return parser.parse_args()


def parse_log_timestamp(line: str) -> datetime | None:
    match = LOG_TIMESTAMP_PATTERN.match(line)
    if match is None:
        return None
    raw = match.group(1)
    with contextlib.suppress(ValueError):
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f")
    with contextlib.suppress(ValueError):
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    return None


def build_model_loader_extra_config(args: argparse.Namespace) -> str:
    payload = {
        "tensorcast_init_mode": args.tensorcast_init_mode,
        "tensorcast_daemon_address": args.tensorcast_daemon_address,
        "tensorcast_show_daemon_logs": args.tensorcast_show_daemon_logs,
        "tensorcast_key_template": args.tensorcast_key_template,
        "tensorcast_model_name": args.model_name,
        "tensorcast_allow_disk_fallback": args.tensorcast_allow_disk_fallback,
        "tensorcast_fallback_prefer": args.tensorcast_fallback_prefer,
        "tensorcast_get_prefer": args.tensorcast_get_prefer,
        "tensorcast_export_policy": args.tensorcast_export_policy,
        "tensorcast_disk_fallback_auto_put": args.tensorcast_disk_fallback_auto_put,
        "tensorcast_trace_tp_slices": args.tensorcast_trace_tp_slices,
        "tensorcast_tp_slice_plan_cache_dir": args.tensorcast_tp_slice_plan_cache_dir,
    }
    return json.dumps(payload)


def build_server_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [*UV_RUN]
    if args.launch_vvv:
        cmd.append("-vvv")
    cmd.extend(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model_path,
            "--port",
            str(args.port),
            "--tp-size",
            str(args.tp_size),
            "--mem-fraction-static",
            str(args.mem_fraction_static),
            "--log-level",
            args.log_level,
            "--weight-version",
            str(args.weight_version),
        ]
    )
    if args.enable_metrics:
        cmd.append("--enable-metrics")
    if args.load_format == "tensorcast":
        cmd.extend(
            [
                "--load-format",
                "tensorcast",
                "--model-loader-extra-config",
                build_model_loader_extra_config(args),
            ]
        )
    if args.extra_server_args.strip():
        cmd.extend(shlex.split(args.extra_server_args))
    return cmd


def is_server_healthy(port: int, health_path: str) -> bool:
    request = urllib_request.Request(
        url=f"http://127.0.0.1:{port}{health_path}", method="GET"
    )
    try:
        with urllib_request.urlopen(request, timeout=1.0) as response:
            return response.status == 200
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError):
        return False


def terminate_process_group(proc: subprocess.Popen[str]) -> None:
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


def main() -> None:
    args = parse_args()
    cmd = build_server_cmd(args)
    load_marker = "store.tensor_dict.materialized" if args.load_format == "tensorcast" else "Load weight end"
    load_begin_marker = "Load weight begin"
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.output_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)

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

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"command: {shlex.join(cmd)}\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
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
                if time.monotonic() - start_mono > args.trial_timeout_s:
                    error_message = f"trial timeout exceeded: trial_timeout_s={args.trial_timeout_s}"
                    break

                for key, _ in selector.select(timeout=0.2):
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    log_file.write(line)
                    log_file.flush()
                    line_ts = parse_log_timestamp(line)
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
                                previous = marker_rank_times.get(rank)
                                if previous is None or line_ts > previous:
                                    marker_rank_times[rank] = line_ts
                        else:
                            marker_count_without_rank += 1
                            if line_ts is not None:
                                last_no_rank_marker_time = line_ts
                        if len(marker_ranks) >= args.tp_size or marker_count_without_rank >= args.tp_size:
                            load_done_mono = time.monotonic()
                            if len(marker_ranks) >= args.tp_size and marker_rank_times:
                                load_done_wall = max(marker_rank_times.values())
                            elif last_no_rank_marker_time is not None:
                                load_done_wall = last_no_rank_marker_time
                            else:
                                load_done_wall = line_ts

                if ready_mono is None and is_server_healthy(args.port, args.health_path):
                    ready_mono = time.monotonic()

                if load_done_mono is not None and ready_mono is not None:
                    status = "success" if load_begin_mono is not None else "failed"
                    if load_begin_mono is None:
                        error_message = (
                            "load completion marker found, but no 'Load weight begin' marker was observed"
                        )
                    break

                returncode = proc.poll()
                if returncode is not None:
                    for line in proc.stdout:
                        log_file.write(line)
                    log_file.flush()
                    if load_done_mono is not None and ready_mono is not None:
                        status = "success"
                    else:
                        error_message = f"server exited before completion, returncode={returncode}"
                    break
        finally:
            selector.close()
            terminate_process_group(proc)

    load_time_s = None
    if load_begin_wall is not None and load_done_wall is not None:
        wall_delta = (load_done_wall - load_begin_wall).total_seconds()
        if wall_delta >= 0:
            load_time_s = wall_delta
    if load_time_s is None and load_begin_mono is not None and load_done_mono is not None:
        load_time_s = load_done_mono - load_begin_mono

    ready_time_s = None if ready_mono is None else ready_mono - start_mono
    payload = TrialPayload(
        status=status,
        error_message=error_message,
        load_time_s=load_time_s,
        ready_time_s=ready_time_s,
        log_path=str(log_path),
        command=shlex.join(cmd),
    )
    result_path.write_text(json.dumps(asdict(payload), indent=2), encoding="utf-8")
    print(json.dumps(asdict(payload)), flush=True)
    if status != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
