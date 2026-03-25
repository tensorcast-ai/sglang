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
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

TP_RANK_PATTERN = re.compile(r"\bTP(\d+)\b")
LOG_TIMESTAMP_PATTERN = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)")
UV_RUN = ["/home/i-zhouyuhan/.local/bin/uv", "run", "--active", "--no-project", "--offline"]


@dataclass(frozen=True)
class TrialRecord:
    trial_id: int
    target_weight_version: int
    endpoint: str
    status: str
    error_message: str
    load_time_s: float | None
    ready_time_s: float | None
    log_path: str


@dataclass(frozen=True)
class RunPayload:
    status: str
    error_message: str
    server_log_path: str
    trial_results: list[TrialRecord]
    command: str


@dataclass
class HttpRequestOutcome:
    done: threading.Event
    status_code: int | None = None
    body: str = ""
    error_message: str = ""
    response_end_mono: float | None = None


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remote SGLang update-weight benchmark trials.")
    parser.add_argument("--load-format", choices=("tensorcast", "default"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--weight-version-start", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mem-fraction-static", type=float, required=True)
    parser.add_argument("--log-level", required=True)
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--server-ready-timeout-s", type=float, default=1800.0)
    parser.add_argument("--health-poll-interval-s", type=float, default=0.5)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--trial-log-dir", required=True)
    parser.add_argument("--trial-log-prefix", required=True)
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument("--launch-vvv", action="store_true")
    parser.add_argument("--enable-metrics", action="store_true")
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--abort-all-requests", action="store_true")
    parser.add_argument("--recapture-cuda-graph", action="store_true")
    parser.add_argument("--clear-cache-script", default="")
    parser.add_argument("--cache-dir", action="append", default=[])
    parser.add_argument("--tensorcast-init-mode", default="connect")
    parser.add_argument("--tensorcast-daemon-address", default="")
    parser.add_argument("--tensorcast-show-daemon-logs", action="store_true")
    parser.add_argument("--tensorcast-key-template", default="model:{model_name}:v{weight_version}")
    parser.add_argument("--tensorcast-allow-disk-fallback", action="store_true")
    parser.add_argument("--tensorcast-fallback-prefer", default="auto")
    parser.add_argument("--tensorcast-get-prefer", default="auto")
    parser.add_argument("--tensorcast-export-policy", default="auto")
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
            str(args.weight_version_start),
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
    req = urllib_request.Request(url=f"http://127.0.0.1:{port}{health_path}", method="GET")
    try:
        with urllib_request.urlopen(req, timeout=1.0) as response:
            return response.status == 200
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError):
        return False


def wait_for_server_ready(args: argparse.Namespace, monitor: ServerLogMonitor) -> None:
    deadline = time.monotonic() + args.server_ready_timeout_s
    while time.monotonic() < deadline:
        monitor.poll(timeout_s=args.health_poll_interval_s)
        if is_server_healthy(args.port, args.health_path):
            return
        if monitor.process.poll() is not None:
            raise RuntimeError(f"Server exited before ready, returncode={monitor.process.returncode}")
    raise RuntimeError(f"Timed out waiting for server health in {args.server_ready_timeout_s:.1f}s")


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


def clear_model_cache(args: argparse.Namespace) -> None:
    if not args.clear_cache_script:
        return
    cmd = ["bash", args.clear_cache_script, "--path", args.model_path]
    for cache_dir in args.cache_dir:
        cmd.extend(["--cache-dir", cache_dir])
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Cache clear failed: {shlex.join(cmd)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def build_update_request(args: argparse.Namespace, target_version: int) -> tuple[str, dict[str, object]]:
    if args.load_format == "tensorcast":
        artifact_key = args.tensorcast_key_template.format(
            model_name=args.model_name,
            version=target_version,
            weight_version=target_version,
        )
        return "/update_weights_from_tensorcast", {
            "weight_version": target_version,
            "artifact_key": artifact_key,
            "flush_cache": args.flush_cache,
            "abort_all_requests": args.abort_all_requests,
            "recapture_cuda_graph": args.recapture_cuda_graph,
        }
    return "/update_weights_from_disk", {
        "model_path": args.model_path,
        "weight_version": str(target_version),
        "flush_cache": args.flush_cache,
        "abort_all_requests": args.abort_all_requests,
        "recapture_cuda_graph": args.recapture_cuda_graph,
    }


def start_http_post_request(*, url: str, payload: dict[str, object], timeout_s: float) -> tuple[threading.Thread, HttpRequestOutcome]:
    outcome = HttpRequestOutcome(done=threading.Event())

    def worker() -> None:
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
            outcome.body = exc.read().decode("utf-8", errors="replace")
            outcome.error_message = f"HTTPError: {exc}"
        except Exception as exc:  # noqa: BLE001
            outcome.error_message = str(exc)
        finally:
            outcome.response_end_mono = time.monotonic()
            outcome.done.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread, outcome


def update_begin_marker(load_format: str) -> str:
    if load_format == "tensorcast":
        return "Update engine weights online from tensorcast begin."
    return "Update engine weights online from disk begin."


def update_end_marker(load_format: str, recapture_cuda_graph: bool) -> str:
    if load_format == "tensorcast":
        return "store.tensor_dict.materialized"
    if recapture_cuda_graph:
        return "Capture cuda graph begin."
    return "Update weights end."


def compute_load_time_from_events(*, events: list[tuple[float, str]], load_format: str, tp_size: int, recapture_cuda_graph: bool) -> float:
    begin_marker = update_begin_marker(load_format)
    end_marker = update_end_marker(load_format, recapture_cuda_graph)
    begin_mono: float | None = None
    end_mono: float | None = None
    begin_wall: datetime | None = None
    end_wall: datetime | None = None
    end_ranks: set[int] = set()
    end_rank_walls: dict[int, datetime] = {}
    end_without_rank = 0
    last_no_rank_end_wall: datetime | None = None

    for ts, line in events:
        line_wall = parse_log_timestamp(line)
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
                prev = end_rank_walls.get(rank)
                if prev is None or line_wall > prev:
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
            else:
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


def wait_for_trial_markers(*, monitor: ServerLogMonitor, start_event_idx: int, load_format: str, tp_size: int, recapture_cuda_graph: bool, timeout_s: float) -> list[tuple[float, str]]:
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        trial_events = monitor.get_events_since(start_event_idx)
        try:
            compute_load_time_from_events(
                events=trial_events,
                load_format=load_format,
                tp_size=tp_size,
                recapture_cuda_graph=recapture_cuda_graph,
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


def write_trial_log(*, path: Path, endpoint: str, payload: dict[str, object], status_code: int | None, response_body: str, response_error: str, log_lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(f"endpoint: {endpoint}\n")
        file.write(f"payload: {json.dumps(payload, ensure_ascii=True)}\n")
        file.write(f"status_code: {status_code}\n")
        if response_error:
            file.write(f"response_error: {response_error}\n")
        file.write(f"response_body: {response_body}\n")
        file.write("--- server log lines captured for this trial ---\n")
        for line in log_lines:
            file.write(line)


def main() -> None:
    args = parse_args()
    server_log_path = Path(args.log_path)
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trial_log_dir = Path(args.trial_log_dir)
    trial_log_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_server_cmd(args)
    monitor: ServerLogMonitor | None = None
    proc: subprocess.Popen[str] | None = None
    trial_results: list[TrialRecord] = []
    status = "failed"
    error_message = ""

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        monitor = ServerLogMonitor(proc, server_log_path, cmd)
        wait_for_server_ready(args, monitor)

        for trial_id in range(1, args.trials + 1):
            target_version = args.weight_version_start + trial_id
            if args.load_format == "default":
                clear_model_cache(args)
            endpoint, payload = build_update_request(args, target_version)
            request_start_mono = time.monotonic()
            start_event_idx = monitor.event_count
            thread, outcome = start_http_post_request(
                url=f"http://127.0.0.1:{args.port}{endpoint}",
                payload=payload,
                timeout_s=args.request_timeout_s,
            )

            while not outcome.done.is_set():
                monitor.poll(timeout_s=0.2)
                if monitor.process.poll() is not None:
                    break
            thread.join(timeout=1.0)
            ready_time_s = (outcome.response_end_mono or time.monotonic()) - request_start_mono

            try:
                trial_events = wait_for_trial_markers(
                    monitor=monitor,
                    start_event_idx=start_event_idx,
                    load_format=args.load_format,
                    tp_size=args.tp_size,
                    recapture_cuda_graph=args.recapture_cuda_graph,
                    timeout_s=60.0,
                )
            except Exception:
                trial_events = monitor.get_events_since(start_event_idx)

            trial_log_path = trial_log_dir / f"{args.trial_log_prefix}_trial{trial_id:03d}.log"
            write_trial_log(
                path=trial_log_path,
                endpoint=endpoint,
                payload=payload,
                status_code=outcome.status_code,
                response_body=outcome.body,
                response_error=outcome.error_message,
                log_lines=[line for _, line in trial_events],
            )

            if monitor.process.poll() is not None:
                error_message = f"trial {trial_id} failed: server exited, returncode={monitor.process.returncode}"
                trial_results.append(TrialRecord(trial_id, target_version, endpoint, "failed", error_message, None, ready_time_s, str(trial_log_path)))
                break
            if outcome.status_code != 200:
                error_message = f"trial {trial_id} failed: HTTP {outcome.status_code}, body={outcome.body}, error={outcome.error_message}"
                trial_results.append(TrialRecord(trial_id, target_version, endpoint, "failed", error_message, None, ready_time_s, str(trial_log_path)))
                break
            try:
                load_time_s = compute_load_time_from_events(
                    events=trial_events,
                    load_format=args.load_format,
                    tp_size=args.tp_size,
                    recapture_cuda_graph=args.recapture_cuda_graph,
                )
            except Exception as exc:  # noqa: BLE001
                error_message = f"trial {trial_id} failed to parse load_time: {exc}"
                trial_results.append(TrialRecord(trial_id, target_version, endpoint, "failed", error_message, None, ready_time_s, str(trial_log_path)))
                break

            trial_results.append(
                TrialRecord(
                    trial_id=trial_id,
                    target_weight_version=target_version,
                    endpoint=endpoint,
                    status="success",
                    error_message="",
                    load_time_s=load_time_s,
                    ready_time_s=ready_time_s,
                    log_path=str(trial_log_path),
                )
            )

        if not error_message:
            status = "success"
    finally:
        if monitor is not None:
            monitor.close()
        if proc is not None:
            terminate_process_group(proc)

    payload = RunPayload(
        status=status,
        error_message=error_message,
        server_log_path=str(server_log_path),
        trial_results=trial_results,
        command=shlex.join(cmd),
    )
    output_path.write_text(json.dumps(asdict(payload), indent=2), encoding="utf-8")
    if status != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
