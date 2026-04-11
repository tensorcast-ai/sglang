from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re

from tensorcast_benchmark.kv.dataset import load_prompts
from tensorcast_benchmark.kv.models import (
    GenerateMetrics,
    PairResult,
    RequestPrompt,
    RunSummary,
)
from tensorcast_benchmark.kv.outputs import write_json, write_jsonl
from tensorcast_benchmark.kv.sgl_client import (
    SGLangClient,
    SGLangHTTPError,
    StreamGenerateResult,
)


_SGLANG_LOG_TIMESTAMP_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: [^\]]+)?\]"
)
_SGLANG_LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
_SGLANG_LOG_TIMESTAMP_TOLERANCE = timedelta(seconds=1)
_LEGACY_SOURCE_PUBLICATION_MARKERS: tuple[str, ...] = (
    "stable_dram upload",
    "Tensorcast batch_set_v1",
)
_STORAGE_BACKUP_ENQUEUE_MARKER = "HiCache storage backup enqueue"
_STORAGE_BACKUP_ACK_MARKER = "HiCache storage backup ack"


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


@dataclass(frozen=True)
class SourcePublicationDrainResult:
    drain_ms: float
    wait_ms: float
    post_completion_upload_count: int
    timed_out: bool


@dataclass(frozen=True)
class _RequestOutcome:
    result: StreamGenerateResult | None
    error_message: str
    completion_timestamp: datetime | None = None


@dataclass
class _PublicationDrainTracker:
    outstanding_backups: int = 0
    saw_backup_queue_markers: bool = False
    saw_request_backup_markers: bool = False
    last_upload_observed_at: float | None = None
    last_upload_timestamp: datetime | None = None
    upload_count: int = 0


def _update_last_upload_timestamp(
    *,
    tracker: _PublicationDrainTracker,
    observed_at: float,
    line_timestamp: datetime | None,
    earliest_counted_timestamp: datetime,
) -> None:
    if line_timestamp is None or line_timestamp < earliest_counted_timestamp:
        return
    tracker.upload_count += 1
    tracker.last_upload_observed_at = observed_at
    if (
        tracker.last_upload_timestamp is None
        or line_timestamp > tracker.last_upload_timestamp
    ):
        tracker.last_upload_timestamp = line_timestamp


def _process_publication_log_chunk(
    *,
    tracker: _PublicationDrainTracker,
    text: str,
    observed_at: float,
    earliest_counted_timestamp: datetime,
    request_id: str | None,
) -> None:
    for line in text.splitlines():
        line_timestamp = parse_sglang_log_timestamp(line)
        if request_id is not None:
            if f"rid={request_id}" in line:
                if _STORAGE_BACKUP_ENQUEUE_MARKER in line:
                    tracker.saw_request_backup_markers = True
                    tracker.outstanding_backups += 1
                    continue
                if _STORAGE_BACKUP_ACK_MARKER in line:
                    tracker.saw_request_backup_markers = True
                    tracker.outstanding_backups = max(
                        tracker.outstanding_backups - 1,
                        0,
                    )
                    _update_last_upload_timestamp(
                        tracker=tracker,
                        observed_at=observed_at,
                        line_timestamp=line_timestamp,
                        earliest_counted_timestamp=earliest_counted_timestamp,
                    )
                    continue
                if any(marker in line for marker in _LEGACY_SOURCE_PUBLICATION_MARKERS):
                    _update_last_upload_timestamp(
                        tracker=tracker,
                        observed_at=observed_at,
                        line_timestamp=line_timestamp,
                        earliest_counted_timestamp=earliest_counted_timestamp,
                    )
                continue
            if tracker.saw_request_backup_markers:
                continue
            if any(marker in line for marker in _LEGACY_SOURCE_PUBLICATION_MARKERS):
                _update_last_upload_timestamp(
                    tracker=tracker,
                    observed_at=observed_at,
                    line_timestamp=line_timestamp,
                    earliest_counted_timestamp=earliest_counted_timestamp,
                )
            continue
        if _STORAGE_BACKUP_ENQUEUE_MARKER in line:
            tracker.saw_backup_queue_markers = True
            tracker.outstanding_backups += 1
            tracker.last_upload_observed_at = observed_at
            continue
        if _STORAGE_BACKUP_ACK_MARKER in line:
            tracker.saw_backup_queue_markers = True
            tracker.outstanding_backups = max(tracker.outstanding_backups - 1, 0)
            _update_last_upload_timestamp(
                tracker=tracker,
                observed_at=observed_at,
                line_timestamp=line_timestamp,
                earliest_counted_timestamp=earliest_counted_timestamp,
            )
            continue
        if tracker.saw_backup_queue_markers:
            continue
        if any(marker in line for marker in _LEGACY_SOURCE_PUBLICATION_MARKERS):
            _update_last_upload_timestamp(
                tracker=tracker,
                observed_at=observed_at,
                line_timestamp=line_timestamp,
                earliest_counted_timestamp=earliest_counted_timestamp,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ordered share-local prompt pairs."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backend", choices=["mooncake", "tensorcast"], required=True)
    parser.add_argument(
        "--tensorcast-daemon-mode", choices=["share", "separate"], default="share"
    )
    parser.add_argument("--instance-a-url", required=True)
    parser.add_argument("--instance-b-url", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--prompt-count", type=int, required=True)
    parser.add_argument("--min-prompt-chars", type=int, default=0)
    parser.add_argument("--max-prompt-chars", type=int, default=0)
    parser.add_argument("--pair-rps", type=float, required=True)
    parser.add_argument("--settle-ms", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--results-json-path", required=True)
    parser.add_argument("--summary-json-path", required=True)
    parser.add_argument("--source-instance-log-path", default="")
    parser.add_argument("--wait-for-source-publication-drain", action="store_true")
    parser.add_argument(
        "--source-publication-drain-timeout-s", type=float, default=120.0
    )
    parser.add_argument("--source-publication-drain-idle-s", type=float, default=5.0)
    parser.add_argument("--source-publication-drain-poll-s", type=float, default=0.25)
    parser.add_argument("--worker-process", required=True)
    parser.add_argument("--worker-host", required=True)
    parser.add_argument("--worker-ip", required=True)
    parser.add_argument("--worker-node", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    return parser


def build_empty_metrics(error_message: str) -> GenerateMetrics:
    return GenerateMetrics(
        success=False,
        text="",
        ttft_ms=None,
        latency_ms=None,
        meta_info={},
        error_message=error_message,
    )


def build_request_id(run_id: str, prompt_id: str) -> str:
    return f"share-local:{run_id}:{prompt_id}"


def build_empty_source_publication_drain_result() -> SourcePublicationDrainResult:
    return SourcePublicationDrainResult(
        drain_ms=0.0,
        wait_ms=0.0,
        post_completion_upload_count=0,
        timed_out=False,
    )


def parse_sglang_log_timestamp(line: str) -> datetime | None:
    match = _SGLANG_LOG_TIMESTAMP_RE.match(line)
    if match is None:
        return None
    try:
        return datetime.strptime(
            match.group("timestamp"),
            _SGLANG_LOG_TIMESTAMP_FORMAT,
        )
    except ValueError:
        return None


def compute_source_publication_drain_ms(
    *,
    completion_timestamp: datetime,
    last_upload_timestamp: datetime | None,
) -> float:
    if last_upload_timestamp is None:
        return 0.0
    delta = last_upload_timestamp - completion_timestamp
    return max(delta.total_seconds() * 1000.0, 0.0)


async def wait_for_source_publication_drain(
    *,
    log_path: Path,
    idle_seconds: float,
    timeout_seconds: float,
    poll_seconds: float,
    completion_timestamp: datetime,
    request_id: str | None = None,
) -> SourcePublicationDrainResult:
    start = time.monotonic()
    file_offset = 0
    earliest_counted_timestamp = completion_timestamp - _SGLANG_LOG_TIMESTAMP_TOLERANCE
    tracker = _PublicationDrainTracker()

    if log_path.exists():
        initial_text = log_path.read_bytes().decode("utf-8", errors="replace")
        _process_publication_log_chunk(
            tracker=tracker,
            text=initial_text,
            observed_at=start,
            earliest_counted_timestamp=earliest_counted_timestamp,
            request_id=request_id,
        )
        file_offset = log_path.stat().st_size

    while True:
        if log_path.exists():
            current_size = log_path.stat().st_size
            if current_size > file_offset:
                with log_path.open("rb") as file:
                    file.seek(file_offset)
                    new_text = file.read().decode("utf-8", errors="replace")
                    file_offset = file.tell()
                _process_publication_log_chunk(
                    tracker=tracker,
                    text=new_text,
                    observed_at=time.monotonic(),
                    earliest_counted_timestamp=earliest_counted_timestamp,
                    request_id=request_id,
                )

        now = time.monotonic()
        elapsed = now - start
        if elapsed >= timeout_seconds:
            return SourcePublicationDrainResult(
                drain_ms=compute_source_publication_drain_ms(
                    completion_timestamp=completion_timestamp,
                    last_upload_timestamp=tracker.last_upload_timestamp,
                ),
                wait_ms=elapsed * 1000.0,
                post_completion_upload_count=tracker.upload_count,
                timed_out=True,
            )

        outstanding_backups = tracker.outstanding_backups
        if outstanding_backups > 0:
            await asyncio.sleep(poll_seconds)
            continue

        if tracker.last_upload_observed_at is not None and (
            now - tracker.last_upload_observed_at >= idle_seconds
        ):
            return SourcePublicationDrainResult(
                drain_ms=compute_source_publication_drain_ms(
                    completion_timestamp=completion_timestamp,
                    last_upload_timestamp=tracker.last_upload_timestamp,
                ),
                wait_ms=elapsed * 1000.0,
                post_completion_upload_count=tracker.upload_count,
                timed_out=False,
            )

        await asyncio.sleep(poll_seconds)


async def _sleep_until(deadline_s: float) -> None:
    sleep_seconds = deadline_s - time.monotonic()
    if sleep_seconds > 0:
        await asyncio.sleep(sleep_seconds)


async def _run_scheduled_request(
    *,
    client: SGLangClient,
    prompt_text: str,
    sampling_params: dict[str, float | int],
    request_id: str,
    scheduled_start_s: float,
) -> _RequestOutcome:
    await _sleep_until(scheduled_start_s)
    try:
        result = await client.generate_stream(
            prompt_text,
            sampling_params=sampling_params,
            rid=request_id,
        )
    except (SGLangHTTPError, TimeoutError) as exc:
        return _RequestOutcome(
            result=None,
            error_message=str(exc),
            completion_timestamp=datetime.now(),
        )
    return _RequestOutcome(
        result=result,
        error_message="",
        completion_timestamp=datetime.now(),
    )


def _build_metrics(
    outcome: _RequestOutcome,
    *,
    default_error_message: str,
) -> GenerateMetrics:
    if outcome.result is not None:
        return outcome.result.to_metrics()
    return build_empty_metrics(outcome.error_message or default_error_message)


def _build_skipped_request_outcome(error_message: str) -> _RequestOutcome:
    return _RequestOutcome(
        result=None,
        error_message=error_message,
        completion_timestamp=None,
    )


def _build_pair_result(
    *,
    args: argparse.Namespace,
    prompt: RequestPrompt,
    outcome_a: _RequestOutcome,
    outcome_b: _RequestOutcome,
    source_drain: SourcePublicationDrainResult,
) -> PairResult:
    error_message = outcome_a.error_message or outcome_b.error_message
    metrics_a = _build_metrics(
        outcome_a,
        default_error_message="instance A request failed",
    )
    metrics_b = _build_metrics(
        outcome_b,
        default_error_message="instance B request failed",
    )

    status = "success"
    if not metrics_a.success or not metrics_b.success:
        status = "failed"
        if not error_message:
            error_message = (
                metrics_a.error_message
                if not metrics_a.success
                else metrics_b.error_message
            )

    improvement_ms = None
    speedup_ratio = None
    if metrics_a.ttft_ms is not None and metrics_b.ttft_ms is not None:
        improvement_ms = metrics_a.ttft_ms - metrics_b.ttft_ms
        if metrics_b.ttft_ms > 0:
            speedup_ratio = metrics_a.ttft_ms / metrics_b.ttft_ms

    return PairResult(
        prompt_id=prompt.prompt_id,
        prompt_chars=len(prompt.prompt_text),
        prompt_length=prompt.prompt_filter_length,
        backend=args.backend,
        tensorcast_daemon_mode=(
            args.tensorcast_daemon_mode if args.backend == "tensorcast" else None
        ),
        instance_a=metrics_a,
        instance_b=metrics_b,
        ttft_improvement_ms=improvement_ms,
        ttft_speedup_ratio=speedup_ratio,
        source_publication_drain_ms=source_drain.drain_ms,
        source_publication_wait_ms=source_drain.wait_ms,
        source_publication_post_completion_upload_count=source_drain.post_completion_upload_count,
        source_publication_drain_timed_out=source_drain.timed_out,
        status=status,
        error_message=error_message,
    )


async def _run_serial_pair(
    *,
    args: argparse.Namespace,
    prompt: RequestPrompt,
    client_a: SGLangClient,
    client_b: SGLangClient,
    sampling_params: dict[str, float | int],
) -> PairResult:
    request_id = build_request_id(args.run_id, prompt.prompt_id)
    source_drain = build_empty_source_publication_drain_result()
    outcome_a = await _run_scheduled_request(
        client=client_a,
        prompt_text=prompt.prompt_text,
        sampling_params=sampling_params,
        request_id=request_id,
        scheduled_start_s=time.monotonic(),
    )
    if outcome_a.result is None:
        return _build_pair_result(
            args=args,
            prompt=prompt,
            outcome_a=outcome_a,
            outcome_b=_build_skipped_request_outcome(
                "instance B request skipped because instance A request failed"
            ),
            source_drain=source_drain,
        )

    if (
        args.source_instance_log_path.strip()
        and outcome_a.completion_timestamp is not None
    ):
        source_drain = await wait_for_source_publication_drain(
            log_path=Path(args.source_instance_log_path),
            idle_seconds=args.source_publication_drain_idle_s,
            timeout_seconds=args.source_publication_drain_timeout_s,
            poll_seconds=args.source_publication_drain_poll_s,
            completion_timestamp=outcome_a.completion_timestamp,
            request_id=request_id,
        )
    if args.settle_ms > 0:
        await asyncio.sleep(args.settle_ms / 1000.0)
    outcome_b = await _run_scheduled_request(
        client=client_b,
        prompt_text=prompt.prompt_text,
        sampling_params=sampling_params,
        request_id=request_id,
        scheduled_start_s=time.monotonic(),
    )
    return _build_pair_result(
        args=args,
        prompt=prompt,
        outcome_a=outcome_a,
        outcome_b=outcome_b,
        source_drain=source_drain,
    )


async def _run_overlapped_pair(
    *,
    args: argparse.Namespace,
    prompt: RequestPrompt,
    client_a: SGLangClient,
    client_b: SGLangClient,
    sampling_params: dict[str, float | int],
    a_start_s: float,
    b_start_s: float,
) -> PairResult:
    request_id = build_request_id(args.run_id, prompt.prompt_id)
    outcome_a, outcome_b = await asyncio.gather(
        asyncio.create_task(
            _run_scheduled_request(
                client=client_a,
                prompt_text=prompt.prompt_text,
                sampling_params=sampling_params,
                request_id=request_id,
                scheduled_start_s=a_start_s,
            )
        ),
        asyncio.create_task(
            _run_scheduled_request(
                client=client_b,
                prompt_text=prompt.prompt_text,
                sampling_params=sampling_params,
                request_id=request_id,
                scheduled_start_s=b_start_s,
            )
        ),
    )
    return _build_pair_result(
        args=args,
        prompt=prompt,
        outcome_a=outcome_a,
        outcome_b=outcome_b,
        source_drain=build_empty_source_publication_drain_result(),
    )


async def run_pairs(args: argparse.Namespace) -> None:
    prompts = load_prompts(
        args.dataset_path,
        args.prompt_count,
        min_prompt_chars=args.min_prompt_chars,
        max_prompt_chars=args.max_prompt_chars,
    )
    results: list[PairResult] = []
    pair_interval_s = 1.0 / args.pair_rps
    next_pair_start = time.monotonic()
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }

    async with (
        SGLangClient(
            args.instance_a_url,
            request_timeout_seconds=args.request_timeout_s,
        ) as client_a,
        SGLangClient(
            args.instance_b_url,
            request_timeout_seconds=args.request_timeout_s,
        ) as client_b,
    ):
        if args.wait_for_source_publication_drain:
            for prompt in prompts:
                sleep_s = next_pair_start - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                next_pair_start = time.monotonic() + pair_interval_s
                results.append(
                    await _run_serial_pair(
                        args=args,
                        prompt=prompt,
                        client_a=client_a,
                        client_b=client_b,
                        sampling_params=sampling_params,
                    )
                )
        else:
            base_start_s = time.monotonic()
            pair_tasks = [
                asyncio.create_task(
                    _run_overlapped_pair(
                        args=args,
                        prompt=prompt,
                        client_a=client_a,
                        client_b=client_b,
                        sampling_params=sampling_params,
                        a_start_s=base_start_s + (index * pair_interval_s),
                        b_start_s=base_start_s
                        + (index * pair_interval_s)
                        + (args.settle_ms / 1000.0),
                    )
                )
                for index, prompt in enumerate(prompts)
            ]
            results = list(await asyncio.gather(*pair_tasks))

    write_jsonl(
        Path(args.results_json_path),
        [result.model_dump(mode="json") for result in results],
    )

    a_ttfts = [
        result.instance_a.ttft_ms
        for result in results
        if result.status == "success" and result.instance_a.ttft_ms is not None
    ]
    b_ttfts = [
        result.instance_b.ttft_ms
        for result in results
        if result.status == "success" and result.instance_b.ttft_ms is not None
    ]
    improvements = [
        result.ttft_improvement_ms
        for result in results
        if result.status == "success" and result.ttft_improvement_ms is not None
    ]
    prompt_lengths = [result.prompt_length for result in results]
    speedups = [
        result.ttft_speedup_ratio
        for result in results
        if result.status == "success" and result.ttft_speedup_ratio is not None
    ]
    drain_durations = [
        result.source_publication_drain_ms
        for result in results
        if result.status == "success" and result.source_publication_drain_ms is not None
    ]
    drain_waits = [
        result.source_publication_wait_ms
        for result in results
        if result.status == "success" and result.source_publication_wait_ms is not None
    ]
    success_pairs = sum(result.status == "success" for result in results)
    failed_pairs = len(results) - success_pairs
    drain_timeout_count = sum(
        result.source_publication_drain_timed_out for result in results
    )

    observation = "No successful TTFT samples recorded."
    if improvements:
        positive = sum(value > 0 for value in improvements)
        observation = (
            f"{positive}/{len(improvements)} successful pairs showed lower TTFT on instance B. "
            f"Mean improvement: {_mean(improvements):.2f} ms."
        )

    summary = RunSummary(
        run_id=args.run_id,
        backend=args.backend,
        tensorcast_daemon_mode=(
            args.tensorcast_daemon_mode if args.backend == "tensorcast" else None
        ),
        prompt_count=len(results),
        avg_prompt_length=_mean(prompt_lengths),
        success_pairs=success_pairs,
        failed_pairs=failed_pairs,
        mean_instance_a_ttft_ms=_mean(a_ttfts),
        mean_instance_b_ttft_ms=_mean(b_ttfts),
        median_instance_a_ttft_ms=_median(a_ttfts),
        median_instance_b_ttft_ms=_median(b_ttfts),
        p95_instance_a_ttft_ms=_p95(a_ttfts),
        p95_instance_b_ttft_ms=_p95(b_ttfts),
        mean_ttft_improvement_ms=_mean(improvements),
        median_ttft_improvement_ms=_median(improvements),
        p95_ttft_improvement_ms=_p95(improvements),
        mean_ttft_speedup_ratio=_mean(speedups),
        mean_source_publication_drain_ms=_mean(drain_durations),
        median_source_publication_drain_ms=_median(drain_durations),
        p95_source_publication_drain_ms=_p95(drain_durations),
        mean_source_publication_wait_ms=_mean(drain_waits),
        source_publication_drain_timeout_count=drain_timeout_count,
        log_dir=str(Path(args.results_json_path).parent / "logs"),
        results_json_path=args.results_json_path,
        worker_process=args.worker_process,
        worker_host=args.worker_host,
        worker_ip=args.worker_ip,
        worker_node=args.worker_node,
        model_path=args.model_path,
        tp_size=args.tp_size,
        observation=observation,
    )
    write_json(Path(args.summary_json_path), summary.model_dump(mode="json"))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run_pairs(args))


if __name__ == "__main__":
    main()
