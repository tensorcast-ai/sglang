from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

from tensorcast.kv.dataset import load_prompts
from tensorcast.kv.models import GenerateMetrics, PairResult, RunSummary
from tensorcast.kv.outputs import write_json, write_jsonl
from tensorcast.kv.sgl_client import SGLangClient, SGLangHTTPError


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
        for prompt in prompts:
            sleep_s = next_pair_start - time.monotonic()
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
            next_pair_start = time.monotonic() + pair_interval_s

            result_a = None
            result_b = None
            error_message = ""
            status = "success"
            try:
                result_a = await client_a.generate_stream(
                    prompt.prompt_text,
                    sampling_params=sampling_params,
                )
                if args.settle_ms > 0:
                    await asyncio.sleep(args.settle_ms / 1000.0)
                result_b = await client_b.generate_stream(
                    prompt.prompt_text,
                    sampling_params=sampling_params,
                )
            except (SGLangHTTPError, TimeoutError) as exc:
                error_message = str(exc)
                status = "failed"

            metrics_a = (
                result_a.to_metrics()
                if result_a is not None
                else build_empty_metrics(error_message or "instance A request failed")
            )
            metrics_b = (
                result_b.to_metrics()
                if result_b is not None
                else build_empty_metrics(error_message or "instance B request failed")
            )

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

            pair_result = PairResult(
                prompt_id=prompt.prompt_id,
                prompt_chars=len(prompt.prompt_text),
                backend=args.backend,
                tensorcast_daemon_mode=(
                    args.tensorcast_daemon_mode
                    if args.backend == "tensorcast"
                    else None
                ),
                instance_a=metrics_a,
                instance_b=metrics_b,
                ttft_improvement_ms=improvement_ms,
                ttft_speedup_ratio=speedup_ratio,
                status=status,
                error_message=error_message,
            )
            results.append(pair_result)

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
    speedups = [
        result.ttft_speedup_ratio
        for result in results
        if result.status == "success" and result.ttft_speedup_ratio is not None
    ]
    success_pairs = sum(result.status == "success" for result in results)
    failed_pairs = len(results) - success_pairs

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
