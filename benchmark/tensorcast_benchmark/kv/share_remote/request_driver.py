from __future__ import annotations

import argparse
import asyncio
import contextlib
import statistics
import time
from pathlib import Path

from tensorcast_benchmark.kv.dataset import load_prompts
from tensorcast_benchmark.kv.models import RequestPrompt
from tensorcast_benchmark.kv.sgl_client import SGLangClient, SGLangHTTPError
from tensorcast_benchmark.kv.share_remote.models import (
    DriverInstanceTarget,
    InstanceRequestResult,
    PromptGroupResult,
    ShareRemoteDriverConfig,
    ShareRemoteRunSummary,
)
from tensorcast_benchmark.kv.share_remote.outputs import load_yaml, write_json, write_jsonl


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


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the multi-worker share-remote request driver."
    )
    parser.add_argument("--config", required=True)
    return parser


def parse_args() -> ShareRemoteDriverConfig:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    payload = load_yaml(config_path)
    if "config" in payload:
        raise RuntimeError("driver config YAML must be a plain object, not nested")
    return ShareRemoteDriverConfig.model_validate(payload)


def _build_request_id(run_id: str, prompt_id: str, group_index: int, position: int) -> str:
    return f"share-remote:{run_id}:{prompt_id}:group{group_index}:pos{position}"


def _extract_cached_tokens(meta_info: dict[str, object]) -> int | None:
    value = meta_info.get("cached_tokens")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


async def _sleep_until(deadline_s: float) -> None:
    sleep_seconds = deadline_s - time.monotonic()
    if sleep_seconds > 0:
        await asyncio.sleep(sleep_seconds)


async def _run_scheduled_request(
    *,
    client: SGLangClient,
    target: DriverInstanceTarget,
    prompt: RequestPrompt,
    run_id: str,
    group_index: int,
    position: int,
    sampling_params: dict[str, float | int],
    scheduled_start_s: float,
) -> InstanceRequestResult:
    rid = _build_request_id(run_id, prompt.prompt_id, group_index, position)
    await _sleep_until(scheduled_start_s)
    log(
        f"Prompt {prompt.prompt_id} group={group_index} position={position} "
        f"scheduled for {target.instance_url}"
    )
    try:
        result = await client.generate_stream(
            prompt.prompt_text,
            sampling_params=sampling_params,
            rid=rid,
        )
        metrics = result.to_metrics()
    except (SGLangHTTPError, TimeoutError) as exc:
        metrics = None
        error_message = str(exc)
        meta_info: dict[str, object] = {}
        text = ""
        ttft_ms = None
        latency_ms = None
        success = False
        cached_tokens = None
    else:
        error_message = metrics.error_message
        meta_info = metrics.meta_info
        text = metrics.text
        ttft_ms = metrics.ttft_ms
        latency_ms = metrics.latency_ms
        success = metrics.success
        cached_tokens = _extract_cached_tokens(metrics.meta_info)
    return InstanceRequestResult(
        position=position,
        worker_index=target.index,
        worker_process=target.worker_process,
        worker_host=target.worker_host,
        worker_ip=target.worker_ip,
        worker_node=target.worker_node,
        instance_url=target.instance_url,
        rid=rid,
        success=success,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        cached_tokens=cached_tokens,
        text=text,
        meta_info=meta_info,
        error_message=error_message,
    )


def _build_group_result(
    *,
    config: ShareRemoteDriverConfig,
    prompt: RequestPrompt,
    group_index: int,
    instance_results: list[InstanceRequestResult],
) -> PromptGroupResult:
    ordered_results = tuple(sorted(instance_results, key=lambda result: result.position))
    first_failure = next(
        (result for result in ordered_results if not result.success),
        None,
    )
    status = "success" if first_failure is None else "failed"
    error_message = "" if first_failure is None else first_failure.error_message
    return PromptGroupResult(
        prompt_id=prompt.prompt_id,
        prompt_chars=len(prompt.prompt_text),
        prompt_length=prompt.prompt_filter_length,
        backend=config.backend,
        group_index=group_index,
        instance_results=ordered_results,
        status=status,
        error_message=error_message,
    )


async def _run_prompt_group(
    *,
    config: ShareRemoteDriverConfig,
    prompt: RequestPrompt,
    group_index: int,
    clients_by_position: dict[int, SGLangClient],
    sampling_params: dict[str, float | int],
    group_start_s: float,
) -> PromptGroupResult:
    instance_tasks = [
        asyncio.create_task(
            _run_scheduled_request(
                client=clients_by_position[target.index],
                target=target,
                prompt=prompt,
                run_id=config.run_id,
                group_index=group_index,
                position=position,
                sampling_params=sampling_params,
                scheduled_start_s=group_start_s + (position * config.settle_ms / 1000.0),
            )
        )
        for position, target in enumerate(config.instance_targets)
    ]
    instance_results = list(await asyncio.gather(*instance_tasks))
    return _build_group_result(
        config=config,
        prompt=prompt,
        group_index=group_index,
        instance_results=instance_results,
    )


def _build_observation(results: list[PromptGroupResult]) -> str:
    if not results:
        return "No prompt groups were executed."
    worker_count = len(results[0].instance_results)
    if worker_count <= 1:
        return "Only one worker position recorded."
    segments: list[str] = []
    for position in range(1, worker_count):
        improvements: list[float] = []
        for result in results:
            first = result.instance_results[0]
            current = result.instance_results[position]
            if first.ttft_ms is None or current.ttft_ms is None:
                continue
            improvements.append(first.ttft_ms - current.ttft_ms)
        if not improvements:
            segments.append(f"position {position}: no TTFT samples")
            continue
        positive = sum(value > 0 for value in improvements)
        segments.append(
            f"position {position}: {positive}/{len(improvements)} lower TTFT than position 0"
        )
    return "; ".join(segments)


def _collect_position_values(
    results: list[PromptGroupResult],
    *,
    extractor,
) -> list[list[float]]:
    if not results:
        return []
    position_count = len(results[0].instance_results)
    values: list[list[float]] = [[] for _ in range(position_count)]
    for result in results:
        for instance_result in result.instance_results:
            value = extractor(result, instance_result)
            if value is None:
                continue
            values[instance_result.position].append(value)
    return values


def _collect_position_improvements(
    results: list[PromptGroupResult],
) -> list[list[float]]:
    if not results:
        return []
    position_count = len(results[0].instance_results)
    values: list[list[float]] = [[] for _ in range(position_count)]
    for result in results:
        first = result.instance_results[0]
        if first.ttft_ms is None:
            continue
        values[0].append(0.0)
        for instance_result in result.instance_results[1:]:
            if instance_result.ttft_ms is None:
                continue
            values[instance_result.position].append(first.ttft_ms - instance_result.ttft_ms)
    return values


async def run_driver(config: ShareRemoteDriverConfig) -> None:
    prompts = load_prompts(
        config.dataset_path,
        config.prompt_count,
        min_prompt_chars=config.min_prompt_chars,
        max_prompt_chars=config.max_prompt_chars,
    )
    sampling_params = {
        "temperature": config.temperature,
        "max_new_tokens": config.max_new_tokens,
    }
    group_interval_s = 1.0 / config.rps
    base_start_s = time.monotonic()
    async with contextlib.AsyncExitStack() as stack:
        client_contexts = [
            await stack.enter_async_context(
                SGLangClient(
                    target.instance_url,
                    request_timeout_seconds=config.request_timeout_s,
                )
            )
            for target in config.instance_targets
        ]
        await asyncio.gather(
            *(
                client.wait_ready(
                    timeout_seconds=config.request_timeout_s,
                    poll_interval_seconds=1.0,
                )
                for client in client_contexts
            )
        )
        clients_by_position = {
            target.index: client
            for target, client in zip(
                config.instance_targets,
                client_contexts,
                strict=True,
            )
        }
        pending_tasks = [
            asyncio.create_task(
                _run_prompt_group(
                    config=config,
                    prompt=prompt,
                    group_index=group_index,
                    clients_by_position=clients_by_position,
                    sampling_params=sampling_params,
                    group_start_s=base_start_s + (group_index * group_interval_s),
                )
            )
            for group_index, prompt in enumerate(prompts)
        ]
        results = list(await asyncio.gather(*pending_tasks))

    write_jsonl(
        Path(config.results_json_path),
        [result.model_dump(mode="json") for result in results],
    )

    ttft_by_position = _collect_position_values(
        results,
        extractor=lambda _group, instance_result: instance_result.ttft_ms,
    )
    cached_tokens_by_position = _collect_position_values(
        results,
        extractor=lambda _group, instance_result: (
            float(instance_result.cached_tokens)
            if instance_result.cached_tokens is not None
            else None
        ),
    )
    improvement_by_position = _collect_position_improvements(results)

    summary = ShareRemoteRunSummary(
        run_id=config.run_id,
        backend=config.backend,
        transport_use_rdma=config.transport_use_rdma,
        worker_count=len(config.instance_targets),
        service_host_worker_index=config.service_host_worker_index,
        prompt_count=len(results),
        avg_prompt_length=_mean([result.prompt_length for result in results]),
        successful_prompt_groups=sum(result.status == "success" for result in results),
        failed_prompt_groups=sum(result.status != "success" for result in results),
        mean_ttft_by_position=tuple(_mean(values) for values in ttft_by_position),
        median_ttft_by_position=tuple(_median(values) for values in ttft_by_position),
        p95_ttft_by_position=tuple(_p95(values) for values in ttft_by_position),
        mean_cached_tokens_by_position=tuple(
            _mean(values) for values in cached_tokens_by_position
        ),
        mean_improvement_vs_first_ms_by_position=tuple(
            _mean(values) for values in improvement_by_position
        ),
        log_dir=str(Path(config.summary_json_path).parent / "logs"),
        results_json_path=config.results_json_path,
        worker_processes=tuple(target.worker_process for target in config.instance_targets),
        worker_hosts=tuple(target.worker_host for target in config.instance_targets),
        worker_ips=tuple(target.worker_ip for target in config.instance_targets),
        worker_nodes=tuple(target.worker_node for target in config.instance_targets),
        model_path=config.model_path,
        tp_size=config.tp_size,
        observation=_build_observation(results),
    )
    write_json(Path(config.summary_json_path), summary.model_dump(mode="json"))


def main() -> None:
    config = parse_args()
    asyncio.run(run_driver(config))


if __name__ == "__main__":
    main()
