from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

import tensorcast as tc
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tensorcast.api.context import CallContext
from tensorcast.api.errors import ArtifactError
from tensorcast.api.plan import Instance, PlanFailedError, Worker
from tensorcast.api.runtime import Runtime
from tensorcast.engine_adapter.artifact_api import (
    HydrateResult,
    PublishManifest,
    PublishResult,
)
from tensorcast_benchmark.kv.dataset import load_prompts
from tensorcast_benchmark.kv.models import GenerateMetrics, RequestPrompt
from tensorcast_benchmark.kv.outputs import write_json, write_jsonl
from tensorcast_benchmark.kv.request_transfer.models import (
    PromptTransferResult,
    RequestTransferRunSummary,
    TopologyMode,
)
from tensorcast_benchmark.kv.sgl_client import SGLangClient, SGLangHTTPError


class CallerDriverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    topology_mode: TopologyMode
    gateway_daemon_address: str = Field(min_length=1)
    source_instance_id: str = Field(min_length=1)
    target_instance_id: str = Field(min_length=1)
    source_instance_url: str = Field(min_length=1)
    target_instance_url: str = Field(min_length=1)
    target_instance_log_path: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
    prompt_count: int = Field(ge=1)
    min_prompt_chars: int = Field(default=0, ge=0)
    max_prompt_chars: int = Field(default=0, ge=0)
    max_new_tokens: int = Field(ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    request_timeout_s: float = Field(default=600.0, gt=0.0)
    plan_deadline_ms: int = Field(default=15_000, gt=0)
    publish_ttl_ms: int = Field(default=60_000, gt=0)
    enable_target_worker_warmup: bool = False
    verify_log_timeout_s: float = Field(default=15.0, gt=0.0)
    verify_log_poll_interval_s: float = Field(default=0.25, gt=0.0)
    post_target_generate_settle_s: float = Field(default=1.0, ge=0.0)
    evict_after_prompt: bool = True
    results_json_path: str = Field(min_length=1)
    summary_json_path: str = Field(min_length=1)
    worker_process_a: str = ""
    worker_process_b: str = ""
    worker_host_a: str = ""
    worker_host_b: str = ""
    worker_ip_a: str = ""
    worker_ip_b: str = ""
    worker_node_a: str = ""
    worker_node_b: str = ""
    model_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_paths(self) -> "CallerDriverConfig":
        dataset_path = Path(self.dataset_path).expanduser()
        if not dataset_path.is_file():
            raise ValueError(f"dataset_path does not exist: {self.dataset_path}")
        if self.max_prompt_chars > 0 and self.max_prompt_chars < self.min_prompt_chars:
            raise ValueError(
                "max_prompt_chars must be 0 or greater than or equal to min_prompt_chars"
            )
        return self


class _PreparedBundleSignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attached: bool = False
    fallback: bool = False
    fail_closed: bool = False
    consume_failed: bool = False


class _EmbeddedArtifactManifestEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifact_id: str = Field(min_length=1)


class _EmbeddedArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifact_manifest_digest: str = Field(min_length=1)
    entries: tuple[_EmbeddedArtifactManifestEntry, ...] = ()


class _EmbeddedPublishManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    publish_manifest_digest: str = Field(min_length=1)
    artifact_manifest: _EmbeddedArtifactManifest
    cutoff_token_count: int = Field(ge=0)
    tail_valid_tokens: int = Field(default=0, ge=0)


_INSTANCE_AGENT_EMBEDDED_MANIFEST_SCHEMA = (
    "sglang.request_bundle.publish_manifest_record.v1"
)


def _sha256_hexdigest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decode_instance_publish_manifest(
    publish_manifest: PublishManifest,
) -> tuple[str, _EmbeddedPublishManifest]:
    if (
        publish_manifest.engine_owned_manifest.schema
        != _INSTANCE_AGENT_EMBEDDED_MANIFEST_SCHEMA
    ):
        raise ArtifactError(
            "unsupported SGLang instance publish manifest schema",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    payload_bytes = bytes(publish_manifest.engine_owned_manifest.payload)
    payload_sha256 = publish_manifest.engine_owned_manifest.payload_sha256
    if payload_sha256 is not None and payload_sha256 != _sha256_hexdigest(
        payload_bytes
    ):
        raise ArtifactError(
            "instance publish manifest payload digest mismatch",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    try:
        decoded_payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            "instance publish manifest payload is not valid JSON",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        ) from exc
    if decoded_payload.get("schema") != _INSTANCE_AGENT_EMBEDDED_MANIFEST_SCHEMA:
        raise ArtifactError(
            "instance publish manifest payload schema mismatch",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    engine_request_id = str(decoded_payload.get("engine_request_id", "")).strip()
    if not engine_request_id:
        raise ArtifactError(
            "instance publish manifest is missing engine_request_id",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    local_publish_manifest = _EmbeddedPublishManifest.model_validate(
        decoded_payload.get("publish_manifest", {})
    )
    if (
        local_publish_manifest.artifact_manifest.artifact_manifest_digest
        != publish_manifest.artifact_manifest.key_set_digest_hex
    ):
        raise ArtifactError(
            "instance publish manifest artifact digest mismatch",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    artifact_ids = tuple(
        entry.artifact_id for entry in local_publish_manifest.artifact_manifest.entries
    )
    if artifact_ids != tuple(publish_manifest.artifact_manifest.artifact_ids):
        raise ArtifactError(
            "instance publish manifest artifact ordering mismatch",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    return engine_request_id, local_publish_manifest


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the controller-driven request-transfer caller benchmark."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--topology-mode", choices=["local", "remote"], required=True)
    parser.add_argument("--gateway-daemon-address", required=True)
    parser.add_argument("--source-instance-id", required=True)
    parser.add_argument("--target-instance-id", required=True)
    parser.add_argument("--source-instance-url", required=True)
    parser.add_argument("--target-instance-url", required=True)
    parser.add_argument("--target-instance-log-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--prompt-count", type=int, required=True)
    parser.add_argument("--min-prompt-chars", type=int, default=0)
    parser.add_argument("--max-prompt-chars", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
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
    parser.add_argument("--results-json-path", required=True)
    parser.add_argument("--summary-json-path", required=True)
    parser.add_argument("--worker-process-a", default="")
    parser.add_argument("--worker-process-b", default="")
    parser.add_argument("--worker-host-a", default="")
    parser.add_argument("--worker-host-b", default="")
    parser.add_argument("--worker-ip-a", default="")
    parser.add_argument("--worker-ip-b", default="")
    parser.add_argument("--worker-node-a", default="")
    parser.add_argument("--worker-node-b", default="")
    parser.add_argument("--model-path", required=True)
    return parser


def parse_args() -> CallerDriverConfig:
    args = build_parser().parse_args()
    payload = vars(args)
    for key in (
        "dataset_path",
        "target_instance_log_path",
        "results_json_path",
        "summary_json_path",
    ):
        payload[key] = str(Path(payload[key]).expanduser().resolve())
    return CallerDriverConfig.model_validate(payload)


def build_empty_metrics(error_message: str) -> GenerateMetrics:
    return GenerateMetrics(
        success=False,
        text="",
        ttft_ms=None,
        latency_ms=None,
        meta_info={},
        error_message=error_message,
    )


def build_logical_request_id(run_id: str, prompt_id: str) -> str:
    return f"request-transfer:{run_id}:{prompt_id}"


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _step_error_message(exc: Exception) -> str:
    if isinstance(exc, PlanFailedError):
        messages = [
            step.status.message
            for step in exc.result.steps.values()
            if step.status.message.strip()
        ]
        if messages:
            return "; ".join(messages)
    return str(exc) or exc.__class__.__name__


def _poll_instance_route(
    runtime: Runtime,
    *,
    instance_id: str,
    timeout_s: float,
    poll_interval_s: float,
) -> Instance:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            route = runtime.directory().resolve_instance_execution(instance_id).value
            return Instance(
                instance_id=route.instance_id,
                daemon_id=route.daemon_id,
                engine=route.engine or "sglang",
                execution_endpoint=route.execution_endpoint,
            )
        except Exception as exc:
            last_error = str(exc)
            time.sleep(poll_interval_s)
    raise RuntimeError(
        f"timed out resolving instance route for {instance_id}: {last_error}"
    )


def _poll_worker_route_for_daemon(
    runtime: Runtime,
    *,
    daemon_id: str,
    timeout_s: float,
    poll_interval_s: float,
) -> Worker:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            workers = runtime.directory().list_workers().value
            for route in workers:
                if route.daemon_id != daemon_id:
                    continue
                return Worker(
                    worker_id=route.worker_id or route.daemon_id,
                    daemon_address=route.daemon_address or "",
                    daemon_id=route.daemon_id,
                )
            last_error = f"daemon_id={daemon_id} not found in worker directory"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(poll_interval_s)
    raise RuntimeError(
        f"timed out resolving worker route for daemon {daemon_id}: {last_error}"
    )


def _publish_plan_result(
    runtime: Runtime,
    *,
    source_instance: Instance,
    logical_request_id: str,
    deadline_ms: int,
    ttl_ms: int,
) -> PublishResult:
    context = CallContext(
        request_id=f"publish:{logical_request_id}",
        deadline_ms=deadline_ms,
        idempotency_key=f"publish:{logical_request_id}",
    )
    plan = runtime.plan(context)
    publish_ref = plan.on_instance(source_instance).publish(
        engine_request_id=logical_request_id,
        ttl_ms=ttl_ms,
    )
    result = plan.run()
    artifact_result = result.step(publish_ref).artifact_result
    if not isinstance(artifact_result, PublishResult):
        raise RuntimeError("publish step did not return PublishResult")
    if artifact_result.publish_manifest is None:
        raise RuntimeError("publish step did not return publish_manifest")
    return artifact_result


def _hydrate_plan_result(
    runtime: Runtime,
    *,
    target_instance: Instance,
    target_worker: Worker | None,
    publish_result: PublishResult,
    logical_request_id: str,
    deadline_ms: int,
    enable_warmup: bool,
) -> HydrateResult:
    publish_manifest = publish_result.publish_manifest
    if publish_manifest is None:
        raise RuntimeError("publish_result.publish_manifest is required")
    context = CallContext(
        request_id=f"hydrate:{logical_request_id}",
        deadline_ms=deadline_ms,
        idempotency_key=f"hydrate:{logical_request_id}",
    )
    plan = runtime.plan(context)
    depends_on = None
    if enable_warmup:
        if target_worker is None:
            raise RuntimeError("target worker is required for warmup")
        warm_ref = plan.on_worker(target_worker).prefetch_manifest_result(
            publish_manifest.artifact_manifest,
            device="cpu",
        )
        depends_on = [warm_ref]
    hydrate_ref = plan.on_instance(target_instance).hydrate(
        publish_manifest=publish_manifest,
        depends_on=depends_on,
    )
    result = plan.run()
    artifact_result = result.step(hydrate_ref).artifact_result
    if not isinstance(artifact_result, HydrateResult):
        raise RuntimeError("hydrate step did not return HydrateResult")
    return artifact_result


def _best_effort_evict(
    runtime: Runtime,
    *,
    instance: Instance,
    logical_request_id: str,
    deadline_ms: int,
) -> None:
    try:
        context = CallContext(
            request_id=f"evict:{logical_request_id}:{instance.instance_id}",
            deadline_ms=deadline_ms,
            idempotency_key=f"evict:{logical_request_id}:{instance.instance_id}",
        )
        plan = runtime.plan(context)
        plan.on_instance(instance).evict_local(engine_request_id=logical_request_id)
        plan.run()
    except Exception as exc:
        log(
            f"Best-effort evict_local failed for instance={instance.instance_id} "
            f"logical_request_id={logical_request_id}: {exc}"
        )


def _read_log_chunk(log_path: Path, offset: int) -> tuple[str, int]:
    if not log_path.exists():
        return "", offset
    current_size = log_path.stat().st_size
    if current_size < offset:
        offset = 0
    if current_size == offset:
        return "", offset
    with log_path.open("rb") as file:
        file.seek(offset)
        payload = file.read()
        new_offset = file.tell()
    return payload.decode("utf-8", errors="replace"), new_offset


def _extract_prepared_bundle_signals(
    *,
    log_chunk: str,
    logical_request_id: str,
    publish_manifest_digest: str,
) -> _PreparedBundleSignals:
    attached = False
    fallback = False
    fail_closed = False
    consume_failed = False
    for line in log_chunk.splitlines():
        if (
            "Tensorcast prepared-bundle attached" in line
            and logical_request_id in line
            and publish_manifest_digest in line
        ):
            attached = True
        if (
            "Tensorcast prepared-bundle falling back to normal generate path:" in line
            and logical_request_id in line
            and publish_manifest_digest in line
        ):
            fallback = True
        if (
            "Tensorcast prepared-bundle fail-closed during ordinary generate admission:"
            in line
        ):
            fail_closed = True
        if (
            "Tensorcast prepared-bundle consume failed rid=" in line
            and logical_request_id in line
            and publish_manifest_digest in line
        ):
            consume_failed = True
    return _PreparedBundleSignals(
        attached=attached,
        fallback=fallback,
        fail_closed=fail_closed,
        consume_failed=consume_failed,
    )


async def _wait_for_prepared_bundle_signals(
    *,
    log_path: Path,
    start_offset: int,
    logical_request_id: str,
    publish_manifest_digest: str,
    timeout_s: float,
    poll_interval_s: float,
) -> _PreparedBundleSignals:
    deadline = time.monotonic() + timeout_s
    offset = start_offset
    aggregate = _PreparedBundleSignals()
    while time.monotonic() < deadline:
        chunk, offset = _read_log_chunk(log_path, offset)
        if chunk:
            observed = _extract_prepared_bundle_signals(
                log_chunk=chunk,
                logical_request_id=logical_request_id,
                publish_manifest_digest=publish_manifest_digest,
            )
            aggregate = _PreparedBundleSignals(
                attached=aggregate.attached or observed.attached,
                fallback=aggregate.fallback or observed.fallback,
                fail_closed=aggregate.fail_closed or observed.fail_closed,
                consume_failed=aggregate.consume_failed or observed.consume_failed,
            )
            if (
                aggregate.attached
                or aggregate.fallback
                or aggregate.fail_closed
                or aggregate.consume_failed
            ):
                return aggregate
        await asyncio.sleep(poll_interval_s)
    return aggregate


def _cached_tokens_from_metrics(metrics: GenerateMetrics) -> int | None:
    raw_value = metrics.meta_info.get("cached_tokens")
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _prompt_length_from_metrics(metrics: GenerateMetrics) -> int:
    raw_value = metrics.meta_info.get("prompt_tokens")
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return 0


def _expected_cached_tokens(
    *,
    published_cutoff_token_count: int | None,
    tail_valid_tokens: int | None,
) -> int | None:
    if published_cutoff_token_count is None:
        return None
    if tail_valid_tokens is None:
        return published_cutoff_token_count
    clamped_tail_valid_tokens = min(
        max(int(tail_valid_tokens), 0),
        published_cutoff_token_count,
    )
    return published_cutoff_token_count - clamped_tail_valid_tokens


def _verify_prepared_bundle(
    *,
    signals: _PreparedBundleSignals,
    target_generate: GenerateMetrics,
    published_cutoff_token_count: int | None,
    tail_valid_tokens: int | None,
) -> bool:
    if not signals.attached:
        return False
    if signals.fallback or signals.fail_closed or signals.consume_failed:
        return False
    expected_cached_tokens = _expected_cached_tokens(
        published_cutoff_token_count=published_cutoff_token_count,
        tail_valid_tokens=tail_valid_tokens,
    )
    if expected_cached_tokens is None:
        return True
    cached_tokens = _cached_tokens_from_metrics(target_generate)
    if cached_tokens is None:
        return False
    return cached_tokens == expected_cached_tokens


async def _run_prompt(
    *,
    config: CallerDriverConfig,
    runtime: Runtime,
    source_client: SGLangClient,
    target_client: SGLangClient,
    source_instance: Instance,
    target_instance: Instance,
    target_worker: Worker | None,
    prompt: RequestPrompt,
    target_log_path: Path,
) -> PromptTransferResult:
    logical_request_id = build_logical_request_id(config.run_id, prompt.prompt_id)
    source_metrics = build_empty_metrics("")
    target_metrics = build_empty_metrics("")
    publish_latency_ms: float | None = None
    publish_error_message = ""
    publish_manifest_digest = ""
    artifact_manifest_digest = ""
    published_cutoff_token_count: int | None = None
    tail_valid_tokens: int | None = None
    hydrate_latency_ms: float | None = None
    hydrate_error_message = ""
    warmup_attempted = bool(config.enable_target_worker_warmup)
    warmup_success: bool | None = None
    warmup_latency_ms: float | None = None
    warmup_error_message = ""
    prepared_signals = _PreparedBundleSignals()
    publish_result: PublishResult | None = None

    try:
        source_result = await source_client.generate_stream(
            prompt.prompt_text,
            sampling_params={
                "temperature": config.temperature,
                "max_new_tokens": config.max_new_tokens,
            },
            rid=logical_request_id,
        )
        source_metrics = source_result.to_metrics()
    except (SGLangHTTPError, TimeoutError) as exc:
        source_metrics = build_empty_metrics(str(exc))
    except Exception as exc:
        source_metrics = build_empty_metrics(str(exc))
    if not source_metrics.success:
        return PromptTransferResult(
            prompt_id=prompt.prompt_id,
            prompt_chars=prompt.prompt_filter_length,
            prompt_length=_prompt_length_from_metrics(source_metrics),
            logical_request_id=logical_request_id,
            source_generate=source_metrics,
            publish_success=False,
            publish_error_message="source generate failed",
            target_generate=target_metrics,
            status="source_generate_failed",
            error_message=source_metrics.error_message or "source generate failed",
        )

    try:
        publish_start = time.perf_counter()
        publish_result = _publish_plan_result(
            runtime,
            source_instance=source_instance,
            logical_request_id=logical_request_id,
            deadline_ms=config.plan_deadline_ms,
            ttl_ms=config.publish_ttl_ms,
        )
        publish_latency_ms = (time.perf_counter() - publish_start) * 1000.0
        _, local_publish_manifest = _decode_instance_publish_manifest(
            publish_result.publish_manifest
        )
        publish_manifest_digest = local_publish_manifest.publish_manifest_digest
        artifact_manifest_digest = (
            local_publish_manifest.artifact_manifest.artifact_manifest_digest
        )
        published_cutoff_token_count = local_publish_manifest.cutoff_token_count
        tail_valid_tokens = local_publish_manifest.tail_valid_tokens
    except Exception as exc:
        publish_error_message = _step_error_message(exc)
        return PromptTransferResult(
            prompt_id=prompt.prompt_id,
            prompt_chars=prompt.prompt_filter_length,
            prompt_length=_prompt_length_from_metrics(source_metrics),
            logical_request_id=logical_request_id,
            source_generate=source_metrics,
            publish_success=False,
            publish_latency_ms=publish_latency_ms,
            publish_error_message=publish_error_message,
            target_generate=target_metrics,
            status="publish_failed",
            error_message=publish_error_message,
        )

    try:
        hydrate_start = time.perf_counter()
        _ = _hydrate_plan_result(
            runtime,
            target_instance=target_instance,
            target_worker=target_worker,
            publish_result=publish_result,
            logical_request_id=logical_request_id,
            deadline_ms=config.plan_deadline_ms,
            enable_warmup=config.enable_target_worker_warmup,
        )
        hydrate_latency_ms = (time.perf_counter() - hydrate_start) * 1000.0
        if warmup_attempted:
            warmup_success = True
            warmup_latency_ms = hydrate_latency_ms
    except Exception as exc:
        hydrate_error_message = _step_error_message(exc)
        if warmup_attempted:
            warmup_success = False
            warmup_error_message = hydrate_error_message
        return PromptTransferResult(
            prompt_id=prompt.prompt_id,
            prompt_chars=prompt.prompt_filter_length,
            prompt_length=_prompt_length_from_metrics(source_metrics),
            logical_request_id=logical_request_id,
            source_generate=source_metrics,
            publish_success=True,
            publish_latency_ms=publish_latency_ms,
            publish_error_message="",
            publish_manifest_digest=publish_manifest_digest,
            artifact_manifest_digest=artifact_manifest_digest,
            published_cutoff_token_count=published_cutoff_token_count,
            tail_valid_tokens=tail_valid_tokens,
            warmup_attempted=warmup_attempted,
            warmup_success=warmup_success,
            warmup_latency_ms=warmup_latency_ms,
            warmup_error_message=warmup_error_message,
            hydrate_success=False,
            hydrate_latency_ms=hydrate_latency_ms,
            hydrate_error_message=hydrate_error_message,
            target_generate=target_metrics,
            status="hydrate_failed",
            error_message=hydrate_error_message,
        )

    target_log_start_offset = (
        target_log_path.stat().st_size if target_log_path.exists() else 0
    )
    try:
        target_result = await target_client.generate_stream(
            prompt.prompt_text,
            sampling_params={
                "temperature": config.temperature,
                "max_new_tokens": config.max_new_tokens,
            },
            rid=logical_request_id,
        )
        target_metrics = target_result.to_metrics()
    except (SGLangHTTPError, TimeoutError) as exc:
        target_metrics = build_empty_metrics(str(exc))
    except Exception as exc:
        target_metrics = build_empty_metrics(str(exc))

    if config.post_target_generate_settle_s > 0:
        await asyncio.sleep(config.post_target_generate_settle_s)
    prepared_signals = await _wait_for_prepared_bundle_signals(
        log_path=target_log_path,
        start_offset=target_log_start_offset,
        logical_request_id=logical_request_id,
        publish_manifest_digest=publish_manifest_digest,
        timeout_s=config.verify_log_timeout_s,
        poll_interval_s=config.verify_log_poll_interval_s,
    )
    target_cached_tokens = _cached_tokens_from_metrics(target_metrics)
    prepared_bundle_verified = _verify_prepared_bundle(
        signals=prepared_signals,
        target_generate=target_metrics,
        published_cutoff_token_count=published_cutoff_token_count,
        tail_valid_tokens=tail_valid_tokens,
    )
    if config.evict_after_prompt:
        _best_effort_evict(
            runtime,
            instance=target_instance,
            logical_request_id=logical_request_id,
            deadline_ms=config.plan_deadline_ms,
        )
        _best_effort_evict(
            runtime,
            instance=source_instance,
            logical_request_id=logical_request_id,
            deadline_ms=config.plan_deadline_ms,
        )

    if not target_metrics.success:
        return PromptTransferResult(
            prompt_id=prompt.prompt_id,
            prompt_chars=prompt.prompt_filter_length,
            prompt_length=_prompt_length_from_metrics(source_metrics),
            logical_request_id=logical_request_id,
            source_generate=source_metrics,
            publish_success=True,
            publish_latency_ms=publish_latency_ms,
            publish_error_message="",
            publish_manifest_digest=publish_manifest_digest,
            artifact_manifest_digest=artifact_manifest_digest,
            published_cutoff_token_count=published_cutoff_token_count,
            tail_valid_tokens=tail_valid_tokens,
            warmup_attempted=warmup_attempted,
            warmup_success=warmup_success,
            warmup_latency_ms=warmup_latency_ms,
            warmup_error_message=warmup_error_message,
            hydrate_success=True,
            hydrate_latency_ms=hydrate_latency_ms,
            hydrate_error_message="",
            target_generate=target_metrics,
            target_cached_tokens=target_cached_tokens,
            prepared_bundle_attached=prepared_signals.attached,
            prepared_bundle_fallback=prepared_signals.fallback,
            prepared_bundle_fail_closed=prepared_signals.fail_closed,
            prepared_bundle_consume_failed=prepared_signals.consume_failed,
            prepared_bundle_verified=prepared_bundle_verified,
            status="target_generate_failed",
            error_message=target_metrics.error_message or "target generate failed",
        )

    if not prepared_bundle_verified:
        return PromptTransferResult(
            prompt_id=prompt.prompt_id,
            prompt_chars=prompt.prompt_filter_length,
            prompt_length=_prompt_length_from_metrics(source_metrics),
            logical_request_id=logical_request_id,
            source_generate=source_metrics,
            publish_success=True,
            publish_latency_ms=publish_latency_ms,
            publish_error_message="",
            publish_manifest_digest=publish_manifest_digest,
            artifact_manifest_digest=artifact_manifest_digest,
            published_cutoff_token_count=published_cutoff_token_count,
            tail_valid_tokens=tail_valid_tokens,
            warmup_attempted=warmup_attempted,
            warmup_success=warmup_success,
            warmup_latency_ms=warmup_latency_ms,
            warmup_error_message=warmup_error_message,
            hydrate_success=True,
            hydrate_latency_ms=hydrate_latency_ms,
            hydrate_error_message="",
            target_generate=target_metrics,
            target_cached_tokens=target_cached_tokens,
            prepared_bundle_attached=prepared_signals.attached,
            prepared_bundle_fallback=prepared_signals.fallback,
            prepared_bundle_fail_closed=prepared_signals.fail_closed,
            prepared_bundle_consume_failed=prepared_signals.consume_failed,
            prepared_bundle_verified=False,
            status="verification_failed",
            error_message=(
                "prepared bundle verification failed"
                f" (expected cached_tokens={_expected_cached_tokens(published_cutoff_token_count=published_cutoff_token_count, tail_valid_tokens=tail_valid_tokens)},"
                f" got {target_cached_tokens})"
            ),
        )

    return PromptTransferResult(
        prompt_id=prompt.prompt_id,
        prompt_chars=prompt.prompt_filter_length,
        prompt_length=_prompt_length_from_metrics(source_metrics),
        logical_request_id=logical_request_id,
        source_generate=source_metrics,
        publish_success=True,
        publish_latency_ms=publish_latency_ms,
        publish_error_message="",
        publish_manifest_digest=publish_manifest_digest,
        artifact_manifest_digest=artifact_manifest_digest,
        published_cutoff_token_count=published_cutoff_token_count,
        tail_valid_tokens=tail_valid_tokens,
        warmup_attempted=warmup_attempted,
        warmup_success=warmup_success,
        warmup_latency_ms=warmup_latency_ms,
        warmup_error_message=warmup_error_message,
        hydrate_success=True,
        hydrate_latency_ms=hydrate_latency_ms,
        hydrate_error_message="",
        target_generate=target_metrics,
        target_cached_tokens=target_cached_tokens,
        prepared_bundle_attached=prepared_signals.attached,
        prepared_bundle_fallback=prepared_signals.fallback,
        prepared_bundle_fail_closed=prepared_signals.fail_closed,
        prepared_bundle_consume_failed=prepared_signals.consume_failed,
        prepared_bundle_verified=True,
        status="ok",
        error_message="",
    )


def _build_summary(
    *,
    config: CallerDriverConfig,
    results: list[PromptTransferResult],
) -> RequestTransferRunSummary:
    successful_results = [result for result in results if result.status == "ok"]
    publish_success_count = sum(1 for result in results if result.publish_success)
    hydrate_success_count = sum(1 for result in results if result.hydrate_success)
    prepared_bundle_verified_count = sum(
        1 for result in results if result.prepared_bundle_verified
    )
    warmup_success_count = sum(1 for result in results if result.warmup_success is True)
    source_ttft_values = [
        result.source_generate.ttft_ms
        for result in successful_results
        if result.source_generate.ttft_ms is not None
    ]
    target_ttft_values = [
        result.target_generate.ttft_ms
        for result in successful_results
        if result.target_generate.ttft_ms is not None
    ]
    publish_latency_values = [
        result.publish_latency_ms
        for result in results
        if result.publish_latency_ms is not None
    ]
    hydrate_latency_values = [
        result.hydrate_latency_ms
        for result in results
        if result.hydrate_latency_ms is not None
    ]
    target_cached_tokens = [
        float(result.target_cached_tokens)
        for result in successful_results
        if result.target_cached_tokens is not None
    ]
    prompt_lengths = [
        float(result.prompt_length) for result in results if result.prompt_length > 0
    ]
    observation = (
        "request_transfer correctness "
        f"success={len(successful_results)}/{len(results)} "
        f"publish={publish_success_count}/{len(results)} "
        f"hydrate={hydrate_success_count}/{len(results)} "
        f"prepared_bundle_verified={prepared_bundle_verified_count}/{len(results)}"
    )
    return RequestTransferRunSummary(
        run_id=config.run_id,
        topology_mode=config.topology_mode,
        prompt_count=len(results),
        avg_prompt_length=_mean(prompt_lengths),
        successful_prompts=len(successful_results),
        failed_prompts=len(results) - len(successful_results),
        publish_success_count=publish_success_count,
        hydrate_success_count=hydrate_success_count,
        prepared_bundle_verified_count=prepared_bundle_verified_count,
        mean_source_ttft_ms=_mean(
            [float(value) for value in source_ttft_values if value is not None]
        ),
        mean_target_ttft_ms=_mean(
            [float(value) for value in target_ttft_values if value is not None]
        ),
        mean_publish_latency_ms=_mean(
            [float(value) for value in publish_latency_values if value is not None]
        ),
        mean_hydrate_latency_ms=_mean(
            [float(value) for value in hydrate_latency_values if value is not None]
        ),
        mean_target_cached_tokens=_mean(target_cached_tokens),
        warmup_enabled=config.enable_target_worker_warmup,
        warmup_success_count=warmup_success_count,
        log_dir=str(Path(config.results_json_path).resolve().parent / "logs"),
        results_json_path=str(Path(config.results_json_path).resolve()),
        worker_process_a=config.worker_process_a,
        worker_process_b=config.worker_process_b or config.worker_process_a,
        worker_host_a=config.worker_host_a,
        worker_host_b=config.worker_host_b or config.worker_host_a,
        worker_ip_a=config.worker_ip_a,
        worker_ip_b=config.worker_ip_b or config.worker_ip_a,
        worker_node_a=config.worker_node_a,
        worker_node_b=config.worker_node_b or config.worker_node_a,
        model_path=config.model_path,
        observation=observation,
    )


async def _run_driver(config: CallerDriverConfig) -> RequestTransferRunSummary:
    prompts = load_prompts(
        config.dataset_path,
        config.prompt_count,
        min_prompt_chars=config.min_prompt_chars,
        max_prompt_chars=config.max_prompt_chars,
    )
    runtime = tc.connect(daemon_address=config.gateway_daemon_address)
    source_instance = _poll_instance_route(
        runtime,
        instance_id=config.source_instance_id,
        timeout_s=config.request_timeout_s,
        poll_interval_s=1.0,
    )
    target_instance = _poll_instance_route(
        runtime,
        instance_id=config.target_instance_id,
        timeout_s=config.request_timeout_s,
        poll_interval_s=1.0,
    )
    target_worker = _poll_worker_route_for_daemon(
        runtime,
        daemon_id=target_instance.daemon_id or "",
        timeout_s=config.request_timeout_s,
        poll_interval_s=1.0,
    )
    results: list[PromptTransferResult] = []
    target_log_path = Path(config.target_instance_log_path)

    async with (
        SGLangClient(
            config.source_instance_url,
            request_timeout_seconds=config.request_timeout_s,
        ) as source_client,
        SGLangClient(
            config.target_instance_url,
            request_timeout_seconds=config.request_timeout_s,
        ) as target_client,
    ):
        await source_client.wait_ready(
            timeout_seconds=config.request_timeout_s,
            poll_interval_seconds=1.0,
        )
        await target_client.wait_ready(
            timeout_seconds=config.request_timeout_s,
            poll_interval_seconds=1.0,
        )
        for prompt in prompts:
            log(f"Running prompt_id={prompt.prompt_id}")
            result = await _run_prompt(
                config=config,
                runtime=runtime,
                source_client=source_client,
                target_client=target_client,
                source_instance=source_instance,
                target_instance=target_instance,
                target_worker=(
                    target_worker if config.enable_target_worker_warmup else None
                ),
                prompt=prompt,
                target_log_path=target_log_path,
            )
            log(
                f"Prompt finished prompt_id={prompt.prompt_id} status={result.status} "
                f"publish_success={result.publish_success} hydrate_success={result.hydrate_success} "
                f"prepared_bundle_verified={result.prepared_bundle_verified}"
            )
            results.append(result)

    summary = _build_summary(config=config, results=results)
    write_jsonl(
        Path(config.results_json_path),
        [result.model_dump(mode="json") for result in results],
    )
    write_json(
        Path(config.summary_json_path),
        summary.model_dump(mode="json"),
    )
    runtime.close()
    return summary


def main() -> None:
    config = parse_args()
    summary = asyncio.run(_run_driver(config))
    log(summary.observation)


if __name__ == "__main__":
    main()
