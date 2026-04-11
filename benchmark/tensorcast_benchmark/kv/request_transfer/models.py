from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tensorcast_benchmark.kv.models import GenerateMetrics

TopologyMode = Literal["local", "remote"]


class RequestTransferBenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    topology_mode: TopologyMode = "local"
    model_path: str
    model_name: str = ""
    prompt_count: int = Field(default=10, ge=1)
    max_new_tokens: int = Field(default=32, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    tp_size: int = Field(default=2, ge=1)
    min_prompt_chars: int = Field(default=0, ge=0)
    max_prompt_chars: int = Field(default=0, ge=0)
    mem_fraction_static: float = Field(default=0.85, gt=0.0, lt=1.0)
    enable_hierarchical_cache: bool = True
    hicache_mem_layout: str = "page_blob_direct"
    hicache_io_backend: str = "direct"
    hicache_ratio: float = Field(default=2.0, gt=0.0)
    hicache_size_gb: int = Field(default=0, ge=0)
    hicache_storage_prefetch_policy: str = "best_effort"
    request_timeout_s: float = Field(default=600.0, gt=0.0)
    instance_ready_timeout_s: float = Field(default=1800.0, gt=0.0)
    instance_health_poll_interval_s: float = Field(default=1.0, gt=0.0)
    plan_deadline_ms: int = Field(default=15_000, gt=0)
    publish_ttl_ms: int = Field(default=60_000, gt=0)
    enable_target_worker_warmup: bool = False
    verify_log_timeout_s: float = Field(default=15.0, gt=0.0)
    verify_log_poll_interval_s: float = Field(default=0.25, gt=0.0)
    post_target_generate_settle_s: float = Field(default=1.0, ge=0.0)
    evict_after_prompt: bool = True
    port_a: int = Field(default=34000, ge=1, le=65535)
    port_b: int = Field(default=34001, ge=1, le=65535)
    instance_a_cuda_visible_devices: str = "0,1"
    instance_b_cuda_visible_devices: str = "2,3"
    trust_remote_code: bool = False
    extra_server_args: str = "--log-level debug"
    data_path: str = (
        "/home/i-zhouyuhan/tot/thirdparty/sglang/benchmark/"
        "tensorcast_benchmark/kv/dataset/LongBench/hotpotqa.jsonl"
    )

    keep_worker: bool = True
    existing_worker_process_a: str = ""
    existing_worker_process_b: str = ""
    brainctl_namespace: str = "shai-core"
    brainctl_charged_group: str = Field(
        default_factory=lambda: os.environ.get("BRAINCTL_CHARGED_GROUP", "").strip()
    )
    brainctl_private_machine: str = "group"
    brainctl_mount: str = (
        "juicefs+s3://oss.i.shaipower.com/step2-alignment-jfs:/mnt/step2-alignment-jfs"
    )
    brainctl_max_wait_duration: str = "10m"
    worker_ready_timeout_s: float = Field(default=900.0, gt=0.0)
    worker_poll_interval_s: float = Field(default=5.0, gt=0.0)
    worker_gpu: int = Field(default=4, ge=1)
    worker_cpu: int = Field(default=64, ge=1)
    worker_memory: int = Field(default=500_000, ge=1)
    worker_positive_tags: str = "H800"
    worker_negative_tags: str = ""

    tensorcast_namespace: str = "request_transfer"
    tensorcast_global_store_port: int = Field(default=50051, ge=1, le=65535)
    tensorcast_daemon_port_a: int = Field(default=50052, ge=1, le=65535)
    tensorcast_daemon_port_b: int = Field(default=50053, ge=1, le=65535)
    tensorcast_instance_agent_port_a: int = Field(default=34110, ge=1, le=65535)
    tensorcast_instance_agent_port_b: int = Field(default=34111, ge=1, le=65535)
    tensorcast_daemon_p2p_port_a: int = Field(default=65090, ge=1, le=65535)
    tensorcast_daemon_p2p_port_b: int = Field(default=65091, ge=1, le=65535)
    tensorcast_source_prefetch_threshold: int = Field(default=1, ge=1)
    tensorcast_target_prefetch_threshold: int = Field(default=1_000_000, ge=1)
    tensorcast_service_ready_timeout_s: float = Field(default=120.0, gt=0.0)
    tensorcast_service_poll_interval_s: float = Field(default=2.0, gt=0.0)
    tensorcast_cuda_home: str = "/usr/local/cuda-12.4"
    tensorcast_nvidia_lib_dirs: str = (
        "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
    )
    tensorcast_daemon_stable_bytes: str = "64GB"
    tensorcast_byte_artifact_shard_count: int = Field(default=8, ge=1)
    tensorcast_byte_artifact_lease_ttl_s: float = Field(default=30.0, gt=0.0)
    tensorcast_byte_artifact_keepalive_interval_s: float = Field(default=10.0, gt=0.0)
    tensorcast_payload_max_chunk_bytes: int = Field(default=(1 << 20), ge=1)
    tensorcast_max_batch_payload_bytes: int = Field(default=(16 << 20), ge=1)
    tensorcast_host_allocator_enabled: bool = True
    tensorcast_host_allocator_region_ttl_ms: int = Field(default=0, ge=0)
    tensorcast_host_allocator_region_name: str = "sglang_tensorcast_host_pool"

    @model_validator(mode="after")
    def validate_config(self) -> "RequestTransferBenchmarkConfig":
        data_path = Path(self.data_path).expanduser()
        if not data_path.is_file():
            raise ValueError(f"data_path does not exist: {self.data_path}")
        if self.max_prompt_chars > 0 and self.max_prompt_chars < self.min_prompt_chars:
            raise ValueError(
                "max_prompt_chars must be 0 or greater than or equal to min_prompt_chars"
            )
        if self.tensorcast_host_allocator_enabled:
            if self.hicache_mem_layout != "page_blob_direct":
                raise ValueError(
                    "tensorcast_host_allocator_enabled requires hicache_mem_layout=page_blob_direct"
                )
            if self.hicache_io_backend != "direct":
                raise ValueError(
                    "tensorcast_host_allocator_enabled requires hicache_io_backend=direct"
                )
        if self.topology_mode == "local" and self.existing_worker_process_b.strip():
            raise ValueError(
                "existing_worker_process_b is only valid when topology_mode=remote"
            )
        needs_launch = self.existing_worker_process_a.strip() == "" or (
            self.topology_mode == "remote"
            and self.existing_worker_process_b.strip() == ""
        )
        if needs_launch and self.brainctl_charged_group == "":
            raise ValueError(
                "brainctl_charged_group is required unless all required workers are provided"
            )
        return self


class PromptTransferResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    prompt_chars: int
    prompt_length: int
    logical_request_id: str
    source_generate: GenerateMetrics
    publish_success: bool
    publish_latency_ms: float | None = None
    publish_error_message: str = ""
    publish_manifest_digest: str = ""
    artifact_manifest_digest: str = ""
    published_cutoff_token_count: int | None = None
    tail_valid_tokens: int | None = None
    warmup_attempted: bool = False
    warmup_success: bool | None = None
    warmup_latency_ms: float | None = None
    warmup_error_message: str = ""
    hydrate_success: bool = False
    hydrate_latency_ms: float | None = None
    hydrate_error_message: str = ""
    target_generate: GenerateMetrics
    target_cached_tokens: int | None = None
    prepared_bundle_attached: bool = False
    prepared_bundle_fallback: bool = False
    prepared_bundle_fail_closed: bool = False
    prepared_bundle_consume_failed: bool = False
    prepared_bundle_verified: bool = False
    status: str
    error_message: str = ""


class RequestTransferRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    topology_mode: TopologyMode
    prompt_count: int
    avg_prompt_length: float | None = None
    successful_prompts: int
    failed_prompts: int
    publish_success_count: int
    hydrate_success_count: int
    prepared_bundle_verified_count: int
    mean_source_ttft_ms: float | None = None
    mean_target_ttft_ms: float | None = None
    mean_publish_latency_ms: float | None = None
    mean_hydrate_latency_ms: float | None = None
    mean_target_cached_tokens: float | None = None
    warmup_enabled: bool
    warmup_success_count: int
    log_dir: str
    results_json_path: str
    worker_process_a: str
    worker_process_b: str
    worker_host_a: str
    worker_host_b: str
    worker_ip_a: str
    worker_ip_b: str
    worker_node_a: str
    worker_node_b: str
    model_path: str
    observation: str
