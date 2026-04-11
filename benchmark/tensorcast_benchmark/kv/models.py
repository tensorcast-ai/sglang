from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


HicacheStorageBackend = Literal["mooncake", "tensorcast"]
TensorcastDaemonMode = Literal["share", "separate"]


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_path: str
    model_name: str = ""
    hicache_storage_backend: HicacheStorageBackend = "mooncake"
    tensorcast_daemon_mode: TensorcastDaemonMode = "share"
    prompt_count: int = Field(default=10, ge=1)
    pair_rps: float = Field(default=1.0, gt=0.0)
    settle_ms: int = Field(default=1000, ge=0)
    max_new_tokens: int = Field(default=32, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    tp_size: int = Field(default=2, ge=1)
    min_prompt_chars: int = Field(default=0, ge=0)
    max_prompt_chars: int = Field(default=0, ge=0)
    mem_fraction_static: float = Field(default=0.85, gt=0.0, lt=1.0)
    enable_hierarchical_cache: bool = True
    hicache_mem_layout: str = "page_first"
    hicache_io_backend: str = "kernel"
    hicache_ratio: float = Field(default=2.0, gt=0.0)
    hicache_size_gb: int = Field(default=0, ge=0)
    hicache_storage_prefetch_policy: str = "best_effort"
    request_timeout_s: float = Field(default=600.0, gt=0.0)
    instance_ready_timeout_s: float = Field(default=1800.0, gt=0.0)
    instance_health_poll_interval_s: float = Field(default=1.0, gt=0.0)
    wait_for_source_publication_drain: bool = False
    source_publication_drain_timeout_s: float = Field(default=120.0, gt=0.0)
    source_publication_drain_idle_s: float = Field(default=5.0, gt=0.0)
    source_publication_drain_poll_s: float = Field(default=0.25, gt=0.0)
    require_positive_ttft_improvement: bool = False
    port_a: int = Field(default=31000, ge=1, le=65535)
    port_b: int = Field(default=31001, ge=1, le=65535)
    host: str = "127.0.0.1"
    instance_a_cuda_visible_devices: str = "0,1,2,3"
    instance_b_cuda_visible_devices: str = "4,5,6,7"
    trust_remote_code: bool = False
    extra_server_args: str = ""
    data_path: str = "/home/i-zhouyuhan/tot/data/test.jsonl"
    keep_worker: bool = True
    existing_worker_process: str = ""

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
    worker_gpu: int = Field(default=8, ge=1)
    worker_cpu: int = Field(default=128, ge=1)
    worker_memory: int = Field(default=1000000, ge=1)
    worker_positive_tags: str = "H800"
    worker_negative_tags: str = ""

    mooncake_http_metadata_server_port: int = Field(default=8080, ge=1, le=65535)
    mooncake_master_port: int = Field(default=60051, ge=1, le=65535)
    mooncake_protocol: str = "tcp"
    mooncake_device_name: str = ""
    mooncake_global_segment_size: str = "4gb"
    mooncake_local_buffer_size: int = Field(default=0, ge=0)
    mooncake_eviction_high_watermark_ratio: float = Field(default=0.9, gt=0.0, lt=1.0)

    tensorcast_global_store_port: int = Field(default=50051, ge=1, le=65535)
    tensorcast_daemon_port_a: int = Field(default=50052, ge=1, le=65535)
    tensorcast_daemon_port_b: int = Field(default=50053, ge=1, le=65535)
    tensorcast_instance_agent_port_a: int = Field(default=31110, ge=1, le=65535)
    tensorcast_instance_agent_port_b: int = Field(default=31111, ge=1, le=65535)
    tensorcast_daemon_p2p_port_a: int = Field(default=65090, ge=1, le=65535)
    tensorcast_daemon_p2p_port_b: int = Field(default=65091, ge=1, le=65535)
    tensorcast_prefetch_threshold: int = Field(default=1, ge=1)
    tensorcast_service_ready_timeout_s: float = Field(default=120.0, gt=0.0)
    tensorcast_service_poll_interval_s: float = Field(default=2.0, gt=0.0)
    tensorcast_cuda_home: str = "/usr/local/cuda-12.4"
    tensorcast_nvidia_lib_dirs: str = (
        "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
    )
    tensorcast_daemon_stable_bytes: str = "64GB"
    tensorcast_byte_artifact_shard_count: int = Field(default=8, ge=1)
    tensorcast_byte_artifact_lease_ttl_s: float = Field(default=30.0, gt=0.0)
    tensorcast_byte_artifact_keepalive_interval_s: float = Field(
        default=5.0, gt=0.0
    )
    tensorcast_payload_max_chunk_bytes: int = Field(default=(1 << 20), ge=1)
    tensorcast_max_batch_payload_bytes: int = Field(default=(16 << 20), ge=1)
    tensorcast_host_allocator_enabled: bool = False
    tensorcast_host_allocator_region_ttl_ms: int = Field(default=0, ge=0)
    tensorcast_host_allocator_region_name: str = "sglang_tensorcast_host_pool"

    @model_validator(mode="after")
    def validate_config(self) -> "BenchmarkConfig":
        if not Path(self.data_path).expanduser().is_file():
            raise ValueError(f"data_path does not exist: {self.data_path}")
        if self.max_prompt_chars > 0 and self.max_prompt_chars < self.min_prompt_chars:
            raise ValueError(
                "max_prompt_chars must be 0 or greater than or equal to min_prompt_chars"
            )
        if (
            self.hicache_storage_backend == "mooncake"
            and self.hicache_mem_layout
            not in {
                "page_first",
                "page_first_direct",
            }
        ):
            raise ValueError(
                "mooncake backend requires hicache_mem_layout to be page_first or "
                f"page_first_direct, got {self.hicache_mem_layout!r}"
            )
        if (
            self.hicache_storage_backend == "tensorcast"
            and self.tensorcast_daemon_mode not in {"share", "separate"}
        ):
            raise ValueError(
                "tensorcast_daemon_mode must be share or separate when backend is tensorcast"
            )
        if self.tensorcast_host_allocator_enabled:
            if self.hicache_storage_backend != "tensorcast":
                raise ValueError(
                    "tensorcast_host_allocator_enabled requires hicache_storage_backend=tensorcast"
                )
            if self.hicache_mem_layout != "page_blob_direct":
                raise ValueError(
                    "tensorcast_host_allocator_enabled requires hicache_mem_layout=page_blob_direct"
                )
            if self.hicache_io_backend != "direct":
                raise ValueError(
                    "tensorcast_host_allocator_enabled requires hicache_io_backend=direct"
                )
        if (
            self.existing_worker_process.strip() == ""
            and self.brainctl_charged_group == ""
        ):
            raise ValueError(
                "brainctl_charged_group is required unless existing_worker_process is provided"
            )
        return self


class BenchmarkPaths(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_root: Path
    benchmark_name: str
    kv_root: Path
    sglang_root: Path
    workspace_root: Path
    outputs_dir: Path
    run_dir: Path
    logs_dir: Path
    generated_configs_dir: Path
    results_json_path: Path
    summary_json_path: Path
    csv_path: Path
    orchestrator_log_path: Path
    venv_python: Path
    uv_bin: Path
    mooncake_master_bin: Path
    scripts_dir: Path


class WorkerInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    process_name: str
    hostname: str
    creator: str
    ready: str
    status: str
    ip: str
    node: str


class RequestPrompt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_id: str
    prompt_text: str
    prompt_filter_length: int = Field(ge=0)


class GenerateMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool = False
    text: str = ""
    ttft_ms: float | None = None
    latency_ms: float | None = None
    meta_info: dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""


class PairResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    prompt_chars: int
    prompt_length: int
    backend: HicacheStorageBackend
    tensorcast_daemon_mode: TensorcastDaemonMode | None = None
    instance_a: GenerateMetrics
    instance_b: GenerateMetrics
    ttft_improvement_ms: float | None = None
    ttft_speedup_ratio: float | None = None
    source_publication_drain_ms: float | None = None
    source_publication_wait_ms: float | None = None
    source_publication_post_completion_upload_count: int = 0
    source_publication_drain_timed_out: bool = False
    status: str
    error_message: str = ""


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    backend: HicacheStorageBackend
    tensorcast_daemon_mode: TensorcastDaemonMode | None = None
    prompt_count: int
    avg_prompt_length: float | None = None
    success_pairs: int
    failed_pairs: int
    mean_instance_a_ttft_ms: float | None = None
    mean_instance_b_ttft_ms: float | None = None
    median_instance_a_ttft_ms: float | None = None
    median_instance_b_ttft_ms: float | None = None
    p95_instance_a_ttft_ms: float | None = None
    p95_instance_b_ttft_ms: float | None = None
    mean_ttft_improvement_ms: float | None = None
    median_ttft_improvement_ms: float | None = None
    p95_ttft_improvement_ms: float | None = None
    mean_ttft_speedup_ratio: float | None = None
    mean_source_publication_drain_ms: float | None = None
    median_source_publication_drain_ms: float | None = None
    p95_source_publication_drain_ms: float | None = None
    mean_source_publication_wait_ms: float | None = None
    source_publication_drain_timeout_count: int = 0
    log_dir: str
    results_json_path: str
    worker_process: str
    worker_host: str
    worker_ip: str
    worker_node: str
    model_path: str
    tp_size: int
    observation: str = ""
