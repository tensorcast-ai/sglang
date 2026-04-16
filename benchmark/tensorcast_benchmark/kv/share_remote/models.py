from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


BackendKind = Literal["tensorcast", "mooncake"]
RDMASmokeMode = Literal["star"]


class WorkerShape(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    gpu: int = Field(default=4, ge=1)
    cpu: int = Field(default=64, ge=1)
    memory: int = Field(default=500_000, ge=1)
    positive_tags: str = "H800"
    negative_tags: str = ""


class WorkerEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    process_name: str = ""
    gpu: int | None = Field(default=None, ge=1)
    cpu: int | None = Field(default=None, ge=1)
    memory: int | None = Field(default=None, ge=1)
    positive_tags: str | None = None
    negative_tags: str | None = None

    def resolve(
        self,
        *,
        index: int,
        default_shape: WorkerShape,
        process_name_override: str = "",
    ) -> "ResolvedWorkerSpec":
        effective_process_name = process_name_override.strip() or self.process_name.strip()
        return ResolvedWorkerSpec(
            index=index,
            process_name=effective_process_name,
            gpu=self.gpu if self.gpu is not None else default_shape.gpu,
            cpu=self.cpu if self.cpu is not None else default_shape.cpu,
            memory=self.memory if self.memory is not None else default_shape.memory,
            positive_tags=(
                self.positive_tags
                if self.positive_tags is not None
                else default_shape.positive_tags
            ),
            negative_tags=(
                self.negative_tags
                if self.negative_tags is not None
                else default_shape.negative_tags
            ),
        )


class ResolvedWorkerSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    process_name: str = ""
    gpu: int = Field(ge=1)
    cpu: int = Field(ge=1)
    memory: int = Field(ge=1)
    positive_tags: str = ""
    negative_tags: str = ""


class WorkersConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int = Field(default=0, ge=0)
    entries: tuple[WorkerEntry, ...] = ()
    existing_worker_processes: tuple[str, ...] = ()
    keep_workers: bool = True
    require_distinct_nodes: bool = True
    rdma_required: bool = True
    verify_rdma_on_reuse: bool = False
    service_host_worker_index: int = Field(default=0, ge=0)
    default_shape: WorkerShape = Field(default_factory=WorkerShape)

    @model_validator(mode="after")
    def validate_worker_layout(self) -> "WorkersConfig":
        if not self.entries and self.count <= 0:
            raise ValueError("workers.count must be positive when workers.entries is empty")
        if self.entries and self.count not in (0, len(self.entries)):
            raise ValueError(
                "workers.count must be 0 or equal to len(workers.entries)"
            )
        resolved_count = self.resolved_count()
        if len(self.existing_worker_processes) > resolved_count:
            raise ValueError(
                "len(workers.existing_worker_processes) must be less than or equal to "
                "the resolved worker count"
            )
        if self.service_host_worker_index >= resolved_count:
            raise ValueError(
                "workers.service_host_worker_index must be smaller than worker count"
            )
        return self

    def resolved_count(self) -> int:
        if self.entries:
            return len(self.entries)
        return self.count

    def resolved_worker_specs(self) -> tuple[ResolvedWorkerSpec, ...]:
        specs: list[ResolvedWorkerSpec] = []
        if self.entries:
            base_entries = list(self.entries)
        else:
            base_entries = [WorkerEntry() for _ in range(self.count)]
        for index, entry in enumerate(base_entries):
            override = ""
            if index < len(self.existing_worker_processes):
                override = self.existing_worker_processes[index]
            if (
                override.strip()
                and entry.process_name.strip()
                and override.strip() != entry.process_name.strip()
            ):
                raise ValueError(
                    f"workers.entries[{index}].process_name conflicts with "
                    "workers.existing_worker_processes"
                )
            specs.append(
                entry.resolve(
                    index=index,
                    default_shape=self.default_shape,
                    process_name_override=override,
                )
            )
        return tuple(specs)


class BrainctlConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: str = "shai-core"
    charged_group: str = Field(
        default_factory=lambda: os.environ.get("BRAINCTL_CHARGED_GROUP", "").strip()
    )
    private_machine: str = "group"
    mount: str = (
        "juicefs+s3://oss.i.shaipower.com/step2-alignment-jfs:"
        "/mnt/step2-alignment-jfs"
    )
    max_wait_duration: str = "10m"
    worker_ready_timeout_s: float = Field(default=900.0, gt=0.0)
    worker_poll_interval_s: float = Field(default=5.0, gt=0.0)


class TransportConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    use_rdma: bool = False
    rdma_smoke_mode: RDMASmokeMode = "star"
    rdma_gid_index: int = Field(default=3, ge=0)
    ib_write_bw_fixed_port: int = Field(default=18517, ge=1, le=65535)
    ib_write_bw_sweep_port: int = Field(default=18516, ge=1, le=65535)
    ib_write_bw_server_ready_sleep_s: float = Field(default=2.0, gt=0.0)


class WorkloadConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_path: str
    model_name: str = ""
    instance_port: int = Field(default=35000, ge=1, le=65535)
    tp_size: int = Field(default=2, ge=1)
    page_size: int = Field(default=32, ge=1)
    prompt_count: int = Field(default=10, ge=1)
    rps: float = Field(default=0.5, gt=0.0)
    settle_ms: int = Field(default=20_000, ge=0)
    max_new_tokens: int = Field(default=32, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    data_path: str = (
        "/home/i-zhouyuhan/tot/thirdparty/sglang/benchmark/"
        "tensorcast_benchmark/kv/dataset/LongBench/hotpotqa.jsonl"
    )
    min_prompt_chars: int = Field(default=0, ge=0)
    max_prompt_chars: int = Field(default=0, ge=0)
    request_timeout_s: float = Field(default=600.0, gt=0.0)
    instance_ready_timeout_s: float = Field(default=1800.0, gt=0.0)
    instance_health_poll_interval_s: float = Field(default=1.0, gt=0.0)
    mem_fraction_static: float = Field(default=0.85, gt=0.0, lt=1.0)
    enable_hierarchical_cache: bool = True
    hicache_mem_layout: str = "page_blob_direct"
    hicache_io_backend: str = "direct"
    hicache_ratio: float = Field(default=2.0, gt=0.0)
    hicache_size_gb: int = Field(default=0, ge=0)
    hicache_storage_prefetch_policy: str = "wait_complete"
    trust_remote_code: bool = False
    extra_server_args: str = "--log-level debug"

    @model_validator(mode="after")
    def validate_paths(self) -> "WorkloadConfig":
        dataset_path = Path(self.data_path).expanduser()
        if not dataset_path.is_file():
            raise ValueError(f"workload.data_path does not exist: {self.data_path}")
        if (
            self.max_prompt_chars > 0
            and self.max_prompt_chars < self.min_prompt_chars
        ):
            raise ValueError(
                "workload.max_prompt_chars must be 0 or greater than or equal to "
                "workload.min_prompt_chars"
            )
        return self


class TensorcastBackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: str = "share_remote"
    global_store_port: int = Field(default=50051, ge=1, le=65535)
    daemon_port: int = Field(default=50052, ge=1, le=65535)
    daemon_p2p_port: int = Field(default=65090, ge=1, le=65535)
    service_ready_timeout_s: float = Field(default=120.0, gt=0.0)
    service_poll_interval_s: float = Field(default=2.0, gt=0.0)
    cuda_home: str = "/usr/local/cuda-12.4"
    nvidia_lib_dirs: str = "/usr/local/cuda-12.9/compat:/usr/local/nvidia/lib64"
    daemon_stable_bytes: str = "64GB"
    byte_artifact_shard_count: int = Field(default=8, ge=1)
    byte_artifact_lease_ttl_s: float = Field(default=30.0, gt=0.0)
    byte_artifact_keepalive_interval_s: float = Field(default=10.0, gt=0.0)
    payload_max_chunk_bytes: int = Field(default=(1 << 20), ge=1)
    max_batch_payload_bytes: int = Field(default=(16 << 20), ge=1)
    prefetch_threshold: int = Field(default=1, ge=1)
    host_allocator_enabled: bool = True
    host_allocator_region_ttl_ms: int = Field(default=0, ge=0)
    host_allocator_region_name: str = "sglang_tensorcast_host_pool"


class MooncakeBackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    http_metadata_server_port: int = Field(default=8080, ge=1, le=65535)
    master_port: int = Field(default=60051, ge=1, le=65535)
    prefetch_threshold: int = Field(default=1, ge=1)
    protocol: str = ""
    device_name: str = ""
    global_segment_size: str = "64gb"
    local_buffer_size: int = Field(default=0, ge=0)
    eviction_high_watermark_ratio: float = Field(default=0.9, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_protocol(self) -> "MooncakeBackendConfig":
        if self.protocol and self.protocol not in {"tcp", "rdma"}:
            raise ValueError("backend_config.mooncake.protocol must be tcp or rdma")
        return self


class BackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tensorcast: TensorcastBackendConfig = Field(default_factory=TensorcastBackendConfig)
    mooncake: MooncakeBackendConfig = Field(default_factory=MooncakeBackendConfig)


class ShareRemoteBenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: BackendKind
    workers: WorkersConfig
    brainctl: BrainctlConfig = Field(default_factory=BrainctlConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    workload: WorkloadConfig
    backend_config: BackendConfig = Field(default_factory=BackendConfig)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "ShareRemoteBenchmarkConfig":
        resolved_worker_specs = self.workers.resolved_worker_specs()
        if len(resolved_worker_specs) < 2:
            raise ValueError("share_remote requires at least two workers")
        for spec in resolved_worker_specs:
            if spec.gpu < self.workload.tp_size:
                raise ValueError(
                    f"worker {spec.index} requests {spec.gpu} GPUs but workload.tp_size="
                    f"{self.workload.tp_size}"
                )
        if self.backend == "tensorcast":
            if self.workload.hicache_mem_layout != "page_blob_direct":
                raise ValueError(
                    "tensorcast backend requires "
                    "workload.hicache_mem_layout=page_blob_direct"
                )
            if self.workload.hicache_io_backend != "direct":
                raise ValueError(
                    "tensorcast backend requires workload.hicache_io_backend=direct"
                )
        if self.backend == "mooncake":
            if self.workload.hicache_mem_layout not in {
                "page_first",
                "page_first_direct",
            }:
                raise ValueError(
                    "mooncake backend requires workload.hicache_mem_layout to be "
                    "page_first or page_first_direct"
                )
        protocol = self.backend_config.mooncake.protocol.strip()
        if protocol:
            expected_protocol = "rdma" if self.transport.use_rdma else "tcp"
            if protocol != expected_protocol:
                raise ValueError(
                    "backend_config.mooncake.protocol conflicts with transport.use_rdma"
                )
        needs_launch = any(not spec.process_name.strip() for spec in resolved_worker_specs)
        if needs_launch and not self.brainctl.charged_group.strip():
            raise ValueError(
                "brainctl.charged_group is required unless all workers are reused"
            )
        return self

    def resolved_worker_specs(self) -> tuple[ResolvedWorkerSpec, ...]:
        return self.workers.resolved_worker_specs()


class ShareRemotePaths(BaseModel):
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
    config_copy_path: Path
    resolved_config_path: Path
    driver_config_path: Path
    worker_inventory_path: Path
    venv_python: Path
    uv_bin: Path
    mooncake_master_bin: Path
    scripts_dir: Path


class WorkerDirectoryInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    process_name: str
    hostname: str
    creator: str
    ready: str
    status: str
    ip: str
    node: str


class RDMAGpuCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    gpu_label: str
    nic_label: str
    hca_name: str
    netdev: str
    distance: str


class WorkerRDMAInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    socket_ifname: str = ""
    nccl_ib_hca_raw: str = ""
    nccl_ib_hca_exact: str = ""
    preferred_ib_device: str = ""
    gpu_candidates: tuple[RDMAGpuCandidate, ...] = ()


class WorkerInventoryRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    process_name: str
    hostname: str
    ip: str
    node: str
    launched_in_run: bool
    gpu: int = Field(ge=1)
    cpu: int = Field(ge=1)
    memory: int = Field(ge=1)
    positive_tags: str = ""
    negative_tags: str = ""
    rdma: WorkerRDMAInfo | None = None


class DriverInstanceTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    worker_process: str
    worker_host: str
    worker_ip: str
    worker_node: str
    instance_url: str
    instance_log_path: str


class ShareRemoteDriverConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    backend: BackendKind
    dataset_path: str = Field(min_length=1)
    prompt_count: int = Field(ge=1)
    min_prompt_chars: int = Field(default=0, ge=0)
    max_prompt_chars: int = Field(default=0, ge=0)
    rps: float = Field(gt=0.0)
    settle_ms: int = Field(default=20_000, ge=0)
    max_new_tokens: int = Field(ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    request_timeout_s: float = Field(default=600.0, gt=0.0)
    results_json_path: str = Field(min_length=1)
    summary_json_path: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    tp_size: int = Field(ge=1)
    transport_use_rdma: bool = False
    service_host_worker_index: int = Field(default=0, ge=0)
    instance_targets: tuple[DriverInstanceTarget, ...]

    @model_validator(mode="after")
    def validate_driver_config(self) -> "ShareRemoteDriverConfig":
        dataset_path = Path(self.dataset_path).expanduser()
        if not dataset_path.is_file():
            raise ValueError(f"dataset_path does not exist: {self.dataset_path}")
        if not self.instance_targets:
            raise ValueError("instance_targets must not be empty")
        if self.max_prompt_chars > 0 and self.max_prompt_chars < self.min_prompt_chars:
            raise ValueError(
                "max_prompt_chars must be 0 or greater than or equal to "
                "min_prompt_chars"
            )
        if self.service_host_worker_index >= len(self.instance_targets):
            raise ValueError(
                "service_host_worker_index must be smaller than len(instance_targets)"
            )
        return self


class InstanceRequestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: int = Field(ge=0)
    worker_index: int = Field(ge=0)
    worker_process: str
    worker_host: str
    worker_ip: str
    worker_node: str
    instance_url: str
    rid: str
    success: bool
    ttft_ms: float | None = None
    latency_ms: float | None = None
    cached_tokens: int | None = None
    text: str = ""
    meta_info: dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""


class PromptGroupResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    prompt_chars: int
    prompt_length: int
    backend: BackendKind
    group_index: int = Field(ge=0)
    instance_results: tuple[InstanceRequestResult, ...]
    status: str
    error_message: str = ""


class ShareRemoteRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    backend: BackendKind
    transport_use_rdma: bool
    worker_count: int
    service_host_worker_index: int
    prompt_count: int
    avg_prompt_length: float | None = None
    successful_prompt_groups: int
    failed_prompt_groups: int
    mean_ttft_by_position: tuple[float | None, ...]
    median_ttft_by_position: tuple[float | None, ...]
    p95_ttft_by_position: tuple[float | None, ...]
    mean_cached_tokens_by_position: tuple[float | None, ...]
    mean_improvement_vs_first_ms_by_position: tuple[float | None, ...]
    log_dir: str
    results_json_path: str
    worker_processes: tuple[str, ...]
    worker_hosts: tuple[str, ...]
    worker_ips: tuple[str, ...]
    worker_nodes: tuple[str, ...]
    model_path: str
    tp_size: int
    observation: str
