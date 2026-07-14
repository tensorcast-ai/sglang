"""Pydantic models for `benchmark.yaml`.

Schema mirrors arch § 9.2. `cluster.yaml` uses a separate Pydantic model
in `resource.base.ClusterConfig` (already implemented in Phase 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    tp_size: int = Field(default=1, ge=1)


class InstancesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int = Field(ge=1)
    base_port: int = Field(default=30001, ge=1024, le=65535)
    # `auto` lets each instance use SGLang's default sizing. Numeric values
    # are reserved for a later mem-fraction-static computation.
    kv_pool_size_gb: str | int = "auto"
    mem_fraction_static: float = Field(default=0.85, gt=0.0, lt=1.0)
    page_size: int = Field(default=32, ge=1)
    sglang_log_level: Optional[
        Literal["debug", "info", "warning", "error", "critical"]
    ] = None


class TransportConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    use_rdma: bool = True


class PoolFilterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_turns: int = Field(default=8, ge=1)
    min_total_tokens: int = Field(default=8000, ge=0)


class InterTurnDelayConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    preset: Literal["agent_fast", "agent_medium", "agent_slow", "custom"] = (
        "agent_medium"
    )
    custom_mu: Optional[float] = None
    custom_sigma: Optional[float] = None

    @model_validator(mode="after")
    def _custom_requires_params(self) -> "InterTurnDelayConfig":
        if self.preset == "custom":
            if self.custom_mu is None or self.custom_sigma is None:
                raise ValueError(
                    "preset=custom requires both custom_mu and custom_sigma"
                )
        return self


class WorkloadConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_path: str = Field(min_length=1)
    pool_filter: PoolFilterConfig = Field(default_factory=PoolFilterConfig)
    inter_turn_delay: InterTurnDelayConfig = Field(default_factory=InterTurnDelayConfig)
    max_new_tokens_clip: int = Field(default=512, ge=1)
    start_jitter_s: float = Field(default=2.0, ge=0.0)
    wall_seconds: float = Field(default=600.0, gt=0.0)
    warmup_seconds: float = Field(default=0.0, ge=0.0)
    warmup_counts: int = Field(default=0, ge=0)
    trials: int = Field(default=1, ge=1)
    c_target_sweep: tuple[int, ...] = Field(min_length=1)


# Per-config policy / extras. Phase 5 only consumes `gw_load_aware` and
# `gw_cache_aware`; later phases extend.
class ConfigSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[
        "gw_load_aware",
        "gw_cache_aware",
        "gw_load_aware_mooncake",
        "tc_router",
    ]
    # Free-form provider-specific knobs. tc_router uses `policy.kind=threshold`.
    policy: Optional[dict] = None


class GatewayConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=30100, ge=1024, le=65535)


class MooncakeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    http_metadata_server_port: int = Field(default=62300, ge=1024, le=65535)
    master_port: int = Field(default=62301, ge=1024, le=65535)
    global_segment_size: str = Field(default="64gb", min_length=1)
    eviction_high_watermark_ratio: float = Field(default=0.9, gt=0.0, lt=1.0)
    device_name: str = ""
    clear_storage_between_cells: bool = True

    @model_validator(mode="after")
    def _ports_must_be_distinct(self) -> "MooncakeConfig":
        if self.http_metadata_server_port == self.master_port:
            raise ValueError(
                "mooncake.http_metadata_server_port and mooncake.master_port "
                "must be distinct"
            )
        return self


class TensorcastConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    global_store_port: int = Field(default=61050, ge=1024, le=65535)
    daemon_port: int = Field(default=61053, ge=1024, le=65535)
    daemon_p2p_port: int = Field(default=61090, ge=1024, le=65535)
    instance_agent_base_port: int = Field(default=61400, ge=1024, le=65535)
    instance_agent_start_timeout_s: float = Field(default=180.0, gt=0.0)
    daemon_stable_bytes: str = Field(default="16GB", min_length=1)
    clear_storage_between_cells: bool = True
    hicache_mem_layout: str = "page_blob_direct"
    hicache_io_backend: str = "direct"
    host_allocator_enabled: bool = True
    host_allocator_region_ttl_ms: int = Field(default=600_000, ge=0)
    host_allocator_region_name_prefix: str = Field(
        default="tc_router_sglang_host_pool",
        min_length=1,
    )

    @model_validator(mode="after")
    def _allocator_requires_direct_page_blob(self) -> "TensorcastConfig":
        if self.host_allocator_enabled:
            if self.hicache_mem_layout != "page_blob_direct":
                raise ValueError(
                    "tensorcast.host_allocator_enabled requires "
                    "tensorcast.hicache_mem_layout=page_blob_direct"
                )
            if self.hicache_io_backend != "direct":
                raise ValueError(
                    "tensorcast.host_allocator_enabled requires "
                    "tensorcast.hicache_io_backend=direct"
                )
        return self


class LoadPollingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    period_ms: int = Field(default=250, ge=10)


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    model: ModelConfig
    instances: InstancesConfig
    transport: TransportConfig = Field(default_factory=TransportConfig)
    workload: WorkloadConfig
    configs: tuple[ConfigSpec, ...] = Field(min_length=1)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    mooncake: MooncakeConfig = Field(default_factory=MooncakeConfig)
    tensorcast: TensorcastConfig = Field(default_factory=TensorcastConfig)
    load_polling: LoadPollingConfig = Field(default_factory=LoadPollingConfig)

    @model_validator(mode="after")
    def _validate_port_layout(self) -> "BenchmarkConfig":
        uses_gateway = any(config.kind != "tc_router" for config in self.configs)
        uses_mooncake = any(
            config.kind == "gw_load_aware_mooncake" for config in self.configs
        )
        uses_tensorcast = any(config.kind == "tc_router" for config in self.configs)
        serving_ports = [
            self.instances.base_port + offset for offset in range(self.instances.count)
        ]
        if serving_ports[-1] > 65535:
            raise ValueError("SGLang serving ports exceed 65535")

        instance_agent_ports: list[int] = []
        if uses_tensorcast:
            instance_agent_ports = [
                self.tensorcast.instance_agent_base_port + offset
                for offset in range(self.instances.count)
            ]
            if instance_agent_ports[-1] > 65535:
                raise ValueError(
                    "tensorcast.instance_agent_base_port + instances.count - 1 "
                    "must be <= 65535"
                )

        ports: dict[int, str] = {}

        def add_port(port: int, label: str) -> None:
            previous = ports.get(port)
            if previous is not None:
                raise ValueError(
                    f"port collision: {label} and {previous} both use {port}"
                )
            ports[port] = label

        for idx, port in enumerate(serving_ports):
            add_port(port, f"instances[{idx}].serving")
            for candidate in (port + 100, port - 100):
                if 1024 <= candidate <= 65535:
                    add_port(candidate, f"instances[{idx}].nccl")
                    break
            else:
                raise ValueError(
                    f"cannot derive a valid nccl_port from serving_port={port}"
                )
        if uses_gateway:
            add_port(self.gateway.port, "gateway.port")
        if uses_mooncake:
            add_port(
                self.mooncake.http_metadata_server_port,
                "mooncake.http_metadata_server_port",
            )
            add_port(self.mooncake.master_port, "mooncake.master_port")
        if uses_tensorcast:
            add_port(self.tensorcast.global_store_port, "tensorcast.global_store_port")
            add_port(self.tensorcast.daemon_port, "tensorcast.daemon_port")
            add_port(self.tensorcast.daemon_p2p_port, "tensorcast.daemon_p2p_port")
            for idx, port in enumerate(instance_agent_ports):
                add_port(port, f"tensorcast.instance_agent[{idx}]")
        return self


def load_benchmark_yaml(path: str | Path) -> BenchmarkConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"benchmark YAML at {path} is not a mapping")
    return BenchmarkConfig.model_validate(raw)
