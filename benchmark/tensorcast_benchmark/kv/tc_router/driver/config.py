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
    load_polling: LoadPollingConfig = Field(default_factory=LoadPollingConfig)


def load_benchmark_yaml(path: str | Path) -> BenchmarkConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"benchmark YAML at {path} is not a mapping")
    return BenchmarkConfig.model_validate(raw)
