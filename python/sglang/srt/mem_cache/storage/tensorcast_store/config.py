# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import json
import re
from typing import Any
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig

PolicyProfile = Literal["cache", "durable", "ha", "cold", "warm", "pinned"]
TensorcastKvMode = Literal["passive_prefix_share", "explicit_request_transfer"]
LogicalSessionIdSource = Literal[
    "disabled",
    "routing_key",
    "routing_key_then_rid_regex",
]


class TensorcastHiCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    daemon_address: str
    namespace: str = "sglang_hicache"
    engine: str = "sglang"
    model_id: str = ""
    model_version: str = ""
    page_tensor_name: str = "page"
    policy_profile: PolicyProfile = "durable"
    page_layout_version: str = "v1"
    batch_exists_timeout_s: float = 30.0
    batch_transfer_timeout_s: float = 600.0
    staging_region_ttl_ms: int = 0
    host_allocator_enabled: bool = False
    host_allocator_region_ttl_ms: int = 0
    host_allocator_region_name: str = "sglang_tensorcast_host_pool"
    instance_directory_address: str = ""
    instance_agent_execution_endpoint: str = ""
    instance_agent_start_timeout_s: float = 30.0
    instance_signals_endpoint: str = ""
    instance_directory_heartbeat_interval_ms: int = 10_000
    tensorcast_kv_mode: TensorcastKvMode = "passive_prefix_share"
    background_page_publish: bool = True
    ordinary_storage_prefetch: bool = True
    record_host_residency_for_publish: bool = True
    logical_session_id_source: LogicalSessionIdSource = "routing_key"
    logical_session_id_rid_regex: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _apply_mode_defaults(cls, raw_payload: Any) -> Any:
        if not isinstance(raw_payload, dict):
            return raw_payload
        payload = dict(raw_payload)
        if payload.get("tensorcast_kv_mode") == "explicit_request_transfer":
            payload.setdefault("background_page_publish", False)
            payload.setdefault("ordinary_storage_prefetch", False)
            payload.setdefault("record_host_residency_for_publish", True)
        return payload

    @field_validator("logical_session_id_rid_regex")
    @classmethod
    def _validate_logical_session_regex(cls, value: str | None) -> str | None:
        if value is None:
            return None
        pattern = value.strip()
        if not pattern:
            return None
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid logical_session_id_rid_regex: {exc}") from exc
        if "session_id" not in compiled.groupindex:
            raise ValueError(
                "logical_session_id_rid_regex must define a named 'session_id' capture"
            )
        return pattern

    @model_validator(mode="after")
    def _validate_mode_consistency(self) -> "TensorcastHiCacheConfig":
        if self.tensorcast_kv_mode == "explicit_request_transfer":
            if self.background_page_publish:
                raise ValueError(
                    "explicit_request_transfer requires background_page_publish=false"
                )
            if self.ordinary_storage_prefetch:
                raise ValueError(
                    "explicit_request_transfer requires ordinary_storage_prefetch=false"
                )
            if not self.record_host_residency_for_publish:
                raise ValueError(
                    "explicit_request_transfer requires record_host_residency_for_publish=true"
                )
        if (
            self.logical_session_id_source == "routing_key_then_rid_regex"
            and not self.logical_session_id_rid_regex
        ):
            raise ValueError(
                "routing_key_then_rid_regex requires logical_session_id_rid_regex"
            )
        return self

    @property
    def explicit_request_transfer_enabled(self) -> bool:
        return self.tensorcast_kv_mode == "explicit_request_transfer"

    @property
    def background_page_publish_enabled(self) -> bool:
        return (
            self.tensorcast_kv_mode == "passive_prefix_share"
            and self.background_page_publish
        )

    @property
    def ordinary_storage_prefetch_enabled(self) -> bool:
        return (
            self.tensorcast_kv_mode == "passive_prefix_share"
            and self.ordinary_storage_prefetch
        )

    @classmethod
    def from_storage_config(
        cls,
        storage_config: HiCacheStorageConfig,
    ) -> "TensorcastHiCacheConfig":
        raw_payload = storage_config.extra_config or {}
        payload = (
            json.loads(raw_payload)
            if isinstance(raw_payload, str)
            else dict(raw_payload)
        )
        model_id = str(payload.get("model_id", "")).strip()
        if not model_id and storage_config.model_name:
            model_id = str(storage_config.model_name)
        model_version = str(payload.get("model_version", "")).strip()
        if not model_version:
            model_version = "default"
        payload["model_id"] = model_id
        payload["model_version"] = model_version
        return cls.model_validate(payload)


class TensorcastHostAllocatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    daemon_address: str
    region_ttl_ms: int = 0
    region_name: str = "sglang_tensorcast_host_pool"


def tensorcast_host_allocator_config_from_extra_config(
    raw_payload: dict[str, Any] | None,
) -> TensorcastHostAllocatorConfig | None:
    payload = dict(raw_payload or {})
    if not bool(payload.get("host_allocator_enabled", False)):
        return None
    daemon_address = str(payload.get("daemon_address", "")).strip()
    if not daemon_address:
        raise ValueError(
            "TensorCast host allocator requires extra_config.daemon_address when host_allocator_enabled=true"
        )
    region_ttl_ms = int(
        payload.get(
            "host_allocator_region_ttl_ms",
            payload.get("staging_region_ttl_ms", 0),
        )
    )
    region_name = str(
        payload.get(
            "host_allocator_region_name",
            TensorcastHostAllocatorConfig.model_fields["region_name"].default,
        )
    ).strip()
    return TensorcastHostAllocatorConfig.model_validate(
        {
            "daemon_address": daemon_address,
            "region_ttl_ms": region_ttl_ms,
            "region_name": region_name
            or TensorcastHostAllocatorConfig.model_fields["region_name"].default,
        }
    )
