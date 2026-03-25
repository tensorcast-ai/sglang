# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig

PolicyProfile = Literal["cache", "durable", "ha", "cold", "warm", "pinned"]


class TensorcastHiCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    daemon_address: str
    namespace: str = "sglang_hicache"
    engine: str = "sglang"
    model_id: str = ""
    page_tensor_name: str = "page"
    policy_profile: PolicyProfile = "durable"

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
        payload["model_id"] = model_id
        return cls.model_validate(payload)
