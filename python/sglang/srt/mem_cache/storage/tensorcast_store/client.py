# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

from typing import Protocol

import torch

from sglang.srt.mem_cache.storage.tensorcast_store.config import (
    TensorcastHiCacheConfig,
)


class TensorcastPageClient(Protocol):
    def exists(self, key: str) -> bool: ...

    def get_into(self, key: str, target: torch.Tensor) -> None: ...

    def get_tensor(self, key: str) -> torch.Tensor: ...

    def put(self, key: str, tensor: torch.Tensor) -> None: ...


def _is_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None and str(status_code).upper() == "NOT_FOUND":
        return True
    return "NOT_FOUND" in str(exc).upper()


class DefaultTensorcastPageClient:
    def __init__(self, config: TensorcastHiCacheConfig) -> None:
        import tensorcast as tc

        self._store = tc.Store(config.daemon_address)
        self._page_tensor_name = config.page_tensor_name
        self._policy_profile = config.policy_profile

    def exists(self, key: str) -> bool:
        try:
            return bool(self._store.artifact(key=key).exists())
        except Exception as exc:  # noqa: BLE001
            if _is_not_found_error(exc):
                return False
            raise

    def get_into(self, key: str, target: torch.Tensor) -> None:
        if target.device.type == "cpu":
            value = self.get_tensor(key)
            target.copy_(value)
            return
        self._store.artifact(key=key).tensor_into(
            self._page_tensor_name,
            target,
            device=target.device,
        )

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._store.artifact(key=key).tensor(
            self._page_tensor_name,
            device="cpu",
        )

    def put(self, key: str, tensor: torch.Tensor) -> None:
        value = tensor.contiguous()
        self._store.put(
            {self._page_tensor_name: value},
            key=key,
            policy=self._policy_profile,
        )
