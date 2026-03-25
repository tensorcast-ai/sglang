# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.storage.tensorcast_store.client import (
    DefaultTensorcastPageClient,
    TensorcastPageClient,
)
from sglang.srt.mem_cache.storage.tensorcast_store.config import (
    TensorcastHiCacheConfig,
)

logger = logging.getLogger(__name__)


def _sanitize_component(value: str) -> str:
    sanitized = str(value).strip().replace("/", "-")
    sanitized = sanitized.replace(":", "-").replace(" ", "_")
    return sanitized or "default"


def _is_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None and str(status_code).upper() == "NOT_FOUND":
        return True
    return "NOT_FOUND" in str(exc).upper()


class TensorcastStore(HiCacheStorage):
    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: HostKVCache,
        page_client: TensorcastPageClient | None = None,
    ) -> None:
        self._storage_config = storage_config
        self._tensorcast_config = TensorcastHiCacheConfig.from_storage_config(
            storage_config
        )
        self.is_mla_backend = storage_config.is_mla_model
        self.local_rank = storage_config.tp_rank
        self.pp_rank = storage_config.pp_rank
        self.pp_size = storage_config.pp_size
        self.layout = mem_pool_host.layout
        self.page_size = mem_pool_host.page_size
        self.dtype = mem_pool_host.dtype
        self.tp_size = storage_config.tp_size
        self._rank_suffix = self._build_rank_suffix()
        self._key_prefix = self._build_key_prefix()
        self._page_client = page_client or DefaultTensorcastPageClient(
            self._tensorcast_config
        )
        self.register_mem_pool_host(mem_pool_host)
        logger.info(
            "Initialized Tensorcast HiCache backend: daemon=%s namespace=%s rank_suffix=%s",
            self._tensorcast_config.daemon_address,
            self._tensorcast_config.namespace,
            self._rank_suffix,
        )

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)

    def _build_rank_suffix(self) -> str:
        if self.is_mla_backend:
            return f"pp{self.pp_rank}of{self.pp_size}"
        if self.pp_size > 1:
            return (
                f"tp{self.local_rank}of{self.tp_size}_"
                f"pp{self.pp_rank}of{self.pp_size}"
            )
        return f"tp{self.local_rank}of{self.tp_size}"

    def _build_key_prefix(self) -> str:
        parts = [
            "sglang",
            "kv_page",
            _sanitize_component(self._tensorcast_config.namespace),
            _sanitize_component(self._tensorcast_config.engine),
            _sanitize_component(self._tensorcast_config.model_id),
            _sanitize_component(self.layout),
            _sanitize_component(str(self.dtype)),
            f"ps{self.page_size}",
            self._rank_suffix,
        ]
        return ":".join(parts)

    def _artifact_key(self, logical_key: str) -> str:
        return f"{self._key_prefix}:{logical_key}"

    def _page_start_indices(
        self,
        host_indices: torch.Tensor,
        expected_pages: int,
    ) -> list[int]:
        if host_indices.numel() != expected_pages * self.page_size:
            raise ValueError(
                "host_indices length must equal number of pages multiplied by page_size"
            )
        return [
            int(host_indices[i * self.page_size].item()) for i in range(expected_pages)
        ]

    def _put_page_if_absent(self, key: str, page: torch.Tensor) -> bool:
        artifact_key = self._artifact_key(key)
        try:
            if self._page_client.exists(artifact_key):
                logger.debug(
                    "Tensorcast page already exists, skipping duplicate put for key=%s",
                    artifact_key,
                )
                return True
            self._page_client.put(artifact_key, page)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Tensorcast page put failed for key=%s: %s", artifact_key, exc
            )
            return False

    def batch_exists(
        self,
        keys: list[str],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> int:
        _ = extra_info
        for index, key in enumerate(keys):
            try:
                if not self._page_client.exists(self._artifact_key(key)):
                    return index
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Tensorcast page exists failed for key=%s: %s", key, exc
                )
                return index
        return len(keys)

    def batch_get_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> list[bool]:
        _ = extra_info
        page_starts = self._page_start_indices(host_indices, len(keys))
        results: list[bool] = []
        for idx, key in zip(page_starts, keys):
            artifact_key = self._artifact_key(key)
            target = self.mem_pool_host.get_dummy_flat_data_page()
            try:
                self._page_client.get_into(artifact_key, target)
                self.mem_pool_host.set_from_flat_data_page(idx, target)
                results.append(True)
            except Exception as exc:  # noqa: BLE001
                if not _is_not_found_error(exc):
                    logger.exception(
                        "Tensorcast page get failed for key=%s: %s",
                        artifact_key,
                        exc,
                    )
                results.append(False)
                results.extend([False] * (len(keys) - len(results)))
                break
        return results

    def batch_set_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> list[bool]:
        _ = extra_info
        page_starts = self._page_start_indices(host_indices, len(keys))
        results: list[bool] = []
        for idx, key in zip(page_starts, keys):
            page = self.mem_pool_host.get_data_page(idx, flat=False).reshape(-1)
            results.append(self._put_page_if_absent(key, page))
        return results

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        _ = target_sizes
        artifact_key = self._artifact_key(key)
        try:
            if isinstance(target_location, torch.Tensor):
                self._page_client.get_into(artifact_key, target_location)
                return target_location
            return self._page_client.get_tensor(artifact_key)
        except Exception as exc:  # noqa: BLE001
            if not _is_not_found_error(exc):
                logger.exception(
                    "Tensorcast page get failed for key=%s: %s", artifact_key, exc
                )
            return None

    def batch_get(
        self,
        keys: list[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> list[torch.Tensor | None] | int:
        _ = target_sizes
        outputs: list[torch.Tensor | None] = []
        for index, key in enumerate(keys):
            target = None
            if target_locations is not None:
                target = target_locations[index]
            outputs.append(self.get(key, target_location=target))
        return outputs

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        _ = target_sizes
        tensor: torch.Tensor | None = None
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(target_location, torch.Tensor):
            tensor = target_location
        if tensor is None:
            raise ValueError("TensorcastStore.set requires a tensor value")
        return self._put_page_if_absent(key, tensor)

    def batch_set(
        self,
        keys: list[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        _ = target_sizes
        tensors = values if values is not None else target_locations
        if tensors is None:
            raise ValueError("TensorcastStore.batch_set requires tensors")
        results = [
            self.set(key, value=tensor)
            for key, tensor in zip(keys, tensors, strict=False)
        ]
        return all(results)

    def exists(self, key: str) -> bool:
        try:
            return self._page_client.exists(self._artifact_key(key))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tensorcast page exists failed for key=%s: %s", key, exc)
            return False

    def clear(self) -> None:
        raise NotImplementedError(
            "Tensorcast HiCache backend does not support destructive global clear"
        )
