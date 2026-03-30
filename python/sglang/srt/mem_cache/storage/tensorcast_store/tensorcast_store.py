# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.storage.tensorcast_store.client import (
    DefaultTensorcastPageClient,
    TensorcastBatchExistsResult,
    TensorcastBatchTransferResult,
    TensorcastPageClient,
)
from sglang.srt.mem_cache.storage.tensorcast_store.config import (
    TensorcastHiCacheConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class _TensorcastPublicationStats:
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    batch_calls: int = 0
    page_calls: int = 0
    duplicate_pages: int = 0
    failed_pages: int = 0
    batch_total_s: float = 0.0
    pack_total_s: float = 0.0
    stage_copy_total_s: float = 0.0
    rpc_total_s: float = 0.0

    def record_batch(
        self,
        *,
        pages: int,
        duplicate_pages: int,
        failed_pages: int,
        batch_elapsed_s: float,
        pack_elapsed_s: float,
        stage_copy_elapsed_s: float,
        rpc_elapsed_s: float,
    ) -> dict[str, float | int]:
        with self.lock:
            self.batch_calls += 1
            self.page_calls += pages
            self.duplicate_pages += duplicate_pages
            self.failed_pages += failed_pages
            self.batch_total_s += batch_elapsed_s
            self.pack_total_s += pack_elapsed_s
            self.stage_copy_total_s += stage_copy_elapsed_s
            self.rpc_total_s += rpc_elapsed_s
            return {
                "batch_calls": self.batch_calls,
                "page_calls": self.page_calls,
                "duplicate_pages": self.duplicate_pages,
                "failed_pages": self.failed_pages,
                "batch_total_ms": self.batch_total_s * 1000.0,
                "pack_total_ms": self.pack_total_s * 1000.0,
                "stage_copy_total_ms": self.stage_copy_total_s * 1000.0,
                "rpc_total_ms": self.rpc_total_s * 1000.0,
            }


def _sanitize_component(value: str) -> str:
    sanitized = str(value).strip().replace("/", "_")
    sanitized = sanitized.replace(":", "_").replace(" ", "_")
    return sanitized or "default"


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
        self._layout_id = self._build_layout_id()
        self._page_client = page_client or DefaultTensorcastPageClient(
            self._tensorcast_config,
            layout_id=self._layout_id,
            engine_key_prefix=self._rank_suffix,
        )
        self._publication_stats = _TensorcastPublicationStats()
        self.register_mem_pool_host(mem_pool_host)
        logger.info(
            "Initialized Tensorcast HiCache backend: daemon=%s namespace=%s rank_suffix=%s layout_id=%s",
            self._tensorcast_config.daemon_address,
            self._tensorcast_config.namespace,
            self._rank_suffix,
            self._layout_id,
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

    def _build_layout_id(self) -> str:
        attention_family = "mla" if self.is_mla_backend else "mha"
        return "_".join(
            [
                "sglang_kv_page",
                _sanitize_component(self._tensorcast_config.page_layout_version),
                _sanitize_component(self.layout),
                _sanitize_component(str(self.dtype)),
                f"ps{self.page_size}",
                attention_family,
            ]
        )

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

    def _host_page_views(self, page_starts: list[int]) -> list[torch.Tensor]:
        return [
            self.mem_pool_host.get_data_page(index, flat=True) for index in page_starts
        ]

    def batch_exists(
        self,
        keys: list[str],
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> int:
        _ = extra_info
        result: TensorcastBatchExistsResult = self._page_client.batch_exists(keys)
        prefix_success = 0
        for exists in result.existence_mask:
            if not exists:
                break
            prefix_success += 1
        first_key = keys[0] if keys else ""
        first_artifact_id = (
            self._page_client.artifact_id_for(first_key) if first_key else ""
        )
        logger.debug(
            "Tensorcast batch_exists pages=%d prefix_success=%d rpc_elapsed_ms=%.2f first_key=%s first_artifact_id=%s",
            len(keys),
            prefix_success,
            result.rpc_elapsed_s * 1000.0,
            first_key,
            first_artifact_id,
        )
        return prefix_success

    def batch_get_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> list[bool]:
        _ = extra_info
        page_starts = self._page_start_indices(host_indices, len(keys))
        targets = self._host_page_views(page_starts)
        result = self._page_client.batch_get_into(keys, targets)
        first_key = keys[0] if keys else ""
        first_artifact_id = (
            self._page_client.artifact_id_for(first_key) if first_key else ""
        )
        logger.debug(
            "Tensorcast batch_get_v1 pages=%d succeeded=%d pack_elapsed_ms=%.2f rpc_elapsed_ms=%.2f host_fill_ms=%.2f operation_id=%s first_key=%s first_artifact_id=%s",
            len(keys),
            sum(1 for item in result.success_mask if item),
            result.pack_elapsed_s * 1000.0,
            result.rpc_elapsed_s * 1000.0,
            result.host_fill_elapsed_s * 1000.0,
            result.operation_id,
            first_key,
            first_artifact_id,
        )
        return list(result.success_mask)

    def batch_set_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> list[bool]:
        _ = extra_info
        page_starts = self._page_start_indices(host_indices, len(keys))
        pages = self._host_page_views(page_starts)
        batch_started_at = time.perf_counter()
        result: TensorcastBatchTransferResult = self._page_client.batch_put(keys, pages)
        batch_elapsed_s = time.perf_counter() - batch_started_at
        succeeded = sum(1 for item in result.success_mask if item)
        failed_pages = len(keys) - succeeded
        first_key = keys[0] if keys else ""
        first_artifact_id = (
            self._page_client.artifact_id_for(first_key) if first_key else ""
        )
        cumulative = self._publication_stats.record_batch(
            pages=len(keys),
            duplicate_pages=result.adopted_duplicate_count,
            failed_pages=failed_pages,
            batch_elapsed_s=batch_elapsed_s,
            pack_elapsed_s=result.pack_elapsed_s,
            stage_copy_elapsed_s=result.stage_copy_elapsed_s,
            rpc_elapsed_s=result.rpc_elapsed_s,
        )
        logger.debug(
            "Tensorcast batch_set_v1 pages=%d succeeded=%d duplicates=%d failed=%d batch_elapsed_ms=%.2f pack_elapsed_ms=%.2f stage_copy_ms=%.2f rpc_elapsed_ms=%.2f first_key=%s first_artifact_id=%s cumulative_pages=%d cumulative_duplicates=%d cumulative_failed=%d cumulative_batch_ms=%.2f cumulative_pack_ms=%.2f cumulative_stage_copy_ms=%.2f cumulative_rpc_ms=%.2f",
            len(keys),
            succeeded,
            result.adopted_duplicate_count,
            failed_pages,
            batch_elapsed_s * 1000.0,
            result.pack_elapsed_s * 1000.0,
            result.stage_copy_elapsed_s * 1000.0,
            result.rpc_elapsed_s * 1000.0,
            first_key,
            first_artifact_id,
            cumulative["page_calls"],
            cumulative["duplicate_pages"],
            cumulative["failed_pages"],
            cumulative["batch_total_ms"],
            cumulative["pack_total_ms"],
            cumulative["stage_copy_total_ms"],
            cumulative["rpc_total_ms"],
        )
        return list(result.success_mask)

    def get(
        self,
        key: str,
        target_location: Any | None = None,
        target_sizes: Any | None = None,
    ) -> torch.Tensor | None:
        _ = target_sizes
        cpu_target = (
            torch.empty_like(target_location, device="cpu")
            if isinstance(target_location, torch.Tensor)
            else self.mem_pool_host.get_dummy_flat_data_page()
        )
        result = self._page_client.batch_get_into([key], [cpu_target])
        if not result.success_mask or not result.success_mask[0]:
            return None
        if isinstance(target_location, torch.Tensor):
            target_location.copy_(cpu_target.reshape_as(target_location))
            return target_location
        return cpu_target

    def batch_get(
        self,
        keys: list[str],
        target_locations: Any | None = None,
        target_sizes: Any | None = None,
    ) -> list[torch.Tensor | None] | int:
        _ = target_sizes
        if target_locations is None:
            outputs: list[torch.Tensor | None] = []
            for key in keys:
                outputs.append(self.get(key))
            return outputs
        outputs = []
        for key, target in zip(keys, target_locations, strict=False):
            outputs.append(self.get(key, target_location=target))
        return outputs

    def set(
        self,
        key: str,
        value: Any | None = None,
        target_location: Any | None = None,
        target_sizes: Any | None = None,
    ) -> bool:
        _ = target_sizes
        tensor = value if isinstance(value, torch.Tensor) else target_location
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("TensorcastStore.set requires a tensor value")
        cpu_tensor = tensor if tensor.device.type == "cpu" else tensor.cpu()
        result = self._page_client.batch_put([key], [cpu_tensor.reshape(-1)])
        return bool(result.success_mask and result.success_mask[0])

    def batch_set(
        self,
        keys: list[str],
        values: Any | None = None,
        target_locations: Any | None = None,
        target_sizes: Any | None = None,
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
        result = self._page_client.batch_exists([key])
        return bool(result.existence_mask and result.existence_mask[0])
