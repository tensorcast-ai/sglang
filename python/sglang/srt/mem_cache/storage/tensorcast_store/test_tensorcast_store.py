"""Unit tests for the Tensorcast HiCache backend."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from types import ModuleType

import torch


memory_pool_host_module = ModuleType("sglang.srt.mem_cache.memory_pool_host")
memory_pool_host_module.HostKVCache = object
sys.modules.setdefault(
    "sglang.srt.mem_cache.memory_pool_host",
    memory_pool_host_module,
)

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.tensorcast_store.client import (
    TensorcastBatchExistsResult,
    TensorcastBatchTransferResult,
    _compact_cgid_segment,
    _engine_key_payload,
)
from sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store import (
    TensorcastStore,
)


class FakeTensorcastPageClient:
    def __init__(self) -> None:
        self.data: dict[str, torch.Tensor] = {}

    def artifact_id_for(self, logical_key: str) -> str:
        return f"artifact::{logical_key}"

    def batch_exists(self, logical_keys: list[str]) -> TensorcastBatchExistsResult:
        return TensorcastBatchExistsResult(
            existence_mask=tuple(key in self.data for key in logical_keys),
            rpc_elapsed_s=0.0,
        )

    def batch_put(
        self,
        logical_keys: list[str],
        pages: list[torch.Tensor],
    ) -> TensorcastBatchTransferResult:
        success_mask: list[bool] = []
        duplicate_count = 0
        for logical_key, page in zip(logical_keys, pages, strict=True):
            if logical_key in self.data:
                duplicate_count += 1
                success_mask.append(True)
                continue
            self.data[logical_key] = page.clone()
            success_mask.append(True)
        return TensorcastBatchTransferResult(
            success_mask=tuple(success_mask),
            adopted_duplicate_count=duplicate_count,
            pack_elapsed_s=0.0,
            stage_copy_elapsed_s=0.0,
            rpc_elapsed_s=0.0,
            host_fill_elapsed_s=0.0,
        )

    def batch_get_into(
        self,
        logical_keys: list[str],
        targets: list[torch.Tensor],
    ) -> TensorcastBatchTransferResult:
        success_mask: list[bool] = []
        stop_copying = False
        for logical_key, target in zip(logical_keys, targets, strict=True):
            if stop_copying or logical_key not in self.data:
                stop_copying = True
                success_mask.append(False)
                continue
            target.copy_(self.data[logical_key])
            success_mask.append(True)
        return TensorcastBatchTransferResult(
            success_mask=tuple(success_mask),
            adopted_duplicate_count=0,
            pack_elapsed_s=0.0,
            stage_copy_elapsed_s=0.0,
            rpc_elapsed_s=0.0,
            host_fill_elapsed_s=0.0,
        )


class FakeHostKVCache:
    def __init__(self, values: list[float], *, page_size: int = 2) -> None:
        self.page_size = page_size
        self.layout = "page_first"
        self.dtype = torch.float32
        self.kv_buffer = torch.tensor(values, dtype=self.dtype).reshape(-1, page_size)

    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        page = self.kv_buffer[index // self.page_size]
        return page.view(-1) if flat else page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros((self.page_size,), dtype=self.dtype)

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        self.kv_buffer[index // self.page_size] = data_page.reshape(self.page_size)


def build_storage_config(
    *,
    is_mla_model: bool = False,
    tp_rank: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    extra_config: dict[str, str] | str | None = None,
) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=4,
        pp_rank=pp_rank,
        pp_size=pp_size,
        is_mla_model=is_mla_model,
        is_page_first_layout=True,
        model_name="Qwen3-32B",
        extra_config=extra_config
        or {
            "daemon_address": "127.0.0.1:50052",
            "namespace": "unit-test",
        },
    )


class TensorcastStoreTest(unittest.TestCase):
    def test_tensorcast_store_batch_set_and_get_v1(self) -> None:
        client = FakeTensorcastPageClient()
        host_cache = FakeHostKVCache([1.0, 2.0, 3.0, 4.0])
        store = TensorcastStore(
            build_storage_config(),
            host_cache,
            page_client=client,
        )

        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        self.assertEqual(store.batch_set_v1(["hash_a", "hash_b"], host_indices), [True, True])
        self.assertEqual(store.batch_exists(["hash_a", "hash_b"]), 2)

        host_cache.kv_buffer.zero_()
        self.assertEqual(store.batch_get_v1(["hash_a", "hash_b"], host_indices), [True, True])
        self.assertTrue(
            torch.equal(
                host_cache.kv_buffer.flatten(),
                torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
            )
        )

    def test_tensorcast_store_duplicate_batch_set_is_idempotent(self) -> None:
        client = FakeTensorcastPageClient()
        host_cache = FakeHostKVCache([7.0, 8.0])
        store = TensorcastStore(
            build_storage_config(),
            host_cache,
            page_client=client,
        )

        host_indices = torch.tensor([0, 1], dtype=torch.int64)
        self.assertEqual(store.batch_set_v1(["hash_dup"], host_indices), [True])

        host_cache.kv_buffer = torch.tensor([9.0, 10.0], dtype=torch.float32).reshape(1, 2)
        self.assertEqual(store.batch_set_v1(["hash_dup"], host_indices), [True])
        self.assertEqual(store._publication_stats.duplicate_pages, 1)
        self.assertTrue(
            torch.equal(client.data["hash_dup"], torch.tensor([7.0, 8.0], dtype=torch.float32))
        )

    def test_tensorcast_store_batch_exists_stops_at_first_missing_key(self) -> None:
        client = FakeTensorcastPageClient()
        host_cache = FakeHostKVCache([11.0, 12.0, 13.0, 14.0])
        store = TensorcastStore(
            build_storage_config(),
            host_cache,
            page_client=client,
        )

        host_indices = torch.tensor([0, 1], dtype=torch.int64)
        self.assertEqual(store.batch_set_v1(["hash_present"], host_indices), [True])
        self.assertEqual(
            store.batch_exists(["hash_present", "hash_missing", "hash_after"]),
            1,
        )

    def test_tensorcast_store_accepts_json_string_extra_config(self) -> None:
        client = FakeTensorcastPageClient()
        host_cache = FakeHostKVCache([15.0, 16.0])
        store = TensorcastStore(
            build_storage_config(
                extra_config='{"daemon_address":"127.0.0.1:50052","namespace":"json-test"}'
            ),
            host_cache,
            page_client=client,
        )

        self.assertEqual(store._tensorcast_config.namespace, "json-test")
        self.assertEqual(store._tensorcast_config.model_id, "Qwen3-32B")

    def test_tensorcast_store_layout_and_rank_suffix_for_mha(self) -> None:
        store = TensorcastStore(
            build_storage_config(tp_rank=1, pp_rank=0, pp_size=1),
            FakeHostKVCache([1.0, 2.0]),
            page_client=FakeTensorcastPageClient(),
        )

        self.assertEqual(store._rank_suffix, "tp1of4")
        self.assertEqual(
            store._layout_id,
            "sglang_kv_page_v1_page_first_torch.float32_ps2_mha",
        )

    def test_tensorcast_store_mla_uses_pp_rank_suffix(self) -> None:
        store = TensorcastStore(
            build_storage_config(is_mla_model=True, tp_rank=3, pp_rank=2, pp_size=4),
            FakeHostKVCache([5.0, 6.0]),
            page_client=FakeTensorcastPageClient(),
        )

        self.assertEqual(store._rank_suffix, "pp2of4")
        self.assertEqual(
            store._layout_id,
            "sglang_kv_page_v1_page_first_torch.float32_ps2_mla",
        )

    def test_cgid_helpers_compact_long_segments_and_hash_keys(self) -> None:
        compact_namespace = _compact_cgid_segment(
            "share_local:20260325-142807_tensorcast_tp2_pairs1",
            prefix="ns",
        )
        compact_layout = _compact_cgid_segment(
            "sglang_kv_page_v1_page_first_torch.bfloat16_ps64_mha",
            prefix="ly",
        )
        engine_key_payload = _engine_key_payload(
            "tp0of2",
            "fc6267db9ba02bbb251960231bc87251c581110e4e2b5f3f8a0d60b884d57ccf",
        )

        self.assertLessEqual(len(compact_namespace), 19)
        self.assertLessEqual(len(compact_layout), 19)
        self.assertEqual(engine_key_payload[:7], b"tp0of2:")
        self.assertEqual(len(engine_key_payload), 39)


if __name__ == "__main__":
    unittest.main()
