from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.tensorcast_store.client import (
    DefaultTensorcastPageClient,
)
from sglang.srt.mem_cache.storage.tensorcast_store.config import (
    TensorcastHiCacheConfig,
)
from sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store import (
    TensorcastStore,
)


class FakeTensorcastPageClient:
    def __init__(self) -> None:
        self.data: dict[str, torch.Tensor] = {}

    def exists(self, key: str) -> bool:
        return key in self.data

    def get_into(self, key: str, target: torch.Tensor) -> None:
        if key not in self.data:
            raise RuntimeError("NOT_FOUND")
        target.copy_(self.data[key])

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self.data:
            raise RuntimeError("NOT_FOUND")
        return self.data[key].clone()

    def put(self, key: str, tensor: torch.Tensor) -> None:
        self.data[key] = tensor.clone()


class FakeHostKVCache:
    def __init__(self, values: list[float], *, page_size: int = 2) -> None:
        self.page_size = page_size
        self.layout = "page_first"
        self.dtype = torch.float32
        self.kv_buffer = torch.tensor(values, dtype=self.dtype).reshape(-1, page_size)

    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        page = self.kv_buffer[index // self.page_size]
        return page.flatten().clone() if flat else page

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


def test_tensorcast_store_batch_set_and_get_v1() -> None:
    client = FakeTensorcastPageClient()
    host_cache = FakeHostKVCache([1.0, 2.0, 3.0, 4.0])
    store = TensorcastStore(
        build_storage_config(),
        host_cache,
        page_client=client,
    )

    host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    assert store.batch_set_v1(["hash_a", "hash_b"], host_indices) == [True, True]
    assert store.batch_exists(["hash_a", "hash_b"]) == 2

    host_cache.kv_buffer.zero_()
    assert store.batch_get_v1(["hash_a", "hash_b"], host_indices) == [True, True]
    assert torch.equal(
        host_cache.kv_buffer.flatten(),
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
    )

    stored_keys = sorted(client.data.keys())
    assert stored_keys == [
        "sglang:kv_page:unit-test:sglang:Qwen3-32B:page_first:torch.float32:ps2:tp1of4:hash_a",
        "sglang:kv_page:unit-test:sglang:Qwen3-32B:page_first:torch.float32:ps2:tp1of4:hash_b",
    ]


def test_tensorcast_store_mla_uses_pp_rank_suffix() -> None:
    client = FakeTensorcastPageClient()
    host_cache = FakeHostKVCache([5.0, 6.0])
    store = TensorcastStore(
        build_storage_config(is_mla_model=True, tp_rank=3, pp_rank=2, pp_size=4),
        host_cache,
        page_client=client,
    )

    host_indices = torch.tensor([0, 1], dtype=torch.int64)
    assert store.batch_set_v1(["hash_mla"], host_indices) == [True]
    assert sorted(client.data.keys()) == [
        "sglang:kv_page:unit-test:sglang:Qwen3-32B:page_first:torch.float32:ps2:pp2of4:hash_mla"
    ]


def test_tensorcast_store_duplicate_batch_set_is_idempotent() -> None:
    client = FakeTensorcastPageClient()
    host_cache = FakeHostKVCache([7.0, 8.0])
    store = TensorcastStore(
        build_storage_config(),
        host_cache,
        page_client=client,
    )

    host_indices = torch.tensor([0, 1], dtype=torch.int64)
    assert store.batch_set_v1(["hash_dup"], host_indices) == [True]

    host_cache.kv_buffer = torch.tensor([9.0, 10.0], dtype=torch.float32).reshape(1, 2)
    assert store.batch_set_v1(["hash_dup"], host_indices) == [True]

    stored = client.data[
        "sglang:kv_page:unit-test:sglang:Qwen3-32B:page_first:torch.float32:ps2:tp1of4:hash_dup"
    ]
    assert torch.equal(stored, torch.tensor([7.0, 8.0], dtype=torch.float32))


def test_tensorcast_store_batch_exists_stops_at_first_missing_key() -> None:
    client = FakeTensorcastPageClient()
    host_cache = FakeHostKVCache([11.0, 12.0, 13.0, 14.0])
    store = TensorcastStore(
        build_storage_config(),
        host_cache,
        page_client=client,
    )

    host_indices = torch.tensor([0, 1], dtype=torch.int64)
    assert store.batch_set_v1(["hash_present"], host_indices) == [True]
    assert store.batch_exists(["hash_present", "hash_missing", "hash_after"]) == 1


def test_tensorcast_store_accepts_json_string_extra_config() -> None:
    client = FakeTensorcastPageClient()
    host_cache = FakeHostKVCache([15.0, 16.0])
    store = TensorcastStore(
        build_storage_config(
            extra_config='{"daemon_address":"127.0.0.1:50052","namespace":"json-test"}'
        ),
        host_cache,
        page_client=client,
    )

    host_indices = torch.tensor([0, 1], dtype=torch.int64)
    assert store.batch_set_v1(["hash_json"], host_indices) == [True]
    assert sorted(client.data.keys()) == [
        "sglang:kv_page:json-test:sglang:Qwen3-32B:page_first:torch.float32:ps2:tp1of4:hash_json"
    ]


def test_default_tensorcast_page_client_get_into_cpu_target_uses_cpu_tensor_copy() -> None:
    cpu_target = torch.zeros(4, dtype=torch.float32)
    expected = torch.arange(4, dtype=torch.float32)

    class FakeArtifact:
        def __init__(self) -> None:
            self.tensor_into_calls = 0
            self.tensor_calls: list[str] = []

        def tensor_into(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.tensor_into_calls += 1
            raise AssertionError("CPU target path should not call tensor_into")

        def tensor(self, name: str, *, device: str) -> torch.Tensor:
            assert name == "page"
            self.tensor_calls.append(device)
            return expected.clone()

    artifact = FakeArtifact()

    class FakeStore:
        def artifact(self, *, key: str):
            assert key == "page-key"
            return artifact

    fake_tensorcast = SimpleNamespace(Store=lambda daemon_address: FakeStore())
    config = TensorcastHiCacheConfig(daemon_address="127.0.0.1:50052")

    with patch(
        "sglang.srt.mem_cache.storage.tensorcast_store.client.importlib.import_module",
        return_value=fake_tensorcast,
    ):
        client = DefaultTensorcastPageClient(config)

    client.get_into("page-key", cpu_target)

    assert torch.equal(cpu_target, expected)
    assert artifact.tensor_into_calls == 0
    assert artifact.tensor_calls == ["cpu"]
