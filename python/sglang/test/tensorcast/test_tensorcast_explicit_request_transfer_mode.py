from __future__ import annotations

# ruff: noqa: E402

import torch

from sglang.test.tensorcast.test_support import install_memory_pool_host_stub

install_memory_pool_host_stub()

from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    PublishInstanceOpRequest,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    PagePublicationState,
)
from sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store import (
    TensorcastStore,
)
from sglang.test.tensorcast.test_tensorcast_store import (
    FakeHostKVCache,
    FakeTensorcastPageClient,
    build_storage_config,
)


def _explicit_store() -> tuple[TensorcastStore, FakeTensorcastPageClient]:
    client = FakeTensorcastPageClient()
    host_cache = FakeHostKVCache([1.0, 2.0, 3.0, 4.0])
    store = TensorcastStore(
        build_storage_config(
            tp_rank=0,
            extra_config={
                "daemon_address": "127.0.0.1:50052",
                "namespace": "explicit-test",
                "tensorcast_kv_mode": "explicit_request_transfer",
            },
        ),
        host_cache,
        page_client=client,
    )
    return store, client


def _page_hashes(prompt_token_ids: list[int], page_size: int) -> list[str]:
    hashes: list[str] = []
    parent_hash: str | None = None
    for page_start in range(0, len(prompt_token_ids), page_size):
        parent_hash = get_hash_str(
            prompt_token_ids[page_start : page_start + page_size],
            prior_hash=parent_hash,
        )
        hashes.append(parent_hash)
    return hashes


def _publish_request(
    prompt_token_ids: list[int], page_size: int
) -> PublishInstanceOpRequest:
    return PublishInstanceOpRequest(
        logical_request_id="rid-explicit-1",
        engine_request_id="rid-explicit-1",
        publish_op_id="publish-explicit-1",
        requested_cutoff_token_count=len(prompt_token_ids),
        prompt_token_digest=get_hash_str(prompt_token_ids),
        dtype="float32",
        page_size=page_size,
        requested_at_ms=200,
    )


def test_explicit_mode_batch_set_records_host_residency_without_put() -> None:
    store, client = _explicit_store()
    prompt_token_ids = [1, 2, 3, 4]
    page_hashes = _page_hashes(prompt_token_ids, store.page_size)

    store.request_bundle_manager.start_live_request_tracking(
        logical_request_id="rid-explicit-1",
        engine_request_id="rid-explicit-1",
        prompt_token_ids=prompt_token_ids,
        requested_at_ms=100,
        routing_key="session-explicit",
    )
    store.request_bundle_manager.observe_live_request_progress(
        logical_request_id="rid-explicit-1",
        visible_prompt_token_count=len(prompt_token_ids),
        emitted_decode_token_count=0,
        now_ms=110,
    )

    host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    assert store.batch_set_v1(page_hashes, host_indices) == [True, True]
    assert client.data == {}
    assert store.batch_exists(page_hashes) == 0

    pages = store.request_bundle_manager.page_publication_registry.snapshot_rank(
        logical_request_id="rid-explicit-1",
        rank=store.request_bundle_manager._current_rank(),
    )
    assert [page.publication_state for page in pages] == [
        PagePublicationState.ABSENT,
        PagePublicationState.ABSENT,
    ]

    result = store.request_bundle_manager.instance_publish_local(
        request=_publish_request(prompt_token_ids, store.page_size)
    )

    assert set(client.data) == set(page_hashes)
    assert result.publish_manifest.engine_owned_manifest.payload.logical_session_id == (
        "session-explicit"
    )
    assert [
        outcome.action.value
        for rank_result in result.rank_results
        for outcome in rank_result.page_outcomes
    ] == ["flushed", "flushed"]


def test_explicit_mode_force_flush_survives_hicache_backup_ack_free() -> None:
    store, client = _explicit_store()
    prompt_token_ids = [1, 2, 3, 4]
    page_hashes = _page_hashes(prompt_token_ids, store.page_size)
    host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    store.request_bundle_manager.start_live_request_tracking(
        logical_request_id="rid-explicit-1",
        engine_request_id="rid-explicit-1",
        prompt_token_ids=prompt_token_ids,
        requested_at_ms=100,
    )
    store.request_bundle_manager.observe_live_request_progress(
        logical_request_id="rid-explicit-1",
        visible_prompt_token_count=len(prompt_token_ids),
        emitted_decode_token_count=0,
        now_ms=110,
    )
    assert store.batch_set_v1(page_hashes, host_indices) == [True, True]
    store.mem_pool_host.free(host_indices)

    result = store.request_bundle_manager.instance_publish_local(
        request=_publish_request(prompt_token_ids, store.page_size)
    )

    assert result.publish_manifest.cutoff_token_count == len(prompt_token_ids)
    assert set(client.data) == set(page_hashes)
    assert store.mem_pool_host.describe_page_slot(0).state.value == "slot_free"
