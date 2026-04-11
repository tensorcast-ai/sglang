# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass
import logging
from types import SimpleNamespace

import pytest
import torch

from sglang.test.tensorcast.test_support import install_memory_pool_host_stub

install_memory_pool_host_stub()

from sglang.srt.managers.scheduler_tensorcast_instance_ops_mixin import (
    SchedulerTensorcastInstanceOpsMixin,
)
from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    EvictLocalInstanceOpRequest,
    HydrateInstanceOpRequest,
    InstanceOpStatus,
    PublishInstanceOpReqInput,
    PublishInstanceOpRequest,
    PublishInstanceOpRespOutput,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    PreparedBundleLifecycleState,
    RankCoord,
    RankInstallState,
    RequestBundleLifecycleState,
)
from sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store import (
    TensorcastStore,
)
from sglang.test.tensorcast.test_tensorcast_store import (
    FakeHostKVCache,
    FakeTensorcastPageClient,
    build_storage_config,
)


class FakeWorldGroup:
    def __init__(
        self,
        *,
        rank_in_group: int = 0,
        broadcast_value: object | None = None,
    ) -> None:
        self.rank_in_group = rank_in_group
        self.broadcast_value = broadcast_value

    def all_gather_object(self, obj: object) -> list[object]:
        return [obj]

    def broadcast_object(self, obj: object, src: int = 0) -> object:
        if self.rank_in_group == src:
            return obj
        if self.broadcast_value is None:
            raise AssertionError("broadcast_value must be set for non-source ranks")
        return self.broadcast_value


class FakeScheduler(SchedulerTensorcastInstanceOpsMixin):
    pass


class FakePreparedHostNode:
    def __init__(self) -> None:
        self.host_ref_counter = 0

    def protect_host(self) -> None:
        self.host_ref_counter += 1

    def release_host(self) -> None:
        self.host_ref_counter -= 1


class FakePreparedTreeCache:
    def __init__(self, storage_backend: TensorcastStore) -> None:
        self.cache_controller = SimpleNamespace(storage_backend=storage_backend)
        self.hicache_storage_pass_prefix_keys = False
        self.inserted_token_ids: list[int] | None = None
        self.inserted_hash_values: list[str] | None = None
        self.last_host_node = FakePreparedHostNode()

    def insert_host_prefix(
        self,
        key,
        host_indices: torch.Tensor,
        hash_values: list[str] | None = None,
    ) -> int:
        assert host_indices.numel() == len(key)
        self.inserted_token_ids = list(key.token_ids)
        self.inserted_hash_values = list(hash_values or [])
        return 0

    def match_prefix(self, params):
        token_ids = list(params.key.token_ids)
        return SimpleNamespace(
            device_indices=torch.empty((0,), dtype=torch.int64),
            last_device_node=SimpleNamespace(backuped=False),
            last_host_node=self.last_host_node,
            host_hit_length=len(token_ids),
        )


@dataclass(eq=False)
class FakeGenerateReq:
    rid: str
    origin_input_ids: list[int]
    kv_committed_len: int
    output_ids: list[int]
    extra_key: str = ""
    aborted_reason: str | None = None

    def __post_init__(self) -> None:
        self.sampling_params = SimpleNamespace(n=1)

    def set_finish_with_abort(self, reason: str) -> None:
        self.aborted_reason = reason


@dataclass(frozen=True)
class FakeTokenizedGenerateReqInput:
    mm_inputs: object | None = None
    input_embeds: object | None = None
    session_params: object | None = None


def _build_scheduler(
    *,
    port: int,
    page_client: FakeTensorcastPageClient | None = None,
    extra_config: dict[str, object] | None = None,
    host: str = "127.0.0.1",
    rank_in_group: int = 0,
    broadcast_value: object | None = None,
) -> tuple[FakeScheduler, TensorcastStore]:
    resolved_page_client = page_client or FakeTensorcastPageClient()
    store = TensorcastStore(
        build_storage_config(tp_rank=0, extra_config=extra_config),
        FakeHostKVCache([1.0, 2.0, 3.0, 4.0], page_size=2),
        page_client=resolved_page_client,
    )
    scheduler = FakeScheduler()
    scheduler.enable_hicache_storage = True
    scheduler.tree_cache = FakePreparedTreeCache(store)
    scheduler.tp_rank = 0
    scheduler.pp_rank = 0
    scheduler.tp_size = 1
    scheduler.pp_size = 1
    scheduler.dp_size = 1
    scheduler.world_group = FakeWorldGroup(
        rank_in_group=rank_in_group,
        broadcast_value=broadcast_value,
    )
    scheduler.server_args = SimpleNamespace(host=host, port=port)
    scheduler.init_tensorcast_instance_ops()
    scheduler._configure_tensorcast_instance_ops_runtime()
    return scheduler, store


class FakeDirectoryRegistration:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


def _seed_ready_pages(store: TensorcastStore, logical_request_id: str) -> None:
    manager = store.request_bundle_manager
    request_state = manager.request_bundle_registry.get(logical_request_id)
    assert request_state is not None
    rank = request_state.required_ranks[0]
    page_client = store._page_client
    pages = manager.page_publication_registry.snapshot_rank(
        logical_request_id=logical_request_id,
        rank=rank,
    )
    for page in pages:
        page_client.data[page.page_hash] = torch.ones(
            (store.page_size,), dtype=torch.float32
        )


def _publish_request(
    prompt_token_ids: list[int],
    *,
    requested_at_ms: int,
    timeout_ms: int | None = None,
) -> PublishInstanceOpRequest:
    return PublishInstanceOpRequest(
        logical_request_id="rid-1",
        engine_request_id="rid-1",
        publish_op_id=f"publish-op-{requested_at_ms}",
        requested_cutoff_token_count=len(prompt_token_ids),
        timeout_ms=timeout_ms,
        prompt_token_digest=get_hash_str(prompt_token_ids),
        attention_arch="mha",
        dtype="torch.float32",
        page_size=2,
        requested_at_ms=requested_at_ms,
    )


def _install_clean_prepared_bundle(
    store: TensorcastStore,
    *,
    logical_request_id: str,
    publish_manifest_digest: str,
    prompt_token_ids: list[int],
) -> None:
    manager = store.request_bundle_manager
    manager.prepared_bundle_registry.begin_prepare(
        logical_request_id=logical_request_id,
        target_instance_id="instance-b",
        publish_manifest_digest=publish_manifest_digest,
        artifact_manifest_digest=f"artifact:{publish_manifest_digest}",
        engine_owned_manifest_sha256=f"engine:{publish_manifest_digest}",
        required_ranks=(RankCoord(tp_rank=0, pp_rank=0),),
        prompt_token_digest=get_hash_str(prompt_token_ids),
        cutoff_token_count=len(prompt_token_ids),
        now_ms=100,
    )
    manager.prepared_bundle_registry.mark_rank_install(
        logical_request_id=logical_request_id,
        publish_manifest_digest=publish_manifest_digest,
        rank=RankCoord(tp_rank=0, pp_rank=0),
        state=RankInstallState.READY,
        hydrated_page_count=max(1, len(prompt_token_ids) // store.page_size),
        runnable_prefix_tokens=len(prompt_token_ids),
    )
    manager.prepared_bundle_registry.mark_prepared(
        logical_request_id=logical_request_id,
        publish_manifest_digest=publish_manifest_digest,
        now_ms=110,
        prepared_hold_set_id=f"hold:{publish_manifest_digest}",
    )


def test_scheduler_instance_publish_engine_request_id_uses_live_request_shape() -> None:
    scheduler, store = _build_scheduler(port=30000)
    prompt_token_ids = [101, 102, 103, 104]
    req = FakeGenerateReq(
        rid="rid-1",
        origin_input_ids=prompt_token_ids,
        kv_committed_len=len(prompt_token_ids),
        output_ids=[],
    )
    scheduler._maybe_start_tensorcast_live_request(
        req,
        FakeTokenizedGenerateReqInput(),
    )
    _seed_ready_pages(store, "rid-1")

    publish_result = scheduler.tensorcast_instance_publish_engine_request_id(
        engine_request_id="rid-1",
        ttl_ms=45_000,
        timeout_ms=75_000,
        requested_at_ms=150,
        publish_op_id="publish-op-150",
    )

    assert publish_result.request_state.engine_request_id == "rid-1"
    assert publish_result.request_state.publish_op_id == "publish-op-150"
    assert publish_result.publish_manifest.prompt_token_digest == get_hash_str(
        prompt_token_ids
    )
    assert (
        publish_result.publish_manifest.engine_owned_manifest.payload.cutoff_token_count
        == len(prompt_token_ids)
    )
    assert publish_result.request_state.publish_op_id == "publish-op-150"


def test_scheduler_publish_instance_op_req_input_dispatches_to_live_request_publish() -> None:
    scheduler, store = _build_scheduler(port=30000)
    prompt_token_ids = [101, 102, 103, 104]
    req = FakeGenerateReq(
        rid="rid-1",
        origin_input_ids=prompt_token_ids,
        kv_committed_len=len(prompt_token_ids),
        output_ids=[],
    )
    scheduler._maybe_start_tensorcast_live_request(
        req,
        FakeTokenizedGenerateReqInput(),
    )
    _seed_ready_pages(store, "rid-1")

    response = scheduler.handle_tensorcast_publish_instance_op_request(
        PublishInstanceOpReqInput(
            engine_request_id="rid-1",
            ttl_ms=45_000,
            timeout_ms=90_000,
            requested_at_ms=150,
            publish_op_id="publish-op-150",
        )
    )

    assert isinstance(response, PublishInstanceOpRespOutput)
    assert response.status == InstanceOpStatus.SUCCESS
    assert response.result is not None
    assert response.result.request_state.engine_request_id == "rid-1"
    assert response.result.request_state.publish_op_id == "publish-op-150"


def test_scheduler_instance_publish_propagates_timeout_to_local_request() -> None:
    scheduler, store = _build_scheduler(port=30000)
    prompt_token_ids = [101, 102, 103, 104]
    req = FakeGenerateReq(
        rid="rid-1",
        origin_input_ids=prompt_token_ids,
        kv_committed_len=len(prompt_token_ids),
        output_ids=[],
    )
    scheduler._maybe_start_tensorcast_live_request(
        req,
        FakeTokenizedGenerateReqInput(),
    )
    _seed_ready_pages(store, "rid-1")

    seen_timeout_ms: list[int | None] = []
    original_publish_local = store.request_bundle_manager.instance_publish_local

    def _wrapped_publish_local(*, request: PublishInstanceOpRequest):
        seen_timeout_ms.append(request.timeout_ms)
        return original_publish_local(request=request)

    store.request_bundle_manager.instance_publish_local = _wrapped_publish_local
    try:
        scheduler.tensorcast_instance_publish_engine_request_id(
            engine_request_id="rid-1",
            ttl_ms=45_000,
            timeout_ms=88_000,
            requested_at_ms=150,
            publish_op_id="publish-op-150",
        )
    finally:
        store.request_bundle_manager.instance_publish_local = original_publish_local

    assert seen_timeout_ms == [88_000]


def test_scheduler_instance_hydrate_succeeds_without_worker_warmup() -> None:
    page_client = FakeTensorcastPageClient()
    source_scheduler, source_store = _build_scheduler(
        port=30000,
        page_client=page_client,
    )
    target_scheduler, _target_store = _build_scheduler(
        port=30001,
        page_client=page_client,
    )
    prompt_token_ids = [101, 102, 103, 104]
    req = FakeGenerateReq(
        rid="rid-1",
        origin_input_ids=prompt_token_ids,
        kv_committed_len=len(prompt_token_ids),
        output_ids=[],
    )
    source_scheduler._maybe_start_tensorcast_live_request(
        req,
        FakeTokenizedGenerateReqInput(),
    )
    _seed_ready_pages(source_store, "rid-1")

    publish_result = source_scheduler.tensorcast_instance_publish(
        _publish_request(prompt_token_ids, requested_at_ms=120)
    )
    assert publish_result.request_state.state == RequestBundleLifecycleState.PUBLISHED

    hydrate_result = target_scheduler.tensorcast_instance_hydrate(
        HydrateInstanceOpRequest(
            logical_request_id="rid-1",
            publish_manifest_digest=publish_result.publish_manifest.publish_manifest_digest,
            requested_at_ms=140,
        ),
        publish_manifest=publish_result.publish_manifest,
    )
    assert hydrate_result.prepared_bundle.state == PreparedBundleLifecycleState.PREPARED

    evict_result = target_scheduler.tensorcast_instance_evict_local(
        EvictLocalInstanceOpRequest(
            logical_request_id="rid-1",
            requested_at_ms=160,
        )
    )
    assert evict_result.status.value == "success"
    assert evict_result.evicted_bundle_count == 1


def test_scheduler_instance_binding_warns_and_falls_back_for_stale_bundle(
    caplog,
) -> None:
    target_scheduler, target_store = _build_scheduler(port=30020)
    prompt_token_ids = [301, 302, 303, 304]
    _install_clean_prepared_bundle(
        target_store,
        logical_request_id="rid-1",
        publish_manifest_digest="manifest-stale",
        prompt_token_ids=prompt_token_ids,
    )
    target_store.request_bundle_manager.prepared_bundle_registry.mark_stale(
        logical_request_id="rid-1",
        publish_manifest_digest="manifest-stale",
    )

    target_req = FakeGenerateReq(
        rid="rid-1",
        origin_input_ids=prompt_token_ids,
        kv_committed_len=len(prompt_token_ids),
        output_ids=[],
    )
    tokenized_req = FakeTokenizedGenerateReqInput()

    with caplog.at_level(logging.WARNING):
        assert target_scheduler._maybe_bind_tensorcast_prepared_bundle(
            target_req,
            tokenized_req,
        )

    assert target_scheduler._tensorcast_instance_state_for_req(target_req) is None
    assert any(
        "falling back to normal generate path" in record.message
        and "only stale or tainted prepared bundle records are available"
        in record.message
        for record in caplog.records
    )


def test_scheduler_instance_admission_hooks_bind_and_cleanup() -> None:
    page_client = FakeTensorcastPageClient()
    source_scheduler, source_store = _build_scheduler(
        port=30010,
        page_client=page_client,
    )
    target_scheduler, target_store = _build_scheduler(
        port=30011,
        page_client=page_client,
    )
    prompt_token_ids = [201, 202, 203, 204]
    source_req = FakeGenerateReq(
        rid="rid-1",
        origin_input_ids=prompt_token_ids,
        kv_committed_len=len(prompt_token_ids),
        output_ids=[],
    )
    tokenized_req = FakeTokenizedGenerateReqInput()
    source_scheduler._maybe_start_tensorcast_live_request(source_req, tokenized_req)
    _seed_ready_pages(source_store, "rid-1")

    publish_result = source_scheduler.tensorcast_instance_publish(
        _publish_request(prompt_token_ids, requested_at_ms=220)
    )
    target_scheduler.tensorcast_instance_hydrate(
        HydrateInstanceOpRequest(
            logical_request_id="rid-1",
            publish_manifest_digest=publish_result.publish_manifest.publish_manifest_digest,
            requested_at_ms=240,
        ),
        publish_manifest=publish_result.publish_manifest,
    )

    target_req = FakeGenerateReq(
        rid="rid-1",
        origin_input_ids=prompt_token_ids,
        kv_committed_len=len(prompt_token_ids),
        output_ids=[],
    )
    assert target_scheduler._maybe_bind_tensorcast_prepared_bundle(
        target_req, tokenized_req
    )
    target_state = target_scheduler._tensorcast_instance_state_for_req(target_req)
    assert target_state is not None
    assert target_state.prepared_bundle_record is not None
    assert target_state.prepared_bundle_key is not None
    assert target_state.prepared_bundle_claim_token is not None
    bound_hold_set_id = target_state.prepared_bundle_record.prepared_hold_set_id
    assert (
        target_state.prepared_bundle_record.state
        == PreparedBundleLifecycleState.ATTACHED
    )
    assert bound_hold_set_id is not None

    target_scheduler._maybe_consume_tensorcast_prepared_bundle(target_req)
    target_state = target_scheduler._tensorcast_instance_state_for_req(target_req)
    assert target_state is not None
    assert target_state.prepared_bundle_record is not None
    assert (
        target_state.prepared_bundle_record.state
        == PreparedBundleLifecycleState.CONSUMED
    )
    assert target_state.prepared_bundle_host_node is not None
    assert target_scheduler.tree_cache.inserted_token_ids == prompt_token_ids
    assert target_scheduler.tree_cache.inserted_hash_values is not None
    assert len(target_scheduler.tree_cache.inserted_hash_values) == 2
    hold_set = target_store.request_bundle_manager.prepared_hold_registry.get(
        bound_hold_set_id
    )
    assert hold_set is not None
    assert hold_set.state.value == "released"
    assert target_scheduler.tree_cache.last_host_node.host_ref_counter == 1

    target_scheduler._maybe_cleanup_tensorcast_prepared_bundle(target_req)
    assert target_scheduler._tensorcast_instance_state_for_req(target_req) is None
    assert target_scheduler.tree_cache.last_host_node.host_ref_counter == 0

    source_scheduler._maybe_cleanup_tensorcast_live_request(source_req)
    assert source_scheduler._tensorcast_instance_state_for_req(source_req) is None
    source_request_state = (
        source_store.request_bundle_manager.request_bundle_registry.get("rid-1")
    )
    assert source_request_state is not None
    assert source_request_state.state == RequestBundleLifecycleState.SOURCE_RETAINED
    target_records = target_store.request_bundle_manager.prepared_bundle_registry.list_request_records(
        logical_request_id="rid-1",
        include_evicted=True,
    )
    assert len(target_records) == 1
    assert target_records[0].state == PreparedBundleLifecycleState.EVICTED


def test_scheduler_instance_directory_registration_is_rank0_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[str, str, FakeDirectoryRegistration]] = []

    def _fake_builder(
        self: FakeScheduler,
        *,
        storage_backend: TensorcastStore,
        instance_id: str,
        execution_endpoint: str,
    ) -> FakeDirectoryRegistration:
        assert storage_backend.tensorcast_config.instance_directory_address
        registration = FakeDirectoryRegistration()
        created.append((instance_id, execution_endpoint, registration))
        return registration

    monkeypatch.setattr(
        FakeScheduler,
        "_build_tensorcast_instance_directory_registration",
        _fake_builder,
    )
    extra_config = {
        "daemon_address": "127.0.0.1:50052",
        "namespace": "unit-test",
        "instance_directory_address": "127.0.0.1:50051",
    }

    scheduler_rank0, _ = _build_scheduler(
        port=30030,
        extra_config=extra_config,
    )
    assert len(created) == 1
    assert created[0][0] == "127.0.0.1:30030"
    assert created[0][1] == "127.0.0.1:30030"
    assert created[0][2].start_calls == 1
    scheduler_rank0._shutdown_tensorcast_instance_ops_runtime()
    assert created[0][2].stop_calls == 1

    created.clear()
    _build_scheduler(
        port=30031,
        extra_config=extra_config,
        rank_in_group=1,
        broadcast_value="epoch-from-rank0",
    )
    assert created == []


def test_scheduler_instance_directory_registration_skips_wildcard_host(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _unexpected_builder(
        self: FakeScheduler,
        *,
        storage_backend: TensorcastStore,
        instance_id: str,
        execution_endpoint: str,
    ) -> FakeDirectoryRegistration:
        raise AssertionError("directory registration builder should not be called")

    monkeypatch.setattr(
        FakeScheduler,
        "_build_tensorcast_instance_directory_registration",
        _unexpected_builder,
    )

    with caplog.at_level(logging.WARNING):
        _build_scheduler(
            port=30032,
            host="0.0.0.0",
            extra_config={
                "daemon_address": "127.0.0.1:50052",
                "namespace": "unit-test",
                "instance_directory_address": "127.0.0.1:50051",
            },
        )

    assert any(
        "directory registration skipped" in record.message for record in caplog.records
    )
