# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from sglang.test.tensorcast.test_support import install_memory_pool_host_stub

install_memory_pool_host_stub()

from sglang.srt.tensorcast.request_bundle.request_bundle_hydrate import (
    RequestBundleHydrator,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_publish import (
    RequestBundlePublishAggregator,
    RequestBundlePublisher,
)
from sglang.srt.tensorcast.instance_ops.instance_ops_runtime import (
    InstanceOpsRuntimeCoordinator,
    _LocalOpEnvelope,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    PagePublicationRegistry,
    PreparedBundleRegistry,
    PreparedHoldRegistry,
    RequestBundleStateRegistry,
    RequestBundleStateError,
)
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    EvictLocalInstanceOpRequest,
    EvictLocalInstanceOpResult,
    HydrateInstanceOpRequest,
    InstanceOpStatus,
    PublishInstanceOpRequest,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    HydratePreparedResult,
    HydrateRankInstallResult,
    HydrateTargetCompatibility,
    PageClosureEntry,
    PagePublicationState,
    PreparedBundleLifecycleState,
    PublishManifestRecord,
    RankCoord,
    RankInstallState,
    RequestBundleLifecycleState,
    SourcePublishClosureResult,
)


def _rank(tp_rank: int, pp_rank: int = 0) -> RankCoord:
    return RankCoord(tp_rank=tp_rank, pp_rank=pp_rank)


def _publish_request() -> PublishInstanceOpRequest:
    return PublishInstanceOpRequest(
        logical_request_id="rid-1",
        engine_request_id="rid-1",
        publish_op_id="publish-op-1",
        requested_cutoff_token_count=70,
        timeout_ms=None,
        ttl_ms=60_000,
        prompt_token_digest="prompt-digest-1",
        transfer_mode="prefill_closed_prompt_reuse",
        attention_arch="mha",
        dtype="float16",
        page_size=32,
        requested_at_ms=100,
    )


def _hydrate_request(
    publish_manifest: PublishManifestRecord,
) -> HydrateInstanceOpRequest:
    return HydrateInstanceOpRequest(
        logical_request_id="rid-1",
        publish_manifest_digest=publish_manifest.publish_manifest_digest,
        requested_at_ms=200,
    )


def _target_from_manifest(
    publish_manifest: PublishManifestRecord,
) -> HydrateTargetCompatibility:
    compatibility = publish_manifest.engine_owned_manifest.payload.compatibility
    return HydrateTargetCompatibility(
        target_instance_id="instance-b",
        model_fingerprint=compatibility.model_fingerprint,
        kv_layout_id=compatibility.kv_layout_id,
        dtype=compatibility.dtype,
        page_size=compatibility.page_size,
        tp_size=compatibility.tp_size,
        pp_size=compatibility.pp_size,
        attention_arch=compatibility.attention_arch,
        required_ranks=compatibility.required_ranks,
    )


def _seed_local_publish_result(rank: RankCoord) -> SourcePublishClosureResult:
    page_registry = PagePublicationRegistry()
    bundle_registry = RequestBundleStateRegistry(
        page_publication_registry=page_registry
    )
    rank_pages = (
        PageClosureEntry(
            logical_page_index=0,
            page_hash=f"r{rank.tp_rank}-p0",
            publication_state=PagePublicationState.READY,
            artifact_id=f"artifact-r{rank.tp_rank}-p0",
            host_resident=True,
        ),
        PageClosureEntry(
            logical_page_index=1,
            page_hash=f"r{rank.tp_rank}-p1",
            publication_state=PagePublicationState.READY,
            artifact_id=f"artifact-r{rank.tp_rank}-p1",
            host_resident=True,
        ),
        PageClosureEntry(
            logical_page_index=2,
            page_hash=f"r{rank.tp_rank}-p2",
            publication_state=PagePublicationState.ABSENT,
            host_resident=True,
        ),
    )
    bundle_registry.upsert_live_request(
        logical_request_id="rid-1",
        instance_id="instance-a",
        engine_request_id="rid-1",
        full_prompt_token_count=70,
        model_fingerprint="model-a",
        kv_layout_id="layout-v1",
        tp_size=2,
        pp_size=1,
        required_ranks=(rank,),
        now_ms=90,
    )
    for page in rank_pages:
        page_registry.set_page_state(
            logical_request_id="rid-1",
            rank=rank,
            logical_page_index=page.logical_page_index,
            page_hash=page.page_hash,
            publication_state=page.publication_state,
            artifact_id=page.artifact_id,
            host_resident=page.host_resident,
            updated_at_ms=95,
        )
    bundle_registry.update_rank_frontier(
        logical_request_id="rid-1",
        rank=rank,
        latest_token_count=70,
        latest_last_page_index=2,
        now_ms=100 + rank.tp_rank,
    )
    return RequestBundlePublisher(
        request_bundle_registry=bundle_registry,
        page_publication_registry=page_registry,
    ).publish(
        request=_publish_request(),
        now_ms=150,
    )


def _seed_group_published_manifest() -> tuple[
    SourcePublishClosureResult,
    SourcePublishClosureResult,
    PublishManifestRecord,
]:
    local_rank0 = _seed_local_publish_result(_rank(0))
    local_rank1 = _seed_local_publish_result(_rank(1))
    aggregated = RequestBundlePublishAggregator().aggregate(
        request=_publish_request(),
        local_results=(local_rank0, local_rank1),
        now_ms=180,
        required_ranks=(_rank(0), _rank(1)),
    )
    return local_rank0, local_rank1, aggregated.publish_manifest


def _local_hydrate_result(
    *,
    rank: RankCoord,
    publish_manifest: PublishManifestRecord,
) -> HydratePreparedResult:
    prepared_bundle_registry = PreparedBundleRegistry()
    prepared_hold_registry = PreparedHoldRegistry()
    return RequestBundleHydrator(
        prepared_bundle_registry=prepared_bundle_registry,
        prepared_hold_registry=prepared_hold_registry,
    ).hydrate(
        request=_hydrate_request(publish_manifest),
        publish_manifest=publish_manifest,
        target=_target_from_manifest(publish_manifest),
        local_rank=rank,
        install_rank=lambda work_item: HydrateRankInstallResult(
            rank=work_item.rank,
            state=RankInstallState.READY,
            hydrated_page_count=len(work_item.artifact_entries),
            runnable_prefix_tokens=(
                work_item.cutoff_token_count - work_item.tail_valid_tokens
            ),
            local_install_handle=f"install-rank{work_item.rank.tp_rank}",
        ),
        now_ms=220,
    )


@dataclass
class FakeObjectGroup:
    gathered_followers: tuple[Any, ...] = ()
    gathered_inputs: list[Any] | None = None
    broadcast_inputs: list[tuple[int, Any]] | None = None

    def __post_init__(self) -> None:
        if self.gathered_inputs is None:
            self.gathered_inputs = []
        if self.broadcast_inputs is None:
            self.broadcast_inputs = []

    def all_gather_object(self, obj: Any) -> list[Any]:
        assert self.gathered_inputs is not None
        self.gathered_inputs.append(obj)
        return [obj, *self.gathered_followers]

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        assert self.broadcast_inputs is not None
        self.broadcast_inputs.append((src, obj))
        return obj


def test_runtime_publish_aggregates_group_result_and_reuses_cache() -> None:
    local_rank0, local_rank1, _ = _seed_group_published_manifest()
    group = FakeObjectGroup(
        gathered_followers=(
            _LocalOpEnvelope(
                rank=_rank(1),
                status=InstanceOpStatus.SUCCESS,
                result=local_rank1,
            ),
        )
    )
    runtime = InstanceOpsRuntimeCoordinator(
        current_rank=_rank(0),
        object_group=group,
    )
    runtime.configure(
        instance_id="instance-a",
        coordinator_epoch="epoch-1",
        required_ranks=(_rank(0), _rank(1)),
        now_ms=100,
    )
    publish_call_count = 0
    seen_timeout_ms: list[int | None] = []

    def execute_local(request: PublishInstanceOpRequest) -> SourcePublishClosureResult:
        nonlocal publish_call_count
        publish_call_count += 1
        seen_timeout_ms.append(request.timeout_ms)
        return local_rank0

    result = runtime.publish(
        request=_publish_request(),
        execute_local=execute_local,
    )

    assert result.request_state.state == RequestBundleLifecycleState.PUBLISHED
    assert result.request_state.required_ranks == (_rank(0), _rank(1))
    assert result.publish_manifest.artifact_manifest.required_ranks == (
        _rank(0),
        _rank(1),
    )
    assert publish_call_count == 1
    assert seen_timeout_ms == [30_000]

    cached = runtime.publish(
        request=_publish_request(),
        execute_local=lambda _request: pytest.fail(
            "cached publish should not re-execute"
        ),
    )

    assert cached == result
    assert publish_call_count == 1


def test_runtime_publish_fails_closed_and_caches_failure() -> None:
    local_rank0, _, _ = _seed_group_published_manifest()
    group = FakeObjectGroup(
        gathered_followers=(
            _LocalOpEnvelope(
                rank=_rank(1),
                status=InstanceOpStatus.FAILED,
                error_message="rank-1 page flush failed",
            ),
        )
    )
    runtime = InstanceOpsRuntimeCoordinator(
        current_rank=_rank(0),
        object_group=group,
    )
    runtime.configure(
        instance_id="instance-a",
        coordinator_epoch="epoch-1",
        required_ranks=(_rank(0), _rank(1)),
        now_ms=100,
    )
    publish_call_count = 0

    def execute_local(_request: PublishInstanceOpRequest) -> SourcePublishClosureResult:
        nonlocal publish_call_count
        publish_call_count += 1
        return local_rank0

    with pytest.raises(RequestBundleStateError, match="rank-1 page flush failed"):
        runtime.publish(
            request=_publish_request(),
            execute_local=execute_local,
        )

    with pytest.raises(RequestBundleStateError, match="rank-1 page flush failed"):
        runtime.publish(
            request=_publish_request(),
            execute_local=lambda _request: pytest.fail(
                "cached failure should not re-execute"
            ),
        )

    assert publish_call_count == 1


def test_runtime_hydrate_aggregates_group_prepared_result() -> None:
    _, _, publish_manifest = _seed_group_published_manifest()
    local_rank0 = _local_hydrate_result(
        rank=_rank(0),
        publish_manifest=publish_manifest,
    )
    local_rank1 = _local_hydrate_result(
        rank=_rank(1),
        publish_manifest=publish_manifest,
    )
    group = FakeObjectGroup(
        gathered_followers=(
            _LocalOpEnvelope(
                rank=_rank(1),
                status=InstanceOpStatus.SUCCESS,
                result=local_rank1,
            ),
        )
    )
    runtime = InstanceOpsRuntimeCoordinator(
        current_rank=_rank(0),
        object_group=group,
    )
    runtime.configure(
        instance_id="instance-b",
        coordinator_epoch="epoch-2",
        required_ranks=(_rank(0), _rank(1)),
        now_ms=200,
    )

    result = runtime.hydrate(
        request=_hydrate_request(publish_manifest),
        publish_manifest=publish_manifest,
        target=_target_from_manifest(publish_manifest),
        execute_local=lambda: local_rank0,
    )

    assert result.prepared_bundle.state == PreparedBundleLifecycleState.PREPARED
    assert result.prepared_bundle.required_ranks == (_rank(0), _rank(1))
    assert len(result.prepared_bundle.rank_installs) == 2


def test_runtime_evict_local_sums_per_rank_counts() -> None:
    group = FakeObjectGroup(
        gathered_followers=(
            _LocalOpEnvelope(
                rank=_rank(1),
                status=InstanceOpStatus.SUCCESS,
                result=EvictLocalInstanceOpResult(
                    status=InstanceOpStatus.SUCCESS,
                    logical_request_id="rid-1",
                    evicted_bundle_count=2,
                    evicted_hold_set_count=3,
                ),
            ),
        )
    )
    runtime = InstanceOpsRuntimeCoordinator(
        current_rank=_rank(0),
        object_group=group,
    )
    runtime.configure(
        instance_id="instance-a",
        coordinator_epoch="epoch-3",
        required_ranks=(_rank(0), _rank(1)),
        now_ms=300,
    )

    result = runtime.evict_local(
        request=EvictLocalInstanceOpRequest(
            logical_request_id="rid-1",
            requested_at_ms=310,
        ),
        execute_local=lambda: EvictLocalInstanceOpResult(
            status=InstanceOpStatus.SUCCESS,
            logical_request_id="rid-1",
            evicted_bundle_count=1,
            evicted_hold_set_count=4,
        ),
    )

    assert result.status == InstanceOpStatus.SUCCESS
    assert result.evicted_bundle_count == 3
    assert result.evicted_hold_set_count == 7
