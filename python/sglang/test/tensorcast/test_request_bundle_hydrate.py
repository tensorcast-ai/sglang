# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import pytest

from sglang.srt.tensorcast.request_bundle.request_bundle_hydrate import (
    RequestBundleHydrateAggregator,
    RequestBundleHydrator,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_publish import (
    RequestBundlePublishAggregator,
    RequestBundlePublisher,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    PagePublicationRegistry,
    PreparedBundleRegistry,
    PreparedHoldRegistry,
    RequestBundleStateRegistry,
    RequestBundleStateError,
)
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    HydrateInstanceOpRequest,
    PublishInstanceOpRequest,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    HydrateRankInstallResult,
    HydrateTargetCompatibility,
    PageClosureEntry,
    PagePublicationState,
    PreparedHoldRef,
    PreparedHoldSetState,
    PreparedSlotToken,
    PublishManifestRecord,
    RankCoord,
    RankInstallState,
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
    **overrides: object,
) -> HydrateTargetCompatibility:
    compatibility = publish_manifest.engine_owned_manifest.payload.compatibility
    payload: dict[str, object] = {
        "target_instance_id": "instance-b",
        "model_fingerprint": compatibility.model_fingerprint,
        "kv_layout_id": compatibility.kv_layout_id,
        "dtype": compatibility.dtype,
        "page_size": compatibility.page_size,
        "tp_size": compatibility.tp_size,
        "pp_size": compatibility.pp_size,
        "attention_arch": compatibility.attention_arch,
        "required_ranks": compatibility.required_ranks,
    }
    payload.update(overrides)
    return HydrateTargetCompatibility.model_validate(payload)


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


def _seed_group_published_manifest() -> PublishManifestRecord:
    local_rank0 = _seed_local_publish_result(_rank(0))
    local_rank1 = _seed_local_publish_result(_rank(1))
    aggregated = RequestBundlePublishAggregator().aggregate(
        request=_publish_request(),
        local_results=(local_rank0, local_rank1),
        now_ms=180,
        required_ranks=(_rank(0), _rank(1)),
    )
    return aggregated.publish_manifest


def _ready_install_result(rank: RankCoord, suffix: str) -> HydrateRankInstallResult:
    return HydrateRankInstallResult(
        rank=rank,
        state=RankInstallState.READY,
        hydrated_page_count=2,
        runnable_prefix_tokens=64,
        local_install_handle=f"install-{suffix}",
        hold_refs=(
            PreparedHoldRef(
                logical_page_index=0,
                page_hash=f"{suffix}-page-0",
                slot_token=PreparedSlotToken(
                    slot_index=rank.tp_rank * 10,
                    slot_generation=1,
                ),
                artifact_id=f"artifact-{suffix}-0",
            ),
            PreparedHoldRef(
                logical_page_index=1,
                page_hash=f"{suffix}-page-1",
                slot_token=PreparedSlotToken(
                    slot_index=rank.tp_rank * 10 + 1,
                    slot_generation=1,
                ),
                artifact_id=f"artifact-{suffix}-1",
            ),
        ),
    )


def test_local_hydrate_fails_closed_on_compatibility_mismatch() -> None:
    publish_manifest = _seed_group_published_manifest()
    prepared_bundle_registry = PreparedBundleRegistry()
    prepared_hold_registry = PreparedHoldRegistry()

    with pytest.raises(RequestBundleStateError, match="model_fingerprint mismatch"):
        RequestBundleHydrator(
            prepared_bundle_registry=prepared_bundle_registry,
            prepared_hold_registry=prepared_hold_registry,
        ).hydrate(
            request=_hydrate_request(publish_manifest),
            publish_manifest=publish_manifest,
            target=_target_from_manifest(
                publish_manifest,
                model_fingerprint="different-model",
            ),
            local_rank=_rank(0),
            install_rank=lambda _: _ready_install_result(_rank(0), "unused"),
            now_ms=200,
        )

    assert (
        prepared_bundle_registry.get(
            logical_request_id="rid-1",
            publish_manifest_digest=publish_manifest.publish_manifest_digest,
        )
        is None
    )


def test_local_hydrate_failure_marks_local_bundle_failed() -> None:
    publish_manifest = _seed_group_published_manifest()
    prepared_bundle_registry = PreparedBundleRegistry()
    prepared_hold_registry = PreparedHoldRegistry()

    with pytest.raises(
        RequestBundleStateError,
        match="decode-usability check failed",
    ):
        RequestBundleHydrator(
            prepared_bundle_registry=prepared_bundle_registry,
            prepared_hold_registry=prepared_hold_registry,
        ).hydrate(
            request=_hydrate_request(publish_manifest),
            publish_manifest=publish_manifest,
            target=_target_from_manifest(publish_manifest),
            local_rank=_rank(1),
            install_rank=lambda work_item: HydrateRankInstallResult(
                rank=work_item.rank,
                state=RankInstallState.FAILED,
                hydrated_page_count=1,
                runnable_prefix_tokens=32,
                error_message="decode-usability check failed",
            ),
            now_ms=200,
        )

    prepared_bundle = prepared_bundle_registry.get(
        logical_request_id="rid-1",
        publish_manifest_digest=publish_manifest.publish_manifest_digest,
    )
    assert prepared_bundle is not None
    assert prepared_bundle.state.value == "failed"
    assert prepared_bundle.required_ranks == (_rank(1),)
    assert prepared_bundle.rank_installs[0].state == RankInstallState.FAILED
    assert prepared_bundle.prepared_hold_set_id is None


def test_local_hydrate_installs_prepared_bundle_and_hold_set() -> None:
    publish_manifest = _seed_group_published_manifest()
    prepared_bundle_registry = PreparedBundleRegistry()
    prepared_hold_registry = PreparedHoldRegistry()

    result = RequestBundleHydrator(
        prepared_bundle_registry=prepared_bundle_registry,
        prepared_hold_registry=prepared_hold_registry,
    ).hydrate(
        request=_hydrate_request(publish_manifest),
        publish_manifest=publish_manifest,
        target=_target_from_manifest(publish_manifest),
        local_rank=_rank(0),
        install_rank=lambda work_item: _ready_install_result(
            work_item.rank,
            f"rank{work_item.rank.tp_rank}",
        ),
        now_ms=200,
    )

    assert result.reused_existing is False
    assert result.prepared_bundle.state.value == "prepared"
    assert result.prepared_bundle.required_ranks == (_rank(0),)
    assert result.prepared_bundle.prompt_token_digest == "prompt-digest-1"
    assert result.prepared_bundle.cutoff_token_count == 70
    assert result.prepared_bundle.tail_valid_tokens == 6
    assert result.hold_set is not None
    assert result.hold_set.state == PreparedHoldSetState.ACTIVE
    assert len(result.hold_set.refs) == 2


def test_local_hydrate_with_same_manifest_is_idempotent_retry() -> None:
    publish_manifest = _seed_group_published_manifest()
    prepared_bundle_registry = PreparedBundleRegistry()
    prepared_hold_registry = PreparedHoldRegistry()
    install_calls: list[tuple[int, int]] = []
    hydrator = RequestBundleHydrator(
        prepared_bundle_registry=prepared_bundle_registry,
        prepared_hold_registry=prepared_hold_registry,
    )

    def install_rank(work_item) -> HydrateRankInstallResult:
        install_calls.append(work_item.rank.as_key())
        return _ready_install_result(
            work_item.rank,
            f"rank{work_item.rank.tp_rank}",
        )

    first = hydrator.hydrate(
        request=_hydrate_request(publish_manifest),
        publish_manifest=publish_manifest,
        target=_target_from_manifest(publish_manifest),
        local_rank=_rank(0),
        install_rank=install_rank,
        now_ms=200,
    )
    second = hydrator.hydrate(
        request=_hydrate_request(publish_manifest),
        publish_manifest=publish_manifest,
        target=_target_from_manifest(publish_manifest),
        local_rank=_rank(0),
        install_rank=install_rank,
        now_ms=210,
    )

    assert first.reused_existing is False
    assert second.reused_existing is True
    assert install_calls == [(0, 0)]
    assert second.prepared_bundle == first.prepared_bundle


def test_hydrate_aggregator_combines_local_rank_results() -> None:
    publish_manifest = _seed_group_published_manifest()
    target = _target_from_manifest(publish_manifest)
    local_results = []
    for rank in target.required_ranks:
        local_results.append(
            RequestBundleHydrator(
                prepared_bundle_registry=PreparedBundleRegistry(),
                prepared_hold_registry=PreparedHoldRegistry(),
            ).hydrate(
                request=_hydrate_request(publish_manifest),
                publish_manifest=publish_manifest,
                target=target,
                local_rank=rank,
                install_rank=lambda work_item, rank=rank: _ready_install_result(
                    rank,
                    f"rank{rank.tp_rank}",
                ),
                now_ms=220 + rank.tp_rank,
            )
        )

    aggregated = RequestBundleHydrateAggregator().aggregate(
        request=_hydrate_request(publish_manifest),
        publish_manifest=publish_manifest,
        target=target,
        local_results=tuple(local_results),
        now_ms=260,
    )

    assert aggregated.prepared_bundle.state.value == "prepared"
    assert aggregated.prepared_bundle.required_ranks == (_rank(0), _rank(1))
    assert [
        install.rank_coord() for install in aggregated.prepared_bundle.rank_installs
    ] == [
        _rank(0),
        _rank(1),
    ]
    assert aggregated.hold_set is not None
    assert aggregated.hold_set.state == PreparedHoldSetState.ACTIVE
    assert len(aggregated.hold_set.refs) == 4


def test_hydrate_aggregator_rejects_missing_required_rank_result() -> None:
    publish_manifest = _seed_group_published_manifest()
    target = _target_from_manifest(publish_manifest)
    local_rank0 = RequestBundleHydrator(
        prepared_bundle_registry=PreparedBundleRegistry(),
        prepared_hold_registry=PreparedHoldRegistry(),
    ).hydrate(
        request=_hydrate_request(publish_manifest),
        publish_manifest=publish_manifest,
        target=target,
        local_rank=_rank(0),
        install_rank=lambda work_item: _ready_install_result(work_item.rank, "rank0"),
        now_ms=220,
    )

    with pytest.raises(
        RequestBundleStateError,
        match="missing local hydrate results",
    ):
        RequestBundleHydrateAggregator().aggregate(
            request=_hydrate_request(publish_manifest),
            publish_manifest=publish_manifest,
            target=target,
            local_results=(local_rank0,),
            now_ms=260,
        )
