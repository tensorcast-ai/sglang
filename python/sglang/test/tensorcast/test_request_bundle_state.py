from __future__ import annotations

import pytest

from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    PagePublicationRegistry,
    PreparedBundleRegistry,
    PreparedHoldRegistry,
    RequestBundleStateRegistry,
    RequestBundleStateError,
    aggregate_required_rank_results,
    resolve_page_closure_cutoff,
    compute_bundle_digest,
)
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    InstanceOpKind,
    InstanceOpStatus,
    PerRankInstanceOpResult,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    PagePublicationState,
    PreparedBundleClaimAction,
    PreparedBundleLifecycleState,
    PreparedHoldRef,
    PreparedHoldSetState,
    PreparedSlotToken,
    RankCoord,
    RankInstallState,
    RequestBundleLifecycleState,
)


def _rank(tp_rank: int, pp_rank: int = 0) -> RankCoord:
    return RankCoord(tp_rank=tp_rank, pp_rank=pp_rank)


def test_resolve_page_closure_cutoff_keeps_full_prompt_and_records_tail() -> None:
    cutoff = resolve_page_closure_cutoff(
        requested_token_count=70,
        materialized_token_count=70,
        page_size=32,
    )

    assert cutoff.cutoff_token_count == 70
    assert cutoff.page_aligned_token_count == 64
    assert cutoff.frozen_last_page_index == 1
    assert cutoff.tail_valid_tokens == 6


def test_page_publication_registry_snapshots_rank_pages_in_order() -> None:
    registry = PagePublicationRegistry()
    registry.set_page_state(
        logical_request_id="rid-1",
        rank=_rank(0),
        logical_page_index=1,
        page_hash="page-1",
        publication_state=PagePublicationState.INFLIGHT,
        updated_at_ms=100,
        host_resident=True,
    )
    registry.set_page_state(
        logical_request_id="rid-1",
        rank=_rank(0),
        logical_page_index=0,
        page_hash="page-0",
        publication_state=PagePublicationState.READY,
        artifact_id="artifact-0",
        updated_at_ms=110,
        host_resident=True,
    )
    registry.set_page_state(
        logical_request_id="rid-1",
        rank=_rank(0),
        logical_page_index=2,
        page_hash="page-2",
        publication_state=PagePublicationState.FAILED,
        updated_at_ms=120,
        host_resident=False,
        last_error="transport timeout",
    )

    snapshot = registry.snapshot_rank(logical_request_id="rid-1", rank=_rank(0))

    assert [entry.logical_page_index for entry in snapshot] == [0, 1, 2]
    assert snapshot[0].publication_state == PagePublicationState.READY
    assert snapshot[0].artifact_id == "artifact-0"
    assert snapshot[2].last_error == "transport timeout"


def test_page_publication_registry_matches_page_hashes_across_requests() -> None:
    registry = PagePublicationRegistry()
    registry.set_page_state(
        logical_request_id="rid-1",
        rank=_rank(0),
        logical_page_index=0,
        page_hash="shared-page",
        publication_state=PagePublicationState.READY,
        artifact_id="artifact-1",
        updated_at_ms=100,
        host_resident=True,
    )
    registry.set_page_state(
        logical_request_id="rid-2",
        rank=_rank(0),
        logical_page_index=1,
        page_hash="shared-page",
        publication_state=PagePublicationState.INFLIGHT,
        updated_at_ms=101,
        host_resident=True,
    )
    registry.set_page_state(
        logical_request_id="rid-3",
        rank=_rank(1),
        logical_page_index=0,
        page_hash="other-page",
        publication_state=PagePublicationState.ABSENT,
        updated_at_ms=102,
        host_resident=True,
    )

    matched = registry.matching_page_hashes(
        page_hashes=("shared-page",),
        rank=_rank(0),
    )

    assert [
        (record.logical_request_id, record.logical_page_index) for record in matched
    ] == [
        ("rid-1", 0),
        ("rid-2", 1),
    ]


def test_request_bundle_publish_state_machine_and_retry_generation() -> None:
    publication_registry = PagePublicationRegistry()
    for rank in (_rank(0), _rank(1)):
        publication_registry.set_page_state(
            logical_request_id="rid-1",
            rank=rank,
            logical_page_index=0,
            page_hash=f"{rank.tp_rank}-page-0",
            publication_state=PagePublicationState.READY,
            artifact_id=f"artifact-{rank.tp_rank}-0",
            updated_at_ms=100,
            host_resident=True,
        )
        publication_registry.set_page_state(
            logical_request_id="rid-1",
            rank=rank,
            logical_page_index=1,
            page_hash=f"{rank.tp_rank}-page-1",
            publication_state=PagePublicationState.INFLIGHT,
            updated_at_ms=101,
            host_resident=True,
        )
        publication_registry.set_page_state(
            logical_request_id="rid-1",
            rank=rank,
            logical_page_index=2,
            page_hash=f"{rank.tp_rank}-page-2",
            publication_state=PagePublicationState.ABSENT,
            updated_at_ms=102,
            host_resident=True,
        )

    registry = RequestBundleStateRegistry(
        page_publication_registry=publication_registry
    )
    registry.upsert_live_request(
        logical_request_id="rid-1",
        instance_id="instance-a",
        engine_request_id="rid-1",
        full_prompt_token_count=70,
        model_fingerprint="model-a",
        kv_layout_id="layout-v1",
        tp_size=2,
        pp_size=1,
        required_ranks=(_rank(0), _rank(1)),
        now_ms=90,
    )
    registry.update_rank_frontier(
        logical_request_id="rid-1",
        rank=_rank(0),
        latest_token_count=70,
        latest_last_page_index=2,
        now_ms=95,
    )
    registry.update_rank_frontier(
        logical_request_id="rid-1",
        rank=_rank(1),
        latest_token_count=80,
        latest_last_page_index=2,
        now_ms=96,
    )

    request_state, cutoff = registry.begin_publish(
        logical_request_id="rid-1",
        publish_op_id="publish-1",
        requested_cutoff_token_count=70,
        page_size=32,
        now_ms=100,
    )

    assert request_state.state == RequestBundleLifecycleState.SNAPSHOT_CLOSING
    assert request_state.snapshot_seq == 1
    assert cutoff.cutoff_token_count == 70
    assert cutoff.page_aligned_token_count == 64
    assert cutoff.frozen_last_page_index == 1
    assert all(
        snapshot.frozen_cutoff_token_count == 70
        for snapshot in request_state.rank_snapshots
    )
    assert all(
        len(snapshot.ordered_pages) == 2 for snapshot in request_state.rank_snapshots
    )

    request_state = registry.mark_closing_tail_flush(
        logical_request_id="rid-1",
        now_ms=110,
    )
    assert request_state.state == RequestBundleLifecycleState.CLOSING_TAIL_FLUSH

    bundle_digest = compute_bundle_digest(rank_snapshots=request_state.rank_snapshots)
    request_state = registry.mark_closure_ready(
        logical_request_id="rid-1",
        bundle_digest=bundle_digest,
        now_ms=120,
    )
    assert request_state.state == RequestBundleLifecycleState.CLOSURE_READY
    assert request_state.bundle_digest == bundle_digest

    request_state = registry.mark_publish_failed(
        logical_request_id="rid-1",
        error_message="rank 1 tail page flush failed",
        now_ms=125,
    )
    assert request_state.state == RequestBundleLifecycleState.PUBLISH_FAILED
    assert request_state.last_error == "rank 1 tail page flush failed"

    retried_state, retried_cutoff = registry.begin_publish(
        logical_request_id="rid-1",
        publish_op_id="publish-2",
        requested_cutoff_token_count=64,
        page_size=32,
        now_ms=130,
    )
    assert retried_state.snapshot_seq == 2
    assert retried_cutoff.cutoff_token_count == 64


def test_request_bundle_can_publish_from_source_retained_state() -> None:
    publication_registry = PagePublicationRegistry()
    publication_registry.set_page_state(
        logical_request_id="rid-retained",
        rank=_rank(0),
        logical_page_index=0,
        page_hash="page-0",
        publication_state=PagePublicationState.READY,
        artifact_id="artifact-0",
        updated_at_ms=100,
        host_resident=True,
    )
    registry = RequestBundleStateRegistry(
        page_publication_registry=publication_registry
    )
    registry.upsert_live_request(
        logical_request_id="rid-retained",
        instance_id="instance-a",
        engine_request_id="rid-retained",
        full_prompt_token_count=32,
        model_fingerprint="model-a",
        kv_layout_id="layout-v1",
        tp_size=1,
        pp_size=1,
        required_ranks=(_rank(0),),
        now_ms=90,
    )
    registry.update_rank_frontier(
        logical_request_id="rid-retained",
        rank=_rank(0),
        latest_token_count=32,
        latest_last_page_index=0,
        now_ms=95,
    )

    retained_state = registry.mark_source_retained(
        logical_request_id="rid-retained",
        retained_until_ms=190,
        now_ms=130,
    )
    assert retained_state.state == RequestBundleLifecycleState.SOURCE_RETAINED
    assert retained_state.retained_until_ms == 190

    request_state, cutoff = registry.begin_publish(
        logical_request_id="rid-retained",
        publish_op_id="publish-retained",
        requested_cutoff_token_count=32,
        page_size=32,
        now_ms=140,
    )

    assert request_state.state == RequestBundleLifecycleState.SNAPSHOT_CLOSING
    assert request_state.snapshot_seq == 1
    assert cutoff.cutoff_token_count == 32


def test_request_bundle_mark_cleaned_accepts_source_retained_state() -> None:
    registry = RequestBundleStateRegistry()
    registry.upsert_live_request(
        logical_request_id="rid-retained-clean",
        instance_id="instance-a",
        engine_request_id="rid-retained-clean",
        full_prompt_token_count=32,
        model_fingerprint="model-a",
        kv_layout_id="layout-v1",
        tp_size=1,
        pp_size=1,
        required_ranks=(_rank(0),),
        now_ms=90,
    )
    registry.mark_source_retained(
        logical_request_id="rid-retained-clean",
        retained_until_ms=190,
        now_ms=130,
    )

    cleaned_state = registry.mark_cleaned(
        logical_request_id="rid-retained-clean",
        now_ms=200,
    )

    assert cleaned_state.state == RequestBundleLifecycleState.CLEANED


def test_aggregate_required_rank_results_fails_closed_on_missing_rank() -> None:
    aggregated = aggregate_required_rank_results(
        kind=InstanceOpKind.HYDRATE,
        logical_request_id="rid-1",
        required_ranks=(_rank(0), _rank(1)),
        rank_results=(
            PerRankInstanceOpResult(
                kind=InstanceOpKind.HYDRATE,
                rank=_rank(0),
                status=InstanceOpStatus.SUCCESS,
            ),
        ),
    )

    assert aggregated.status == InstanceOpStatus.FAILED
    assert "missing required rank results" in str(aggregated.error_message)


def test_prepared_bundle_claim_is_one_shot() -> None:
    registry = PreparedBundleRegistry()
    record = registry.begin_prepare(
        logical_request_id="rid-1",
        target_instance_id="instance-b",
        publish_manifest_digest="publish-digest-1",
        artifact_manifest_digest="artifact-digest-1",
        engine_owned_manifest_sha256="engine-digest-1",
        required_ranks=(_rank(0), _rank(1)),
        prompt_token_digest="prompt-digest-1",
        cutoff_token_count=64,
        now_ms=100,
    )
    assert record.state == PreparedBundleLifecycleState.PREPARING

    registry.mark_rank_install(
        logical_request_id="rid-1",
        publish_manifest_digest="publish-digest-1",
        rank=_rank(0),
        state=RankInstallState.READY,
        hydrated_page_count=2,
        runnable_prefix_tokens=64,
    )
    registry.mark_rank_install(
        logical_request_id="rid-1",
        publish_manifest_digest="publish-digest-1",
        rank=_rank(1),
        state=RankInstallState.READY,
        hydrated_page_count=2,
        runnable_prefix_tokens=64,
    )
    registry.mark_prepared(
        logical_request_id="rid-1",
        publish_manifest_digest="publish-digest-1",
        now_ms=120,
        prepared_hold_set_id="hold-set-1",
    )

    decision = registry.claim_prepared_bundle(
        logical_request_id="rid-1",
        incoming_prompt_token_digest="prompt-digest-1",
        incoming_cutoff_token_count=64,
        scheduler_rid="rid-1",
        claim_token="claim-1",
        now_ms=130,
    )
    assert decision.action == PreparedBundleClaimAction.CLAIMED
    assert decision.record is not None
    assert decision.record.state == PreparedBundleLifecycleState.CLAIMED

    second_decision = registry.claim_prepared_bundle(
        logical_request_id="rid-1",
        incoming_prompt_token_digest="prompt-digest-1",
        incoming_cutoff_token_count=64,
        scheduler_rid="rid-1",
        claim_token="claim-2",
        now_ms=131,
    )
    assert second_decision.action == PreparedBundleClaimAction.FAIL_CLOSED
    assert "already claimed or attached" in second_decision.reason


def test_prepared_bundle_claim_falls_back_for_stale_record() -> None:
    registry = PreparedBundleRegistry()
    registry.begin_prepare(
        logical_request_id="rid-stale",
        target_instance_id="instance-b",
        publish_manifest_digest="publish-digest-stale",
        artifact_manifest_digest="artifact-digest-stale",
        engine_owned_manifest_sha256="engine-digest-stale",
        required_ranks=(_rank(0),),
        prompt_token_digest="prompt-digest-stale",
        cutoff_token_count=32,
        now_ms=100,
    )
    registry.mark_rank_install(
        logical_request_id="rid-stale",
        publish_manifest_digest="publish-digest-stale",
        rank=_rank(0),
        state=RankInstallState.READY,
        hydrated_page_count=1,
        runnable_prefix_tokens=32,
    )
    registry.mark_prepared(
        logical_request_id="rid-stale",
        publish_manifest_digest="publish-digest-stale",
        now_ms=110,
    )
    registry.mark_stale(
        logical_request_id="rid-stale",
        publish_manifest_digest="publish-digest-stale",
    )

    decision = registry.claim_prepared_bundle(
        logical_request_id="rid-stale",
        incoming_prompt_token_digest="prompt-digest-stale",
        incoming_cutoff_token_count=32,
        scheduler_rid="rid-stale",
        claim_token="claim-stale",
        now_ms=120,
    )

    assert decision.action == PreparedBundleClaimAction.FALLBACK
    assert (
        "only stale or tainted prepared bundle records are available" in decision.reason
    )
    assert decision.record is not None
    assert decision.record.stale is True


def test_prepared_bundle_claim_fails_closed_on_live_request_conflict() -> None:
    registry = PreparedBundleRegistry()
    registry.begin_prepare(
        logical_request_id="rid-conflict",
        target_instance_id="instance-b",
        publish_manifest_digest="publish-digest-conflict",
        artifact_manifest_digest="artifact-digest-conflict",
        engine_owned_manifest_sha256="engine-digest-conflict",
        required_ranks=(_rank(0),),
        prompt_token_digest="prompt-digest-conflict",
        cutoff_token_count=32,
        now_ms=100,
    )
    registry.mark_rank_install(
        logical_request_id="rid-conflict",
        publish_manifest_digest="publish-digest-conflict",
        rank=_rank(0),
        state=RankInstallState.READY,
        hydrated_page_count=1,
        runnable_prefix_tokens=32,
    )
    registry.mark_prepared(
        logical_request_id="rid-conflict",
        publish_manifest_digest="publish-digest-conflict",
        now_ms=110,
    )

    decision = registry.claim_prepared_bundle(
        logical_request_id="rid-conflict",
        incoming_prompt_token_digest="prompt-digest-conflict",
        incoming_cutoff_token_count=32,
        scheduler_rid="rid-conflict",
        claim_token="claim-conflict",
        now_ms=120,
        live_request_exists=True,
    )

    assert decision.action == PreparedBundleClaimAction.FAIL_CLOSED
    assert "live request already exists" in decision.reason


def test_prepared_hold_registry_tracks_hold_sets_separately() -> None:
    registry = PreparedHoldRegistry()
    installed = registry.install_hold_set(
        hold_set_id="hold-set-1",
        refs=(
            PreparedHoldRef(
                logical_page_index=0,
                page_hash="page-0",
                slot_token=PreparedSlotToken(slot_index=2, slot_generation=7),
                artifact_id="artifact-0",
            ),
        ),
        now_ms=100,
    )
    assert installed.state == PreparedHoldSetState.ACTIVE
    assert registry.get("hold-set-1") == installed

    released = registry.release_hold_set(hold_set_id="hold-set-1", now_ms=120)
    assert released.state == PreparedHoldSetState.RELEASED
    assert released.released_at_ms == 120


def test_begin_publish_requires_complete_rank_snapshots() -> None:
    registry = RequestBundleStateRegistry()
    registry.upsert_live_request(
        logical_request_id="rid-missing-rank",
        instance_id="instance-a",
        engine_request_id="rid-missing-rank",
        full_prompt_token_count=32,
        model_fingerprint="model-a",
        kv_layout_id="layout-v1",
        tp_size=2,
        pp_size=1,
        required_ranks=(_rank(0), _rank(1)),
        now_ms=100,
    )
    registry.update_rank_frontier(
        logical_request_id="rid-missing-rank",
        rank=_rank(0),
        latest_token_count=32,
        latest_last_page_index=0,
        now_ms=110,
        ordered_pages=(),
    )

    with pytest.raises(RequestBundleStateError, match="missing rank snapshots"):
        registry.begin_publish(
            logical_request_id="rid-missing-rank",
            publish_op_id="publish-missing-rank",
            requested_cutoff_token_count=32,
            page_size=32,
            now_ms=120,
        )
