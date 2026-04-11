from __future__ import annotations

from sglang.srt.tensorcast.request_bundle.prepared_bundle_evict import (
    PreparedBundleLocalEvictor,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    PagePublicationRegistry,
    PreparedBundleRegistry,
    PreparedHoldRegistry,
    RequestBundleStateRegistry,
)
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    EvictLocalInstanceOpRequest,
    InstanceOpStatus,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    PagePublicationState,
    PreparedBundleLifecycleState,
    PreparedHoldRef,
    PreparedHoldSetState,
    PreparedSlotToken,
    RankCoord,
    RankInstallState,
)


def _rank(tp_rank: int, pp_rank: int = 0) -> RankCoord:
    return RankCoord(tp_rank=tp_rank, pp_rank=pp_rank)


def _install_prepared_bundle(
    *,
    prepared_bundle_registry: PreparedBundleRegistry,
    prepared_hold_registry: PreparedHoldRegistry,
    logical_request_id: str,
    publish_manifest_digest: str,
    now_ms: int = 100,
) -> None:
    prepared_bundle_registry.begin_prepare(
        logical_request_id=logical_request_id,
        target_instance_id="instance-b",
        publish_manifest_digest=publish_manifest_digest,
        artifact_manifest_digest=f"artifact:{publish_manifest_digest}",
        engine_owned_manifest_sha256=f"engine:{publish_manifest_digest}",
        required_ranks=(_rank(0),),
        prompt_token_digest=f"prompt:{logical_request_id}",
        cutoff_token_count=64,
        now_ms=now_ms,
    )
    prepared_bundle_registry.mark_rank_install(
        logical_request_id=logical_request_id,
        publish_manifest_digest=publish_manifest_digest,
        rank=_rank(0),
        state=RankInstallState.READY,
        hydrated_page_count=2,
        runnable_prefix_tokens=64,
        local_install_handle=f"install:{publish_manifest_digest}",
    )
    hold_set_id = f"hold:{publish_manifest_digest}"
    prepared_hold_registry.install_hold_set(
        hold_set_id=hold_set_id,
        refs=(
            PreparedHoldRef(
                logical_page_index=0,
                page_hash=f"page:{publish_manifest_digest}:0",
                slot_token=PreparedSlotToken(slot_index=1, slot_generation=1),
                artifact_id=f"artifact:{publish_manifest_digest}:0",
            ),
        ),
        now_ms=now_ms + 1,
    )
    prepared_bundle_registry.mark_prepared(
        logical_request_id=logical_request_id,
        publish_manifest_digest=publish_manifest_digest,
        now_ms=now_ms + 2,
        prepared_hold_set_id=hold_set_id,
    )


def test_evict_local_releases_prepared_hold_and_evicts_bundle() -> None:
    page_publication_registry = PagePublicationRegistry()
    request_bundle_registry = RequestBundleStateRegistry(
        page_publication_registry=page_publication_registry
    )
    prepared_bundle_registry = PreparedBundleRegistry()
    prepared_hold_registry = PreparedHoldRegistry()
    _install_prepared_bundle(
        prepared_bundle_registry=prepared_bundle_registry,
        prepared_hold_registry=prepared_hold_registry,
        logical_request_id="rid-1",
        publish_manifest_digest="manifest-1",
    )

    result = PreparedBundleLocalEvictor(
        request_bundle_registry=request_bundle_registry,
        page_publication_registry=page_publication_registry,
        prepared_bundle_registry=prepared_bundle_registry,
        prepared_hold_registry=prepared_hold_registry,
    ).evict(
        request=EvictLocalInstanceOpRequest(
            logical_request_id="rid-1",
            requested_at_ms=200,
        )
    )

    assert result.status == InstanceOpStatus.SUCCESS
    assert result.evicted_bundle_count == 1
    assert result.evicted_hold_set_count == 1
    bundle = prepared_bundle_registry.get(
        logical_request_id="rid-1",
        publish_manifest_digest="manifest-1",
    )
    assert bundle is not None
    assert bundle.state == PreparedBundleLifecycleState.EVICTED
    hold_set = prepared_hold_registry.get("hold:manifest-1")
    assert hold_set is not None
    assert hold_set.state == PreparedHoldSetState.RELEASED


def test_evict_local_cleans_live_request_without_deleting_shared_pages() -> None:
    page_publication_registry = PagePublicationRegistry()
    request_bundle_registry = RequestBundleStateRegistry(
        page_publication_registry=page_publication_registry
    )
    prepared_bundle_registry = PreparedBundleRegistry()
    prepared_hold_registry = PreparedHoldRegistry()
    _install_prepared_bundle(
        prepared_bundle_registry=prepared_bundle_registry,
        prepared_hold_registry=prepared_hold_registry,
        logical_request_id="rid-2",
        publish_manifest_digest="manifest-2",
    )
    request_bundle_registry.upsert_live_request(
        logical_request_id="rid-2",
        instance_id="instance-b",
        engine_request_id="rid-2",
        full_prompt_token_count=32,
        model_fingerprint="model-a",
        kv_layout_id="layout-v1",
        tp_size=1,
        pp_size=1,
        required_ranks=(_rank(0),),
        now_ms=90,
    )
    page_publication_registry.set_page_state(
        logical_request_id="rid-2",
        rank=_rank(0),
        logical_page_index=0,
        page_hash="page-hash-0",
        publication_state=PagePublicationState.READY,
        artifact_id="artifact-0",
        host_resident=True,
        updated_at_ms=95,
    )

    result = PreparedBundleLocalEvictor(
        request_bundle_registry=request_bundle_registry,
        page_publication_registry=page_publication_registry,
        prepared_bundle_registry=prepared_bundle_registry,
        prepared_hold_registry=prepared_hold_registry,
    ).evict(
        request=EvictLocalInstanceOpRequest(
            logical_request_id="rid-2",
            requested_at_ms=200,
        )
    )

    assert result.status == InstanceOpStatus.SUCCESS
    assert request_bundle_registry.get("rid-2") is None
    assert (
        page_publication_registry.snapshot_rank(
            logical_request_id="rid-2",
            rank=_rank(0),
        )
        == ()
    )
    bundle = prepared_bundle_registry.get(
        logical_request_id="rid-2",
        publish_manifest_digest="manifest-2",
    )
    assert bundle is not None
    assert bundle.state == PreparedBundleLifecycleState.EVICTED
    assert bundle.prepared_hold_set_id == "hold:manifest-2"


def test_evict_local_by_manifest_digest_supports_post_consume_cleanup() -> None:
    page_publication_registry = PagePublicationRegistry()
    request_bundle_registry = RequestBundleStateRegistry(
        page_publication_registry=page_publication_registry
    )
    prepared_bundle_registry = PreparedBundleRegistry()
    prepared_hold_registry = PreparedHoldRegistry()
    _install_prepared_bundle(
        prepared_bundle_registry=prepared_bundle_registry,
        prepared_hold_registry=prepared_hold_registry,
        logical_request_id="rid-3",
        publish_manifest_digest="manifest-3",
    )
    prepared_bundle_registry.claim_prepared_bundle(
        logical_request_id="rid-3",
        incoming_prompt_token_digest="prompt:rid-3",
        incoming_cutoff_token_count=64,
        scheduler_rid="sgl-rid-3",
        claim_token="claim-3",
        now_ms=120,
    )
    prepared_bundle_registry.mark_attached(
        logical_request_id="rid-3",
        publish_manifest_digest="manifest-3",
        prepared_bundle_key="prepared-key-3",
    )
    prepared_bundle_registry.mark_consumed(
        logical_request_id="rid-3",
        publish_manifest_digest="manifest-3",
    )

    result = PreparedBundleLocalEvictor(
        request_bundle_registry=request_bundle_registry,
        page_publication_registry=page_publication_registry,
        prepared_bundle_registry=prepared_bundle_registry,
        prepared_hold_registry=prepared_hold_registry,
    ).evict(
        request=EvictLocalInstanceOpRequest(
            publish_manifest_digest="manifest-3",
            requested_at_ms=200,
        )
    )

    assert result.status == InstanceOpStatus.SUCCESS
    assert result.evicted_bundle_count == 1
    assert result.evicted_hold_set_count == 1
    bundle = prepared_bundle_registry.get(
        logical_request_id="rid-3",
        publish_manifest_digest="manifest-3",
    )
    assert bundle is not None
    assert bundle.state == PreparedBundleLifecycleState.EVICTED


def test_evict_local_rejects_selector_mismatch() -> None:
    page_publication_registry = PagePublicationRegistry()
    request_bundle_registry = RequestBundleStateRegistry(
        page_publication_registry=page_publication_registry
    )
    prepared_bundle_registry = PreparedBundleRegistry()
    prepared_hold_registry = PreparedHoldRegistry()
    _install_prepared_bundle(
        prepared_bundle_registry=prepared_bundle_registry,
        prepared_hold_registry=prepared_hold_registry,
        logical_request_id="rid-4",
        publish_manifest_digest="manifest-4",
    )

    result = PreparedBundleLocalEvictor(
        request_bundle_registry=request_bundle_registry,
        page_publication_registry=page_publication_registry,
        prepared_bundle_registry=prepared_bundle_registry,
        prepared_hold_registry=prepared_hold_registry,
    ).evict(
        request=EvictLocalInstanceOpRequest(
            logical_request_id="different-rid",
            publish_manifest_digest="manifest-4",
            requested_at_ms=200,
        )
    )

    assert result.status == InstanceOpStatus.FAILED
    assert "selector mismatch" in str(result.error_message)
