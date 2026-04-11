# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import pytest

from sglang.srt.tensorcast.request_bundle.request_bundle_publish import (
    RequestBundlePublishAggregator,
    RequestBundlePublisher,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    PagePublicationRegistry,
    RequestBundleStateRegistry,
    RequestBundleStateError,
)
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    PublishInstanceOpRequest,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    PageClosureEntry,
    PagePublicationState,
    PagePublishAction,
    RankCoord,
    RequestBundleLifecycleState,
    SourcePublishClosureResult,
)


def _rank(tp_rank: int, pp_rank: int = 0) -> RankCoord:
    return RankCoord(tp_rank=tp_rank, pp_rank=pp_rank)


def _publish_request(**overrides: object) -> PublishInstanceOpRequest:
    payload: dict[str, object] = {
        "logical_request_id": "rid-1",
        "engine_request_id": "rid-1",
        "publish_op_id": "publish-op-1",
        "requested_cutoff_token_count": 70,
        "prompt_token_digest": "prompt-digest-1",
        "dtype": "float16",
        "page_size": 32,
        "requested_at_ms": 100,
    }
    payload.update(overrides)
    return PublishInstanceOpRequest.model_validate(payload)


def _seed_local_live_request(
    *,
    rank: RankCoord,
    rank_pages: tuple[PageClosureEntry, ...],
    latest_token_count: int = 70,
    latest_last_page_index: int = 2,
) -> tuple[PagePublicationRegistry, RequestBundleStateRegistry]:
    page_registry = PagePublicationRegistry()
    bundle_registry = RequestBundleStateRegistry(
        page_publication_registry=page_registry
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
            last_error=page.last_error,
            updated_at_ms=100,
        )
    bundle_registry.upsert_live_request(
        logical_request_id="rid-1",
        instance_id="instance-a",
        engine_request_id="rid-1",
        full_prompt_token_count=latest_token_count,
        model_fingerprint="model-a",
        kv_layout_id="layout-v1",
        tp_size=2,
        pp_size=1,
        required_ranks=(rank,),
        now_ms=90,
    )
    bundle_registry.update_rank_frontier(
        logical_request_id="rid-1",
        rank=rank,
        latest_token_count=latest_token_count,
        latest_last_page_index=latest_last_page_index,
        now_ms=95 + rank.tp_rank,
    )
    return page_registry, bundle_registry


def _publish_local_result(
    *,
    rank: RankCoord,
    rank_pages: tuple[PageClosureEntry, ...],
    now_ms: int = 200,
    resolve_inflight_page=None,
    force_flush_absent_page=None,
    latest_token_count: int = 70,
    latest_last_page_index: int = 2,
    request: PublishInstanceOpRequest | None = None,
) -> SourcePublishClosureResult:
    page_registry, bundle_registry = _seed_local_live_request(
        rank=rank,
        rank_pages=rank_pages,
        latest_token_count=latest_token_count,
        latest_last_page_index=latest_last_page_index,
    )
    return RequestBundlePublisher(
        request_bundle_registry=bundle_registry,
        page_publication_registry=page_registry,
    ).publish(
        request=_publish_request() if request is None else request,
        now_ms=now_ms,
        resolve_inflight_page=resolve_inflight_page,
        force_flush_absent_page=force_flush_absent_page,
    )


def test_local_publish_ready_only_succeeds_and_reuses_pages() -> None:
    result = _publish_local_result(
        rank=_rank(0),
        rank_pages=(
            PageClosureEntry(
                logical_page_index=0,
                page_hash="r0-p0",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r0-p0",
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=1,
                page_hash="r0-p1",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r0-p1",
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=2,
                page_hash="r0-p2",
                publication_state=PagePublicationState.ABSENT,
                host_resident=True,
            ),
        ),
    )

    assert result.request_state.state == RequestBundleLifecycleState.PUBLISHED
    assert result.request_state.required_ranks == (_rank(0),)
    assert result.cutoff.cutoff_token_count == 70
    assert result.cutoff.page_aligned_token_count == 64
    assert result.publish_manifest.tail_valid_tokens == 6
    assert len(result.publish_manifest.artifact_manifest.entries) == 2
    assert all(
        outcome.action == PagePublishAction.REUSED
        for rank_result in result.rank_results
        for outcome in rank_result.page_outcomes
    )


def test_local_publish_allows_mid_decode_requests_but_keeps_prompt_prefix_only() -> (
    None
):
    result = _publish_local_result(
        rank=_rank(0),
        rank_pages=(
            PageClosureEntry(
                logical_page_index=0,
                page_hash="r0-p0",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r0-p0",
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=1,
                page_hash="r0-p1",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r0-p1",
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=2,
                page_hash="r0-p2",
                publication_state=PagePublicationState.ABSENT,
                host_resident=True,
            ),
        ),
        request=_publish_request(emitted_decode_token_count=7),
    )

    assert result.request_state.state == RequestBundleLifecycleState.PUBLISHED
    assert result.cutoff.cutoff_token_count == 70
    assert len(result.publish_manifest.artifact_manifest.entries) == 2


def test_local_publish_waits_on_inflight_pages() -> None:
    result = _publish_local_result(
        rank=_rank(0),
        rank_pages=(
            PageClosureEntry(
                logical_page_index=0,
                page_hash="r0-p0",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r0-p0",
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=1,
                page_hash="r0-p1",
                publication_state=PagePublicationState.INFLIGHT,
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=2,
                page_hash="r0-p2",
                publication_state=PagePublicationState.ABSENT,
                host_resident=True,
            ),
        ),
        resolve_inflight_page=lambda rank, page: page.model_copy(
            update={
                "publication_state": PagePublicationState.READY,
                "artifact_id": f"artifact-{rank.tp_rank}-p{page.logical_page_index}",
            }
        ),
    )

    assert result.rank_results[0].rank == _rank(0)
    assert [outcome.action for outcome in result.rank_results[0].page_outcomes] == [
        PagePublishAction.REUSED,
        PagePublishAction.WAITED,
    ]


def test_local_publish_force_flushes_absent_tail_pages() -> None:
    result = _publish_local_result(
        rank=_rank(1),
        rank_pages=(
            PageClosureEntry(
                logical_page_index=0,
                page_hash="r1-p0",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r1-p0",
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=1,
                page_hash="r1-p1",
                publication_state=PagePublicationState.ABSENT,
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=2,
                page_hash="r1-p2",
                publication_state=PagePublicationState.ABSENT,
                host_resident=True,
            ),
        ),
        force_flush_absent_page=lambda rank, page: page.model_copy(
            update={
                "publication_state": PagePublicationState.READY,
                "artifact_id": f"artifact-{rank.tp_rank}-p{page.logical_page_index}",
            }
        ),
    )

    assert [outcome.action for outcome in result.rank_results[0].page_outcomes] == [
        PagePublishAction.REUSED,
        PagePublishAction.FLUSHED,
    ]


def test_local_publish_records_partial_tail_without_transferring_tail_page() -> None:
    result = _publish_local_result(
        rank=_rank(0),
        rank_pages=(
            PageClosureEntry(
                logical_page_index=0,
                page_hash="r0-p0",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r0-p0",
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=1,
                page_hash="r0-p1",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r0-p1",
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=2,
                page_hash="r0-p2",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r0-p2",
                host_resident=True,
            ),
        ),
        latest_token_count=70,
        latest_last_page_index=2,
        request=_publish_request(requested_cutoff_token_count=70),
    )

    assert [
        entry.logical_page_index
        for entry in result.publish_manifest.artifact_manifest.entries
    ] == [0, 1]
    assert result.publish_manifest.cutoff_token_count == 70
    assert result.publish_manifest.tail_valid_tokens == 6


def test_local_publish_fails_if_live_request_is_released_before_closure_finishes() -> (
    None
):
    page_registry, bundle_registry = _seed_local_live_request(
        rank=_rank(0),
        rank_pages=(
            PageClosureEntry(
                logical_page_index=0,
                page_hash="r0-p0",
                publication_state=PagePublicationState.INFLIGHT,
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=1,
                page_hash="r0-p1",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r0-p1",
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=2,
                page_hash="r0-p2",
                publication_state=PagePublicationState.ABSENT,
                host_resident=True,
            ),
        ),
    )

    def clear_live_request(rank: RankCoord, page: PageClosureEntry) -> PageClosureEntry:
        bundle_registry.clear("rid-1")
        return page.model_copy(
            update={
                "publication_state": PagePublicationState.READY,
                "artifact_id": f"artifact-{rank.tp_rank}-p{page.logical_page_index}",
            }
        )

    with pytest.raises(
        RequestBundleStateError,
        match="source request state was released before publish closure completed",
    ):
        RequestBundlePublisher(
            request_bundle_registry=bundle_registry,
            page_publication_registry=page_registry,
        ).publish(
            request=_publish_request(),
            now_ms=200,
            resolve_inflight_page=clear_live_request,
        )


def test_local_publish_rejects_unsupported_v1_batch_shape_early() -> None:
    with pytest.raises(
        RequestBundleStateError,
        match="does not support batch request transfer",
    ):
        _publish_local_result(
            rank=_rank(0),
            rank_pages=(
                PageClosureEntry(
                    logical_page_index=0,
                    page_hash="r0-p0",
                    publication_state=PagePublicationState.READY,
                    artifact_id="artifact-r0-p0",
                    host_resident=True,
                ),
            ),
            latest_token_count=32,
            latest_last_page_index=0,
            request=_publish_request(
                requested_cutoff_token_count=32,
                batch_request_count=2,
            ),
        )


def test_publish_aggregator_combines_local_rank_results_into_group_manifest() -> None:
    local_rank0 = _publish_local_result(
        rank=_rank(0),
        rank_pages=(
            PageClosureEntry(
                logical_page_index=0,
                page_hash="r0-p0",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r0-p0",
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=1,
                page_hash="r0-p1",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r0-p1",
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=2,
                page_hash="r0-p2",
                publication_state=PagePublicationState.ABSENT,
                host_resident=True,
            ),
        ),
    )
    local_rank1 = _publish_local_result(
        rank=_rank(1),
        rank_pages=(
            PageClosureEntry(
                logical_page_index=0,
                page_hash="r1-p0",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r1-p0",
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=1,
                page_hash="r1-p1",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r1-p1",
                host_resident=True,
            ),
            PageClosureEntry(
                logical_page_index=2,
                page_hash="r1-p2",
                publication_state=PagePublicationState.ABSENT,
                host_resident=True,
            ),
        ),
    )

    result = RequestBundlePublishAggregator().aggregate(
        request=_publish_request(),
        local_results=(local_rank0, local_rank1),
        now_ms=260,
        required_ranks=(_rank(0), _rank(1)),
    )

    assert result.request_state.state == RequestBundleLifecycleState.PUBLISHED
    assert result.request_state.required_ranks == (_rank(0), _rank(1))
    assert len(result.request_state.rank_snapshots) == 2
    assert len(result.publish_manifest.artifact_manifest.entries) == 4
    assert (
        result.publish_manifest.engine_owned_manifest.payload.compatibility.required_ranks
        == (
            _rank(0),
            _rank(1),
        )
    )
    assert [
        entry.logical_page_index
        for entry in result.publish_manifest.artifact_manifest.entries
    ] == [
        0,
        1,
        0,
        1,
    ]


def test_publish_aggregator_rejects_missing_required_rank_result() -> None:
    local_rank0 = _publish_local_result(
        rank=_rank(0),
        rank_pages=(
            PageClosureEntry(
                logical_page_index=0,
                page_hash="r0-p0",
                publication_state=PagePublicationState.READY,
                artifact_id="artifact-r0-p0",
                host_resident=True,
            ),
        ),
        latest_token_count=32,
        latest_last_page_index=0,
        request=_publish_request(requested_cutoff_token_count=32),
    )

    with pytest.raises(
        RequestBundleStateError,
        match="missing local publish results",
    ):
        RequestBundlePublishAggregator().aggregate(
            request=_publish_request(requested_cutoff_token_count=32),
            local_results=(local_rank0,),
            now_ms=260,
            required_ranks=(_rank(0), _rank(1)),
        )
