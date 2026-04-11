from __future__ import annotations

import pytest

from sglang.srt.tensorcast.instance_ops.instance_ops_coordinator import (
    InstanceOpsCoordinator,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    RequestBundleStateError,
)
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    CoordinatorFanoutResult,
    CoordinatorInstanceOpRequest,
    InstanceOpKind,
    InstanceOpStatus,
    PerRankInstanceOpResult,
    PublishInstanceOpRequest,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    RankCoord,
)


def _rank(tp_rank: int, pp_rank: int = 0) -> RankCoord:
    return RankCoord(tp_rank=tp_rank, pp_rank=pp_rank)


def _publish_request(
    *,
    op_id: str = "publish-op-1",
    instance_id: str = "instance-a",
    required_ranks: tuple[RankCoord, ...] = (_rank(0), _rank(1)),
    requested_at_ms: int = 100,
    timeout_ms: int = 5_000,
) -> CoordinatorInstanceOpRequest:
    return CoordinatorInstanceOpRequest(
        kind=InstanceOpKind.PUBLISH,
        op_id=op_id,
        instance_id=instance_id,
        required_ranks=required_ranks,
        timeout_ms=timeout_ms,
        requested_at_ms=requested_at_ms,
        payload=PublishInstanceOpRequest(
            logical_request_id="rid-1",
            engine_request_id="rid-1",
            publish_op_id=op_id,
            requested_cutoff_token_count=64,
            requested_at_ms=requested_at_ms,
        ),
    )


def test_coordinator_fans_out_and_aggregates_success() -> None:
    coordinator = InstanceOpsCoordinator()
    required_ranks = (_rank(0), _rank(1))
    coordinator.register_instance(
        instance_id="instance-a",
        coordinator_rank=_rank(0),
        coordinator_epoch="epoch-1",
        required_ranks=required_ranks,
        now_ms=10,
    )
    seen_rank_requests: list[tuple[int, int]] = []

    def fanout_handler(
        rank_requests,
    ) -> CoordinatorFanoutResult:
        seen_rank_requests.extend(request.rank.as_key() for request in rank_requests)
        return CoordinatorFanoutResult(
            rank_results=tuple(
                PerRankInstanceOpResult(
                    kind=InstanceOpKind.PUBLISH,
                    rank=request.rank,
                    status=InstanceOpStatus.SUCCESS,
                )
                for request in rank_requests
            )
        )

    aggregated = coordinator.coordinate(
        request=_publish_request(required_ranks=required_ranks),
        fanout_handler=fanout_handler,
        now_ms=200,
    )

    assert aggregated.status == InstanceOpStatus.SUCCESS
    assert seen_rank_requests == [(0, 0), (1, 0)]


def test_coordinator_fails_closed_when_one_rank_fails() -> None:
    coordinator = InstanceOpsCoordinator()
    required_ranks = (_rank(0), _rank(1))
    coordinator.register_instance(
        instance_id="instance-a",
        coordinator_rank=_rank(0),
        coordinator_epoch="epoch-1",
        required_ranks=required_ranks,
        now_ms=10,
    )

    aggregated = coordinator.coordinate(
        request=_publish_request(required_ranks=required_ranks),
        fanout_handler=lambda rank_requests: CoordinatorFanoutResult(
            rank_results=(
                PerRankInstanceOpResult(
                    kind=InstanceOpKind.PUBLISH,
                    rank=rank_requests[0].rank,
                    status=InstanceOpStatus.SUCCESS,
                ),
                PerRankInstanceOpResult(
                    kind=InstanceOpKind.PUBLISH,
                    rank=rank_requests[1].rank,
                    status=InstanceOpStatus.FAILED,
                    error_message="page flush failed",
                ),
            )
        ),
        now_ms=200,
    )

    assert aggregated.status == InstanceOpStatus.FAILED
    assert "page flush failed" in str(aggregated.error_message)


def test_coordinator_converts_rank_timeout_to_failed_result() -> None:
    coordinator = InstanceOpsCoordinator()
    required_ranks = (_rank(0), _rank(1))
    coordinator.register_instance(
        instance_id="instance-a",
        coordinator_rank=_rank(0),
        coordinator_epoch="epoch-1",
        required_ranks=required_ranks,
        now_ms=10,
    )

    aggregated = coordinator.coordinate(
        request=_publish_request(required_ranks=required_ranks, timeout_ms=1_000),
        fanout_handler=lambda rank_requests: CoordinatorFanoutResult(
            rank_results=(
                PerRankInstanceOpResult(
                    kind=InstanceOpKind.PUBLISH,
                    rank=rank_requests[0].rank,
                    status=InstanceOpStatus.SUCCESS,
                ),
            ),
            timed_out_ranks=(rank_requests[1].rank,),
        ),
        now_ms=1_200,
    )

    assert aggregated.status == InstanceOpStatus.FAILED
    assert "timed out after 1000 ms" in str(aggregated.error_message)


def test_coordinator_fails_closed_when_required_rank_result_is_missing() -> None:
    coordinator = InstanceOpsCoordinator()
    required_ranks = (_rank(0), _rank(1))
    coordinator.register_instance(
        instance_id="instance-a",
        coordinator_rank=_rank(0),
        coordinator_epoch="epoch-1",
        required_ranks=required_ranks,
        now_ms=10,
    )

    aggregated = coordinator.coordinate(
        request=_publish_request(required_ranks=required_ranks),
        fanout_handler=lambda rank_requests: CoordinatorFanoutResult(
            rank_results=(
                PerRankInstanceOpResult(
                    kind=InstanceOpKind.PUBLISH,
                    rank=rank_requests[0].rank,
                    status=InstanceOpStatus.SUCCESS,
                ),
            )
        ),
        now_ms=200,
    )

    assert aggregated.status == InstanceOpStatus.FAILED
    assert "missing required rank results" in str(aggregated.error_message)


def test_coordinator_reuses_cached_result_for_idempotent_retry() -> None:
    coordinator = InstanceOpsCoordinator()
    required_ranks = (_rank(0), _rank(1))
    coordinator.register_instance(
        instance_id="instance-a",
        coordinator_rank=_rank(0),
        coordinator_epoch="epoch-1",
        required_ranks=required_ranks,
        now_ms=10,
    )
    dispatch_call_count = 0

    def fanout_handler(
        rank_requests,
    ) -> CoordinatorFanoutResult:
        nonlocal dispatch_call_count
        dispatch_call_count += 1
        return CoordinatorFanoutResult(
            rank_results=tuple(
                PerRankInstanceOpResult(
                    kind=InstanceOpKind.PUBLISH,
                    rank=request.rank,
                    status=InstanceOpStatus.SUCCESS,
                )
                for request in rank_requests
            )
        )

    request = _publish_request(required_ranks=required_ranks)
    first = coordinator.coordinate(
        request=request,
        fanout_handler=fanout_handler,
        now_ms=200,
    )
    second = coordinator.coordinate(
        request=request,
        fanout_handler=fanout_handler,
        now_ms=210,
    )
    record = coordinator.get_op_record(
        instance_id="instance-a",
        kind=InstanceOpKind.PUBLISH,
        op_id="publish-op-1",
    )

    assert first == second
    assert dispatch_call_count == 1
    assert record is not None
    assert record.retry_hit_count == 1


def test_coordinator_rejects_mismatched_retry_payload_for_same_op_id() -> None:
    coordinator = InstanceOpsCoordinator()
    required_ranks = (_rank(0), _rank(1))
    coordinator.register_instance(
        instance_id="instance-a",
        coordinator_rank=_rank(0),
        coordinator_epoch="epoch-1",
        required_ranks=required_ranks,
        now_ms=10,
    )
    request = _publish_request(required_ranks=required_ranks)

    coordinator.coordinate(
        request=request,
        fanout_handler=lambda rank_requests: CoordinatorFanoutResult(
            rank_results=tuple(
                PerRankInstanceOpResult(
                    kind=InstanceOpKind.PUBLISH,
                    rank=rank_request.rank,
                    status=InstanceOpStatus.SUCCESS,
                )
                for rank_request in rank_requests
            )
        ),
        now_ms=200,
    )

    with pytest.raises(
        RequestBundleStateError,
        match="idempotent retry must reuse the exact same coordinator request",
    ):
        coordinator.coordinate(
            request=_publish_request(
                required_ranks=required_ranks,
                timeout_ms=10_000,
            ),
            fanout_handler=lambda _: CoordinatorFanoutResult(),
            now_ms=210,
        )
