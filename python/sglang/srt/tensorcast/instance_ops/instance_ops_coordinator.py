# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    RequestBundleStateError,
    aggregate_required_rank_results,
)
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    AggregatedInstanceOpResult,
    CoordinatorFanoutResult,
    CoordinatorInstanceOpRequest,
    CoordinatorOpRecord,
    CoordinatorRegistrationRecord,
    InstanceOpKind,
    InstanceOpStatus,
    PerRankInstanceOpResult,
    RankScopedInstanceOpRequest,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    RankCoord,
)

InstanceOpsFanoutHandler: TypeAlias = Callable[
    [tuple[RankScopedInstanceOpRequest, ...]],
    CoordinatorFanoutResult,
]


class InstanceOpsCoordinator:
    def __init__(self) -> None:
        self._registration_by_instance: dict[str, CoordinatorRegistrationRecord] = {}
        self._op_record_by_key: dict[tuple[str, str, str], CoordinatorOpRecord] = {}

    def register_instance(
        self,
        *,
        instance_id: str,
        coordinator_rank: RankCoord,
        coordinator_epoch: str,
        required_ranks: tuple[RankCoord, ...],
        now_ms: int,
    ) -> CoordinatorRegistrationRecord:
        existing = self._registration_by_instance.get(instance_id)
        record = CoordinatorRegistrationRecord(
            instance_id=instance_id,
            coordinator_rank=coordinator_rank,
            coordinator_epoch=coordinator_epoch,
            required_ranks=required_ranks,
            created_at_ms=(
                existing.created_at_ms
                if existing is not None
                and existing.coordinator_rank == coordinator_rank
                and existing.coordinator_epoch == coordinator_epoch
                and existing.required_ranks == required_ranks
                else int(now_ms)
            ),
            updated_at_ms=int(now_ms),
        )
        self._registration_by_instance[instance_id] = record
        if existing is not None and existing != record:
            self._clear_instance_op_records(instance_id)
        return record

    def get_registration(
        self, instance_id: str
    ) -> CoordinatorRegistrationRecord | None:
        return self._registration_by_instance.get(instance_id)

    def get_op_record(
        self,
        *,
        instance_id: str,
        kind: InstanceOpKind,
        op_id: str,
    ) -> CoordinatorOpRecord | None:
        return self._op_record_by_key.get((instance_id, kind.value, op_id))

    def coordinate(
        self,
        *,
        request: CoordinatorInstanceOpRequest,
        fanout_handler: InstanceOpsFanoutHandler,
        now_ms: int,
    ) -> AggregatedInstanceOpResult:
        registration = self._require_registration(request.instance_id)
        if registration.required_ranks != request.required_ranks:
            raise RequestBundleStateError(
                "request required_ranks does not match registered coordinator instance"
            )
        record_key = (request.instance_id, request.kind.value, request.op_id)
        existing_record = self._op_record_by_key.get(record_key)
        if existing_record is not None:
            if existing_record.request != request:
                raise RequestBundleStateError(
                    "idempotent retry must reuse the exact same coordinator request"
                )
            updated_record = existing_record.model_copy(
                update={"retry_hit_count": existing_record.retry_hit_count + 1}
            )
            self._op_record_by_key[record_key] = updated_record
            return updated_record.aggregated_result

        rank_requests = tuple(
            RankScopedInstanceOpRequest(
                kind=request.kind,
                op_id=request.op_id,
                instance_id=request.instance_id,
                rank=rank,
                payload=request.payload,
            )
            for rank in request.required_ranks
        )
        try:
            fanout_result = fanout_handler(rank_requests)
        except Exception as exc:
            aggregated_result = AggregatedInstanceOpResult(
                kind=request.kind,
                logical_request_id=request.logical_request_id,
                status=InstanceOpStatus.FAILED,
                required_ranks=request.required_ranks,
                rank_results=(),
                error_message=f"coordinator fanout failed: {exc}",
            )
            self._op_record_by_key[record_key] = CoordinatorOpRecord(
                request=request,
                aggregated_result=aggregated_result,
                dispatch_invocation_count=1,
                retry_hit_count=0,
                started_at_ms=request.requested_at_ms,
                completed_at_ms=int(now_ms),
            )
            return aggregated_result

        self._validate_fanout_membership(
            required_ranks=request.required_ranks,
            fanout_result=fanout_result,
        )
        rank_results = list(fanout_result.rank_results)
        for rank in fanout_result.timed_out_ranks:
            rank_results.append(
                PerRankInstanceOpResult(
                    kind=request.kind,
                    rank=rank,
                    status=InstanceOpStatus.FAILED,
                    error_message=(
                        f"rank={rank.as_key()} timed out after {request.timeout_ms} ms"
                    ),
                )
            )
        aggregated_result = aggregate_required_rank_results(
            kind=request.kind,
            logical_request_id=request.logical_request_id,
            required_ranks=request.required_ranks,
            rank_results=tuple(rank_results),
        )
        self._op_record_by_key[record_key] = CoordinatorOpRecord(
            request=request,
            aggregated_result=aggregated_result,
            dispatch_invocation_count=1,
            retry_hit_count=0,
            started_at_ms=request.requested_at_ms,
            completed_at_ms=int(now_ms),
        )
        return aggregated_result

    def clear_instance(self, instance_id: str) -> None:
        self._registration_by_instance.pop(instance_id, None)
        self._clear_instance_op_records(instance_id)

    def _require_registration(self, instance_id: str) -> CoordinatorRegistrationRecord:
        record = self._registration_by_instance.get(instance_id)
        if record is None:
            raise RequestBundleStateError(
                f"coordinator instance_id={instance_id} is not registered"
            )
        return record

    def _validate_fanout_membership(
        self,
        *,
        required_ranks: tuple[RankCoord, ...],
        fanout_result: CoordinatorFanoutResult,
    ) -> None:
        required_rank_keys = {rank.as_key() for rank in required_ranks}
        unexpected_result_ranks = [
            result.rank.as_key()
            for result in fanout_result.rank_results
            if result.rank.as_key() not in required_rank_keys
        ]
        if unexpected_result_ranks:
            raise RequestBundleStateError(
                "fanout returned results for unexpected ranks: "
                + ", ".join(str(rank_key) for rank_key in unexpected_result_ranks)
            )
        unexpected_timeout_ranks = [
            rank.as_key()
            for rank in fanout_result.timed_out_ranks
            if rank.as_key() not in required_rank_keys
        ]
        if unexpected_timeout_ranks:
            raise RequestBundleStateError(
                "fanout returned timeouts for unexpected ranks: "
                + ", ".join(str(rank_key) for rank_key in unexpected_timeout_ranks)
            )

    def _clear_instance_op_records(self, instance_id: str) -> None:
        stale_keys = [
            record_key
            for record_key in self._op_record_by_key
            if record_key[0] == instance_id
        ]
        for record_key in stale_keys:
            self._op_record_by_key.pop(record_key, None)
