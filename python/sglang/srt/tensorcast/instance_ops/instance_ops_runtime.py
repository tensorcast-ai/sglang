# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast

from sglang.srt.tensorcast.instance_ops.instance_ops_coordinator import (
    InstanceOpsCoordinator,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_hydrate import (
    RequestBundleHydrateAggregator,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_publish import (
    RequestBundlePublishAggregator,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    RequestBundleStateError,
)
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    CoordinatorFanoutResult,
    CoordinatorInstanceOpRequest,
    EvictLocalInstanceOpRequest,
    EvictLocalInstanceOpResult,
    HydrateInstanceOpRequest,
    InstanceOpKind,
    InstanceOpStatus,
    PerRankInstanceOpResult,
    PublishInstanceOpRequest,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    HydratePreparedResult,
    HydrateTargetCompatibility,
    PublishManifestRecord,
    RankCoord,
    SourcePublishClosureResult,
)

DEFAULT_INSTANCE_OP_TIMEOUT_MS = 30_000
GroupOpResult: TypeAlias = (
    SourcePublishClosureResult | HydratePreparedResult | EvictLocalInstanceOpResult
)
LocalPublishExecutor: TypeAlias = Callable[
    [PublishInstanceOpRequest], SourcePublishClosureResult
]
LocalHydrateExecutor: TypeAlias = Callable[[], HydratePreparedResult]
LocalEvictExecutor: TypeAlias = Callable[[], EvictLocalInstanceOpResult]


class InstanceOpsObjectGroup(Protocol):
    def all_gather_object(self, obj: Any) -> list[Any]: ...

    def broadcast_object(self, obj: Any, src: int = 0) -> Any: ...


@dataclass(frozen=True)
class _RuntimeCacheEntry:
    request: CoordinatorInstanceOpRequest
    result: GroupOpResult | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class _LocalOpEnvelope:
    rank: RankCoord
    status: InstanceOpStatus
    result: GroupOpResult | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class _GroupOutcome:
    result: GroupOpResult | None = None
    error_message: str | None = None


class InstanceOpsRuntimeCoordinator:
    def __init__(
        self,
        *,
        current_rank: RankCoord,
        object_group: InstanceOpsObjectGroup,
        coordinator_rank: RankCoord = RankCoord(tp_rank=0, pp_rank=0),
        coordinator_group_src: int = 0,
        timeout_ms: int = DEFAULT_INSTANCE_OP_TIMEOUT_MS,
    ) -> None:
        self._current_rank = current_rank
        self._object_group = object_group
        self._coordinator_rank = coordinator_rank
        self._coordinator_group_src = int(coordinator_group_src)
        self._timeout_ms = int(timeout_ms)
        self._instance_id = "unconfigured"
        self._coordinator_epoch = "unconfigured"
        self._required_ranks: tuple[RankCoord, ...] = (current_rank,)
        self._status_coordinator = InstanceOpsCoordinator()
        self._publish_aggregator = RequestBundlePublishAggregator()
        self._hydrate_aggregator = RequestBundleHydrateAggregator()
        self._cache_by_key: dict[tuple[str, str, str], _RuntimeCacheEntry] = {}

    def configure(
        self,
        *,
        instance_id: str,
        coordinator_epoch: str,
        required_ranks: Sequence[RankCoord],
        now_ms: int,
    ) -> None:
        resolved_required_ranks = tuple(required_ranks)
        if self._current_rank.as_key() not in {
            rank.as_key() for rank in resolved_required_ranks
        }:
            raise RequestBundleStateError(
                f"current rank={self._current_rank.as_key()} is not part of the required rank set"
            )
        self._instance_id = instance_id
        self._coordinator_epoch = coordinator_epoch
        self._required_ranks = resolved_required_ranks
        self._cache_by_key.clear()
        self._status_coordinator.register_instance(
            instance_id=instance_id,
            coordinator_rank=self._coordinator_rank,
            coordinator_epoch=coordinator_epoch,
            required_ranks=resolved_required_ranks,
            now_ms=int(now_ms),
        )

    def publish(
        self,
        *,
        request: PublishInstanceOpRequest,
        execute_local: LocalPublishExecutor,
    ) -> SourcePublishClosureResult:
        resolved_timeout_ms = int(request.timeout_ms or self._timeout_ms)
        resolved_request = request.model_copy(
            update={"timeout_ms": resolved_timeout_ms}
        )
        coordinator_request = CoordinatorInstanceOpRequest(
            kind=InstanceOpKind.PUBLISH,
            op_id=resolved_request.publish_op_id,
            instance_id=self._instance_id,
            required_ranks=self._required_ranks,
            timeout_ms=resolved_timeout_ms,
            requested_at_ms=resolved_request.requested_at_ms,
            payload=resolved_request,
        )
        return cast(
            SourcePublishClosureResult,
            self._run_group_op(
                request=coordinator_request,
                execute_local=lambda: execute_local(resolved_request),
                aggregate_results=lambda local_results: self._publish_aggregator.aggregate(
                    request=resolved_request,
                    local_results=tuple(
                        cast(SourcePublishClosureResult, local_result)
                        for local_result in local_results
                    ),
                    now_ms=int(resolved_request.requested_at_ms),
                    required_ranks=self._required_ranks,
                ),
            ),
        )

    def hydrate(
        self,
        *,
        request: HydrateInstanceOpRequest,
        publish_manifest: PublishManifestRecord,
        target: HydrateTargetCompatibility,
        execute_local: LocalHydrateExecutor,
    ) -> HydratePreparedResult:
        coordinator_request = CoordinatorInstanceOpRequest(
            kind=InstanceOpKind.HYDRATE,
            op_id=request.publish_manifest_digest,
            instance_id=self._instance_id,
            required_ranks=self._required_ranks,
            timeout_ms=self._timeout_ms,
            requested_at_ms=request.requested_at_ms,
            payload=request,
        )
        return cast(
            HydratePreparedResult,
            self._run_group_op(
                request=coordinator_request,
                execute_local=execute_local,
                aggregate_results=lambda local_results: self._hydrate_aggregator.aggregate(
                    request=request,
                    publish_manifest=publish_manifest,
                    target=target,
                    local_results=tuple(
                        cast(HydratePreparedResult, local_result)
                        for local_result in local_results
                    ),
                    now_ms=int(request.requested_at_ms),
                ),
            ),
        )

    def evict_local(
        self,
        *,
        request: EvictLocalInstanceOpRequest,
        execute_local: LocalEvictExecutor,
    ) -> EvictLocalInstanceOpResult:
        selector = (
            request.publish_manifest_digest
            if request.publish_manifest_digest is not None
            else request.logical_request_id
        )
        if selector is None:
            raise RequestBundleStateError(
                "evict_local requires logical_request_id or publish_manifest_digest"
            )
        coordinator_request = CoordinatorInstanceOpRequest(
            kind=InstanceOpKind.EVICT_LOCAL,
            op_id=f"evict::{selector}",
            instance_id=self._instance_id,
            required_ranks=self._required_ranks,
            timeout_ms=self._timeout_ms,
            requested_at_ms=request.requested_at_ms,
            payload=request,
        )
        return cast(
            EvictLocalInstanceOpResult,
            self._run_group_op(
                request=coordinator_request,
                execute_local=execute_local,
                aggregate_results=self._aggregate_evict_results,
            ),
        )

    def _run_group_op(
        self,
        *,
        request: CoordinatorInstanceOpRequest,
        execute_local: Callable[[], GroupOpResult],
        aggregate_results: Callable[[tuple[GroupOpResult, ...]], GroupOpResult],
    ) -> GroupOpResult:
        cache_key = self._cache_key(request)
        cached = self._cache_by_key.get(cache_key)
        if cached is not None:
            self._validate_cached_request(cached.request, request)
            if cached.error_message is not None:
                raise RequestBundleStateError(cached.error_message)
            assert cached.result is not None
            return cached.result

        local_envelope = self._execute_local(
            rank=self._current_rank,
            execute_local=execute_local,
        )
        gathered_envelopes = tuple(
            cast(
                _LocalOpEnvelope,
                item,
            )
            for item in self._object_group.all_gather_object(local_envelope)
        )
        group_outcome = None
        if self._current_rank == self._coordinator_rank:
            group_outcome = self._coordinator_finalize(
                request=request,
                gathered_envelopes=gathered_envelopes,
                aggregate_results=aggregate_results,
            )
        group_outcome = cast(
            _GroupOutcome,
            self._object_group.broadcast_object(
                group_outcome,
                src=self._coordinator_group_src,
            ),
        )
        self._cache_by_key[cache_key] = _RuntimeCacheEntry(
            request=request,
            result=group_outcome.result,
            error_message=group_outcome.error_message,
        )
        if group_outcome.error_message is not None:
            raise RequestBundleStateError(group_outcome.error_message)
        assert group_outcome.result is not None
        return group_outcome.result

    def _coordinator_finalize(
        self,
        *,
        request: CoordinatorInstanceOpRequest,
        gathered_envelopes: tuple[_LocalOpEnvelope, ...],
        aggregate_results: Callable[[tuple[GroupOpResult, ...]], GroupOpResult],
    ) -> _GroupOutcome:
        envelope_by_rank = {
            envelope.rank.as_key(): envelope for envelope in gathered_envelopes
        }
        if len(envelope_by_rank) != len(gathered_envelopes):
            return _GroupOutcome(
                error_message="runtime gather returned duplicate rank envelopes"
            )
        per_rank_results = tuple(
            envelope_to_rank_result(
                kind=request.kind,
                envelope=envelope,
            )
            for rank in self._required_ranks
            if (envelope := envelope_by_rank.get(rank.as_key())) is not None
        )
        aggregated_status = self._status_coordinator.coordinate(
            request=request,
            fanout_handler=lambda _rank_requests: CoordinatorFanoutResult(
                rank_results=per_rank_results,
            ),
            now_ms=int(request.requested_at_ms),
        )
        if aggregated_status.status != InstanceOpStatus.SUCCESS:
            return _GroupOutcome(error_message=aggregated_status.error_message)
        ordered_local_results = tuple(
            envelope_by_rank[rank.as_key()].result for rank in self._required_ranks
        )
        if any(local_result is None for local_result in ordered_local_results):
            return _GroupOutcome(
                error_message="successful group operation is missing one or more local results"
            )
        return _GroupOutcome(
            result=aggregate_results(
                tuple(
                    cast(GroupOpResult, local_result)
                    for local_result in ordered_local_results
                )
            )
        )

    def _aggregate_evict_results(
        self, local_results: tuple[GroupOpResult, ...]
    ) -> EvictLocalInstanceOpResult:
        evict_results = tuple(
            cast(EvictLocalInstanceOpResult, local_result)
            for local_result in local_results
        )
        first_result = evict_results[0]
        return EvictLocalInstanceOpResult(
            status=InstanceOpStatus.SUCCESS,
            logical_request_id=first_result.logical_request_id,
            publish_manifest_digest=first_result.publish_manifest_digest,
            evicted_bundle_count=sum(
                result.evicted_bundle_count for result in evict_results
            ),
            evicted_hold_set_count=sum(
                result.evicted_hold_set_count for result in evict_results
            ),
        )

    def _execute_local(
        self,
        *,
        rank: RankCoord,
        execute_local: Callable[[], GroupOpResult],
    ) -> _LocalOpEnvelope:
        try:
            return _LocalOpEnvelope(
                rank=rank,
                status=InstanceOpStatus.SUCCESS,
                result=execute_local(),
            )
        except Exception as exc:
            return _LocalOpEnvelope(
                rank=rank,
                status=InstanceOpStatus.FAILED,
                error_message=str(exc),
            )

    def _cache_key(self, request: CoordinatorInstanceOpRequest) -> tuple[str, str, str]:
        return (request.instance_id, request.kind.value, request.op_id)

    def _validate_cached_request(
        self,
        cached_request: CoordinatorInstanceOpRequest,
        request: CoordinatorInstanceOpRequest,
    ) -> None:
        if cached_request != request:
            raise RequestBundleStateError(
                "idempotent retry must reuse the exact same coordinator request"
            )


def envelope_to_rank_result(
    *,
    kind: InstanceOpKind,
    envelope: _LocalOpEnvelope,
) -> PerRankInstanceOpResult:
    return PerRankInstanceOpResult(
        kind=kind,
        rank=envelope.rank,
        status=envelope.status,
        error_message=envelope.error_message,
    )
