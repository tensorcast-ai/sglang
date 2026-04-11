# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping, Sequence

from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    AggregatedInstanceOpResult,
    InstanceOpKind,
    InstanceOpStatus,
    PerRankInstanceOpResult,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    PageClosureCutoff,
    PageClosureEntry,
    PagePublicationRecord,
    PagePublicationState,
    PreparedBundleClaimAction,
    PreparedBundleClaimDecision,
    PreparedBundleLifecycleState,
    PreparedBundleRecord,
    PreparedHoldRef,
    PreparedHoldSetRecord,
    PreparedHoldSetState,
    RankCoord,
    RankInstallRecord,
    RankInstallState,
    RankSnapshotCursor,
    RequestBundleLifecycleState,
    RequestBundleState,
)


class RequestBundleStateError(RuntimeError):
    pass


def resolve_page_closure_cutoff(
    *,
    requested_token_count: int,
    materialized_token_count: int,
    page_size: int,
) -> PageClosureCutoff:
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if requested_token_count < 0:
        raise ValueError(
            f"requested_token_count must be non-negative, got {requested_token_count}"
        )
    if materialized_token_count < 0:
        raise ValueError(
            "materialized_token_count must be non-negative, got "
            f"{materialized_token_count}"
        )
    if int(requested_token_count) > int(materialized_token_count):
        raise ValueError(
            "requested_token_count must not exceed materialized_token_count, got "
            f"{requested_token_count} > {materialized_token_count}"
        )
    cutoff_token_count = int(requested_token_count)
    tail_valid_tokens = cutoff_token_count % page_size
    page_aligned_token_count = cutoff_token_count - tail_valid_tokens
    frozen_last_page_index = (
        (page_aligned_token_count // page_size) - 1
        if page_aligned_token_count > 0
        else None
    )
    return PageClosureCutoff(
        requested_token_count=int(requested_token_count),
        materialized_token_count=int(materialized_token_count),
        cutoff_token_count=int(cutoff_token_count),
        page_aligned_token_count=int(page_aligned_token_count),
        frozen_last_page_index=frozen_last_page_index,
        tail_valid_tokens=int(tail_valid_tokens),
    )


def aggregate_required_rank_results(
    *,
    kind: InstanceOpKind,
    logical_request_id: str,
    required_ranks: Sequence[RankCoord],
    rank_results: Sequence[PerRankInstanceOpResult],
) -> AggregatedInstanceOpResult:
    required_by_key = {rank.as_key(): rank for rank in required_ranks}
    if len(required_by_key) != len(required_ranks):
        raise RequestBundleStateError("required_ranks must not contain duplicates")
    results_by_key: dict[tuple[int, int], PerRankInstanceOpResult] = {}
    for result in rank_results:
        result_key = result.rank.as_key()
        if result_key not in required_by_key:
            raise RequestBundleStateError(
                f"unexpected rank result for rank={result_key}"
            )
        if result_key in results_by_key:
            raise RequestBundleStateError(
                f"duplicate rank result for rank={result_key}"
            )
        results_by_key[result_key] = result
    missing_keys = [
        rank_key for rank_key in required_by_key if rank_key not in results_by_key
    ]
    if missing_keys:
        return AggregatedInstanceOpResult(
            kind=kind,
            logical_request_id=logical_request_id,
            status=InstanceOpStatus.FAILED,
            required_ranks=tuple(required_ranks),
            rank_results=tuple(rank_results),
            error_message=(
                "missing required rank results for "
                + ", ".join(str(rank_key) for rank_key in sorted(missing_keys))
            ),
        )
    failed_results = [
        result for result in rank_results if result.status == InstanceOpStatus.FAILED
    ]
    if failed_results:
        return AggregatedInstanceOpResult(
            kind=kind,
            logical_request_id=logical_request_id,
            status=InstanceOpStatus.FAILED,
            required_ranks=tuple(required_ranks),
            rank_results=tuple(rank_results),
            error_message=(
                "one or more required ranks failed: "
                + ", ".join(
                    f"{result.rank.as_key()}={result.error_message or 'failed'}"
                    for result in failed_results
                )
            ),
        )
    return AggregatedInstanceOpResult(
        kind=kind,
        logical_request_id=logical_request_id,
        status=InstanceOpStatus.SUCCESS,
        required_ranks=tuple(required_ranks),
        rank_results=tuple(rank_results),
    )


class PagePublicationRegistry:
    def __init__(self) -> None:
        self._records: dict[
            tuple[str, tuple[int, int], int], PagePublicationRecord
        ] = {}

    def set_page_state(
        self,
        *,
        logical_request_id: str,
        rank: RankCoord,
        logical_page_index: int,
        page_hash: str,
        publication_state: PagePublicationState,
        updated_at_ms: int,
        host_resident: bool,
        artifact_id: str | None = None,
        last_error: str | None = None,
    ) -> PagePublicationRecord:
        key = (logical_request_id, rank.as_key(), int(logical_page_index))
        existing = self._records.get(key)
        if existing is not None and existing.page_hash != page_hash:
            raise RequestBundleStateError(
                "page_hash must stay stable for the same logical request page entry"
            )
        record = PagePublicationRecord(
            logical_request_id=logical_request_id,
            rank=rank,
            logical_page_index=int(logical_page_index),
            page_hash=page_hash,
            publication_state=publication_state,
            artifact_id=artifact_id,
            host_resident=host_resident,
            last_error=last_error,
            updated_at_ms=int(updated_at_ms),
        )
        self._records[key] = record
        return record

    def snapshot_rank(
        self,
        *,
        logical_request_id: str,
        rank: RankCoord,
        max_page_index: int | None = None,
    ) -> tuple[PageClosureEntry, ...]:
        selected = []
        rank_key = rank.as_key()
        for (
            request_id,
            entry_rank_key,
            logical_page_index,
        ), record in self._records.items():
            if request_id != logical_request_id or entry_rank_key != rank_key:
                continue
            if max_page_index is not None and logical_page_index > int(max_page_index):
                continue
            selected.append(record)
        selected.sort(key=lambda record: record.logical_page_index)
        return tuple(record.to_closure_entry() for record in selected)

    def matching_page_hashes(
        self,
        *,
        page_hashes: Collection[str],
        rank: RankCoord | None = None,
    ) -> tuple[PagePublicationRecord, ...]:
        requested_hashes = set(page_hashes)
        if not requested_hashes:
            return ()
        rank_key = rank.as_key() if rank is not None else None
        selected = []
        for (_, entry_rank_key, _), record in self._records.items():
            if rank_key is not None and entry_rank_key != rank_key:
                continue
            if record.page_hash not in requested_hashes:
                continue
            selected.append(record)
        selected.sort(
            key=lambda record: (
                record.logical_request_id,
                record.rank.tp_rank,
                record.rank.pp_rank,
                record.logical_page_index,
            )
        )
        return tuple(selected)

    def clear_request(self, logical_request_id: str) -> None:
        stale_keys = [
            record_key
            for record_key in self._records
            if record_key[0] == logical_request_id
        ]
        for record_key in stale_keys:
            self._records.pop(record_key, None)


class RequestBundleStateRegistry:
    def __init__(
        self,
        *,
        page_publication_registry: PagePublicationRegistry | None = None,
    ) -> None:
        self._page_publication_registry = page_publication_registry
        self._records: dict[str, RequestBundleState] = {}

    def upsert_live_request(
        self,
        *,
        logical_request_id: str,
        instance_id: str,
        engine_request_id: str,
        full_prompt_token_count: int,
        model_fingerprint: str,
        kv_layout_id: str,
        tp_size: int,
        pp_size: int,
        required_ranks: Sequence[RankCoord],
        now_ms: int,
    ) -> RequestBundleState:
        existing = self._records.get(logical_request_id)
        if existing is None:
            record = RequestBundleState(
                logical_request_id=logical_request_id,
                instance_id=instance_id,
                engine_request_id=engine_request_id,
                full_prompt_token_count=int(full_prompt_token_count),
                model_fingerprint=model_fingerprint,
                kv_layout_id=kv_layout_id,
                tp_size=int(tp_size),
                pp_size=int(pp_size),
                state=RequestBundleLifecycleState.LIVE_TRACKING,
                snapshot_seq=0,
                required_ranks=tuple(required_ranks),
                rank_snapshots=(),
                created_at_ms=int(now_ms),
                updated_at_ms=int(now_ms),
            )
            self._records[logical_request_id] = record
            return record
        if existing.required_ranks != tuple(required_ranks):
            raise RequestBundleStateError(
                "required_ranks must stay stable for a live logical request"
            )
        updated = existing.model_copy(
            update={
                "instance_id": instance_id,
                "engine_request_id": engine_request_id,
                "full_prompt_token_count": int(full_prompt_token_count),
                "model_fingerprint": model_fingerprint,
                "kv_layout_id": kv_layout_id,
                "tp_size": int(tp_size),
                "pp_size": int(pp_size),
                "state": RequestBundleLifecycleState.LIVE_TRACKING,
                "retained_until_ms": None,
                "updated_at_ms": int(now_ms),
            }
        )
        self._records[logical_request_id] = updated
        return updated

    def get(self, logical_request_id: str) -> RequestBundleState | None:
        return self._records.get(logical_request_id)

    def update_rank_frontier(
        self,
        *,
        logical_request_id: str,
        rank: RankCoord,
        latest_token_count: int,
        latest_last_page_index: int,
        now_ms: int,
        ordered_pages: Sequence[PageClosureEntry] | None = None,
    ) -> RequestBundleState:
        record = self._require_record(logical_request_id)
        self._require_rank(record.required_ranks, rank)
        existing_by_rank = {
            snapshot.rank_coord().as_key(): snapshot
            for snapshot in record.rank_snapshots
        }
        prior_snapshot = existing_by_rank.get(rank.as_key())
        resolved_ordered_pages = (
            tuple(ordered_pages)
            if ordered_pages is not None
            else self._resolved_rank_pages(
                logical_request_id=logical_request_id,
                rank=rank,
                max_page_index=None,
            )
        )
        next_snapshot = RankSnapshotCursor(
            tp_rank=rank.tp_rank,
            pp_rank=rank.pp_rank,
            latest_token_count=int(latest_token_count),
            latest_last_page_index=int(latest_last_page_index),
            frozen_cutoff_token_count=(
                prior_snapshot.frozen_cutoff_token_count if prior_snapshot else None
            ),
            frozen_last_page_index=(
                prior_snapshot.frozen_last_page_index if prior_snapshot else None
            ),
            force_flush_cursor=prior_snapshot.force_flush_cursor
            if prior_snapshot
            else None,
            ordered_pages=resolved_ordered_pages,
        )
        existing_by_rank[rank.as_key()] = next_snapshot
        updated = record.model_copy(
            update={
                "rank_snapshots": self._ordered_present_rank_snapshots(
                    record.required_ranks, existing_by_rank
                ),
                "updated_at_ms": int(now_ms),
            }
        )
        self._records[logical_request_id] = updated
        return updated

    def begin_publish(
        self,
        *,
        logical_request_id: str,
        publish_op_id: str,
        requested_cutoff_token_count: int | None,
        page_size: int,
        now_ms: int,
    ) -> tuple[RequestBundleState, PageClosureCutoff]:
        record = self._require_record(logical_request_id)
        if record.state not in {
            RequestBundleLifecycleState.LIVE_TRACKING,
            RequestBundleLifecycleState.SOURCE_RETAINED,
            RequestBundleLifecycleState.PUBLISHED,
            RequestBundleLifecycleState.PUBLISH_FAILED,
        }:
            raise RequestBundleStateError(
                f"cannot begin publish from state={record.state}"
            )
        snapshots = self._require_all_rank_snapshots(record)
        materialized_token_count = min(
            snapshot.latest_token_count for snapshot in snapshots.values()
        )
        resolved_request_count = (
            int(record.full_prompt_token_count)
            if requested_cutoff_token_count is None
            else int(requested_cutoff_token_count)
        )
        if resolved_request_count > int(record.full_prompt_token_count):
            raise RequestBundleStateError(
                "requested_cutoff_token_count exceeds the full prompt token count"
            )
        if resolved_request_count > materialized_token_count:
            raise RequestBundleStateError(
                "requested_cutoff_token_count exceeds the currently materialized prompt tokens across required ranks"
            )
        cutoff = resolve_page_closure_cutoff(
            requested_token_count=resolved_request_count,
            materialized_token_count=materialized_token_count,
            page_size=page_size,
        )
        updated_snapshots = {}
        for rank in record.required_ranks:
            snapshot = snapshots[rank.as_key()]
            updated_snapshots[rank.as_key()] = snapshot.model_copy(
                update={
                    "frozen_cutoff_token_count": cutoff.cutoff_token_count,
                    "frozen_last_page_index": cutoff.frozen_last_page_index,
                    "force_flush_cursor": 0,
                    "ordered_pages": self._resolved_rank_pages(
                        logical_request_id=logical_request_id,
                        rank=rank,
                        max_page_index=cutoff.frozen_last_page_index,
                        fallback_ordered_pages=snapshot.ordered_pages,
                    ),
                }
            )
        updated = record.model_copy(
            update={
                "state": RequestBundleLifecycleState.SNAPSHOT_CLOSING,
                "snapshot_seq": record.snapshot_seq + 1,
                "publish_op_id": publish_op_id,
                "frozen_cutoff_token_count": cutoff.cutoff_token_count,
                "frozen_last_page_index": cutoff.frozen_last_page_index,
                "bundle_digest": None,
                "rank_snapshots": self._ordered_rank_snapshots(
                    record.required_ranks, updated_snapshots
                ),
                "updated_at_ms": int(now_ms),
                "last_error": None,
            }
        )
        self._records[logical_request_id] = updated
        return updated, cutoff

    def mark_source_retained(
        self,
        *,
        logical_request_id: str,
        retained_until_ms: int,
        now_ms: int,
    ) -> RequestBundleState:
        record = self._require_record(logical_request_id)
        if record.state in {
            RequestBundleLifecycleState.SNAPSHOT_CLOSING,
            RequestBundleLifecycleState.CLOSING_TAIL_FLUSH,
            RequestBundleLifecycleState.CLOSURE_READY,
        }:
            raise RequestBundleStateError(
                "cannot retain source request bundle while it is in a transient publish state"
            )
        updated = record.model_copy(
            update={
                "state": RequestBundleLifecycleState.SOURCE_RETAINED,
                "retained_until_ms": int(retained_until_ms),
                "updated_at_ms": int(now_ms),
            }
        )
        self._records[logical_request_id] = updated
        return updated

    def mark_closing_tail_flush(
        self, *, logical_request_id: str, now_ms: int
    ) -> RequestBundleState:
        return self._transition_request_state(
            logical_request_id=logical_request_id,
            from_states=(RequestBundleLifecycleState.SNAPSHOT_CLOSING,),
            to_state=RequestBundleLifecycleState.CLOSING_TAIL_FLUSH,
            now_ms=now_ms,
        )

    def replace_closure_pages(
        self,
        *,
        logical_request_id: str,
        rank_pages: Mapping[tuple[int, int], Sequence[PageClosureEntry]],
        now_ms: int,
    ) -> RequestBundleState:
        record = self._require_record(logical_request_id)
        if record.state not in {
            RequestBundleLifecycleState.SNAPSHOT_CLOSING,
            RequestBundleLifecycleState.CLOSING_TAIL_FLUSH,
        }:
            raise RequestBundleStateError(
                f"cannot replace closure pages from state={record.state}"
            )
        snapshots = self._require_all_rank_snapshots(record)
        updated_snapshots: dict[tuple[int, int], RankSnapshotCursor] = {}
        for rank in record.required_ranks:
            rank_key = rank.as_key()
            snapshot = snapshots[rank_key]
            updated_snapshots[rank_key] = snapshot.model_copy(
                update={
                    "ordered_pages": tuple(
                        rank_pages.get(rank_key, snapshot.ordered_pages)
                    )
                }
            )
        updated = record.model_copy(
            update={
                "rank_snapshots": self._ordered_rank_snapshots(
                    record.required_ranks, updated_snapshots
                ),
                "updated_at_ms": int(now_ms),
            }
        )
        self._records[logical_request_id] = updated
        return updated

    def mark_closure_ready(
        self,
        *,
        logical_request_id: str,
        bundle_digest: str,
        now_ms: int,
    ) -> RequestBundleState:
        record = self._require_record(logical_request_id)
        if record.state != RequestBundleLifecycleState.CLOSING_TAIL_FLUSH:
            raise RequestBundleStateError(
                f"cannot mark closure ready from state={record.state}"
            )
        updated = record.model_copy(
            update={
                "state": RequestBundleLifecycleState.CLOSURE_READY,
                "bundle_digest": bundle_digest,
                "updated_at_ms": int(now_ms),
                "last_error": None,
            }
        )
        self._records[logical_request_id] = updated
        return updated

    def mark_published(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
        now_ms: int,
    ) -> RequestBundleState:
        record = self._require_record(logical_request_id)
        if record.state != RequestBundleLifecycleState.CLOSURE_READY:
            raise RequestBundleStateError(
                f"cannot mark published from state={record.state}"
            )
        updated = record.model_copy(
            update={
                "state": RequestBundleLifecycleState.PUBLISHED,
                "latest_publish_manifest_digest": publish_manifest_digest,
                "updated_at_ms": int(now_ms),
                "last_error": None,
            }
        )
        self._records[logical_request_id] = updated
        return updated

    def mark_publish_failed(
        self,
        *,
        logical_request_id: str,
        error_message: str,
        now_ms: int,
    ) -> RequestBundleState:
        record = self._require_record(logical_request_id)
        if record.state not in {
            RequestBundleLifecycleState.SNAPSHOT_CLOSING,
            RequestBundleLifecycleState.CLOSING_TAIL_FLUSH,
            RequestBundleLifecycleState.CLOSURE_READY,
        }:
            raise RequestBundleStateError(
                f"cannot mark publish failed from state={record.state}"
            )
        updated = record.model_copy(
            update={
                "state": RequestBundleLifecycleState.PUBLISH_FAILED,
                "updated_at_ms": int(now_ms),
                "last_error": error_message,
            }
        )
        self._records[logical_request_id] = updated
        return updated

    def mark_cleaned(
        self, *, logical_request_id: str, now_ms: int
    ) -> RequestBundleState:
        return self._transition_request_state(
            logical_request_id=logical_request_id,
            from_states=(
                RequestBundleLifecycleState.LIVE_TRACKING,
                RequestBundleLifecycleState.SOURCE_RETAINED,
                RequestBundleLifecycleState.PUBLISHED,
                RequestBundleLifecycleState.PUBLISH_FAILED,
            ),
            to_state=RequestBundleLifecycleState.CLEANED,
            now_ms=now_ms,
        )

    def clear(self, logical_request_id: str) -> None:
        self._records.pop(logical_request_id, None)

    def _transition_request_state(
        self,
        *,
        logical_request_id: str,
        from_states: Sequence[RequestBundleLifecycleState],
        to_state: RequestBundleLifecycleState,
        now_ms: int,
    ) -> RequestBundleState:
        record = self._require_record(logical_request_id)
        if record.state not in set(from_states):
            raise RequestBundleStateError(
                f"cannot transition request bundle from state={record.state}"
            )
        updated = record.model_copy(
            update={
                "state": to_state,
                "updated_at_ms": int(now_ms),
            }
        )
        self._records[logical_request_id] = updated
        return updated

    def _resolved_rank_pages(
        self,
        *,
        logical_request_id: str,
        rank: RankCoord,
        max_page_index: int | None,
        fallback_ordered_pages: Sequence[PageClosureEntry] = (),
    ) -> tuple[PageClosureEntry, ...]:
        if self._page_publication_registry is not None:
            resolved = self._page_publication_registry.snapshot_rank(
                logical_request_id=logical_request_id,
                rank=rank,
                max_page_index=max_page_index,
            )
            if resolved:
                return resolved
        if max_page_index is None:
            return tuple(fallback_ordered_pages)
        return tuple(
            page
            for page in fallback_ordered_pages
            if page.logical_page_index <= int(max_page_index)
        )

    def _require_record(self, logical_request_id: str) -> RequestBundleState:
        record = self._records.get(logical_request_id)
        if record is None:
            raise RequestBundleStateError(
                f"unknown logical_request_id={logical_request_id}"
            )
        return record

    def _require_rank(
        self, required_ranks: Sequence[RankCoord], rank: RankCoord
    ) -> None:
        if rank.as_key() not in {
            required_rank.as_key() for required_rank in required_ranks
        }:
            raise RequestBundleStateError(
                f"rank={rank.as_key()} is not part of the required rank set"
            )

    def _require_all_rank_snapshots(
        self, record: RequestBundleState
    ) -> Mapping[tuple[int, int], RankSnapshotCursor]:
        snapshot_by_rank = {
            snapshot.rank_coord().as_key(): snapshot
            for snapshot in record.rank_snapshots
        }
        missing_ranks = [
            rank.as_key()
            for rank in record.required_ranks
            if rank.as_key() not in snapshot_by_rank
        ]
        if missing_ranks:
            raise RequestBundleStateError(
                "missing rank snapshots for "
                + ", ".join(str(rank_key) for rank_key in sorted(missing_ranks))
            )
        return snapshot_by_rank

    def _ordered_rank_snapshots(
        self,
        required_ranks: Sequence[RankCoord],
        snapshot_by_rank: Mapping[tuple[int, int], RankSnapshotCursor],
    ) -> tuple[RankSnapshotCursor, ...]:
        return tuple(snapshot_by_rank[rank.as_key()] for rank in required_ranks)

    def _ordered_present_rank_snapshots(
        self,
        required_ranks: Sequence[RankCoord],
        snapshot_by_rank: Mapping[tuple[int, int], RankSnapshotCursor],
    ) -> tuple[RankSnapshotCursor, ...]:
        ordered_snapshots: list[RankSnapshotCursor] = []
        for rank in required_ranks:
            snapshot = snapshot_by_rank.get(rank.as_key())
            if snapshot is not None:
                ordered_snapshots.append(snapshot)
        return tuple(ordered_snapshots)


class PreparedHoldRegistry:
    def __init__(self) -> None:
        self._records: dict[str, PreparedHoldSetRecord] = {}

    def install_hold_set(
        self,
        *,
        hold_set_id: str,
        refs: Sequence[PreparedHoldRef],
        now_ms: int,
    ) -> PreparedHoldSetRecord:
        record = PreparedHoldSetRecord(
            hold_set_id=hold_set_id,
            state=PreparedHoldSetState.ACTIVE,
            refs=tuple(refs),
            created_at_ms=int(now_ms),
        )
        self._records[hold_set_id] = record
        return record

    def get(self, hold_set_id: str) -> PreparedHoldSetRecord | None:
        return self._records.get(hold_set_id)

    def release_hold_set(
        self, *, hold_set_id: str, now_ms: int
    ) -> PreparedHoldSetRecord:
        record = self._records.get(hold_set_id)
        if record is None:
            raise RequestBundleStateError(f"unknown hold_set_id={hold_set_id}")
        updated = record.model_copy(
            update={
                "state": PreparedHoldSetState.RELEASED,
                "released_at_ms": int(now_ms),
            }
        )
        self._records[hold_set_id] = updated
        return updated


class PreparedBundleRegistry:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], PreparedBundleRecord] = {}

    def begin_prepare(
        self,
        *,
        logical_request_id: str,
        target_instance_id: str,
        publish_manifest_digest: str,
        artifact_manifest_digest: str,
        engine_owned_manifest_sha256: str,
        required_ranks: Sequence[RankCoord],
        now_ms: int,
        prompt_token_digest: str | None = None,
        cutoff_token_count: int | None = None,
        tail_valid_tokens: int = 0,
        prepared_hold_set_id: str | None = None,
        live_request_exists: bool = False,
    ) -> PreparedBundleRecord:
        if live_request_exists:
            raise RequestBundleStateError(
                "cannot install a prepared bundle while a live request already exists"
            )
        record_key = (logical_request_id, publish_manifest_digest)
        existing = self._records.get(record_key)
        if existing is not None:
            return existing
        conflicting = [
            record
            for (request_id, _digest), record in self._records.items()
            if request_id == logical_request_id
            and record.publish_manifest_digest != publish_manifest_digest
            and record.state != PreparedBundleLifecycleState.EVICTED
        ]
        if conflicting:
            raise RequestBundleStateError(
                "cannot install a different prepared bundle generation before existing generations are cleaned"
            )
        record = PreparedBundleRecord(
            logical_request_id=logical_request_id,
            target_instance_id=target_instance_id,
            publish_manifest_digest=publish_manifest_digest,
            artifact_manifest_digest=artifact_manifest_digest,
            engine_owned_manifest_sha256=engine_owned_manifest_sha256,
            prepared_hold_set_id=prepared_hold_set_id,
            state=PreparedBundleLifecycleState.PREPARING,
            required_ranks=tuple(required_ranks),
            rank_installs=tuple(
                RankInstallRecord(
                    tp_rank=rank.tp_rank,
                    pp_rank=rank.pp_rank,
                    state=RankInstallState.PREPARING,
                    hydrated_page_count=0,
                    runnable_prefix_tokens=0,
                )
                for rank in required_ranks
            ),
            created_at_ms=int(now_ms),
            prompt_token_digest=prompt_token_digest,
            cutoff_token_count=cutoff_token_count,
            tail_valid_tokens=int(tail_valid_tokens),
        )
        self._records[record_key] = record
        return record

    def get(
        self, *, logical_request_id: str, publish_manifest_digest: str
    ) -> PreparedBundleRecord | None:
        return self._records.get((logical_request_id, publish_manifest_digest))

    def list_request_records(
        self, *, logical_request_id: str, include_evicted: bool = False
    ) -> tuple[PreparedBundleRecord, ...]:
        return tuple(
            record
            for (request_id, _digest), record in self._records.items()
            if request_id == logical_request_id
            and (
                include_evicted or record.state != PreparedBundleLifecycleState.EVICTED
            )
        )

    def get_by_publish_manifest_digest(
        self, *, publish_manifest_digest: str, include_evicted: bool = False
    ) -> PreparedBundleRecord | None:
        matched_records = [
            record
            for (_request_id, digest), record in self._records.items()
            if digest == publish_manifest_digest
            and (
                include_evicted or record.state != PreparedBundleLifecycleState.EVICTED
            )
        ]
        if not matched_records:
            return None
        if len(matched_records) > 1:
            raise RequestBundleStateError(
                "publish_manifest_digest must resolve to at most one prepared bundle"
            )
        return matched_records[0]

    def mark_rank_install(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
        rank: RankCoord,
        state: RankInstallState,
        hydrated_page_count: int,
        runnable_prefix_tokens: int,
        local_install_handle: str | None = None,
        last_error: str | None = None,
    ) -> PreparedBundleRecord:
        record = self._require_record(
            logical_request_id=logical_request_id,
            publish_manifest_digest=publish_manifest_digest,
        )
        installs_by_rank = {
            install.rank_coord().as_key(): install for install in record.rank_installs
        }
        if rank.as_key() not in installs_by_rank:
            raise RequestBundleStateError(
                f"rank={rank.as_key()} is not part of the prepared bundle"
            )
        installs_by_rank[rank.as_key()] = RankInstallRecord(
            tp_rank=rank.tp_rank,
            pp_rank=rank.pp_rank,
            state=state,
            hydrated_page_count=int(hydrated_page_count),
            runnable_prefix_tokens=int(runnable_prefix_tokens),
            local_install_handle=local_install_handle,
            last_error=last_error,
        )
        updated = record.model_copy(
            update={
                "rank_installs": tuple(
                    installs_by_rank[required_rank.as_key()]
                    for required_rank in record.required_ranks
                )
            }
        )
        self._records[(logical_request_id, publish_manifest_digest)] = updated
        return updated

    def mark_prepared(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
        now_ms: int,
        prepared_hold_set_id: str | None = None,
    ) -> PreparedBundleRecord:
        record = self._require_record(
            logical_request_id=logical_request_id,
            publish_manifest_digest=publish_manifest_digest,
        )
        if any(
            install.state != RankInstallState.READY for install in record.rank_installs
        ):
            raise RequestBundleStateError(
                "cannot mark prepared until all required ranks are ready"
            )
        updated = record.model_copy(
            update={
                "state": PreparedBundleLifecycleState.PREPARED,
                "prepared_at_ms": int(now_ms),
                "prepared_hold_set_id": prepared_hold_set_id
                if prepared_hold_set_id is not None
                else record.prepared_hold_set_id,
                "last_error": None,
            }
        )
        self._records[(logical_request_id, publish_manifest_digest)] = updated
        return updated

    def mark_failed(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
        error_message: str,
    ) -> PreparedBundleRecord:
        record = self._require_record(
            logical_request_id=logical_request_id,
            publish_manifest_digest=publish_manifest_digest,
        )
        updated = record.model_copy(
            update={
                "state": PreparedBundleLifecycleState.FAILED,
                "last_error": error_message,
            }
        )
        self._records[(logical_request_id, publish_manifest_digest)] = updated
        return updated

    def mark_stale(
        self, *, logical_request_id: str, publish_manifest_digest: str
    ) -> PreparedBundleRecord:
        return self._patch_record(
            logical_request_id=logical_request_id,
            publish_manifest_digest=publish_manifest_digest,
            stale=True,
        )

    def mark_tainted(
        self, *, logical_request_id: str, publish_manifest_digest: str
    ) -> PreparedBundleRecord:
        return self._patch_record(
            logical_request_id=logical_request_id,
            publish_manifest_digest=publish_manifest_digest,
            tainted=True,
        )

    def claim_prepared_bundle(
        self,
        *,
        logical_request_id: str,
        incoming_prompt_token_digest: str,
        incoming_cutoff_token_count: int,
        scheduler_rid: str,
        claim_token: str,
        now_ms: int,
        live_request_exists: bool = False,
    ) -> PreparedBundleClaimDecision:
        records = self._records_for_request(logical_request_id)
        if live_request_exists:
            return PreparedBundleClaimDecision(
                action=PreparedBundleClaimAction.FAIL_CLOSED,
                reason="a live request already exists for this logical request id",
            )
        active_claims = [
            record
            for record in records
            if not record.stale
            and not record.tainted
            and record.state
            in {
                PreparedBundleLifecycleState.CLAIMED,
                PreparedBundleLifecycleState.ATTACHED,
            }
        ]
        if active_claims:
            return PreparedBundleClaimDecision(
                action=PreparedBundleClaimAction.FAIL_CLOSED,
                reason="a clean prepared bundle is already claimed or attached",
            )
        prepared_records = [
            record
            for record in records
            if record.state == PreparedBundleLifecycleState.PREPARED
        ]
        stale_or_tainted_candidates = [
            record for record in prepared_records if record.stale or record.tainted
        ]
        clean_candidates = [
            record
            for record in prepared_records
            if not record.stale and not record.tainted
        ]
        if len(clean_candidates) > 1:
            return PreparedBundleClaimDecision(
                action=PreparedBundleClaimAction.FAIL_CLOSED,
                reason="multiple clean prepared bundle generations exist",
            )
        if not clean_candidates:
            if stale_or_tainted_candidates:
                stale_or_tainted_candidates.sort(
                    key=lambda record: (
                        int(record.prepared_at_ms or record.created_at_ms),
                        record.publish_manifest_digest,
                    )
                )
                return PreparedBundleClaimDecision(
                    action=PreparedBundleClaimAction.FALLBACK,
                    reason="only stale or tainted prepared bundle records are available",
                    record=stale_or_tainted_candidates[-1],
                )
            return PreparedBundleClaimDecision(
                action=PreparedBundleClaimAction.FALLBACK,
                reason="no prepared bundle is available for this logical request id",
            )
        candidate = clean_candidates[0]
        if candidate.prompt_token_digest != incoming_prompt_token_digest:
            return PreparedBundleClaimDecision(
                action=PreparedBundleClaimAction.FALLBACK,
                reason="prepared bundle prompt token digest does not match the incoming request",
                record=candidate,
            )
        if candidate.cutoff_token_count != int(incoming_cutoff_token_count):
            return PreparedBundleClaimDecision(
                action=PreparedBundleClaimAction.FALLBACK,
                reason="prepared bundle cutoff token count does not match the incoming request",
                record=candidate,
            )
        claimed = candidate.model_copy(
            update={
                "state": PreparedBundleLifecycleState.CLAIMED,
                "claim_token": claim_token,
                "active_scheduler_rid": scheduler_rid,
                "claimed_at_ms": int(now_ms),
            }
        )
        self._records[(logical_request_id, candidate.publish_manifest_digest)] = claimed
        return PreparedBundleClaimDecision(
            action=PreparedBundleClaimAction.CLAIMED,
            reason="prepared bundle claimed successfully",
            record=claimed,
            claim_token=claim_token,
        )

    def rollback_claim(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
    ) -> PreparedBundleRecord:
        record = self._require_record(
            logical_request_id=logical_request_id,
            publish_manifest_digest=publish_manifest_digest,
        )
        if record.state != PreparedBundleLifecycleState.CLAIMED:
            raise RequestBundleStateError(
                f"cannot roll back claim from state={record.state}"
            )
        updated = record.model_copy(
            update={
                "state": PreparedBundleLifecycleState.PREPARED,
                "claim_token": None,
                "active_scheduler_rid": None,
                "claimed_at_ms": None,
            }
        )
        self._records[(logical_request_id, publish_manifest_digest)] = updated
        return updated

    def mark_attached(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
        prepared_bundle_key: str,
    ) -> PreparedBundleRecord:
        record = self._require_record(
            logical_request_id=logical_request_id,
            publish_manifest_digest=publish_manifest_digest,
        )
        if record.state != PreparedBundleLifecycleState.CLAIMED:
            raise RequestBundleStateError(
                f"cannot mark attached from state={record.state}"
            )
        updated = record.model_copy(
            update={
                "state": PreparedBundleLifecycleState.ATTACHED,
                "prepared_bundle_key": prepared_bundle_key,
            }
        )
        self._records[(logical_request_id, publish_manifest_digest)] = updated
        return updated

    def mark_consumed(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
        clear_prepared_hold_set_id: bool = False,
    ) -> PreparedBundleRecord:
        record = self._require_record(
            logical_request_id=logical_request_id,
            publish_manifest_digest=publish_manifest_digest,
        )
        if record.state != PreparedBundleLifecycleState.ATTACHED:
            raise RequestBundleStateError(
                f"cannot mark consumed from state={record.state}"
            )
        updated = record.model_copy(
            update={
                "state": PreparedBundleLifecycleState.CONSUMED,
                "prepared_hold_set_id": None
                if clear_prepared_hold_set_id
                else record.prepared_hold_set_id,
            }
        )
        self._records[(logical_request_id, publish_manifest_digest)] = updated
        return updated

    def evict_bundle(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
        now_ms: int,
    ) -> PreparedBundleRecord:
        record = self._require_record(
            logical_request_id=logical_request_id,
            publish_manifest_digest=publish_manifest_digest,
        )
        updated = record.model_copy(
            update={
                "state": PreparedBundleLifecycleState.EVICTED,
                "cleaned_at_ms": int(now_ms),
            }
        )
        self._records[(logical_request_id, publish_manifest_digest)] = updated
        return updated

    def _patch_record(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
        stale: bool | None = None,
        tainted: bool | None = None,
    ) -> PreparedBundleRecord:
        record = self._require_record(
            logical_request_id=logical_request_id,
            publish_manifest_digest=publish_manifest_digest,
        )
        update_payload: dict[str, bool] = {}
        if stale is not None:
            update_payload["stale"] = stale
        if tainted is not None:
            update_payload["tainted"] = tainted
        updated = record.model_copy(update=update_payload)
        self._records[(logical_request_id, publish_manifest_digest)] = updated
        return updated

    def _records_for_request(
        self, logical_request_id: str
    ) -> list[PreparedBundleRecord]:
        return [
            record
            for (request_id, _digest), record in self._records.items()
            if request_id == logical_request_id
            and record.state != PreparedBundleLifecycleState.EVICTED
        ]

    def _require_record(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
    ) -> PreparedBundleRecord:
        record = self._records.get((logical_request_id, publish_manifest_digest))
        if record is None:
            raise RequestBundleStateError(
                "unknown prepared bundle generation "
                f"logical_request_id={logical_request_id} "
                f"publish_manifest_digest={publish_manifest_digest}"
            )
        return record


def compute_bundle_digest(
    *,
    rank_snapshots: Sequence[RankSnapshotCursor],
) -> str:
    digest = hashlib.sha256()
    for snapshot in rank_snapshots:
        digest.update(f"{snapshot.tp_rank}:{snapshot.pp_rank}|".encode("utf-8"))
        for page in snapshot.ordered_pages:
            digest.update(
                f"{page.logical_page_index}:{page.page_hash}:{page.publication_state.value}|".encode(
                    "utf-8"
                )
            )
            if page.artifact_id is not None:
                digest.update(page.artifact_id.encode("utf-8"))
            digest.update(b";")
    return digest.hexdigest()
