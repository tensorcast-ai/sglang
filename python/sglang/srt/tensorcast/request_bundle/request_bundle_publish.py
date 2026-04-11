# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import TypeAlias

from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    PagePublicationRegistry,
    RequestBundleStateRegistry,
    RequestBundleStateError,
    compute_bundle_digest,
)
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    PublishInstanceOpRequest,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    EngineOwnedManifestPayload,
    EngineOwnedManifestRecord,
    PageClosureEntry,
    PagePublishAction,
    PagePublishOutcome,
    PagePublicationState,
    PublishArtifactManifestEntry,
    PublishArtifactManifestRecord,
    PublishCompatibilityEnvelope,
    PublishManifestRecord,
    RankCoord,
    RankPublishClosureResult,
    RequestBundleLifecycleState,
    RequestBundleState,
    SourcePublishClosureResult,
)

InflightPageResolver: TypeAlias = Callable[
    [RankCoord, PageClosureEntry], PageClosureEntry
]
AbsentPageResolver: TypeAlias = Callable[
    [RankCoord, PageClosureEntry], PageClosureEntry
]


class RequestBundlePublisher:
    def __init__(
        self,
        *,
        request_bundle_registry: RequestBundleStateRegistry,
        page_publication_registry: PagePublicationRegistry,
    ) -> None:
        self._request_bundle_registry = request_bundle_registry
        self._page_publication_registry = page_publication_registry

    def publish(
        self,
        *,
        request: PublishInstanceOpRequest,
        now_ms: int,
        resolve_inflight_page: InflightPageResolver | None = None,
        force_flush_absent_page: AbsentPageResolver | None = None,
    ) -> SourcePublishClosureResult:
        self._require_source_request_state(request)
        self._validate_v1_publish_shape(request)
        closure_state, cutoff = self._request_bundle_registry.begin_publish(
            logical_request_id=request.logical_request_id,
            publish_op_id=request.publish_op_id,
            requested_cutoff_token_count=request.requested_cutoff_token_count,
            page_size=request.page_size,
            now_ms=now_ms,
        )
        closure_state = self._request_bundle_registry.mark_closing_tail_flush(
            logical_request_id=request.logical_request_id,
            now_ms=now_ms,
        )
        rank_pages: dict[tuple[int, int], tuple[PageClosureEntry, ...]] = {}
        rank_results: list[RankPublishClosureResult] = []
        try:
            for snapshot in closure_state.rank_snapshots:
                rank = snapshot.rank_coord()
                self._assert_snapshot_covers_cutoff(
                    rank=rank,
                    closure_state=closure_state,
                    ordered_pages=snapshot.ordered_pages,
                )
                resolved_pages: list[PageClosureEntry] = []
                page_outcomes: list[PagePublishOutcome] = []
                for page in snapshot.ordered_pages:
                    self._require_request_still_publishable(request.logical_request_id)
                    resolved_page, action = self._resolve_page_closure(
                        logical_request_id=request.logical_request_id,
                        rank=rank,
                        page=page,
                        now_ms=now_ms,
                        resolve_inflight_page=resolve_inflight_page,
                        force_flush_absent_page=force_flush_absent_page,
                    )
                    resolved_pages.append(resolved_page)
                    page_outcomes.append(
                        PagePublishOutcome(
                            rank=rank,
                            logical_page_index=resolved_page.logical_page_index,
                            page_hash=resolved_page.page_hash,
                            action=action,
                            final_publication_state=resolved_page.publication_state,
                            artifact_id=resolved_page.artifact_id,
                        )
                    )
                rank_pages[rank.as_key()] = tuple(resolved_pages)
                rank_results.append(
                    RankPublishClosureResult(
                        rank=rank,
                        page_outcomes=tuple(page_outcomes),
                    )
                )
        except Exception as exc:
            self._mark_publish_failed_if_present(
                logical_request_id=request.logical_request_id,
                error_message=str(exc),
                now_ms=now_ms,
            )
            raise

        closure_state = self._request_bundle_registry.replace_closure_pages(
            logical_request_id=request.logical_request_id,
            rank_pages=rank_pages,
            now_ms=now_ms,
        )
        bundle_digest = compute_bundle_digest(
            rank_snapshots=closure_state.rank_snapshots
        )
        closure_state = self._request_bundle_registry.mark_closure_ready(
            logical_request_id=request.logical_request_id,
            bundle_digest=bundle_digest,
            now_ms=now_ms,
        )
        publish_manifest = self._build_publish_manifest(
            request_state=closure_state,
            request=request,
            created_at_ms=now_ms,
        )
        closure_state = self._request_bundle_registry.mark_published(
            logical_request_id=request.logical_request_id,
            publish_manifest_digest=publish_manifest.publish_manifest_digest,
            now_ms=now_ms,
        )
        return SourcePublishClosureResult(
            request_state=closure_state,
            cutoff=cutoff,
            publish_manifest=publish_manifest,
            rank_results=tuple(rank_results),
        )

    def _require_source_request_state(
        self, request: PublishInstanceOpRequest
    ) -> RequestBundleState:
        request_state = self._request_bundle_registry.get(request.logical_request_id)
        if request_state is None:
            raise RequestBundleStateError(
                f"unknown logical_request_id={request.logical_request_id}"
            )
        if request_state.engine_request_id != request.engine_request_id:
            raise RequestBundleStateError(
                "publish engine_request_id does not match the source request state"
            )
        if request_state.state == RequestBundleLifecycleState.CLEANED:
            raise RequestBundleStateError(
                "cannot publish a request that has already been cleaned"
            )
        return request_state

    def _validate_v1_publish_shape(self, request: PublishInstanceOpRequest) -> None:
        if request.batch_request_count != 1:
            raise RequestBundleStateError(
                "v1 publish does not support batch request transfer"
            )
        if request.parallel_sampling_count != 1:
            raise RequestBundleStateError(
                "v1 publish does not support parallel-sampling transfer"
            )
        if request.session_lineage_depth != 0:
            raise RequestBundleStateError(
                "v1 publish does not support session-lineage transfer"
            )
        if not request.prompt_token_digest:
            raise RequestBundleStateError(
                "v1 publish requires a non-empty prompt_token_digest"
            )

    def _require_request_still_publishable(
        self, logical_request_id: str
    ) -> RequestBundleState:
        request_state = self._request_bundle_registry.get(logical_request_id)
        if request_state is None:
            raise RequestBundleStateError(
                "source request state was released before publish closure completed"
            )
        return request_state

    def _assert_snapshot_covers_cutoff(
        self,
        *,
        rank: RankCoord,
        closure_state: RequestBundleState,
        ordered_pages: Sequence[PageClosureEntry],
    ) -> None:
        if closure_state.frozen_last_page_index is None:
            if ordered_pages:
                raise RequestBundleStateError(
                    f"rank={rank.as_key()} has pages beyond an empty publish cutoff"
                )
            return
        expected_indices = tuple(range(closure_state.frozen_last_page_index + 1))
        actual_indices = tuple(page.logical_page_index for page in ordered_pages)
        if actual_indices != expected_indices:
            raise RequestBundleStateError(
                "publish snapshot does not cover the full cutoff for "
                f"rank={rank.as_key()}: expected={expected_indices} actual={actual_indices}"
            )

    def _resolve_page_closure(
        self,
        *,
        logical_request_id: str,
        rank: RankCoord,
        page: PageClosureEntry,
        now_ms: int,
        resolve_inflight_page: InflightPageResolver | None,
        force_flush_absent_page: AbsentPageResolver | None,
    ) -> tuple[PageClosureEntry, PagePublishAction]:
        if page.publication_state == PagePublicationState.READY:
            resolved_page = page
            action = PagePublishAction.REUSED
        elif page.publication_state == PagePublicationState.INFLIGHT:
            if resolve_inflight_page is None:
                raise RequestBundleStateError(
                    f"rank={rank.as_key()} page={page.logical_page_index} is inflight without a resolver"
                )
            resolved_page = resolve_inflight_page(rank, page)
            action = PagePublishAction.WAITED
        elif page.publication_state == PagePublicationState.ABSENT:
            if not page.host_resident:
                raise RequestBundleStateError(
                    f"rank={rank.as_key()} page={page.logical_page_index} is absent and no host-resident source exists"
                )
            if force_flush_absent_page is None:
                raise RequestBundleStateError(
                    f"rank={rank.as_key()} page={page.logical_page_index} is absent without a force-flush resolver"
                )
            resolved_page = force_flush_absent_page(rank, page)
            action = PagePublishAction.FLUSHED
        else:
            raise RequestBundleStateError(
                f"rank={rank.as_key()} page={page.logical_page_index} is in terminal failed state"
            )
        self._validate_resolved_page(
            rank=rank, original_page=page, resolved_page=resolved_page
        )
        self._page_publication_registry.set_page_state(
            logical_request_id=logical_request_id,
            rank=rank,
            logical_page_index=resolved_page.logical_page_index,
            page_hash=resolved_page.page_hash,
            publication_state=resolved_page.publication_state,
            artifact_id=resolved_page.artifact_id,
            host_resident=resolved_page.host_resident,
            last_error=resolved_page.last_error,
            updated_at_ms=int(now_ms),
        )
        return resolved_page, action

    def _validate_resolved_page(
        self,
        *,
        rank: RankCoord,
        original_page: PageClosureEntry,
        resolved_page: PageClosureEntry,
    ) -> None:
        if resolved_page.logical_page_index != original_page.logical_page_index:
            raise RequestBundleStateError(
                f"rank={rank.as_key()} page resolver changed logical_page_index"
            )
        if resolved_page.page_hash != original_page.page_hash:
            raise RequestBundleStateError(
                f"rank={rank.as_key()} page resolver changed page_hash"
            )
        if resolved_page.publication_state != PagePublicationState.READY:
            raise RequestBundleStateError(
                f"rank={rank.as_key()} page={original_page.logical_page_index} did not reach ready state"
            )
        if not resolved_page.artifact_id:
            raise RequestBundleStateError(
                f"rank={rank.as_key()} page={original_page.logical_page_index} is ready without artifact_id"
            )

    def _build_publish_manifest(
        self,
        *,
        request_state: RequestBundleState,
        request: PublishInstanceOpRequest,
        created_at_ms: int,
    ) -> PublishManifestRecord:
        return build_publish_manifest_record(
            request_state=request_state,
            request=request,
            created_at_ms=created_at_ms,
        )

    def _mark_publish_failed_if_present(
        self,
        *,
        logical_request_id: str,
        error_message: str,
        now_ms: int,
    ) -> None:
        if self._request_bundle_registry.get(logical_request_id) is None:
            return
        self._request_bundle_registry.mark_publish_failed(
            logical_request_id=logical_request_id,
            error_message=error_message,
            now_ms=now_ms,
        )

    def _sha256_hexdigest(self, parts: Sequence[str]) -> str:
        return _sha256_parts_hexdigest(parts)


class RequestBundlePublishAggregator:
    def aggregate(
        self,
        *,
        request: PublishInstanceOpRequest,
        local_results: Sequence[SourcePublishClosureResult],
        now_ms: int,
        required_ranks: Sequence[RankCoord] | None = None,
    ) -> SourcePublishClosureResult:
        if not local_results:
            raise RequestBundleStateError(
                "publish aggregation requires at least one local result"
            )
        first_result = local_results[0]
        aggregate_cutoff = first_result.cutoff
        aggregate_state = first_result.request_state
        local_result_by_rank: dict[tuple[int, int], SourcePublishClosureResult] = {}
        rank_snapshots_by_key: dict[tuple[int, int], object] = {}
        rank_results_by_key: dict[tuple[int, int], RankPublishClosureResult] = {}
        for local_result in local_results:
            local_state = local_result.request_state
            self._validate_local_result(
                request=request,
                aggregate_cutoff=aggregate_cutoff,
                aggregate_state=aggregate_state,
                local_result=local_result,
            )
            local_rank = self._extract_single_rank(local_result)
            local_rank_key = local_rank.as_key()
            if local_rank_key in local_result_by_rank:
                raise RequestBundleStateError(
                    f"duplicate local publish result for rank={local_rank_key}"
                )
            local_result_by_rank[local_rank_key] = local_result
            rank_snapshots_by_key[local_rank_key] = local_state.rank_snapshots[0]
            rank_results_by_key[local_rank_key] = local_result.rank_results[0]
        ordered_required_ranks = self._resolve_required_ranks(
            local_result_by_rank=local_result_by_rank,
            required_ranks=required_ranks,
        )
        aggregate_rank_snapshots = tuple(
            rank_snapshots_by_key[rank.as_key()] for rank in ordered_required_ranks
        )
        aggregate_bundle_digest = compute_bundle_digest(
            rank_snapshots=aggregate_rank_snapshots
        )
        aggregate_request_state = RequestBundleState(
            logical_request_id=request.logical_request_id,
            instance_id=aggregate_state.instance_id,
            engine_request_id=request.engine_request_id,
            full_prompt_token_count=aggregate_state.full_prompt_token_count,
            model_fingerprint=aggregate_state.model_fingerprint,
            kv_layout_id=aggregate_state.kv_layout_id,
            tp_size=aggregate_state.tp_size,
            pp_size=aggregate_state.pp_size,
            state=RequestBundleLifecycleState.PUBLISHED,
            snapshot_seq=aggregate_state.snapshot_seq,
            publish_op_id=request.publish_op_id,
            frozen_cutoff_token_count=aggregate_cutoff.cutoff_token_count,
            frozen_last_page_index=aggregate_cutoff.frozen_last_page_index,
            bundle_digest=aggregate_bundle_digest,
            latest_publish_manifest_digest=None,
            required_ranks=ordered_required_ranks,
            rank_snapshots=aggregate_rank_snapshots,
            created_at_ms=aggregate_state.created_at_ms,
            updated_at_ms=int(now_ms),
        )
        publish_manifest = build_publish_manifest_record(
            request_state=aggregate_request_state,
            request=request,
            created_at_ms=now_ms,
        )
        aggregate_request_state = aggregate_request_state.model_copy(
            update={
                "latest_publish_manifest_digest": publish_manifest.publish_manifest_digest
            }
        )
        return SourcePublishClosureResult(
            request_state=aggregate_request_state,
            cutoff=aggregate_cutoff,
            publish_manifest=publish_manifest,
            rank_results=tuple(
                rank_results_by_key[rank.as_key()] for rank in ordered_required_ranks
            ),
        )

    def _validate_local_result(
        self,
        *,
        request: PublishInstanceOpRequest,
        aggregate_cutoff: object,
        aggregate_state: RequestBundleState,
        local_result: SourcePublishClosureResult,
    ) -> None:
        local_state = local_result.request_state
        if local_state.logical_request_id != request.logical_request_id:
            raise RequestBundleStateError(
                "local publish result logical_request_id does not match aggregate request"
            )
        if local_state.engine_request_id != request.engine_request_id:
            raise RequestBundleStateError(
                "local publish result engine_request_id does not match aggregate request"
            )
        if local_state.instance_id != aggregate_state.instance_id:
            raise RequestBundleStateError(
                "all local publish results must belong to the same logical instance"
            )
        if local_state.model_fingerprint != aggregate_state.model_fingerprint:
            raise RequestBundleStateError(
                "all local publish results must share the same model_fingerprint"
            )
        if local_state.kv_layout_id != aggregate_state.kv_layout_id:
            raise RequestBundleStateError(
                "all local publish results must share the same kv_layout_id"
            )
        if local_state.tp_size != aggregate_state.tp_size:
            raise RequestBundleStateError(
                "all local publish results must share the same tp_size"
            )
        if local_state.pp_size != aggregate_state.pp_size:
            raise RequestBundleStateError(
                "all local publish results must share the same pp_size"
            )
        if local_state.snapshot_seq != aggregate_state.snapshot_seq:
            raise RequestBundleStateError(
                "all local publish results must share the same snapshot_seq"
            )
        if local_state.publish_op_id != request.publish_op_id:
            raise RequestBundleStateError(
                "local publish result publish_op_id does not match aggregate request"
            )
        if local_state.state != RequestBundleLifecycleState.PUBLISHED:
            raise RequestBundleStateError(
                f"cannot aggregate local publish result from state={local_state.state}"
            )
        if local_result.cutoff != aggregate_cutoff:
            raise RequestBundleStateError(
                "all local publish results must resolve the same publish closure cutoff"
            )
        if (
            local_result.publish_manifest.prompt_token_digest
            != request.prompt_token_digest
        ):
            raise RequestBundleStateError(
                "local publish result prompt_token_digest does not match aggregate request"
            )
        if local_result.publish_manifest.cutoff_token_count != getattr(
            aggregate_cutoff, "cutoff_token_count"
        ):
            raise RequestBundleStateError(
                "local publish result cutoff_token_count does not match aggregate cutoff"
            )

    def _extract_single_rank(
        self, local_result: SourcePublishClosureResult
    ) -> RankCoord:
        local_state = local_result.request_state
        if len(local_state.required_ranks) != 1:
            raise RequestBundleStateError(
                "local publish result must contain exactly one required rank"
            )
        if len(local_state.rank_snapshots) != 1:
            raise RequestBundleStateError(
                "local publish result must contain exactly one rank snapshot"
            )
        if len(local_result.rank_results) != 1:
            raise RequestBundleStateError(
                "local publish result must contain exactly one rank result"
            )
        local_rank = local_state.required_ranks[0]
        if local_state.rank_snapshots[0].rank_coord() != local_rank:
            raise RequestBundleStateError(
                "local publish result rank snapshot does not match the local required rank"
            )
        if local_result.rank_results[0].rank != local_rank:
            raise RequestBundleStateError(
                "local publish result rank outcome does not match the local required rank"
            )
        return local_rank

    def _resolve_required_ranks(
        self,
        *,
        local_result_by_rank: dict[tuple[int, int], SourcePublishClosureResult],
        required_ranks: Sequence[RankCoord] | None,
    ) -> tuple[RankCoord, ...]:
        if required_ranks is None:
            return tuple(
                sorted(
                    (
                        local_result.request_state.required_ranks[0]
                        for local_result in local_result_by_rank.values()
                    ),
                    key=lambda rank: rank.as_key(),
                )
            )
        ordered_required_ranks = tuple(required_ranks)
        missing_rank_keys = [
            rank.as_key()
            for rank in ordered_required_ranks
            if rank.as_key() not in local_result_by_rank
        ]
        if missing_rank_keys:
            raise RequestBundleStateError(
                "missing local publish results for "
                + ", ".join(str(rank_key) for rank_key in sorted(missing_rank_keys))
            )
        unexpected_rank_keys = [
            rank_key
            for rank_key in local_result_by_rank
            if rank_key not in {rank.as_key() for rank in ordered_required_ranks}
        ]
        if unexpected_rank_keys:
            raise RequestBundleStateError(
                "unexpected local publish results for "
                + ", ".join(str(rank_key) for rank_key in sorted(unexpected_rank_keys))
            )
        return ordered_required_ranks


def build_publish_manifest_record(
    *,
    request_state: RequestBundleState,
    request: PublishInstanceOpRequest,
    created_at_ms: int,
) -> PublishManifestRecord:
    cutoff_token_count = int(request_state.frozen_cutoff_token_count or 0)
    tail_valid_tokens = cutoff_token_count % int(request.page_size)
    artifact_entries = tuple(
        PublishArtifactManifestEntry(
            rank=snapshot.rank_coord(),
            logical_page_index=page.logical_page_index,
            page_hash=page.page_hash,
            artifact_id=page.artifact_id or "",
        )
        for snapshot in request_state.rank_snapshots
        for page in snapshot.ordered_pages
    )
    artifact_manifest_digest = _sha256_parts_hexdigest(
        [
            request_state.logical_request_id,
            str(request_state.snapshot_seq),
            request_state.bundle_digest or "",
            *[
                "|".join(
                    (
                        str(entry.rank.tp_rank),
                        str(entry.rank.pp_rank),
                        str(entry.logical_page_index),
                        entry.page_hash,
                        entry.artifact_id,
                    )
                )
                for entry in artifact_entries
            ],
        ]
    )
    artifact_manifest = PublishArtifactManifestRecord(
        artifact_manifest_digest=artifact_manifest_digest,
        logical_request_id=request_state.logical_request_id,
        snapshot_seq=request_state.snapshot_seq,
        required_ranks=request_state.required_ranks,
        entries=artifact_entries,
    )
    compatibility = PublishCompatibilityEnvelope(
        model_fingerprint=request_state.model_fingerprint,
        kv_layout_id=request_state.kv_layout_id,
        dtype=request.dtype,
        page_size=request.page_size,
        tp_size=request_state.tp_size,
        pp_size=request_state.pp_size,
        attention_arch=request.attention_arch,
        required_ranks=request_state.required_ranks,
    )
    engine_owned_payload = EngineOwnedManifestPayload(
        transfer_mode=request.transfer_mode,
        logical_request_id=request_state.logical_request_id,
        cutoff_token_count=cutoff_token_count,
        frozen_last_page_index=request_state.frozen_last_page_index,
        tail_valid_tokens=tail_valid_tokens,
        prompt_token_digest=request.prompt_token_digest,
        compatibility=compatibility,
    )
    payload_sha256 = hashlib.sha256(
        json.dumps(
            engine_owned_payload.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    engine_owned_manifest = EngineOwnedManifestRecord(
        created_at_ms=int(created_at_ms),
        expires_at_ms=int(created_at_ms + request.ttl_ms),
        artifact_manifest_digest=artifact_manifest.artifact_manifest_digest,
        payload_sha256=payload_sha256,
        payload=engine_owned_payload,
    )
    publish_manifest_digest = _sha256_parts_hexdigest(
        [
            artifact_manifest.artifact_manifest_digest,
            payload_sha256,
            request_state.bundle_digest or "",
            str(request_state.snapshot_seq),
        ]
    )
    return PublishManifestRecord(
        publish_manifest_digest=publish_manifest_digest,
        artifact_manifest=artifact_manifest,
        engine_owned_manifest=engine_owned_manifest,
        prompt_token_digest=request.prompt_token_digest,
        cutoff_token_count=cutoff_token_count,
        tail_valid_tokens=tail_valid_tokens,
    )


def _sha256_parts_hexdigest(parts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
