# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import TypeAlias

from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    PreparedBundleRegistry,
    PreparedHoldRegistry,
    RequestBundleStateError,
)
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    HydrateInstanceOpRequest,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    HydratePreparedResult,
    HydrateRankInstallResult,
    HydrateRankWorkItem,
    HydrateTargetCompatibility,
    PreparedBundleRecord,
    PreparedBundleLifecycleState,
    PreparedHoldSetRecord,
    PreparedHoldSetState,
    PublishCompatibilityEnvelope,
    PublishManifestRecord,
    RankCoord,
    RankInstallState,
)

HydrateRankInstaller: TypeAlias = Callable[
    [HydrateRankWorkItem],
    HydrateRankInstallResult,
]
HydrateRankCleanup: TypeAlias = Callable[[HydrateRankInstallResult], None]


class RequestBundleHydrator:
    def __init__(
        self,
        *,
        prepared_bundle_registry: PreparedBundleRegistry,
        prepared_hold_registry: PreparedHoldRegistry,
    ) -> None:
        self._prepared_bundle_registry = prepared_bundle_registry
        self._prepared_hold_registry = prepared_hold_registry

    def hydrate(
        self,
        *,
        request: HydrateInstanceOpRequest,
        publish_manifest: PublishManifestRecord,
        target: HydrateTargetCompatibility,
        local_rank: RankCoord,
        install_rank: HydrateRankInstaller,
        now_ms: int,
        live_request_exists: bool = False,
        cleanup_rank_install: HydrateRankCleanup | None = None,
    ) -> HydratePreparedResult:
        self._validate_manifest_binding(
            request=request,
            publish_manifest=publish_manifest,
            target=target,
            now_ms=now_ms,
        )
        self._validate_local_rank_membership(
            local_rank=local_rank,
            publish_manifest=publish_manifest,
            target=target,
        )
        existing = self._prepared_bundle_registry.get(
            logical_request_id=request.logical_request_id,
            publish_manifest_digest=request.publish_manifest_digest,
        )
        if existing is not None:
            if existing.state == PreparedBundleLifecycleState.PREPARED:
                hold_set = (
                    self._prepared_hold_registry.get(existing.prepared_hold_set_id)
                    if existing.prepared_hold_set_id is not None
                    else None
                )
                return HydratePreparedResult(
                    prepared_bundle=existing,
                    hold_set=hold_set,
                    reused_existing=True,
                )
            raise RequestBundleStateError(
                "repeated hydrate with the same manifest is only idempotent once a prepared bundle exists"
            )

        prepared_bundle = self._prepared_bundle_registry.begin_prepare(
            logical_request_id=request.logical_request_id,
            target_instance_id=target.target_instance_id,
            publish_manifest_digest=publish_manifest.publish_manifest_digest,
            artifact_manifest_digest=publish_manifest.artifact_manifest.artifact_manifest_digest,
            engine_owned_manifest_sha256=self._sha256_hexdigest(
                json.dumps(
                    publish_manifest.engine_owned_manifest.model_dump(
                        mode="json",
                        by_alias=True,
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            required_ranks=(local_rank,),
            now_ms=now_ms,
            prompt_token_digest=publish_manifest.prompt_token_digest,
            cutoff_token_count=publish_manifest.cutoff_token_count,
            tail_valid_tokens=publish_manifest.tail_valid_tokens,
            live_request_exists=live_request_exists,
        )

        install_result: HydrateRankInstallResult | None = None
        try:
            work_item = self._build_rank_work_item(
                rank=local_rank,
                publish_manifest=publish_manifest,
            )
            try:
                install_result = install_rank(work_item)
            except Exception as exc:
                self._prepared_bundle_registry.mark_rank_install(
                    logical_request_id=request.logical_request_id,
                    publish_manifest_digest=publish_manifest.publish_manifest_digest,
                    rank=local_rank,
                    state=RankInstallState.FAILED,
                    hydrated_page_count=0,
                    runnable_prefix_tokens=0,
                    last_error=str(exc),
                )
                raise
            if install_result.rank != local_rank:
                raise RequestBundleStateError(
                    f"hydrate installer returned mismatched rank {install_result.rank.as_key()} for expected rank {local_rank.as_key()}"
                )
            self._prepared_bundle_registry.mark_rank_install(
                logical_request_id=request.logical_request_id,
                publish_manifest_digest=publish_manifest.publish_manifest_digest,
                rank=local_rank,
                state=install_result.state,
                hydrated_page_count=install_result.hydrated_page_count,
                runnable_prefix_tokens=install_result.runnable_prefix_tokens,
                local_install_handle=install_result.local_install_handle,
                last_error=install_result.error_message,
            )
            if install_result.state != RankInstallState.READY:
                raise RequestBundleStateError(
                    install_result.error_message
                    or f"rank={local_rank.as_key()} failed to install runnable hydrate state"
                )
        except Exception as exc:
            self._cleanup_failed_install(
                logical_request_id=request.logical_request_id,
                publish_manifest_digest=publish_manifest.publish_manifest_digest,
                install_results=() if install_result is None else (install_result,),
                cleanup_rank_install=cleanup_rank_install,
            )
            self._prepared_bundle_registry.mark_failed(
                logical_request_id=request.logical_request_id,
                publish_manifest_digest=publish_manifest.publish_manifest_digest,
                error_message=str(exc),
            )
            raise

        assert install_result is not None
        hold_refs = install_result.hold_refs
        hold_set = None
        prepared_hold_set_id = None
        if hold_refs:
            prepared_hold_set_id = self._build_local_hold_set_id(
                logical_request_id=request.logical_request_id,
                publish_manifest_digest=publish_manifest.publish_manifest_digest,
                local_rank=local_rank,
            )
            hold_set = self._prepared_hold_registry.install_hold_set(
                hold_set_id=prepared_hold_set_id,
                refs=hold_refs,
                now_ms=now_ms,
            )
        prepared_bundle = self._prepared_bundle_registry.mark_prepared(
            logical_request_id=request.logical_request_id,
            publish_manifest_digest=publish_manifest.publish_manifest_digest,
            now_ms=now_ms,
            prepared_hold_set_id=prepared_hold_set_id,
        )
        return HydratePreparedResult(
            prepared_bundle=prepared_bundle,
            hold_set=hold_set,
            reused_existing=False,
        )

    def _validate_manifest_binding(
        self,
        *,
        request: HydrateInstanceOpRequest,
        publish_manifest: PublishManifestRecord,
        target: HydrateTargetCompatibility,
        now_ms: int,
    ) -> None:
        if (
            request.logical_request_id
            != publish_manifest.engine_owned_manifest.payload.logical_request_id
        ):
            raise RequestBundleStateError(
                "hydrate logical_request_id does not match publish manifest payload"
            )
        if request.publish_manifest_digest != publish_manifest.publish_manifest_digest:
            raise RequestBundleStateError(
                "hydrate publish_manifest_digest does not match the provided publish manifest"
            )
        if (
            publish_manifest.artifact_manifest.logical_request_id
            != request.logical_request_id
        ):
            raise RequestBundleStateError(
                "artifact manifest logical_request_id does not match hydrate request"
            )
        if (
            publish_manifest.engine_owned_manifest.artifact_manifest_digest
            != publish_manifest.artifact_manifest.artifact_manifest_digest
        ):
            raise RequestBundleStateError(
                "engine-owned manifest is not bound to the artifact manifest digest"
            )
        if publish_manifest.engine_owned_manifest.expires_at_ms < int(now_ms):
            raise RequestBundleStateError("publish manifest is expired")
        payload = publish_manifest.engine_owned_manifest.payload
        if payload.cutoff_token_count != publish_manifest.cutoff_token_count:
            raise RequestBundleStateError(
                "publish manifest cutoff_token_count does not match the engine-owned payload"
            )
        if payload.tail_valid_tokens != publish_manifest.tail_valid_tokens:
            raise RequestBundleStateError(
                "publish manifest tail_valid_tokens does not match the engine-owned payload"
            )
        if payload.prompt_token_digest != publish_manifest.prompt_token_digest:
            raise RequestBundleStateError(
                "publish manifest prompt_token_digest does not match the engine-owned payload"
            )
        expected_payload_sha = self._sha256_hexdigest(
            json.dumps(
                payload.model_dump(mode="json", by_alias=True),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if (
            publish_manifest.engine_owned_manifest.payload_sha256
            != expected_payload_sha
        ):
            raise RequestBundleStateError(
                "engine-owned manifest payload_sha256 does not match the serialized payload"
            )
        self._validate_compatibility(
            manifest_compatibility=payload.compatibility,
            target=target,
        )
        self._validate_artifact_manifest_entries(
            publish_manifest=publish_manifest,
            required_ranks=target.required_ranks,
            cutoff_token_count=payload.cutoff_token_count,
            tail_valid_tokens=payload.tail_valid_tokens,
            page_size=payload.compatibility.page_size,
        )

    def _validate_local_rank_membership(
        self,
        *,
        local_rank: RankCoord,
        publish_manifest: PublishManifestRecord,
        target: HydrateTargetCompatibility,
    ) -> None:
        if local_rank not in target.required_ranks:
            raise RequestBundleStateError(
                f"local hydrate rank={local_rank.as_key()} is not part of the target required_ranks"
            )
        matching_entries = [
            entry
            for entry in publish_manifest.artifact_manifest.entries
            if entry.rank == local_rank
        ]
        if not matching_entries:
            raise RequestBundleStateError(
                f"publish manifest has no entries for local hydrate rank={local_rank.as_key()}"
            )

    def _validate_compatibility(
        self,
        *,
        manifest_compatibility: PublishCompatibilityEnvelope,
        target: HydrateTargetCompatibility,
    ) -> None:
        if manifest_compatibility.model_fingerprint != target.model_fingerprint:
            raise RequestBundleStateError("hydrate model_fingerprint mismatch")
        if manifest_compatibility.kv_layout_id != target.kv_layout_id:
            raise RequestBundleStateError("hydrate kv_layout_id mismatch")
        if manifest_compatibility.dtype != target.dtype:
            raise RequestBundleStateError("hydrate dtype mismatch")
        if manifest_compatibility.page_size != target.page_size:
            raise RequestBundleStateError("hydrate page_size mismatch")
        if manifest_compatibility.tp_size != target.tp_size:
            raise RequestBundleStateError("hydrate tp_size mismatch")
        if manifest_compatibility.pp_size != target.pp_size:
            raise RequestBundleStateError("hydrate pp_size mismatch")
        if manifest_compatibility.attention_arch != target.attention_arch:
            raise RequestBundleStateError("hydrate attention_arch mismatch")
        if manifest_compatibility.required_ranks != target.required_ranks:
            raise RequestBundleStateError("hydrate required_ranks mismatch")

    def _validate_artifact_manifest_entries(
        self,
        *,
        publish_manifest: PublishManifestRecord,
        required_ranks: tuple[RankCoord, ...],
        cutoff_token_count: int,
        tail_valid_tokens: int,
        page_size: int,
    ) -> None:
        if tail_valid_tokens < 0 or tail_valid_tokens >= page_size:
            raise RequestBundleStateError(
                "tail_valid_tokens must be within one page for hydrate"
            )
        expected_page_count = (cutoff_token_count - tail_valid_tokens) // page_size
        entries_by_rank = {
            rank.as_key(): [
                entry
                for entry in publish_manifest.artifact_manifest.entries
                if entry.rank == rank
            ]
            for rank in required_ranks
        }
        if publish_manifest.artifact_manifest.required_ranks != required_ranks:
            raise RequestBundleStateError(
                "artifact manifest required_ranks do not match the hydrate target"
            )
        for rank in required_ranks:
            rank_entries = sorted(
                entries_by_rank[rank.as_key()],
                key=lambda entry: entry.logical_page_index,
            )
            if len(rank_entries) != expected_page_count:
                raise RequestBundleStateError(
                    "artifact manifest does not have the expected page count for "
                    f"rank={rank.as_key()}: expected={expected_page_count} actual={len(rank_entries)}"
                )
            expected_indices = tuple(range(expected_page_count))
            actual_indices = tuple(entry.logical_page_index for entry in rank_entries)
            if actual_indices != expected_indices:
                raise RequestBundleStateError(
                    "artifact manifest page indices are not contiguous for "
                    f"rank={rank.as_key()}: expected={expected_indices} actual={actual_indices}"
                )

    def _build_rank_work_item(
        self,
        *,
        rank: RankCoord,
        publish_manifest: PublishManifestRecord,
    ) -> HydrateRankWorkItem:
        artifact_entries = tuple(
            entry
            for entry in publish_manifest.artifact_manifest.entries
            if entry.rank == rank
        )
        return HydrateRankWorkItem(
            rank=rank,
            logical_request_id=publish_manifest.engine_owned_manifest.payload.logical_request_id,
            publish_manifest_digest=publish_manifest.publish_manifest_digest,
            artifact_manifest_digest=publish_manifest.artifact_manifest.artifact_manifest_digest,
            cutoff_token_count=publish_manifest.cutoff_token_count,
            tail_valid_tokens=publish_manifest.tail_valid_tokens,
            prompt_token_digest=publish_manifest.prompt_token_digest,
            artifact_entries=artifact_entries,
        )

    def _cleanup_failed_install(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
        install_results: tuple[HydrateRankInstallResult, ...],
        cleanup_rank_install: HydrateRankCleanup | None,
    ) -> None:
        for install_result in install_results:
            if install_result.state != RankInstallState.READY:
                continue
            if cleanup_rank_install is not None:
                with suppress(Exception):
                    cleanup_rank_install(install_result)
            self._prepared_bundle_registry.mark_rank_install(
                logical_request_id=logical_request_id,
                publish_manifest_digest=publish_manifest_digest,
                rank=install_result.rank,
                state=RankInstallState.CLEANED,
                hydrated_page_count=install_result.hydrated_page_count,
                runnable_prefix_tokens=install_result.runnable_prefix_tokens,
                local_install_handle=install_result.local_install_handle,
                last_error="cleaned after group hydrate failure",
            )

    def _build_local_hold_set_id(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
        local_rank: RankCoord,
    ) -> str:
        return self._sha256_hexdigest(
            (
                f"{logical_request_id}:{publish_manifest_digest}:"
                f"{local_rank.tp_rank}:{local_rank.pp_rank}:prepared_hold_set"
            )
        )

    def _sha256_hexdigest(self, payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RequestBundleHydrateAggregator:
    def aggregate(
        self,
        *,
        request: HydrateInstanceOpRequest,
        publish_manifest: PublishManifestRecord,
        target: HydrateTargetCompatibility,
        local_results: Sequence[HydratePreparedResult],
        now_ms: int,
    ) -> HydratePreparedResult:
        if not local_results:
            raise RequestBundleStateError(
                "hydrate aggregation requires at least one local result"
            )
        first_result = local_results[0]
        install_by_rank: dict[tuple[int, int], HydrateRankInstallResult] = {}
        prepared_result_by_rank: dict[tuple[int, int], HydratePreparedResult] = {}
        engine_owned_manifest_sha256 = (
            first_result.prepared_bundle.engine_owned_manifest_sha256
        )
        created_at_ms = first_result.prepared_bundle.created_at_ms
        prepared_at_ms = first_result.prepared_bundle.prepared_at_ms
        for local_result in local_results:
            self._validate_local_result(
                request=request,
                publish_manifest=publish_manifest,
                target=target,
                local_result=local_result,
                expected_engine_owned_manifest_sha256=engine_owned_manifest_sha256,
            )
            local_rank = self._extract_local_rank(local_result.prepared_bundle)
            local_rank_key = local_rank.as_key()
            if local_rank_key in prepared_result_by_rank:
                raise RequestBundleStateError(
                    f"duplicate local hydrate result for rank={local_rank_key}"
                )
            prepared_result_by_rank[local_rank_key] = local_result
            install_by_rank[local_rank_key] = (
                local_result.prepared_bundle.rank_installs[0]
            )
            created_at_ms = min(
                created_at_ms,
                local_result.prepared_bundle.created_at_ms,
            )
            if local_result.prepared_bundle.prepared_at_ms is not None:
                if prepared_at_ms is None:
                    prepared_at_ms = local_result.prepared_bundle.prepared_at_ms
                else:
                    prepared_at_ms = max(
                        prepared_at_ms,
                        local_result.prepared_bundle.prepared_at_ms,
                    )
        missing_rank_keys = [
            rank.as_key()
            for rank in target.required_ranks
            if rank.as_key() not in prepared_result_by_rank
        ]
        if missing_rank_keys:
            raise RequestBundleStateError(
                "missing local hydrate results for "
                + ", ".join(str(rank_key) for rank_key in sorted(missing_rank_keys))
            )
        local_hold_refs = tuple(
            hold_ref
            for local_result in local_results
            if local_result.hold_set is not None
            for hold_ref in local_result.hold_set.refs
        )
        aggregate_hold_set = (
            PreparedHoldSetRecord(
                hold_set_id=self._build_aggregate_hold_set_id(
                    logical_request_id=request.logical_request_id,
                    publish_manifest_digest=request.publish_manifest_digest,
                ),
                state=PreparedHoldSetState.ACTIVE,
                refs=local_hold_refs,
                created_at_ms=int(now_ms),
            )
            if local_hold_refs
            else None
        )
        prepared_bundle = PreparedBundleRecord(
            logical_request_id=request.logical_request_id,
            target_instance_id=target.target_instance_id,
            publish_manifest_digest=publish_manifest.publish_manifest_digest,
            artifact_manifest_digest=publish_manifest.artifact_manifest.artifact_manifest_digest,
            engine_owned_manifest_sha256=engine_owned_manifest_sha256,
            prepared_hold_set_id=(
                aggregate_hold_set.hold_set_id
                if aggregate_hold_set is not None
                else None
            ),
            state=PreparedBundleLifecycleState.PREPARED,
            required_ranks=target.required_ranks,
            rank_installs=tuple(
                install_by_rank[rank.as_key()] for rank in target.required_ranks
            ),
            created_at_ms=int(created_at_ms),
            prepared_at_ms=prepared_at_ms
            if prepared_at_ms is not None
            else int(now_ms),
            prompt_token_digest=publish_manifest.prompt_token_digest,
            cutoff_token_count=publish_manifest.cutoff_token_count,
            tail_valid_tokens=publish_manifest.tail_valid_tokens,
        )
        return HydratePreparedResult(
            prepared_bundle=prepared_bundle,
            hold_set=aggregate_hold_set,
            reused_existing=all(result.reused_existing for result in local_results),
        )

    def _validate_local_result(
        self,
        *,
        request: HydrateInstanceOpRequest,
        publish_manifest: PublishManifestRecord,
        target: HydrateTargetCompatibility,
        local_result: HydratePreparedResult,
        expected_engine_owned_manifest_sha256: str,
    ) -> None:
        prepared_bundle = local_result.prepared_bundle
        if prepared_bundle.logical_request_id != request.logical_request_id:
            raise RequestBundleStateError(
                "local hydrate result logical_request_id does not match aggregate request"
            )
        if prepared_bundle.publish_manifest_digest != request.publish_manifest_digest:
            raise RequestBundleStateError(
                "local hydrate result publish_manifest_digest does not match aggregate request"
            )
        if prepared_bundle.target_instance_id != target.target_instance_id:
            raise RequestBundleStateError(
                "all local hydrate results must belong to the same target instance"
            )
        if (
            prepared_bundle.artifact_manifest_digest
            != publish_manifest.artifact_manifest.artifact_manifest_digest
        ):
            raise RequestBundleStateError(
                "local hydrate result artifact_manifest_digest does not match the publish manifest"
            )
        if (
            prepared_bundle.engine_owned_manifest_sha256
            != expected_engine_owned_manifest_sha256
        ):
            raise RequestBundleStateError(
                "all local hydrate results must share the same engine_owned_manifest_sha256"
            )
        if prepared_bundle.state != PreparedBundleLifecycleState.PREPARED:
            raise RequestBundleStateError(
                f"cannot aggregate local hydrate result from state={prepared_bundle.state}"
            )
        if prepared_bundle.prompt_token_digest != publish_manifest.prompt_token_digest:
            raise RequestBundleStateError(
                "local hydrate result prompt_token_digest does not match the publish manifest"
            )
        if prepared_bundle.cutoff_token_count != publish_manifest.cutoff_token_count:
            raise RequestBundleStateError(
                "local hydrate result cutoff_token_count does not match the publish manifest"
            )
        if prepared_bundle.tail_valid_tokens != publish_manifest.tail_valid_tokens:
            raise RequestBundleStateError(
                "local hydrate result tail_valid_tokens does not match the publish manifest"
            )
        local_rank = self._extract_local_rank(prepared_bundle)
        if local_rank not in target.required_ranks:
            raise RequestBundleStateError(
                f"unexpected local hydrate rank={local_rank.as_key()}"
            )

    def _extract_local_rank(self, prepared_bundle: PreparedBundleRecord) -> RankCoord:
        if len(prepared_bundle.required_ranks) != 1:
            raise RequestBundleStateError(
                "local hydrate result must contain exactly one required rank"
            )
        if len(prepared_bundle.rank_installs) != 1:
            raise RequestBundleStateError(
                "local hydrate result must contain exactly one rank install"
            )
        local_rank = prepared_bundle.required_ranks[0]
        if prepared_bundle.rank_installs[0].rank_coord() != local_rank:
            raise RequestBundleStateError(
                "local hydrate rank install does not match the local required rank"
            )
        return local_rank

    def _build_aggregate_hold_set_id(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
    ) -> str:
        return hashlib.sha256(
            (
                f"{logical_request_id}:{publish_manifest_digest}:"
                "aggregate_prepared_hold_set"
            ).encode("utf-8")
        ).hexdigest()
