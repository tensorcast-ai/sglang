# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.mem_cache.host_shared_slot_state import HostSharedPageSlotToken
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    EvictLocalInstanceOpRequest,
    EvictLocalInstanceOpResult,
    HydrateInstanceOpRequest,
    InstanceOpStatus,
    PublishInstanceOpRequest,
)
from sglang.srt.tensorcast.request_bundle.prepared_bundle_admission import (
    PreparedBundleAdmissionBinder,
)
from sglang.srt.tensorcast.request_bundle.prepared_bundle_evict import (
    PreparedBundleLocalEvictor,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_hydrate import (
    RequestBundleHydrator,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_publish import (
    RequestBundlePublisher,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    PagePublicationRegistry,
    PreparedBundleRegistry,
    PreparedHoldRegistry,
    RequestBundleStateError,
    RequestBundleStateRegistry,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    HydratePreparedResult,
    HydrateRankInstallResult,
    HydrateRankWorkItem,
    HydrateTargetCompatibility,
    OrdinaryGenerateBindingRequest,
    OrdinaryGenerateBindingResult,
    PagePublicationState,
    PreparedBundleBindAction,
    PreparedBundleLifecycleState,
    PreparedBundleRecord,
    PreparedHoldRef,
    PreparedHoldSetRecord,
    PreparedHoldSetState,
    PreparedSlotToken,
    PublishManifestRecord,
    RankCoord,
    RankInstallState,
    SourcePublishClosureResult,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.storage.tensorcast_store.client import (
        TensorcastPageClient,
    )
    from sglang.srt.mem_cache.storage.tensorcast_store.config import (
        TensorcastHiCacheConfig,
    )

logger = logging.getLogger(__name__)

_DEFAULT_RETAINED_SOURCE_REQUEST_TTL_MS = 60_000
_DEFAULT_PUBLISH_CLOSURE_WAIT_TIMEOUT_MS = 30_000
_PUBLISH_CLOSURE_RETAIN_GRACE_MS = 1_000
_PUBLISH_CLOSURE_POLL_INTERVAL_S = 0.02


@dataclass
class _PublishableSourceRequestState:
    logical_request_id: str
    engine_request_id: str
    prompt_token_ids: tuple[int, ...]
    prompt_token_digest: str
    requested_at_ms: int
    visible_prompt_token_count: int = 0
    emitted_decode_token_count: int = 0
    batch_request_count: int = 1
    parallel_sampling_count: int = 1
    session_lineage_depth: int = 0
    retained_until_ms: int | None = None


class RequestBundleManager:
    def __init__(
        self,
        *,
        tensorcast_config: TensorcastHiCacheConfig,
        page_client: TensorcastPageClient,
        mem_pool_host: Any,
        page_size: int,
        dtype: Any,
        tp_size: int,
        pp_size: int,
        local_rank: int,
        pp_rank: int,
        is_mla_backend: bool,
        layout_id: str,
        model_fingerprint: str,
    ) -> None:
        self._tensorcast_config = tensorcast_config
        self._page_client = page_client
        self._mem_pool_host = mem_pool_host
        self._page_size = int(page_size)
        self._dtype = dtype
        self._tp_size = int(tp_size)
        self._pp_size = int(pp_size)
        self._local_rank = int(local_rank)
        self._pp_rank = int(pp_rank)
        self._is_mla_backend = is_mla_backend
        self._layout_id = layout_id
        self._model_fingerprint = model_fingerprint

        self._lock = threading.Lock()
        self._page_publication_registry = PagePublicationRegistry()
        self._request_bundle_registry = RequestBundleStateRegistry(
            page_publication_registry=self._page_publication_registry
        )
        self._prepared_bundle_registry = PreparedBundleRegistry()
        self._prepared_hold_registry = PreparedHoldRegistry()
        self._prepared_bundle_admission_binder = PreparedBundleAdmissionBinder(
            prepared_bundle_registry=self._prepared_bundle_registry
        )
        self._prepared_bundle_evictor = PreparedBundleLocalEvictor(
            request_bundle_registry=self._request_bundle_registry,
            page_publication_registry=self._page_publication_registry,
            prepared_bundle_registry=self._prepared_bundle_registry,
            prepared_hold_registry=self._prepared_hold_registry,
            discard_publishable_source_request=(
                self._discard_publishable_source_request_locked
            ),
            release_prepared_hold_set=self._release_prepared_hold_set_local,
        )
        self._request_bundle_publisher = RequestBundlePublisher(
            request_bundle_registry=self._request_bundle_registry,
            page_publication_registry=self._page_publication_registry,
        )
        self._request_bundle_hydrator = RequestBundleHydrator(
            prepared_bundle_registry=self._prepared_bundle_registry,
            prepared_hold_registry=self._prepared_hold_registry,
        )
        self._instance_ops_instance_id = (
            f"{self._tensorcast_config.namespace}:unconfigured"
        )
        self._instance_ops_coordinator_epoch = "unconfigured"
        self._live_source_requests: dict[str, _PublishableSourceRequestState] = {}
        self._retained_source_requests: dict[str, _PublishableSourceRequestState] = {}

    @property
    def prepared_bundle_registry(self) -> PreparedBundleRegistry:
        return self._prepared_bundle_registry

    @property
    def prepared_hold_registry(self) -> PreparedHoldRegistry:
        return self._prepared_hold_registry

    @property
    def page_publication_registry(self) -> PagePublicationRegistry:
        return self._page_publication_registry

    @property
    def request_bundle_registry(self) -> RequestBundleStateRegistry:
        return self._request_bundle_registry

    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        self._mem_pool_host = mem_pool_host

    def host_indices_for_prepared_hold_refs(
        self,
        refs: Sequence[PreparedHoldRef],
    ) -> torch.Tensor:
        if not refs:
            return torch.empty((0,), dtype=torch.int64)
        index_slices = [
            torch.arange(
                int(ref.slot_token.slot_index) * self._page_size,
                (int(ref.slot_token.slot_index) + 1) * self._page_size,
                dtype=torch.int64,
            )
            for ref in refs
        ]
        return torch.cat(index_slices)

    def release_prepared_hold_refs(self, refs: Sequence[PreparedHoldRef]) -> None:
        host_indices = self.host_indices_for_prepared_hold_refs(refs)
        if host_indices.numel() == 0:
            return
        self._mem_pool_host.free(host_indices)

    def configure_instance_ops_runtime(
        self,
        *,
        instance_id: str,
        coordinator_epoch: str,
    ) -> None:
        with self._lock:
            self._instance_ops_instance_id = str(instance_id).strip() or (
                f"{self._tensorcast_config.namespace}:unconfigured"
            )
            self._instance_ops_coordinator_epoch = (
                str(coordinator_epoch).strip() or "unconfigured"
            )

    def start_live_request_tracking(
        self,
        *,
        logical_request_id: str,
        engine_request_id: str,
        prompt_token_ids: list[int],
        requested_at_ms: int,
        batch_request_count: int = 1,
        parallel_sampling_count: int = 1,
        session_lineage_depth: int = 0,
    ) -> None:
        prompt_token_digest = get_hash_str(prompt_token_ids)
        with self._lock:
            self._discard_publishable_source_request_locked(logical_request_id)
            self._live_source_requests[logical_request_id] = (
                _PublishableSourceRequestState(
                    logical_request_id=logical_request_id,
                    engine_request_id=engine_request_id,
                    prompt_token_ids=tuple(prompt_token_ids),
                    prompt_token_digest=prompt_token_digest,
                    requested_at_ms=int(requested_at_ms),
                    batch_request_count=int(batch_request_count),
                    parallel_sampling_count=int(parallel_sampling_count),
                    session_lineage_depth=int(session_lineage_depth),
                )
            )
            self._request_bundle_registry.upsert_live_request(
                logical_request_id=logical_request_id,
                instance_id=self._instance_ops_instance_id,
                engine_request_id=engine_request_id,
                full_prompt_token_count=len(prompt_token_ids),
                model_fingerprint=self._model_fingerprint,
                kv_layout_id=self._layout_id,
                tp_size=self._tp_size,
                pp_size=self._pp_size,
                required_ranks=self._required_ranks(),
                now_ms=int(requested_at_ms),
            )

    def observe_live_request_progress(
        self,
        *,
        logical_request_id: str,
        visible_prompt_token_count: int,
        emitted_decode_token_count: int,
        now_ms: int,
    ) -> None:
        with self._lock:
            live_request = self._require_live_source_request(logical_request_id)
            clamped_visible_prompt_token_count = min(
                int(visible_prompt_token_count),
                len(live_request.prompt_token_ids),
            )
            live_request.visible_prompt_token_count = max(
                clamped_visible_prompt_token_count,
                live_request.visible_prompt_token_count,
            )
            live_request.emitted_decode_token_count = max(
                int(emitted_decode_token_count),
                live_request.emitted_decode_token_count,
            )
            self._sync_live_request_page_state(
                live_request=live_request,
                now_ms=int(now_ms),
                refresh_from_storage=False,
            )

    def cleanup_live_request_tracking(
        self,
        *,
        logical_request_id: str,
        now_ms: int,
    ) -> None:
        with self._lock:
            live_request = self._live_source_requests.pop(logical_request_id, None)
            request_state = self._request_bundle_registry.get(logical_request_id)
            if live_request is None and request_state is None:
                return
            retained_until_ms = int(now_ms) + _DEFAULT_RETAINED_SOURCE_REQUEST_TTL_MS
            if live_request is not None:
                self._retained_source_requests[logical_request_id] = (
                    live_request.__class__(
                        logical_request_id=live_request.logical_request_id,
                        engine_request_id=live_request.engine_request_id,
                        prompt_token_ids=live_request.prompt_token_ids,
                        prompt_token_digest=live_request.prompt_token_digest,
                        requested_at_ms=live_request.requested_at_ms,
                        visible_prompt_token_count=live_request.visible_prompt_token_count,
                        emitted_decode_token_count=live_request.emitted_decode_token_count,
                        batch_request_count=live_request.batch_request_count,
                        parallel_sampling_count=live_request.parallel_sampling_count,
                        session_lineage_depth=live_request.session_lineage_depth,
                        retained_until_ms=retained_until_ms,
                    )
                )
            if request_state is None:
                return
            with suppress(RequestBundleStateError):
                self._request_bundle_registry.mark_source_retained(
                    logical_request_id=logical_request_id,
                    retained_until_ms=retained_until_ms,
                    now_ms=int(now_ms),
                )

    def publish_live_request(
        self,
        *,
        logical_request_id: str,
        publish_op_id: str,
        requested_at_ms: int,
        requested_cutoff_token_count: int | None = None,
        timeout_ms: int | None = None,
        ttl_ms: int = 60_000,
    ) -> SourcePublishClosureResult:
        _, resolved_cutoff_token_count = self._await_publish_page_closure(
            logical_request_id=logical_request_id,
            requested_cutoff_token_count=requested_cutoff_token_count,
            requested_at_ms=int(requested_at_ms),
            wait_timeout_ms=(
                int(timeout_ms)
                if timeout_ms is not None
                else _DEFAULT_PUBLISH_CLOSURE_WAIT_TIMEOUT_MS
            ),
        )
        with self._lock:
            live_request = self._require_publishable_source_request(
                logical_request_id=logical_request_id,
                now_ms=int(requested_at_ms),
            )
            self._sync_live_request_page_state(
                live_request=live_request,
                now_ms=int(requested_at_ms),
                refresh_from_storage=True,
            )
            return self._request_bundle_publisher.publish(
                request=self._build_publish_request(
                    live_request=live_request,
                    publish_op_id=publish_op_id,
                    requested_at_ms=int(requested_at_ms),
                    requested_cutoff_token_count=resolved_cutoff_token_count,
                    timeout_ms=None if timeout_ms is None else int(timeout_ms),
                    ttl_ms=int(ttl_ms),
                ),
                now_ms=int(requested_at_ms),
            )

    def build_live_request_publish_request(
        self,
        *,
        logical_request_id: str,
        publish_op_id: str,
        requested_at_ms: int,
        requested_cutoff_token_count: int | None = None,
        timeout_ms: int | None = None,
        ttl_ms: int = 60_000,
    ) -> PublishInstanceOpRequest:
        with self._lock:
            live_request = self._require_publishable_source_request(
                logical_request_id=logical_request_id,
                now_ms=int(requested_at_ms),
            )
            resolved_cutoff_token_count = self._resolve_publish_cutoff_token_count(
                live_request=live_request,
                requested_cutoff_token_count=requested_cutoff_token_count,
            )
            return self._build_publish_request(
                live_request=live_request,
                publish_op_id=publish_op_id,
                requested_at_ms=int(requested_at_ms),
                requested_cutoff_token_count=resolved_cutoff_token_count,
                timeout_ms=None if timeout_ms is None else int(timeout_ms),
                ttl_ms=int(ttl_ms),
            )

    def instance_publish_local(
        self,
        *,
        request: PublishInstanceOpRequest,
    ) -> SourcePublishClosureResult:
        _, resolved_cutoff_token_count = self._await_publish_page_closure(
            logical_request_id=request.logical_request_id,
            requested_cutoff_token_count=request.requested_cutoff_token_count,
            requested_at_ms=int(request.requested_at_ms),
            wait_timeout_ms=(
                int(request.timeout_ms)
                if request.timeout_ms is not None
                else _DEFAULT_PUBLISH_CLOSURE_WAIT_TIMEOUT_MS
            ),
        )
        with self._lock:
            live_request = self._require_publishable_source_request(
                logical_request_id=request.logical_request_id,
                now_ms=int(request.requested_at_ms),
            )
            self._sync_live_request_page_state(
                live_request=live_request,
                now_ms=int(request.requested_at_ms),
                refresh_from_storage=True,
            )
            return self._request_bundle_publisher.publish(
                request=request.model_copy(
                    update={
                        "requested_cutoff_token_count": resolved_cutoff_token_count
                    }
                ),
                now_ms=int(request.requested_at_ms),
            )

    def instance_hydrate_local(
        self,
        *,
        request: HydrateInstanceOpRequest,
        publish_manifest: PublishManifestRecord,
    ) -> HydratePreparedResult:
        with self._lock:
            result = self._request_bundle_hydrator.hydrate(
                request=request,
                publish_manifest=publish_manifest,
                target=self.instance_hydrate_target(
                    required_ranks=publish_manifest.engine_owned_manifest.payload.compatibility.required_ranks
                ),
                local_rank=self._current_rank(),
                install_rank=self._install_prepared_bundle_local,
                now_ms=int(request.requested_at_ms),
                live_request_exists=(
                    request.logical_request_id in self._live_source_requests
                ),
            )
            local_records = self._prepared_bundle_registry.list_request_records(
                logical_request_id=request.logical_request_id,
                include_evicted=True,
            )
            logger.debug(
                "Tensorcast local hydrate installed logical_request_id=%s manifest=%s local_rank=%s local_prepared_records=%s states=%s",
                request.logical_request_id,
                request.publish_manifest_digest,
                self._current_rank().as_key(),
                len(local_records),
                [
                    f"{record.publish_manifest_digest}:{record.state.value}"
                    for record in local_records
                ],
            )
            return result

    def instance_evict_local(
        self,
        *,
        request: EvictLocalInstanceOpRequest,
    ) -> EvictLocalInstanceOpResult:
        with self._lock:
            return self._prepared_bundle_evictor.evict(request=request)

    def bind_prepared_bundle_for_generate(
        self,
        *,
        logical_request_id: str,
        scheduler_rid: str,
        prompt_token_ids: list[int],
        requested_at_ms: int,
    ) -> OrdinaryGenerateBindingResult:
        prompt_token_digest = get_hash_str(prompt_token_ids)
        binding_request = OrdinaryGenerateBindingRequest(
            logical_request_id=logical_request_id,
            scheduler_rid=scheduler_rid,
            prompt_token_digest=prompt_token_digest,
            cutoff_token_count=len(prompt_token_ids),
            requested_at_ms=int(requested_at_ms),
        )
        with self._lock:
            existing_records = self._prepared_bundle_registry.list_request_records(
                logical_request_id=logical_request_id,
                include_evicted=True,
            )
            bind_result = self._prepared_bundle_admission_binder.bind(
                request=binding_request,
                attach_prepared_bundle=self._attach_prepared_bundle,
            )
        logger.debug(
            "Tensorcast prepared-bundle bind logical_request_id=%s scheduler_rid=%s records=%s record_states=%s action=%s reason=%s",
            logical_request_id,
            scheduler_rid,
            len(existing_records),
            [
                (
                    record.publish_manifest_digest,
                    record.state.value,
                    record.stale,
                    record.tainted,
                )
                for record in existing_records
            ],
            bind_result.action.value,
            bind_result.reason,
        )
        self._log_generate_binding_result(bind_result)
        return bind_result

    def consume_attached_prepared_bundle_for_generate(
        self,
        *,
        logical_request_id: str,
        publish_manifest_digest: str,
        prompt_token_ids: Sequence[int],
        requested_at_ms: int,
        install_prepared_bundle: Callable[
            [PreparedBundleRecord, PreparedHoldSetRecord | None, tuple[int, ...]], None
        ],
    ) -> PreparedBundleRecord:
        with self._lock:
            record = self._prepared_bundle_registry.get(
                logical_request_id=logical_request_id,
                publish_manifest_digest=publish_manifest_digest,
            )
            if record is None:
                raise RequestBundleStateError(
                    "attached prepared bundle disappeared before ordinary generate consume"
                )
            if record.state != PreparedBundleLifecycleState.ATTACHED:
                raise RequestBundleStateError(
                    f"cannot consume prepared bundle from state={record.state}"
                )
            cutoff_token_count = int(record.cutoff_token_count or 0)
            if cutoff_token_count > len(prompt_token_ids):
                raise RequestBundleStateError(
                    "prepared bundle cutoff exceeds the incoming prompt length"
                )
            tail_valid_tokens = int(record.tail_valid_tokens)
            if tail_valid_tokens < 0 or tail_valid_tokens > cutoff_token_count:
                raise RequestBundleStateError(
                    "prepared bundle tail_valid_tokens is not valid for the cutoff"
                )
            page_aligned_cutoff_token_count = cutoff_token_count - tail_valid_tokens
            if page_aligned_cutoff_token_count % self._page_size != 0:
                raise RequestBundleStateError(
                    "prepared bundle reusable prefix must remain page aligned for ordinary consume"
                )
            hold_set = (
                self._prepared_hold_registry.get(record.prepared_hold_set_id)
                if record.prepared_hold_set_id is not None
                else None
            )
            if record.prepared_hold_set_id is not None:
                if hold_set is None:
                    raise RequestBundleStateError(
                        "prepared bundle references a missing prepared hold set"
                    )
                if hold_set.state != PreparedHoldSetState.ACTIVE:
                    raise RequestBundleStateError(
                        f"prepared hold set is not active: state={hold_set.state}"
                    )
                expected_page_count = page_aligned_cutoff_token_count // self._page_size
                if len(hold_set.refs) != expected_page_count:
                    raise RequestBundleStateError(
                        "prepared hold refs do not match the page-granular reusable prefix"
                    )
            cutoff_prompt_token_ids = tuple(
                int(token_id)
                for token_id in prompt_token_ids[:page_aligned_cutoff_token_count]
            )
            install_prepared_bundle(record, hold_set, cutoff_prompt_token_ids)
            if record.prepared_hold_set_id is not None:
                self._prepared_hold_registry.release_hold_set(
                    hold_set_id=record.prepared_hold_set_id,
                    now_ms=int(requested_at_ms),
                )
            return self._prepared_bundle_registry.mark_consumed(
                logical_request_id=record.logical_request_id,
                publish_manifest_digest=record.publish_manifest_digest,
                clear_prepared_hold_set_id=record.prepared_hold_set_id is not None,
            )

    def cleanup_bound_prepared_bundle(
        self,
        *,
        publish_manifest_digest: str,
        requested_at_ms: int,
    ) -> None:
        evict_result = self.instance_evict_local(
            request=EvictLocalInstanceOpRequest(
                publish_manifest_digest=publish_manifest_digest,
                requested_at_ms=int(requested_at_ms),
            )
        )
        if evict_result.status != InstanceOpStatus.SUCCESS:
            logger.warning(
                "Tensorcast prepared-bundle cleanup failed manifest=%s error=%s",
                publish_manifest_digest,
                evict_result.error_message,
            )

    def mark_pages_inflight(
        self,
        *,
        page_hashes: Sequence[str],
        now_ms: int,
    ) -> None:
        with self._lock:
            self._update_live_page_publication_state_for_hashes(
                page_hashes=list(page_hashes),
                publication_state=PagePublicationState.INFLIGHT,
                now_ms=int(now_ms),
            )

    def mark_pages_result(
        self,
        *,
        succeeded_page_hashes: Sequence[str],
        failed_page_hashes: Sequence[str],
        now_ms: int,
        failure_reason: str | None = None,
    ) -> None:
        with self._lock:
            if succeeded_page_hashes:
                self._update_live_page_publication_state_for_hashes(
                    page_hashes=list(succeeded_page_hashes),
                    publication_state=PagePublicationState.READY,
                    now_ms=int(now_ms),
                )
            if failed_page_hashes:
                self._update_live_page_publication_state_for_hashes(
                    page_hashes=list(failed_page_hashes),
                    publication_state=PagePublicationState.ABSENT,
                    now_ms=int(now_ms),
                    last_error=failure_reason,
                )

    def instance_hydrate_target(
        self,
        *,
        required_ranks: tuple[RankCoord, ...],
    ) -> HydrateTargetCompatibility:
        return HydrateTargetCompatibility(
            target_instance_id=self._instance_ops_instance_id,
            model_fingerprint=self._model_fingerprint,
            kv_layout_id=self._layout_id,
            dtype=str(self._dtype),
            page_size=self._page_size,
            tp_size=self._tp_size,
            pp_size=self._pp_size,
            attention_arch="mla" if self._is_mla_backend else "mha",
            required_ranks=required_ranks,
        )

    def _release_prepared_hold_set_local(
        self,
        hold_set: PreparedHoldSetRecord,
    ) -> None:
        self.release_prepared_hold_refs(hold_set.refs)

    def _page_start_indices(
        self,
        host_indices: torch.Tensor,
        expected_pages: int,
    ) -> list[int]:
        if host_indices.numel() != expected_pages * self._page_size:
            raise ValueError(
                "host_indices length must equal number of pages multiplied by page_size"
            )
        return [
            int(host_indices[index * self._page_size].item())
            for index in range(expected_pages)
        ]

    def _host_page_views(self, page_starts: list[int]) -> list[torch.Tensor]:
        return [
            self._mem_pool_host.get_data_page(index, flat=True) for index in page_starts
        ]

    def _allocator_backed_direct_get_enabled(self) -> bool:
        return (
            self._tensorcast_config.host_allocator_enabled
            and self._mem_pool_host.host_region_binding is not None
        )

    def _success_prefix_count(self, success_mask: tuple[bool, ...]) -> int:
        prefix_success = 0
        for success in success_mask:
            if not success:
                break
            prefix_success += 1
        return prefix_success

    def _install_prepared_bundle_local(
        self,
        work_item: HydrateRankWorkItem,
    ) -> HydrateRankInstallResult:
        runnable_prefix_tokens = (
            int(work_item.cutoff_token_count) - int(work_item.tail_valid_tokens)
        )
        if runnable_prefix_tokens < 0:
            raise RequestBundleStateError(
                "hydrate work item tail_valid_tokens exceeds cutoff_token_count"
            )
        artifact_entries = tuple(
            sorted(
                work_item.artifact_entries,
                key=lambda entry: entry.logical_page_index,
            )
        )
        if not artifact_entries:
            return HydrateRankInstallResult(
                rank=work_item.rank,
                state=RankInstallState.READY,
                hydrated_page_count=0,
                runnable_prefix_tokens=runnable_prefix_tokens,
                local_install_handle=(
                    "prepared::"
                    f"{work_item.logical_request_id}::"
                    f"{work_item.publish_manifest_digest}::"
                    f"{work_item.rank.tp_rank}:{work_item.rank.pp_rank}"
                ),
            )

        expected_page_indices = tuple(range(len(artifact_entries)))
        actual_page_indices = tuple(
            entry.logical_page_index for entry in artifact_entries
        )
        if actual_page_indices != expected_page_indices:
            raise RequestBundleStateError(
                "hydrate work item pages must be contiguous and start at logical_page_index=0"
            )

        needed_tokens = len(artifact_entries) * self._page_size
        host_indices = self._mem_pool_host.alloc(needed_tokens)
        if host_indices is None:
            raise RequestBundleStateError(
                "insufficient host KV capacity to install the hydrated prepared bundle"
            )
        slot_tokens: list[HostSharedPageSlotToken] = []
        try:
            page_starts = self._page_start_indices(host_indices, len(artifact_entries))
            logical_keys = [entry.page_hash for entry in artifact_entries]
            slot_tokens = list(
                self._mem_pool_host.reserve_page_slots(page_starts, logical_keys)
            )
            self._mem_pool_host.mark_page_get_inflight(slot_tokens)
            targets = self._host_page_views(page_starts)
            target_region_binding = (
                self._mem_pool_host.host_region_binding
                if self._allocator_backed_direct_get_enabled()
                else None
            )
            result = self._page_client.batch_get_into(
                logical_keys,
                targets,
                slot_tokens=slot_tokens,
                target_region_binding=target_region_binding,
            )
        except Exception:
            with suppress(Exception):
                if slot_tokens:
                    self._mem_pool_host.fail_page_get(slot_tokens)
            with suppress(Exception):
                self._mem_pool_host.free(host_indices)
            raise

        prefix_success = self._success_prefix_count(result.success_mask)
        if prefix_success > 0:
            self._mem_pool_host.commit_page_get_success(
                slot_tokens[:prefix_success],
                logical_keys[:prefix_success],
            )
        if prefix_success < len(slot_tokens):
            self._mem_pool_host.fail_page_get(slot_tokens[prefix_success:])
            self._mem_pool_host.free(host_indices)
            raise RequestBundleStateError(
                "hydrate failed to fetch the full prepared prefix into local host pages"
            )

        hold_refs = tuple(
            PreparedHoldRef(
                logical_page_index=entry.logical_page_index,
                page_hash=entry.page_hash,
                slot_token=PreparedSlotToken(
                    slot_index=slot_tokens[offset].slot_index,
                    slot_generation=slot_tokens[offset].slot_generation,
                ),
                artifact_id=entry.artifact_id,
            )
            for offset, entry in enumerate(artifact_entries)
        )
        return HydrateRankInstallResult(
            rank=work_item.rank,
            state=RankInstallState.READY,
            hydrated_page_count=len(artifact_entries),
            runnable_prefix_tokens=runnable_prefix_tokens,
            local_install_handle=(
                "prepared::"
                f"{work_item.logical_request_id}::"
                f"{work_item.publish_manifest_digest}::"
                f"{work_item.rank.tp_rank}:{work_item.rank.pp_rank}"
            ),
            hold_refs=hold_refs,
        )

    def _current_rank(self) -> RankCoord:
        return RankCoord(tp_rank=self._local_rank, pp_rank=self._pp_rank)

    def _required_ranks(self) -> tuple[RankCoord, ...]:
        return (self._current_rank(),)

    def _require_live_source_request(
        self, logical_request_id: str
    ) -> _PublishableSourceRequestState:
        live_request = self._live_source_requests.get(logical_request_id)
        if live_request is None:
            raise RequestBundleStateError(
                f"unknown live logical_request_id={logical_request_id}"
            )
        return live_request

    def _require_publishable_source_request(
        self,
        *,
        logical_request_id: str,
        now_ms: int,
    ) -> _PublishableSourceRequestState:
        live_request = self._live_source_requests.get(logical_request_id)
        if live_request is not None:
            return live_request
        retained_request = self._retained_source_requests.get(logical_request_id)
        if retained_request is None:
            raise RequestBundleStateError(
                f"unknown publishable logical_request_id={logical_request_id}"
            )
        if (
            retained_request.retained_until_ms is not None
            and int(now_ms) >= retained_request.retained_until_ms
        ):
            self._clear_source_request_tracking_locked(
                logical_request_id=logical_request_id,
                now_ms=int(now_ms),
            )
            raise RequestBundleStateError(
                f"retained publish window expired for logical_request_id={logical_request_id}"
            )
        return retained_request

    def _discard_publishable_source_request_locked(
        self, logical_request_id: str
    ) -> None:
        self._live_source_requests.pop(logical_request_id, None)
        self._retained_source_requests.pop(logical_request_id, None)

    def _pin_retained_source_request_for_publish_locked(
        self,
        *,
        logical_request_id: str,
        live_request: _PublishableSourceRequestState,
        requested_at_ms: int,
        wait_timeout_ms: int,
        now_ms: int,
    ) -> None:
        if live_request.retained_until_ms is None:
            return
        pinned_retained_until_ms = max(
            int(live_request.retained_until_ms),
            int(requested_at_ms)
            + int(wait_timeout_ms)
            + _PUBLISH_CLOSURE_RETAIN_GRACE_MS,
        )
        if pinned_retained_until_ms == int(live_request.retained_until_ms):
            return
        live_request.retained_until_ms = pinned_retained_until_ms
        if self._request_bundle_registry.get(logical_request_id) is None:
            return
        with suppress(RequestBundleStateError):
            self._request_bundle_registry.mark_source_retained(
                logical_request_id=logical_request_id,
                retained_until_ms=pinned_retained_until_ms,
                now_ms=int(now_ms),
            )

    def _clear_source_request_tracking_locked(
        self,
        *,
        logical_request_id: str,
        now_ms: int,
    ) -> None:
        self._discard_publishable_source_request_locked(logical_request_id)
        if self._request_bundle_registry.get(logical_request_id) is not None:
            with suppress(RequestBundleStateError):
                self._request_bundle_registry.mark_cleaned(
                    logical_request_id=logical_request_id,
                    now_ms=int(now_ms),
                )
            self._request_bundle_registry.clear(logical_request_id)
        self._page_publication_registry.clear_request(logical_request_id)

    def _resolve_publish_cutoff_token_count(
        self,
        *,
        live_request: _PublishableSourceRequestState,
        requested_cutoff_token_count: int | None,
    ) -> int:
        full_prompt_token_count = len(live_request.prompt_token_ids)
        if requested_cutoff_token_count is None:
            return full_prompt_token_count
        resolved_cutoff_token_count = int(requested_cutoff_token_count)
        if resolved_cutoff_token_count > full_prompt_token_count:
            raise RequestBundleStateError(
                "requested_cutoff_token_count exceeds the full prompt token count"
            )
        return resolved_cutoff_token_count

    def _publish_page_closure_ready_locked(
        self,
        *,
        live_request: _PublishableSourceRequestState,
        cutoff_token_count: int,
        now_ms: int,
    ) -> bool:
        self._sync_live_request_page_state(
            live_request=live_request,
            now_ms=int(now_ms),
            refresh_from_storage=True,
        )
        if live_request.visible_prompt_token_count < int(cutoff_token_count):
            return False
        page_aligned_token_count = (
            int(cutoff_token_count) // self._page_size
        ) * self._page_size
        expected_page_count = page_aligned_token_count // self._page_size
        if expected_page_count == 0:
            return True
        current_rank = self._current_rank()
        pages = self._page_publication_registry.snapshot_rank(
            logical_request_id=live_request.logical_request_id,
            rank=current_rank,
            max_page_index=expected_page_count - 1,
        )
        if len(pages) != expected_page_count:
            return False
        for expected_index, page in enumerate(pages):
            if page.logical_page_index != expected_index:
                return False
            if page.publication_state == PagePublicationState.FAILED:
                raise RequestBundleStateError(
                    "page publication failed while waiting for full prompt closure: "
                    f"rank={current_rank.as_key()} page={page.logical_page_index} "
                    f"error={page.last_error or 'unknown'}"
                )
            if page.publication_state != PagePublicationState.READY:
                if not page.host_resident:
                    raise RequestBundleStateError(
                        "page publication cannot close full prompt boundary because "
                        f"rank={current_rank.as_key()} page={page.logical_page_index} "
                        "is not host resident"
                    )
                return False
            if not page.artifact_id:
                return False
        return True

    def _await_publish_page_closure(
        self,
        *,
        logical_request_id: str,
        requested_cutoff_token_count: int | None,
        requested_at_ms: int,
        wait_timeout_ms: int,
    ) -> tuple[_PublishableSourceRequestState, int]:
        started_monotonic = time.monotonic()
        deadline_monotonic = started_monotonic + (float(wait_timeout_ms) / 1000.0)
        while True:
            now_ms = int(requested_at_ms) + int(
                (time.monotonic() - started_monotonic) * 1000.0
            )
            with self._lock:
                live_request = self._require_publishable_source_request(
                    logical_request_id=logical_request_id,
                    now_ms=now_ms,
                )
                self._pin_retained_source_request_for_publish_locked(
                    logical_request_id=logical_request_id,
                    live_request=live_request,
                    requested_at_ms=int(requested_at_ms),
                    wait_timeout_ms=int(wait_timeout_ms),
                    now_ms=now_ms,
                )
                cutoff_token_count = self._resolve_publish_cutoff_token_count(
                    live_request=live_request,
                    requested_cutoff_token_count=requested_cutoff_token_count,
                )
                if self._publish_page_closure_ready_locked(
                    live_request=live_request,
                    cutoff_token_count=cutoff_token_count,
                    now_ms=now_ms,
                ):
                    return live_request, cutoff_token_count
                latest_materialized_token_count = int(
                    live_request.visible_prompt_token_count
                )
            if time.monotonic() >= deadline_monotonic:
                raise RequestBundleStateError(
                    "timed out waiting for full prompt page-granular closure: "
                    f"logical_request_id={logical_request_id} "
                    f"cutoff_token_count={cutoff_token_count} "
                    f"materialized_prompt_token_count={latest_materialized_token_count}"
                )
            time.sleep(_PUBLISH_CLOSURE_POLL_INTERVAL_S)

    def _stable_visible_page_hashes(
        self,
        *,
        prompt_token_ids: tuple[int, ...],
        visible_prompt_token_count: int,
    ) -> tuple[str, ...]:
        resolved_visible_prompt_token_count = min(
            int(visible_prompt_token_count),
            len(prompt_token_ids),
        )
        full_page_count = resolved_visible_prompt_token_count // self._page_size
        stable_token_count = full_page_count * self._page_size
        prompt_is_fully_visible = resolved_visible_prompt_token_count == len(
            prompt_token_ids
        )
        if prompt_is_fully_visible and len(prompt_token_ids) % self._page_size != 0:
            stable_token_count = len(prompt_token_ids)
        hash_values: list[str] = []
        parent_hash: str | None = None
        for page_start in range(0, stable_token_count, self._page_size):
            page_tokens = prompt_token_ids[page_start : page_start + self._page_size]
            if not page_tokens:
                continue
            parent_hash = get_hash_str(list(page_tokens), prior_hash=parent_hash)
            hash_values.append(parent_hash)
        return tuple(hash_values)

    def _refresh_live_request_snapshot(
        self,
        *,
        logical_request_id: str,
        now_ms: int,
    ) -> None:
        live_request = self._live_source_requests.get(logical_request_id)
        if live_request is None:
            return
        stable_page_hashes = self._stable_visible_page_hashes(
            prompt_token_ids=live_request.prompt_token_ids,
            visible_prompt_token_count=live_request.visible_prompt_token_count,
        )
        current_rank = self._current_rank()
        ordered_pages = self._page_publication_registry.snapshot_rank(
            logical_request_id=logical_request_id,
            rank=current_rank,
            max_page_index=len(stable_page_hashes) - 1,
        )
        self._request_bundle_registry.update_rank_frontier(
            logical_request_id=logical_request_id,
            rank=current_rank,
            latest_token_count=live_request.visible_prompt_token_count,
            latest_last_page_index=len(stable_page_hashes) - 1,
            now_ms=int(now_ms),
            ordered_pages=ordered_pages,
        )

    def _update_live_page_publication_state_for_hashes(
        self,
        *,
        page_hashes: list[str],
        publication_state: PagePublicationState,
        now_ms: int,
        last_error: str | None = None,
    ) -> None:
        current_rank = self._current_rank()
        matching_records = self._page_publication_registry.matching_page_hashes(
            page_hashes=page_hashes,
            rank=current_rank,
        )
        affected_request_ids: set[str] = set()
        for record in matching_records:
            next_state = publication_state
            if publication_state == PagePublicationState.INFLIGHT:
                if record.publication_state == PagePublicationState.READY:
                    continue
                artifact_id = None
                resolved_last_error = None
            elif publication_state == PagePublicationState.READY:
                artifact_id = self._page_client.artifact_id_for(record.page_hash)
                resolved_last_error = None
            else:
                if record.publication_state == PagePublicationState.READY:
                    continue
                next_state = PagePublicationState.ABSENT
                artifact_id = None
                resolved_last_error = last_error
            self._page_publication_registry.set_page_state(
                logical_request_id=record.logical_request_id,
                rank=record.rank,
                logical_page_index=record.logical_page_index,
                page_hash=record.page_hash,
                publication_state=next_state,
                artifact_id=artifact_id,
                host_resident=record.host_resident,
                last_error=resolved_last_error,
                updated_at_ms=int(now_ms),
            )
            affected_request_ids.add(record.logical_request_id)
        for logical_request_id in affected_request_ids:
            self._refresh_live_request_snapshot(
                logical_request_id=logical_request_id,
                now_ms=int(now_ms),
            )

    def _sync_live_request_page_state(
        self,
        *,
        live_request: _PublishableSourceRequestState,
        now_ms: int,
        refresh_from_storage: bool,
    ) -> None:
        page_hashes = self._stable_visible_page_hashes(
            prompt_token_ids=live_request.prompt_token_ids,
            visible_prompt_token_count=live_request.visible_prompt_token_count,
        )
        ready_prefix_count = 0
        if refresh_from_storage and page_hashes:
            existence = self._page_client.batch_exists(list(page_hashes))
            for exists in existence.existence_mask:
                if not exists:
                    break
                ready_prefix_count += 1
        current_rank = self._current_rank()
        existing_page_records = self._page_publication_registry.snapshot_rank(
            logical_request_id=live_request.logical_request_id,
            rank=current_rank,
        )
        existing_pages_by_index = {
            page.logical_page_index: page for page in existing_page_records
        }
        updated_pages = []
        for logical_page_index, page_hash in enumerate(page_hashes):
            existing_page = existing_pages_by_index.get(logical_page_index)
            page_is_ready = logical_page_index < ready_prefix_count or (
                existing_page is not None
                and existing_page.publication_state == PagePublicationState.READY
            )
            preserved_state = (
                existing_page.publication_state if existing_page is not None else None
            )
            resolved_state = PagePublicationState.ABSENT
            if page_is_ready:
                resolved_state = PagePublicationState.READY
            elif preserved_state in {
                PagePublicationState.INFLIGHT,
                PagePublicationState.FAILED,
            }:
                resolved_state = preserved_state
            record = self._page_publication_registry.set_page_state(
                logical_request_id=live_request.logical_request_id,
                rank=current_rank,
                logical_page_index=logical_page_index,
                page_hash=page_hash,
                publication_state=resolved_state,
                artifact_id=(
                    self._page_client.artifact_id_for(page_hash)
                    if resolved_state == PagePublicationState.READY
                    else None
                ),
                host_resident=True,
                last_error=(
                    None
                    if resolved_state == PagePublicationState.READY
                    else existing_page.last_error
                    if existing_page is not None
                    else None
                ),
                updated_at_ms=int(now_ms),
            )
            updated_pages.append(record.to_closure_entry())
        self._request_bundle_registry.update_rank_frontier(
            logical_request_id=live_request.logical_request_id,
            rank=current_rank,
            latest_token_count=live_request.visible_prompt_token_count,
            latest_last_page_index=len(page_hashes) - 1,
            now_ms=int(now_ms),
            ordered_pages=tuple(updated_pages),
        )

    def _build_publish_request(
        self,
        *,
        live_request: _PublishableSourceRequestState,
        publish_op_id: str,
        requested_at_ms: int,
        requested_cutoff_token_count: int | None,
        timeout_ms: int | None,
        ttl_ms: int,
    ) -> PublishInstanceOpRequest:
        return PublishInstanceOpRequest(
            logical_request_id=live_request.logical_request_id,
            engine_request_id=live_request.engine_request_id,
            publish_op_id=publish_op_id,
            requested_cutoff_token_count=requested_cutoff_token_count,
            timeout_ms=timeout_ms,
            ttl_ms=ttl_ms,
            prompt_token_digest=live_request.prompt_token_digest,
            transfer_mode="prefill_closed_prompt_reuse",
            batch_request_count=live_request.batch_request_count,
            parallel_sampling_count=live_request.parallel_sampling_count,
            session_lineage_depth=live_request.session_lineage_depth,
            emitted_decode_token_count=live_request.emitted_decode_token_count,
            attention_arch="mla" if self._is_mla_backend else "mha",
            dtype=str(self._dtype),
            page_size=self._page_size,
            requested_at_ms=int(requested_at_ms),
        )

    def _attach_prepared_bundle(self, record: PreparedBundleRecord) -> str:
        if record.state != PreparedBundleLifecycleState.CLAIMED:
            raise RuntimeError(
                f"cannot attach prepared bundle from state={record.state}"
            )
        if record.prepared_hold_set_id is not None:
            hold_set = self._prepared_hold_registry.get(record.prepared_hold_set_id)
            if hold_set is None:
                raise RuntimeError(
                    "prepared bundle references a missing prepared hold set"
                )
            if hold_set.state != PreparedHoldSetState.ACTIVE:
                raise RuntimeError(
                    f"prepared hold set is not active: state={hold_set.state}"
                )
        return (
            f"prepared::{record.logical_request_id}::{record.publish_manifest_digest}"
        )

    def _log_generate_binding_result(
        self, bind_result: OrdinaryGenerateBindingResult
    ) -> None:
        if bind_result.action == PreparedBundleBindAction.ATTACHED:
            record = bind_result.record
            if record is None:
                return
            logger.debug(
                "Tensorcast prepared-bundle attached logical_request_id=%s manifest=%s prepared_bundle_key=%s",
                record.logical_request_id,
                record.publish_manifest_digest,
                bind_result.prepared_bundle_key,
            )
            return
        if bind_result.action == PreparedBundleBindAction.FAIL_CLOSED:
            logger.warning(
                "Tensorcast prepared-bundle fail-closed during ordinary generate admission: %s",
                bind_result.reason,
            )
            return
        if bind_result.record is not None:
            logger.warning(
                "Tensorcast prepared-bundle falling back to normal generate path: %s logical_request_id=%s manifest=%s stale=%s tainted=%s",
                bind_result.reason,
                bind_result.record.logical_request_id,
                bind_result.record.publish_manifest_digest,
                bind_result.record.stale,
                bind_result.record.tainted,
            )
            return
        logger.debug(
            "Tensorcast prepared-bundle found no prepared bundle for ordinary generate admission: %s",
            bind_result.reason,
        )
