# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

from collections.abc import Callable

from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    PagePublicationRegistry,
    PreparedBundleRegistry,
    PreparedHoldRegistry,
    RequestBundleStateRegistry,
    RequestBundleStateError,
)
from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    EvictLocalInstanceOpRequest,
    EvictLocalInstanceOpResult,
    InstanceOpStatus,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    PreparedBundleLifecycleState,
    PreparedHoldSetRecord,
    PreparedHoldSetState,
    RequestBundleLifecycleState,
)


class PreparedBundleLocalEvictor:
    def __init__(
        self,
        *,
        request_bundle_registry: RequestBundleStateRegistry,
        page_publication_registry: PagePublicationRegistry,
        prepared_bundle_registry: PreparedBundleRegistry,
        prepared_hold_registry: PreparedHoldRegistry,
        discard_publishable_source_request: Callable[[str], None] | None = None,
        release_prepared_hold_set: Callable[[PreparedHoldSetRecord], None]
        | None = None,
    ) -> None:
        self._request_bundle_registry = request_bundle_registry
        self._page_publication_registry = page_publication_registry
        self._prepared_bundle_registry = prepared_bundle_registry
        self._prepared_hold_registry = prepared_hold_registry
        self._discard_publishable_source_request = discard_publishable_source_request
        self._release_prepared_hold_set = release_prepared_hold_set

    def evict(
        self,
        *,
        request: EvictLocalInstanceOpRequest,
    ) -> EvictLocalInstanceOpResult:
        try:
            evicted_bundle_count = 0
            evicted_hold_set_count = 0

            prepared_records = self._resolve_prepared_records(request)
            for record in prepared_records:
                if record.prepared_hold_set_id is not None:
                    hold_set = self._prepared_hold_registry.get(
                        record.prepared_hold_set_id
                    )
                    if (
                        hold_set is not None
                        and hold_set.state == PreparedHoldSetState.ACTIVE
                    ):
                        if self._release_prepared_hold_set is not None:
                            self._release_prepared_hold_set(hold_set)
                        self._prepared_hold_registry.release_hold_set(
                            hold_set_id=record.prepared_hold_set_id,
                            now_ms=request.requested_at_ms,
                        )
                        evicted_hold_set_count += 1
                if record.state != PreparedBundleLifecycleState.EVICTED:
                    self._prepared_bundle_registry.evict_bundle(
                        logical_request_id=record.logical_request_id,
                        publish_manifest_digest=record.publish_manifest_digest,
                        now_ms=request.requested_at_ms,
                    )
                    evicted_bundle_count += 1

            if request.logical_request_id is not None:
                live_request = self._request_bundle_registry.get(
                    request.logical_request_id
                )
                if live_request is not None:
                    if live_request.state in {
                        RequestBundleLifecycleState.LIVE_TRACKING,
                        RequestBundleLifecycleState.SOURCE_RETAINED,
                        RequestBundleLifecycleState.PUBLISHED,
                        RequestBundleLifecycleState.PUBLISH_FAILED,
                    }:
                        self._request_bundle_registry.mark_cleaned(
                            logical_request_id=request.logical_request_id,
                            now_ms=request.requested_at_ms,
                        )
                    elif live_request.state != RequestBundleLifecycleState.CLEANED:
                        raise RequestBundleStateError(
                            "cannot evict local request bundle while it is in a transient publish state"
                        )
                    self._request_bundle_registry.clear(request.logical_request_id)
                    self._page_publication_registry.clear_request(
                        request.logical_request_id
                    )
                    if self._discard_publishable_source_request is not None:
                        self._discard_publishable_source_request(
                            request.logical_request_id
                        )

            return EvictLocalInstanceOpResult(
                status=InstanceOpStatus.SUCCESS,
                logical_request_id=request.logical_request_id,
                publish_manifest_digest=request.publish_manifest_digest,
                evicted_bundle_count=evicted_bundle_count,
                evicted_hold_set_count=evicted_hold_set_count,
            )
        except Exception as exc:
            return EvictLocalInstanceOpResult(
                status=InstanceOpStatus.FAILED,
                logical_request_id=request.logical_request_id,
                publish_manifest_digest=request.publish_manifest_digest,
                error_message=str(exc),
            )

    def _resolve_prepared_records(self, request: EvictLocalInstanceOpRequest) -> tuple:
        if request.publish_manifest_digest is not None:
            record = self._prepared_bundle_registry.get_by_publish_manifest_digest(
                publish_manifest_digest=request.publish_manifest_digest,
                include_evicted=True,
            )
            if record is None:
                return ()
            if (
                request.logical_request_id is not None
                and record.logical_request_id != request.logical_request_id
            ):
                raise RequestBundleStateError(
                    "evict_local selector mismatch between logical_request_id and publish_manifest_digest"
                )
            return (record,)
        if request.logical_request_id is None:
            return ()
        return self._prepared_bundle_registry.list_request_records(
            logical_request_id=request.logical_request_id,
            include_evicted=False,
        )
