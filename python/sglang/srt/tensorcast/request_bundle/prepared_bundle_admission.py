# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TypeAlias

from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    PreparedBundleRegistry,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    OrdinaryGenerateBindingRequest,
    OrdinaryGenerateBindingResult,
    PreparedBundleBindAction,
    PreparedBundleClaimAction,
    PreparedBundleRecord,
)

AttachPreparedBundle: TypeAlias = Callable[[PreparedBundleRecord], str]


class PreparedBundleAdmissionBinder:
    def __init__(self, *, prepared_bundle_registry: PreparedBundleRegistry) -> None:
        self._prepared_bundle_registry = prepared_bundle_registry

    def bind(
        self,
        *,
        request: OrdinaryGenerateBindingRequest,
        attach_prepared_bundle: AttachPreparedBundle,
        live_request_exists: bool = False,
    ) -> OrdinaryGenerateBindingResult:
        claim_token = self._build_claim_token(request)
        claim_decision = self._prepared_bundle_registry.claim_prepared_bundle(
            logical_request_id=request.logical_request_id,
            incoming_prompt_token_digest=request.prompt_token_digest,
            incoming_cutoff_token_count=request.cutoff_token_count,
            scheduler_rid=request.scheduler_rid,
            claim_token=claim_token,
            now_ms=request.requested_at_ms,
            live_request_exists=live_request_exists,
        )
        if claim_decision.action == PreparedBundleClaimAction.FALLBACK:
            return OrdinaryGenerateBindingResult(
                action=PreparedBundleBindAction.FALLBACK,
                reason=claim_decision.reason,
                record=claim_decision.record,
            )
        if claim_decision.action == PreparedBundleClaimAction.FAIL_CLOSED:
            return OrdinaryGenerateBindingResult(
                action=PreparedBundleBindAction.FAIL_CLOSED,
                reason=claim_decision.reason,
                record=claim_decision.record,
            )
        claimed_record = claim_decision.record
        if claimed_record is None:
            return OrdinaryGenerateBindingResult(
                action=PreparedBundleBindAction.FAIL_CLOSED,
                reason="claim succeeded without a prepared bundle record",
            )

        try:
            prepared_bundle_key = attach_prepared_bundle(claimed_record)
        except Exception as exc:
            self._prepared_bundle_registry.rollback_claim(
                logical_request_id=request.logical_request_id,
                publish_manifest_digest=claimed_record.publish_manifest_digest,
            )
            tainted = self._prepared_bundle_registry.mark_tainted(
                logical_request_id=request.logical_request_id,
                publish_manifest_digest=claimed_record.publish_manifest_digest,
            )
            return OrdinaryGenerateBindingResult(
                action=PreparedBundleBindAction.FAIL_CLOSED,
                reason=f"failed to attach claimed prepared bundle: {exc}",
                record=tainted,
                claim_token=claim_token,
            )

        attached = self._prepared_bundle_registry.mark_attached(
            logical_request_id=request.logical_request_id,
            publish_manifest_digest=claimed_record.publish_manifest_digest,
            prepared_bundle_key=prepared_bundle_key,
        )
        return OrdinaryGenerateBindingResult(
            action=PreparedBundleBindAction.ATTACHED,
            reason="prepared bundle attached successfully",
            record=attached,
            claim_token=claim_token,
            prepared_bundle_key=prepared_bundle_key,
        )

    def _build_claim_token(self, request: OrdinaryGenerateBindingRequest) -> str:
        payload = (
            f"{request.logical_request_id}:"
            f"{request.scheduler_rid}:"
            f"{request.prompt_token_digest}:"
            f"{request.cutoff_token_count}:"
            f"{request.requested_at_ms}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
