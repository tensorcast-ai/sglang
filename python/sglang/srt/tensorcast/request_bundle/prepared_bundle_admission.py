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


def _hash_token_ids(token_ids: list[int]) -> str:
    hasher = hashlib.sha256()
    for token_id in token_ids:
        hasher.update(int(token_id).to_bytes(4, byteorder="little", signed=False))
    return hasher.hexdigest()


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
        if (
            claim_decision.action == PreparedBundleClaimAction.FALLBACK
            and request.logical_session_id is not None
        ):
            claim_decision = (
                self._prepared_bundle_registry.claim_prepared_bundle_for_session(
                    logical_session_id=request.logical_session_id,
                    incoming_prompt_token_count=len(request.prompt_token_ids)
                    if request.prompt_token_ids
                    else request.cutoff_token_count,
                    incoming_prompt_token_digest_for_count=(
                        self._build_prefix_digest_resolver(request)
                    ),
                    scheduler_rid=request.scheduler_rid,
                    claim_token=claim_token,
                    now_ms=request.requested_at_ms,
                    live_request_exists=live_request_exists,
                )
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

        claimed_logical_request_id = claimed_record.logical_request_id
        try:
            prepared_bundle_key = attach_prepared_bundle(claimed_record)
        except Exception as exc:
            self._prepared_bundle_registry.rollback_claim(
                logical_request_id=claimed_logical_request_id,
                publish_manifest_digest=claimed_record.publish_manifest_digest,
            )
            tainted = self._prepared_bundle_registry.mark_tainted(
                logical_request_id=claimed_logical_request_id,
                publish_manifest_digest=claimed_record.publish_manifest_digest,
            )
            return OrdinaryGenerateBindingResult(
                action=PreparedBundleBindAction.FAIL_CLOSED,
                reason=f"failed to attach claimed prepared bundle: {exc}",
                record=tainted,
                claim_token=claim_token,
            )

        attached = self._prepared_bundle_registry.mark_attached(
            logical_request_id=claimed_logical_request_id,
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

    def _build_prefix_digest_resolver(
        self, request: OrdinaryGenerateBindingRequest
    ) -> Callable[[int], str]:
        def digest_for_count(token_count: int) -> str:
            resolved_token_count = int(token_count)
            if resolved_token_count == int(request.cutoff_token_count):
                return request.prompt_token_digest
            if not request.prompt_token_ids:
                raise ValueError("prompt_token_ids are required for prefix digest")
            if resolved_token_count > len(request.prompt_token_ids):
                raise ValueError("prefix token count exceeds prompt_token_ids length")
            return _hash_token_ids(
                list(request.prompt_token_ids[:resolved_token_count])
            )

        return digest_for_count

    def _build_claim_token(self, request: OrdinaryGenerateBindingRequest) -> str:
        payload = (
            f"{request.logical_request_id}:"
            f"{request.scheduler_rid}:"
            f"{request.logical_session_id or ''}:"
            f"{request.session_generation if request.session_generation is not None else ''}:"
            f"{request.prompt_token_digest}:"
            f"{request.cutoff_token_count}:"
            f"{request.requested_at_ms}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
