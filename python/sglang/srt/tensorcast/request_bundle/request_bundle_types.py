# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PagePublicationState(StrEnum):
    ABSENT = "absent"
    INFLIGHT = "inflight"
    READY = "ready"
    FAILED = "failed"


class RequestBundleLifecycleState(StrEnum):
    LIVE_TRACKING = "live_tracking"
    SOURCE_RETAINED = "source_retained"
    SNAPSHOT_CLOSING = "snapshot_closing"
    CLOSING_TAIL_FLUSH = "closing_tail_flush"
    CLOSURE_READY = "closure_ready"
    PUBLISHED = "published"
    PUBLISH_FAILED = "publish_failed"
    CLEANED = "cleaned"


class PreparedBundleLifecycleState(StrEnum):
    PREPARING = "preparing"
    PREPARED = "prepared"
    CLAIMED = "claimed"
    ATTACHED = "attached"
    CONSUMED = "consumed"
    FAILED = "failed"
    EVICTED = "evicted"


class PreparedHoldSetState(StrEnum):
    ACTIVE = "active"
    RELEASED = "released"


class RankInstallState(StrEnum):
    PREPARING = "preparing"
    READY = "ready"
    FAILED = "failed"
    CLEANED = "cleaned"


def _required_ranks_are_unique(required_ranks: tuple["RankCoord", ...]) -> None:
    required_keys = [rank.as_key() for rank in required_ranks]
    if len(set(required_keys)) != len(required_keys):
        raise ValueError("required_ranks must be unique")


class PreparedBundleClaimAction(StrEnum):
    CLAIMED = "claimed"
    FALLBACK = "fallback"
    FAIL_CLOSED = "fail_closed"


class RankCoord(_FrozenModel):
    tp_rank: int = Field(ge=0)
    pp_rank: int = Field(ge=0)

    def as_key(self) -> tuple[int, int]:
        return (self.tp_rank, self.pp_rank)


class PageClosureEntry(_FrozenModel):
    logical_page_index: int = Field(ge=0)
    page_hash: str = Field(min_length=1)
    publication_state: PagePublicationState
    artifact_id: str | None = None
    host_resident: bool
    last_error: str | None = None


class PagePublicationRecord(_FrozenModel):
    logical_request_id: str = Field(min_length=1)
    rank: RankCoord
    logical_page_index: int = Field(ge=0)
    page_hash: str = Field(min_length=1)
    publication_state: PagePublicationState
    artifact_id: str | None = None
    host_resident: bool
    last_error: str | None = None
    updated_at_ms: int = Field(ge=0)

    def to_closure_entry(self) -> PageClosureEntry:
        return PageClosureEntry(
            logical_page_index=self.logical_page_index,
            page_hash=self.page_hash,
            publication_state=self.publication_state,
            artifact_id=self.artifact_id,
            host_resident=self.host_resident,
            last_error=self.last_error,
        )


class RankSnapshotCursor(_FrozenModel):
    tp_rank: int = Field(ge=0)
    pp_rank: int = Field(ge=0)
    latest_token_count: int = Field(ge=0)
    latest_last_page_index: int = Field(ge=-1)
    frozen_cutoff_token_count: int | None = Field(default=None, ge=0)
    frozen_last_page_index: int | None = Field(default=None, ge=-1)
    force_flush_cursor: int | None = Field(default=None, ge=0)
    ordered_pages: tuple[PageClosureEntry, ...] = ()

    def rank_coord(self) -> RankCoord:
        return RankCoord(tp_rank=self.tp_rank, pp_rank=self.pp_rank)


class RequestBundleState(_FrozenModel):
    logical_request_id: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    engine_request_id: str = Field(min_length=1)
    full_prompt_token_count: int = Field(ge=0)
    model_fingerprint: str = Field(min_length=1)
    kv_layout_id: str = Field(min_length=1)
    tp_size: int = Field(gt=0)
    pp_size: int = Field(gt=0)
    state: RequestBundleLifecycleState
    snapshot_seq: int = Field(ge=0)
    publish_op_id: str | None = None
    frozen_cutoff_token_count: int | None = Field(default=None, ge=0)
    frozen_last_page_index: int | None = Field(default=None, ge=-1)
    retained_until_ms: int | None = Field(default=None, ge=0)
    bundle_digest: str | None = None
    latest_publish_manifest_digest: str | None = None
    required_ranks: tuple[RankCoord, ...]
    rank_snapshots: tuple[RankSnapshotCursor, ...] = ()
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)
    last_error: str | None = None

    @model_validator(mode="after")
    def _validate_required_ranks(self) -> "RequestBundleState":
        _required_ranks_are_unique(self.required_ranks)
        snapshot_keys = [
            snapshot.rank_coord().as_key() for snapshot in self.rank_snapshots
        ]
        if len(set(snapshot_keys)) != len(snapshot_keys):
            raise ValueError("rank_snapshots must be unique per rank")
        return self


class PageClosureCutoff(_FrozenModel):
    requested_token_count: int = Field(ge=0)
    materialized_token_count: int = Field(ge=0)
    cutoff_token_count: int = Field(ge=0)
    page_aligned_token_count: int = Field(ge=0)
    frozen_last_page_index: int | None = Field(default=None, ge=-1)
    tail_valid_tokens: int = Field(default=0, ge=0)


class RankInstallRecord(_FrozenModel):
    tp_rank: int = Field(ge=0)
    pp_rank: int = Field(ge=0)
    state: RankInstallState
    hydrated_page_count: int = Field(ge=0)
    runnable_prefix_tokens: int = Field(ge=0)
    local_install_handle: str | None = None
    last_error: str | None = None

    def rank_coord(self) -> RankCoord:
        return RankCoord(tp_rank=self.tp_rank, pp_rank=self.pp_rank)


class PreparedSlotToken(_FrozenModel):
    slot_index: int = Field(ge=0)
    slot_generation: int = Field(ge=0)


class PreparedHoldRef(_FrozenModel):
    logical_page_index: int = Field(ge=0)
    page_hash: str = Field(min_length=1)
    slot_token: PreparedSlotToken
    artifact_id: str | None = None


class PreparedHoldSetRecord(_FrozenModel):
    hold_set_id: str = Field(min_length=1)
    state: PreparedHoldSetState
    refs: tuple[PreparedHoldRef, ...]
    created_at_ms: int = Field(ge=0)
    released_at_ms: int | None = Field(default=None, ge=0)


class PreparedBundleRecord(_FrozenModel):
    logical_request_id: str = Field(min_length=1)
    target_instance_id: str = Field(min_length=1)
    publish_manifest_digest: str = Field(min_length=1)
    artifact_manifest_digest: str = Field(min_length=1)
    engine_owned_manifest_sha256: str = Field(min_length=1)
    prepared_hold_set_id: str | None = None
    state: PreparedBundleLifecycleState
    required_ranks: tuple[RankCoord, ...]
    rank_installs: tuple[RankInstallRecord, ...]
    claim_token: str | None = None
    active_scheduler_rid: str | None = None
    prepared_bundle_key: str | None = None
    created_at_ms: int = Field(ge=0)
    prepared_at_ms: int | None = Field(default=None, ge=0)
    claimed_at_ms: int | None = Field(default=None, ge=0)
    cleaned_at_ms: int | None = Field(default=None, ge=0)
    prompt_token_digest: str | None = None
    cutoff_token_count: int | None = Field(default=None, ge=0)
    tail_valid_tokens: int = Field(default=0, ge=0)
    stale: bool = False
    tainted: bool = False
    last_error: str | None = None

    @model_validator(mode="after")
    def _validate_required_ranks(self) -> "PreparedBundleRecord":
        _required_ranks_are_unique(self.required_ranks)
        install_keys = [record.rank_coord().as_key() for record in self.rank_installs]
        if len(set(install_keys)) != len(install_keys):
            raise ValueError("rank_installs must be unique per rank")
        return self


class PreparedBundleClaimDecision(_FrozenModel):
    action: PreparedBundleClaimAction
    reason: str = Field(min_length=1)
    record: PreparedBundleRecord | None = None
    claim_token: str | None = None


class PreparedBundleBindAction(StrEnum):
    ATTACHED = "attached"
    FALLBACK = "fallback"
    FAIL_CLOSED = "fail_closed"


class OrdinaryGenerateBindingRequest(_FrozenModel):
    logical_request_id: str = Field(min_length=1)
    scheduler_rid: str = Field(min_length=1)
    prompt_token_digest: str = Field(min_length=1)
    cutoff_token_count: int = Field(ge=0)
    requested_at_ms: int = Field(ge=0)


class OrdinaryGenerateBindingResult(_FrozenModel):
    action: PreparedBundleBindAction
    reason: str = Field(min_length=1)
    record: PreparedBundleRecord | None = None
    claim_token: str | None = None
    prepared_bundle_key: str | None = None


class PagePublishAction(StrEnum):
    REUSED = "reused"
    WAITED = "waited"
    FLUSHED = "flushed"


class PagePublishOutcome(_FrozenModel):
    rank: RankCoord
    logical_page_index: int = Field(ge=0)
    page_hash: str = Field(min_length=1)
    action: PagePublishAction
    final_publication_state: PagePublicationState
    artifact_id: str | None = None


class RankPublishClosureResult(_FrozenModel):
    rank: RankCoord
    page_outcomes: tuple[PagePublishOutcome, ...]


class PublishArtifactManifestEntry(_FrozenModel):
    rank: RankCoord
    logical_page_index: int = Field(ge=0)
    page_hash: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)


class PublishArtifactManifestRecord(_FrozenModel):
    artifact_manifest_digest: str = Field(min_length=1)
    logical_request_id: str = Field(min_length=1)
    snapshot_seq: int = Field(ge=1)
    required_ranks: tuple[RankCoord, ...]
    entries: tuple[PublishArtifactManifestEntry, ...]

    @model_validator(mode="after")
    def _validate_manifest(self) -> "PublishArtifactManifestRecord":
        _required_ranks_are_unique(self.required_ranks)
        entry_keys = [
            (entry.rank.as_key(), entry.logical_page_index) for entry in self.entries
        ]
        if len(set(entry_keys)) != len(entry_keys):
            raise ValueError("artifact manifest entries must be unique per rank/page")
        return self


class PublishCompatibilityEnvelope(_FrozenModel):
    model_fingerprint: str = Field(min_length=1)
    kv_layout_id: str = Field(min_length=1)
    dtype: str = ""
    page_size: int = Field(gt=0)
    tp_size: int = Field(gt=0)
    pp_size: int = Field(gt=0)
    attention_arch: str = Field(min_length=1)
    required_ranks: tuple[RankCoord, ...]

    @model_validator(mode="after")
    def _validate_envelope(self) -> "PublishCompatibilityEnvelope":
        _required_ranks_are_unique(self.required_ranks)
        return self


class EngineOwnedManifestPayload(_FrozenModel):
    schema_name: str = Field(
        default="sglang.request_bundle_payload.v1",
        alias="schema",
        serialization_alias="schema",
    )
    transfer_mode: str = Field(min_length=1)
    logical_request_id: str = Field(min_length=1)
    cutoff_token_count: int = Field(ge=0)
    frozen_last_page_index: int | None = Field(default=None, ge=-1)
    tail_valid_tokens: int = Field(default=0, ge=0)
    prompt_token_digest: str = Field(min_length=1)
    compatibility: PublishCompatibilityEnvelope


class EngineOwnedManifestRecord(_FrozenModel):
    engine: str = "sglang"
    schema_name: str = Field(
        default="sglang.engine_owned_manifest.v1",
        alias="schema",
        serialization_alias="schema",
    )
    version: int = Field(default=1, ge=1)
    encoding: str = "json"
    created_at_ms: int = Field(ge=0)
    expires_at_ms: int = Field(ge=0)
    artifact_manifest_digest: str = Field(min_length=1)
    payload_sha256: str = Field(min_length=1)
    payload: EngineOwnedManifestPayload


class PublishManifestRecord(_FrozenModel):
    schema_name: str = Field(
        default="tensorcast.publish_manifest.v1",
        alias="schema",
        serialization_alias="schema",
    )
    publish_manifest_digest: str = Field(min_length=1)
    artifact_manifest: PublishArtifactManifestRecord
    engine_owned_manifest: EngineOwnedManifestRecord
    prompt_token_digest: str = Field(min_length=1)
    cutoff_token_count: int = Field(ge=0)
    tail_valid_tokens: int = Field(default=0, ge=0)


class SourcePublishClosureResult(_FrozenModel):
    request_state: RequestBundleState
    cutoff: PageClosureCutoff
    publish_manifest: PublishManifestRecord
    rank_results: tuple[RankPublishClosureResult, ...]


class HydrateTargetCompatibility(_FrozenModel):
    target_instance_id: str = Field(min_length=1)
    model_fingerprint: str = Field(min_length=1)
    kv_layout_id: str = Field(min_length=1)
    dtype: str = ""
    page_size: int = Field(gt=0)
    tp_size: int = Field(gt=0)
    pp_size: int = Field(gt=0)
    attention_arch: str = Field(min_length=1)
    required_ranks: tuple[RankCoord, ...]

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "HydrateTargetCompatibility":
        _required_ranks_are_unique(self.required_ranks)
        return self


class HydrateRankWorkItem(_FrozenModel):
    rank: RankCoord
    logical_request_id: str = Field(min_length=1)
    publish_manifest_digest: str = Field(min_length=1)
    artifact_manifest_digest: str = Field(min_length=1)
    cutoff_token_count: int = Field(ge=0)
    tail_valid_tokens: int = Field(default=0, ge=0)
    prompt_token_digest: str = Field(min_length=1)
    artifact_entries: tuple[PublishArtifactManifestEntry, ...]


class HydrateRankInstallResult(_FrozenModel):
    rank: RankCoord
    state: RankInstallState
    hydrated_page_count: int = Field(ge=0)
    runnable_prefix_tokens: int = Field(ge=0)
    local_install_handle: str | None = None
    hold_refs: tuple[PreparedHoldRef, ...] = ()
    error_message: str | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> "HydrateRankInstallResult":
        if self.state != RankInstallState.READY and self.hold_refs:
            raise ValueError("only READY hydrate installs may carry hold_refs")
        return self


class HydratePreparedResult(_FrozenModel):
    prepared_bundle: PreparedBundleRecord
    hold_set: PreparedHoldSetRecord | None = None
    reused_existing: bool = False
