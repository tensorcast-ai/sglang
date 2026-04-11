# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    HydratePreparedResult,
    PublishManifestRecord,
    RankCoord,
    SourcePublishClosureResult,
    _FrozenModel,
    _required_ranks_are_unique,
)


class InstanceOpKind(StrEnum):
    PUBLISH = "publish"
    HYDRATE = "hydrate"
    EVICT_LOCAL = "evict_local"


class InstanceOpStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


def _kind_for_payload(
    payload: "PublishInstanceOpRequest"
    | "HydrateInstanceOpRequest"
    | "EvictLocalInstanceOpRequest",
) -> InstanceOpKind:
    payload_kind_by_type = {
        PublishInstanceOpRequest: InstanceOpKind.PUBLISH,
        HydrateInstanceOpRequest: InstanceOpKind.HYDRATE,
        EvictLocalInstanceOpRequest: InstanceOpKind.EVICT_LOCAL,
    }
    resolved_kind = payload_kind_by_type.get(type(payload))
    if resolved_kind is None:
        raise ValueError(f"unsupported instance-op payload type: {type(payload)!r}")
    return resolved_kind


class PublishInstanceOpRequest(_FrozenModel):
    logical_request_id: str = Field(min_length=1)
    engine_request_id: str = Field(min_length=1)
    publish_op_id: str = Field(min_length=1)
    requested_cutoff_token_count: int | None = Field(default=None, ge=0)
    timeout_ms: int | None = Field(default=None, gt=0)
    ttl_ms: int = Field(default=60_000, gt=0)
    prompt_token_digest: str = ""
    transfer_mode: str = "prefill_closed_prompt_reuse"
    batch_request_count: int = Field(default=1, ge=1)
    parallel_sampling_count: int = Field(default=1, ge=1)
    session_lineage_depth: int = Field(default=0, ge=0)
    emitted_decode_token_count: int = Field(default=0, ge=0)
    attention_arch: str = "mha"
    dtype: str = ""
    page_size: int = Field(default=32, gt=0)
    requested_at_ms: int = Field(ge=0)


class PublishInstanceOpResult(_FrozenModel):
    logical_request_id: str = Field(min_length=1)
    status: InstanceOpStatus
    publish_op_id: str = Field(min_length=1)
    snapshot_seq: int | None = Field(default=None, ge=0)
    publish_manifest_digest: str | None = None
    error_message: str | None = None


class PublishInstanceOpReqInput(_FrozenModel):
    engine_request_id: str = Field(min_length=1)
    ttl_ms: int | None = Field(default=None, gt=0)
    requested_cutoff_token_count: int | None = Field(default=None, ge=0)
    timeout_ms: int | None = Field(default=None, gt=0)
    requested_at_ms: int = Field(ge=0)
    publish_op_id: str = Field(min_length=1)


class PublishInstanceOpRespOutput(_FrozenModel):
    status: InstanceOpStatus
    result: SourcePublishClosureResult | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def _validate_response(self) -> "PublishInstanceOpRespOutput":
        if self.status == InstanceOpStatus.SUCCESS and self.result is None:
            raise ValueError("successful publish response requires result")
        if self.status == InstanceOpStatus.FAILED and not self.error_message:
            raise ValueError("failed publish response requires error_message")
        return self


class HydrateInstanceOpRequest(_FrozenModel):
    logical_request_id: str = Field(min_length=1)
    publish_manifest_digest: str = Field(min_length=1)
    requested_at_ms: int = Field(ge=0)


class HydrateInstanceOpResult(_FrozenModel):
    logical_request_id: str = Field(min_length=1)
    publish_manifest_digest: str = Field(min_length=1)
    status: InstanceOpStatus
    prepared_hold_set_id: str | None = None
    error_message: str | None = None


class HydrateInstanceOpReqInput(_FrozenModel):
    request: HydrateInstanceOpRequest
    publish_manifest: "PublishManifestRecord"


class HydrateInstanceOpRespOutput(_FrozenModel):
    status: InstanceOpStatus
    result: HydratePreparedResult | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def _validate_response(self) -> "HydrateInstanceOpRespOutput":
        if self.status == InstanceOpStatus.SUCCESS and self.result is None:
            raise ValueError("successful hydrate response requires result")
        if self.status == InstanceOpStatus.FAILED and not self.error_message:
            raise ValueError("failed hydrate response requires error_message")
        return self


class EvictLocalInstanceOpRequest(_FrozenModel):
    logical_request_id: str | None = None
    publish_manifest_digest: str | None = None
    requested_at_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_request_selector(self) -> "EvictLocalInstanceOpRequest":
        if self.logical_request_id is None and self.publish_manifest_digest is None:
            raise ValueError(
                "either logical_request_id or publish_manifest_digest must be provided"
            )
        return self


class EvictLocalInstanceOpResult(_FrozenModel):
    status: InstanceOpStatus
    logical_request_id: str | None = None
    publish_manifest_digest: str | None = None
    evicted_bundle_count: int = Field(default=0, ge=0)
    evicted_hold_set_count: int = Field(default=0, ge=0)
    error_message: str | None = None


class EvictLocalInstanceOpReqInput(_FrozenModel):
    request: EvictLocalInstanceOpRequest


class EvictLocalInstanceOpRespOutput(_FrozenModel):
    status: InstanceOpStatus
    result: EvictLocalInstanceOpResult | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def _validate_response(self) -> "EvictLocalInstanceOpRespOutput":
        if self.status == InstanceOpStatus.SUCCESS and self.result is None:
            raise ValueError("successful evict_local response requires result")
        if self.status == InstanceOpStatus.FAILED and not self.error_message:
            raise ValueError("failed evict_local response requires error_message")
        return self


class PerRankInstanceOpResult(_FrozenModel):
    kind: InstanceOpKind
    rank: RankCoord
    status: InstanceOpStatus
    error_message: str | None = None


class AggregatedInstanceOpResult(_FrozenModel):
    kind: InstanceOpKind
    logical_request_id: str = Field(min_length=1)
    status: InstanceOpStatus
    required_ranks: tuple[RankCoord, ...]
    rank_results: tuple[PerRankInstanceOpResult, ...]
    error_message: str | None = None


class CoordinatorRegistrationRecord(_FrozenModel):
    instance_id: str = Field(min_length=1)
    coordinator_rank: RankCoord
    coordinator_epoch: str = Field(min_length=1)
    required_ranks: tuple[RankCoord, ...]
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_registration(self) -> "CoordinatorRegistrationRecord":
        _required_ranks_are_unique(self.required_ranks)
        if self.coordinator_rank.as_key() not in {
            rank.as_key() for rank in self.required_ranks
        }:
            raise ValueError("coordinator_rank must be part of required_ranks")
        if self.coordinator_rank.tp_rank != 0:
            raise ValueError("coordinator_rank must be a TP-rank-0 coordinator")
        return self


class CoordinatorInstanceOpRequest(_FrozenModel):
    kind: InstanceOpKind
    op_id: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    required_ranks: tuple[RankCoord, ...]
    timeout_ms: int = Field(gt=0)
    requested_at_ms: int = Field(ge=0)
    payload: (
        PublishInstanceOpRequest
        | HydrateInstanceOpRequest
        | EvictLocalInstanceOpRequest
    )

    @property
    def logical_request_id(self) -> str:
        if isinstance(self.payload, EvictLocalInstanceOpRequest):
            if self.payload.logical_request_id is None:
                raise ValueError(
                    "coordinator-scoped evict_local requires logical_request_id"
                )
            return self.payload.logical_request_id
        return self.payload.logical_request_id

    @model_validator(mode="after")
    def _validate_request(self) -> "CoordinatorInstanceOpRequest":
        _required_ranks_are_unique(self.required_ranks)
        resolved_kind = _kind_for_payload(self.payload)
        if self.kind != resolved_kind:
            raise ValueError(
                f"kind={self.kind} does not match payload type {type(self.payload).__name__}"
            )
        if (
            self.kind == InstanceOpKind.EVICT_LOCAL
            and self.payload.logical_request_id is None
        ):
            raise ValueError(
                "coordinator-scoped evict_local requires payload.logical_request_id"
            )
        return self


class RankScopedInstanceOpRequest(_FrozenModel):
    kind: InstanceOpKind
    op_id: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    rank: RankCoord
    payload: (
        PublishInstanceOpRequest
        | HydrateInstanceOpRequest
        | EvictLocalInstanceOpRequest
    )

    @model_validator(mode="after")
    def _validate_request(self) -> "RankScopedInstanceOpRequest":
        resolved_kind = _kind_for_payload(self.payload)
        if self.kind != resolved_kind:
            raise ValueError(
                f"kind={self.kind} does not match payload type {type(self.payload).__name__}"
            )
        return self


class CoordinatorFanoutResult(_FrozenModel):
    rank_results: tuple[PerRankInstanceOpResult, ...] = ()
    timed_out_ranks: tuple[RankCoord, ...] = ()

    @model_validator(mode="after")
    def _validate_results(self) -> "CoordinatorFanoutResult":
        rank_result_keys = [result.rank.as_key() for result in self.rank_results]
        if len(set(rank_result_keys)) != len(rank_result_keys):
            raise ValueError("rank_results must be unique per rank")
        timeout_keys = [rank.as_key() for rank in self.timed_out_ranks]
        if len(set(timeout_keys)) != len(timeout_keys):
            raise ValueError("timed_out_ranks must be unique per rank")
        if set(rank_result_keys).intersection(timeout_keys):
            raise ValueError("timed_out_ranks must not overlap rank_results")
        return self


class CoordinatorOpRecord(_FrozenModel):
    request: CoordinatorInstanceOpRequest
    aggregated_result: AggregatedInstanceOpResult
    dispatch_invocation_count: int = Field(ge=1)
    retry_hit_count: int = Field(default=0, ge=0)
    started_at_ms: int = Field(ge=0)
    completed_at_ms: int = Field(ge=0)
