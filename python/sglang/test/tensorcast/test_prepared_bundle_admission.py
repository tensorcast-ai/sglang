from __future__ import annotations

from sglang.srt.tensorcast.request_bundle.prepared_bundle_admission import (
    PreparedBundleAdmissionBinder,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_state import (
    PreparedBundleRegistry,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    OrdinaryGenerateBindingRequest,
    PreparedBundleBindAction,
    PreparedBundleLifecycleState,
    RankCoord,
    RankInstallState,
)


def _rank(tp_rank: int, pp_rank: int = 0) -> RankCoord:
    return RankCoord(tp_rank=tp_rank, pp_rank=pp_rank)


def _prepare_clean_bundle(
    registry: PreparedBundleRegistry,
    *,
    logical_request_id: str,
    publish_manifest_digest: str,
    prompt_token_digest: str,
    cutoff_token_count: int,
) -> None:
    registry.begin_prepare(
        logical_request_id=logical_request_id,
        target_instance_id="instance-b",
        publish_manifest_digest=publish_manifest_digest,
        artifact_manifest_digest=f"artifact:{publish_manifest_digest}",
        engine_owned_manifest_sha256=f"engine:{publish_manifest_digest}",
        required_ranks=(_rank(0),),
        prompt_token_digest=prompt_token_digest,
        cutoff_token_count=cutoff_token_count,
        now_ms=100,
    )
    registry.mark_rank_install(
        logical_request_id=logical_request_id,
        publish_manifest_digest=publish_manifest_digest,
        rank=_rank(0),
        state=RankInstallState.READY,
        hydrated_page_count=2,
        runnable_prefix_tokens=cutoff_token_count,
    )
    registry.mark_prepared(
        logical_request_id=logical_request_id,
        publish_manifest_digest=publish_manifest_digest,
        now_ms=110,
        prepared_hold_set_id=f"hold:{publish_manifest_digest}",
    )


def test_binding_attaches_matching_prepared_bundle() -> None:
    registry = PreparedBundleRegistry()
    _prepare_clean_bundle(
        registry,
        logical_request_id="rid-1",
        publish_manifest_digest="manifest-1",
        prompt_token_digest="prompt-digest-1",
        cutoff_token_count=64,
    )
    attached_records: list[str] = []

    result = PreparedBundleAdmissionBinder(prepared_bundle_registry=registry).bind(
        request=OrdinaryGenerateBindingRequest(
            logical_request_id="rid-1",
            scheduler_rid="sgl-rid-1",
            prompt_token_digest="prompt-digest-1",
            cutoff_token_count=64,
            requested_at_ms=200,
        ),
        attach_prepared_bundle=lambda record: attached_records.append(
            record.publish_manifest_digest
        )
        or "prepared-bundle-key-1",
    )

    assert result.action == PreparedBundleBindAction.ATTACHED
    assert result.record is not None
    assert result.record.state == PreparedBundleLifecycleState.ATTACHED
    assert result.record.active_scheduler_rid == "sgl-rid-1"
    assert result.prepared_bundle_key == "prepared-bundle-key-1"
    assert attached_records == ["manifest-1"]


def test_binding_falls_back_on_prompt_digest_mismatch() -> None:
    registry = PreparedBundleRegistry()
    _prepare_clean_bundle(
        registry,
        logical_request_id="rid-2",
        publish_manifest_digest="manifest-2",
        prompt_token_digest="prompt-digest-2",
        cutoff_token_count=64,
    )
    attach_calls: list[str] = []

    result = PreparedBundleAdmissionBinder(prepared_bundle_registry=registry).bind(
        request=OrdinaryGenerateBindingRequest(
            logical_request_id="rid-2",
            scheduler_rid="sgl-rid-2",
            prompt_token_digest="prompt-digest-mismatch",
            cutoff_token_count=64,
            requested_at_ms=200,
        ),
        attach_prepared_bundle=lambda record: attach_calls.append(
            record.publish_manifest_digest
        )
        or "unused",
    )

    assert result.action == PreparedBundleBindAction.FALLBACK
    assert "prompt token digest does not match" in result.reason
    assert attach_calls == []


def test_binding_falls_back_for_stale_prepared_bundle() -> None:
    registry = PreparedBundleRegistry()
    _prepare_clean_bundle(
        registry,
        logical_request_id="rid-3",
        publish_manifest_digest="manifest-3",
        prompt_token_digest="prompt-digest-3",
        cutoff_token_count=32,
    )
    registry.mark_stale(
        logical_request_id="rid-3",
        publish_manifest_digest="manifest-3",
    )

    result = PreparedBundleAdmissionBinder(prepared_bundle_registry=registry).bind(
        request=OrdinaryGenerateBindingRequest(
            logical_request_id="rid-3",
            scheduler_rid="sgl-rid-3",
            prompt_token_digest="prompt-digest-3",
            cutoff_token_count=32,
            requested_at_ms=200,
        ),
        attach_prepared_bundle=lambda _: "unused",
    )

    assert result.action == PreparedBundleBindAction.FALLBACK
    assert (
        "only stale or tainted prepared bundle records are available" in result.reason
    )
    assert result.record is not None
    assert result.record.stale is True


def test_binding_falls_back_for_tainted_prepared_bundle() -> None:
    registry = PreparedBundleRegistry()
    _prepare_clean_bundle(
        registry,
        logical_request_id="rid-3b",
        publish_manifest_digest="manifest-3b",
        prompt_token_digest="prompt-digest-3b",
        cutoff_token_count=32,
    )
    registry.mark_tainted(
        logical_request_id="rid-3b",
        publish_manifest_digest="manifest-3b",
    )

    result = PreparedBundleAdmissionBinder(prepared_bundle_registry=registry).bind(
        request=OrdinaryGenerateBindingRequest(
            logical_request_id="rid-3b",
            scheduler_rid="sgl-rid-3b",
            prompt_token_digest="prompt-digest-3b",
            cutoff_token_count=32,
            requested_at_ms=200,
        ),
        attach_prepared_bundle=lambda _: "unused",
    )

    assert result.action == PreparedBundleBindAction.FALLBACK
    assert (
        "only stale or tainted prepared bundle records are available" in result.reason
    )
    assert result.record is not None
    assert result.record.tainted is True


def test_binding_fails_closed_when_clean_bundle_is_already_claimed() -> None:
    registry = PreparedBundleRegistry()
    _prepare_clean_bundle(
        registry,
        logical_request_id="rid-4",
        publish_manifest_digest="manifest-4",
        prompt_token_digest="prompt-digest-4",
        cutoff_token_count=64,
    )
    registry.claim_prepared_bundle(
        logical_request_id="rid-4",
        incoming_prompt_token_digest="prompt-digest-4",
        incoming_cutoff_token_count=64,
        scheduler_rid="other-rid",
        claim_token="claim-4",
        now_ms=120,
    )

    result = PreparedBundleAdmissionBinder(prepared_bundle_registry=registry).bind(
        request=OrdinaryGenerateBindingRequest(
            logical_request_id="rid-4",
            scheduler_rid="sgl-rid-4",
            prompt_token_digest="prompt-digest-4",
            cutoff_token_count=64,
            requested_at_ms=200,
        ),
        attach_prepared_bundle=lambda _: "unused",
    )

    assert result.action == PreparedBundleBindAction.FAIL_CLOSED
    assert "already claimed or attached" in result.reason


def test_binding_fails_closed_when_multiple_clean_generations_exist() -> None:
    registry = PreparedBundleRegistry()
    _prepare_clean_bundle(
        registry,
        logical_request_id="rid-5",
        publish_manifest_digest="manifest-5a",
        prompt_token_digest="prompt-digest-5",
        cutoff_token_count=64,
    )
    registry.evict_bundle(
        logical_request_id="rid-5",
        publish_manifest_digest="manifest-5a",
        now_ms=115,
    )
    _prepare_clean_bundle(
        registry,
        logical_request_id="rid-5",
        publish_manifest_digest="manifest-5b",
        prompt_token_digest="prompt-digest-5",
        cutoff_token_count=64,
    )
    first_generation = registry.get(
        logical_request_id="rid-5",
        publish_manifest_digest="manifest-5a",
    )
    assert first_generation is not None
    registry._records[("rid-5", "manifest-5a")] = first_generation.model_copy(  # noqa: SLF001
        update={"state": PreparedBundleLifecycleState.PREPARED}
    )

    result = PreparedBundleAdmissionBinder(prepared_bundle_registry=registry).bind(
        request=OrdinaryGenerateBindingRequest(
            logical_request_id="rid-5",
            scheduler_rid="sgl-rid-5",
            prompt_token_digest="prompt-digest-5",
            cutoff_token_count=64,
            requested_at_ms=200,
        ),
        attach_prepared_bundle=lambda _: "unused",
    )

    assert result.action == PreparedBundleBindAction.FAIL_CLOSED
    assert "multiple clean prepared bundle generations exist" in result.reason


def test_binding_attach_failure_taints_bundle_and_fails_closed() -> None:
    registry = PreparedBundleRegistry()
    _prepare_clean_bundle(
        registry,
        logical_request_id="rid-6",
        publish_manifest_digest="manifest-6",
        prompt_token_digest="prompt-digest-6",
        cutoff_token_count=64,
    )

    def _raise_attach_failure(_: object) -> str:
        raise RuntimeError("attach failed")

    result = PreparedBundleAdmissionBinder(prepared_bundle_registry=registry).bind(
        request=OrdinaryGenerateBindingRequest(
            logical_request_id="rid-6",
            scheduler_rid="sgl-rid-6",
            prompt_token_digest="prompt-digest-6",
            cutoff_token_count=64,
            requested_at_ms=200,
        ),
        attach_prepared_bundle=_raise_attach_failure,
    )

    assert result.action == PreparedBundleBindAction.FAIL_CLOSED
    assert "failed to attach claimed prepared bundle" in result.reason
    assert result.record is not None
    assert result.record.state == PreparedBundleLifecycleState.PREPARED
    assert result.record.tainted is True
