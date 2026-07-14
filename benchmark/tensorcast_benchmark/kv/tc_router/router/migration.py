"""Tensorcast publish/hydrate migration helper for tc_router."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


_INSTANCE_AGENT_EMBEDDED_MANIFEST_SCHEMA = (
    "sglang.request_bundle.publish_manifest_record.v1"
)


class _EmbeddedArtifactManifestEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifact_id: str = Field(min_length=1)


class _EmbeddedArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    artifact_manifest_digest: str = Field(min_length=1)
    entries: tuple[_EmbeddedArtifactManifestEntry, ...] = ()


class _EmbeddedPublishManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    publish_manifest_digest: str = Field(min_length=1)
    artifact_manifest: _EmbeddedArtifactManifest
    cutoff_token_count: int = Field(ge=0)
    tail_valid_tokens: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class MigrationResult:
    publish_latency_ms: float
    hydrate_latency_ms: float
    publish_manifest_digest: str
    artifact_manifest_digest: str
    published_cutoff_token_count: int | None
    tail_valid_tokens: int | None


def _sha256_hexdigest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decode_sglang_publish_manifest(publish_manifest) -> _EmbeddedPublishManifest:
    engine_owned_manifest = publish_manifest.engine_owned_manifest
    if engine_owned_manifest.schema != _INSTANCE_AGENT_EMBEDDED_MANIFEST_SCHEMA:
        raise RuntimeError(
            "unsupported SGLang instance publish manifest schema: "
            f"{engine_owned_manifest.schema}"
        )
    payload_bytes = bytes(engine_owned_manifest.payload)
    payload_sha256 = engine_owned_manifest.payload_sha256
    if payload_sha256 is not None and payload_sha256 != _sha256_hexdigest(
        payload_bytes
    ):
        raise RuntimeError("instance publish manifest payload digest mismatch")
    try:
        decoded_payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "instance publish manifest payload is not valid JSON"
        ) from exc
    if decoded_payload.get("schema") != _INSTANCE_AGENT_EMBEDDED_MANIFEST_SCHEMA:
        raise RuntimeError("instance publish manifest payload schema mismatch")
    local_publish_manifest = _EmbeddedPublishManifest.model_validate(
        decoded_payload.get("publish_manifest", {})
    )
    artifact_manifest_digest = (
        local_publish_manifest.artifact_manifest.artifact_manifest_digest
    )
    if (
        artifact_manifest_digest
        != publish_manifest.artifact_manifest.key_set_digest_hex
    ):
        raise RuntimeError("instance publish manifest artifact digest mismatch")
    artifact_ids = tuple(
        entry.artifact_id for entry in local_publish_manifest.artifact_manifest.entries
    )
    if artifact_ids != tuple(publish_manifest.artifact_manifest.artifact_ids):
        raise RuntimeError("instance publish manifest artifact ordering mismatch")
    return local_publish_manifest


class TensorcastMigrationClient:
    """Thin synchronous wrapper around Tensorcast request-transfer plans."""

    def __init__(self, runtime) -> None:
        self._runtime = runtime

    def migrate(
        self,
        *,
        migration_id: str,
        source_instance_id: str,
        target_instance_id: str,
        engine_request_id: str,
        deadline_ms: int,
        publish_ttl_ms: int,
    ) -> MigrationResult:
        source_instance = self._resolve_instance(source_instance_id)
        target_instance = self._resolve_instance(target_instance_id)

        publish_start = time.perf_counter()
        publish_result = self._publish(
            migration_id=migration_id,
            source_instance=source_instance,
            engine_request_id=engine_request_id,
            deadline_ms=deadline_ms,
            publish_ttl_ms=publish_ttl_ms,
        )
        publish_latency_ms = (time.perf_counter() - publish_start) * 1000.0
        publish_manifest = publish_result.publish_manifest
        if publish_manifest is None:
            raise RuntimeError("publish step did not return publish_manifest")
        local_publish_manifest = _decode_sglang_publish_manifest(publish_manifest)

        hydrate_start = time.perf_counter()
        self._hydrate(
            migration_id=migration_id,
            target_instance=target_instance,
            publish_manifest=publish_manifest,
            deadline_ms=deadline_ms,
        )
        hydrate_latency_ms = (time.perf_counter() - hydrate_start) * 1000.0

        return MigrationResult(
            publish_latency_ms=publish_latency_ms,
            hydrate_latency_ms=hydrate_latency_ms,
            publish_manifest_digest=local_publish_manifest.publish_manifest_digest,
            artifact_manifest_digest=(
                local_publish_manifest.artifact_manifest.artifact_manifest_digest
            ),
            published_cutoff_token_count=local_publish_manifest.cutoff_token_count,
            tail_valid_tokens=local_publish_manifest.tail_valid_tokens,
        )

    def _resolve_instance(self, instance_id: str):
        from tensorcast.api.plan import Instance

        route = self._runtime.directory().resolve_instance_execution(instance_id).value
        return Instance(
            instance_id=route.instance_id,
            daemon_id=route.daemon_id,
            engine=route.engine or "sglang",
            execution_endpoint=route.execution_endpoint,
        )

    def _publish(
        self,
        *,
        migration_id: str,
        source_instance,
        engine_request_id: str,
        deadline_ms: int,
        publish_ttl_ms: int,
    ):
        from tensorcast.api.context import CallContext
        from tensorcast.engine_adapter.artifact_api import PublishResult

        context = CallContext(
            request_id=f"migrate-publish:{migration_id}",
            deadline_ms=deadline_ms,
            idempotency_key=f"migrate-publish:{migration_id}",
        )
        plan = self._runtime.plan(context)
        publish_ref = plan.on_instance(source_instance).publish(
            engine_request_id=engine_request_id,
            ttl_ms=publish_ttl_ms,
        )
        result = plan.run()
        artifact_result = result.step(publish_ref).artifact_result
        if not isinstance(artifact_result, PublishResult):
            raise RuntimeError("publish step did not return PublishResult")
        return artifact_result

    def _hydrate(
        self,
        *,
        migration_id: str,
        target_instance,
        publish_manifest,
        deadline_ms: int,
    ) -> None:
        from tensorcast.api.context import CallContext
        from tensorcast.engine_adapter.artifact_api import HydrateResult

        context = CallContext(
            request_id=f"migrate-hydrate:{migration_id}",
            deadline_ms=deadline_ms,
            idempotency_key=f"migrate-hydrate:{migration_id}",
        )
        plan = self._runtime.plan(context)
        hydrate_ref = plan.on_instance(target_instance).hydrate(
            publish_manifest=publish_manifest,
        )
        result = plan.run()
        artifact_result = result.step(hydrate_ref).artifact_result
        if not isinstance(artifact_result, HydrateResult):
            raise RuntimeError("hydrate step did not return HydrateResult")
