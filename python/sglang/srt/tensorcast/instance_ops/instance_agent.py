# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass
import hashlib
import json
import logging
import time
from typing import Protocol

import grpc
import zmq

from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    EvictLocalInstanceOpReqInput,
    EvictLocalInstanceOpRequest,
    EvictLocalInstanceOpResult,
    EvictLocalInstanceOpRespOutput,
    HydrateInstanceOpReqInput,
    HydrateInstanceOpRequest,
    HydrateInstanceOpRespOutput,
    InstanceOpStatus,
    PublishInstanceOpReqInput,
    PublishInstanceOpRespOutput,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    HydratePreparedResult,
    PublishManifestRecord,
    SourcePublishClosureResult,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import get_zmq_socket
from tensorcast.api.context import CallContext
from tensorcast.api.errors import ArtifactError
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.engine_adapter import (
    BatchOutcome,
    BatchResult,
    EngineAdapter,
    EngineOwnedManifest,
    HydrateResult,
    ManifestResult,
    PublishManifest,
    PublishResult,
    SealedByteArtifact,
)
from tensorcast.node_agent.executor import NodeAgentExecutor
from tensorcast.node_agent.server import add_servicer_to_server

logger = logging.getLogger(__name__)

INSTANCE_AGENT_EMBEDDED_MANIFEST_SCHEMA = (
    "sglang.request_bundle.publish_manifest_record.v1"
)
INSTANCE_AGENT_ARTIFACT_MANIFEST_ALG = "sglang.request_bundle.artifact_manifest.v1"


class SchedulerRpcClientProtocol(Protocol):
    def publish_engine_request_id(
        self,
        *,
        engine_request_id: str,
        ttl_ms: int | None,
        timeout_ms: int | None,
        requested_at_ms: int,
        publish_op_id: str,
    ) -> SourcePublishClosureResult: ...

    def hydrate(
        self,
        *,
        request: HydrateInstanceOpRequest,
        publish_manifest: PublishManifestRecord,
    ) -> HydratePreparedResult: ...

    def evict_local(
        self,
        *,
        request: EvictLocalInstanceOpRequest,
    ) -> EvictLocalInstanceOpResult: ...

    def close(self) -> None: ...


class NodeAgentClientFactoryProtocol(Protocol):
    def __call__(self, daemon_address: str) -> object: ...


@dataclass(frozen=True, slots=True)
class TensorcastInstanceAgentConfig:
    daemon_address: str
    instance_id: str
    engine: str
    execution_endpoint: str
    instance_ops_ipc_name: str


def _json_dumps_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_hexdigest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _replace_endpoint_port(endpoint: str, port: int) -> str:
    host, separator, _ = endpoint.rpartition(":")
    if not separator or not host:
        return endpoint
    return f"{host}:{int(port)}"


def _artifact_manifest_from_record(
    publish_manifest: PublishManifestRecord,
    *,
    engine_request_id: str,
) -> ManifestResult:
    compatibility = publish_manifest.engine_owned_manifest.payload.compatibility
    artifact_ids = tuple(
        entry.artifact_id for entry in publish_manifest.artifact_manifest.entries
    )
    return ManifestResult(
        engine_request_id=engine_request_id,
        layout_id=compatibility.kv_layout_id,
        artifact_ids=artifact_ids,
        key_set_digest_alg=INSTANCE_AGENT_ARTIFACT_MANIFEST_ALG,
        key_set_digest_hex=publish_manifest.artifact_manifest.artifact_manifest_digest,
    )


def instance_publish_manifest_record_to_wire_manifest(
    publish_manifest: PublishManifestRecord,
    *,
    engine_request_id: str,
) -> PublishManifest:
    artifact_manifest = _artifact_manifest_from_record(
        publish_manifest,
        engine_request_id=engine_request_id,
    )
    embedded_payload = {
        "schema": INSTANCE_AGENT_EMBEDDED_MANIFEST_SCHEMA,
        "engine_request_id": engine_request_id,
        "publish_manifest": publish_manifest.model_dump(mode="json", by_alias=True),
    }
    payload_bytes = _json_dumps_bytes(embedded_payload)
    engine_owned_manifest = EngineOwnedManifest(
        engine=publish_manifest.engine_owned_manifest.engine,
        schema=INSTANCE_AGENT_EMBEDDED_MANIFEST_SCHEMA,
        version=publish_manifest.engine_owned_manifest.version,
        encoding="json",
        created_at_ms=publish_manifest.engine_owned_manifest.created_at_ms,
        expires_at_ms=publish_manifest.engine_owned_manifest.expires_at_ms,
        artifact_manifest_digest=artifact_manifest.key_set_digest_hex,
        payload_sha256=_sha256_hexdigest(payload_bytes),
        payload=payload_bytes,
    )
    return PublishManifest(
        schema=publish_manifest.schema_name,
        artifact_manifest=artifact_manifest,
        engine_owned_manifest=engine_owned_manifest,
    )


def wire_manifest_to_instance_publish_manifest_record(
    publish_manifest: PublishManifest,
) -> tuple[str, PublishManifestRecord]:
    if (
        publish_manifest.engine_owned_manifest.schema
        != INSTANCE_AGENT_EMBEDDED_MANIFEST_SCHEMA
    ):
        raise ArtifactError(
            "unsupported SGLang instance publish manifest schema",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    payload_bytes = bytes(publish_manifest.engine_owned_manifest.payload)
    payload_sha256 = publish_manifest.engine_owned_manifest.payload_sha256
    if payload_sha256 is not None and payload_sha256 != _sha256_hexdigest(
        payload_bytes
    ):
        raise ArtifactError(
            "instance publish manifest payload digest mismatch",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    try:
        decoded_payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            "instance publish manifest payload is not valid JSON",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        ) from exc
    if decoded_payload.get("schema") != INSTANCE_AGENT_EMBEDDED_MANIFEST_SCHEMA:
        raise ArtifactError(
            "instance publish manifest payload schema mismatch",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    engine_request_id = str(decoded_payload.get("engine_request_id", "")).strip()
    if not engine_request_id:
        raise ArtifactError(
            "instance publish manifest is missing engine_request_id",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    local_publish_manifest = PublishManifestRecord.model_validate(
        decoded_payload.get("publish_manifest", {})
    )
    if (
        local_publish_manifest.artifact_manifest.artifact_manifest_digest
        != publish_manifest.artifact_manifest.key_set_digest_hex
    ):
        raise ArtifactError(
            "instance publish manifest artifact digest mismatch",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    artifact_ids = tuple(
        entry.artifact_id for entry in local_publish_manifest.artifact_manifest.entries
    )
    if artifact_ids != tuple(publish_manifest.artifact_manifest.artifact_ids):
        raise ArtifactError(
            "instance publish manifest artifact ordering mismatch",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    return engine_request_id, local_publish_manifest


class TensorcastInstanceOpsSchedulerRpcClient:
    def __init__(self, *, instance_ops_ipc_name: str) -> None:
        self._instance_ops_ipc_name = instance_ops_ipc_name
        self._context = zmq.Context()

    def _call(self, request: object) -> object:
        socket = get_zmq_socket(
            self._context,
            zmq.DEALER,
            self._instance_ops_ipc_name,
            False,
        )
        assert not isinstance(socket, tuple)
        try:
            socket.send_pyobj(request)
            response = socket.recv_pyobj()
        finally:
            socket.close(linger=0)
        return response

    def publish_engine_request_id(
        self,
        *,
        engine_request_id: str,
        ttl_ms: int | None,
        timeout_ms: int | None,
        requested_at_ms: int,
        publish_op_id: str,
    ) -> SourcePublishClosureResult:
        response = self._call(
            PublishInstanceOpReqInput(
                engine_request_id=engine_request_id,
                ttl_ms=ttl_ms,
                timeout_ms=timeout_ms,
                requested_at_ms=requested_at_ms,
                publish_op_id=publish_op_id,
            )
        )
        if not isinstance(response, PublishInstanceOpRespOutput):
            raise RuntimeError(
                "unexpected scheduler publish response type: "
                f"{type(response).__name__}"
            )
        if response.status != InstanceOpStatus.SUCCESS:
            raise RuntimeError(
                response.error_message or "scheduler publish returned failure"
            )
        assert response.result is not None
        return response.result

    def hydrate(
        self,
        *,
        request: HydrateInstanceOpRequest,
        publish_manifest: PublishManifestRecord,
    ) -> HydratePreparedResult:
        response = self._call(
            HydrateInstanceOpReqInput(
                request=request,
                publish_manifest=publish_manifest,
            )
        )
        if not isinstance(response, HydrateInstanceOpRespOutput):
            raise RuntimeError(
                "unexpected scheduler hydrate response type: "
                f"{type(response).__name__}"
            )
        if response.status != InstanceOpStatus.SUCCESS:
            raise RuntimeError(
                response.error_message or "scheduler hydrate returned failure"
            )
        assert response.result is not None
        return response.result

    def evict_local(
        self,
        *,
        request: EvictLocalInstanceOpRequest,
    ) -> EvictLocalInstanceOpResult:
        response = self._call(
            EvictLocalInstanceOpReqInput(request=request)
        )
        if not isinstance(response, EvictLocalInstanceOpRespOutput):
            raise RuntimeError(
                "unexpected scheduler evict_local response type: "
                f"{type(response).__name__}"
            )
        if response.status != InstanceOpStatus.SUCCESS:
            raise RuntimeError(
                response.error_message or "scheduler evict_local returned failure"
            )
        assert response.result is not None
        return response.result

    def close(self) -> None:
        self._context.term()


class SGLangInstanceOpsEngineAdapterBridge:
    def __init__(
        self,
        *,
        scheduler_rpc_client: SchedulerRpcClientProtocol,
    ) -> None:
        self._scheduler_rpc_client = scheduler_rpc_client

    def publish(
        self,
        engine_request_id: str,
        ttl_ms: int | None,
        sealed_artifacts: tuple[SealedByteArtifact, ...],
        ctx: CallContext | None,
    ) -> PublishResult:
        if sealed_artifacts:
            raise ArtifactError(
                "SGLang instance publish does not accept caller-supplied sealed artifacts",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        requested_at_ms = int(time.time() * 1000)
        publish_op_id = self._publish_op_id(
            engine_request_id=engine_request_id,
            requested_at_ms=requested_at_ms,
            ctx=ctx,
        )
        publish_result = self._scheduler_rpc_client.publish_engine_request_id(
            engine_request_id=engine_request_id,
            ttl_ms=ttl_ms,
            timeout_ms=ctx.deadline_ms if ctx is not None else None,
            requested_at_ms=requested_at_ms,
            publish_op_id=publish_op_id,
        )
        logger.debug(
            "Tensorcast instance-agent publish completed engine_request_id=%s publish_op_id=%s manifest=%s",
            engine_request_id,
            publish_op_id,
            publish_result.publish_manifest.publish_manifest_digest,
        )
        wire_manifest = instance_publish_manifest_record_to_wire_manifest(
            publish_result.publish_manifest,
            engine_request_id=publish_result.request_state.engine_request_id,
        )
        put_outcomes = tuple(
            BatchOutcome(
                artifact_id=entry.artifact_id,
                status_code="OK",
            )
            for entry in publish_result.publish_manifest.artifact_manifest.entries
        )
        return PublishResult(
            manifest=wire_manifest.artifact_manifest,
            put_outcomes=put_outcomes,
            publish_manifest=wire_manifest,
        )

    def hydrate(
        self,
        engine_request_id: str | None,
        publish_manifest: PublishManifest | None,
        ctx: CallContext | None,
    ) -> HydrateResult:
        if publish_manifest is None:
            raise ArtifactError(
                "SGLang instance hydrate requires an explicit publish_manifest",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        _ = engine_request_id
        requested_at_ms = int(time.time() * 1000)
        decoded_engine_request_id, local_publish_manifest = (
            wire_manifest_to_instance_publish_manifest_record(publish_manifest)
        )
        hydrate_request = HydrateInstanceOpRequest(
            logical_request_id=local_publish_manifest.engine_owned_manifest.payload.logical_request_id,
            publish_manifest_digest=local_publish_manifest.publish_manifest_digest,
            requested_at_ms=requested_at_ms,
        )
        logger.debug(
            "Tensorcast instance-agent hydrate dispatch logical_request_id=%s manifest=%s deadline_ms=%s",
            hydrate_request.logical_request_id,
            hydrate_request.publish_manifest_digest,
            ctx.deadline_ms if ctx is not None else None,
        )
        self._scheduler_rpc_client.hydrate(
            request=hydrate_request,
            publish_manifest=local_publish_manifest,
        )
        logger.debug(
            "Tensorcast instance-agent hydrate completed logical_request_id=%s manifest=%s",
            hydrate_request.logical_request_id,
            hydrate_request.publish_manifest_digest,
        )
        get_outcomes = tuple(
            BatchOutcome(
                artifact_id=entry.artifact_id,
                status_code="OK",
            )
            for entry in local_publish_manifest.artifact_manifest.entries
        )
        return HydrateResult(
            manifest=_artifact_manifest_from_record(
                local_publish_manifest,
                engine_request_id=decoded_engine_request_id,
            ),
            get_outcomes=get_outcomes,
            missing_artifact_ids=(),
        )

    def evict_local(
        self,
        engine_request_id: str | None,
        ctx: CallContext | None,
    ) -> BatchResult:
        del ctx
        normalized_engine_request_id = (
            str(engine_request_id).strip() if engine_request_id is not None else ""
        )
        if not normalized_engine_request_id:
            raise ArtifactError(
                "SGLang instance evict_local requires engine_request_id",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        evict_request = EvictLocalInstanceOpRequest(
            logical_request_id=normalized_engine_request_id,
            requested_at_ms=int(time.time() * 1000),
        )
        self._scheduler_rpc_client.evict_local(request=evict_request)
        return BatchResult(
            engine_request_id=normalized_engine_request_id,
            outcomes=(),
        )

    @staticmethod
    def _publish_op_id(
        *,
        engine_request_id: str,
        requested_at_ms: int,
        ctx: CallContext | None,
    ) -> str:
        if ctx is not None and ctx.idempotency_key:
            return str(ctx.idempotency_key)
        if ctx is not None and ctx.request_id:
            return f"{ctx.request_id}:publish:{engine_request_id}"
        return f"publish::{engine_request_id}::{requested_at_ms}"


class _NoopNodeAgentClient:
    pass


class TensorcastInstanceAgent:
    def __init__(
        self,
        config: TensorcastInstanceAgentConfig,
        *,
        scheduler_rpc_client: SchedulerRpcClientProtocol | None = None,
        daemon_id: str | None = None,
        node_agent_client_factory: NodeAgentClientFactoryProtocol | None = None,
        version: str = "sglang",
    ) -> None:
        self._config = config
        self._scheduler_rpc_client = scheduler_rpc_client or (
            TensorcastInstanceOpsSchedulerRpcClient(
                instance_ops_ipc_name=config.instance_ops_ipc_name,
            )
        )
        self._owned_scheduler_rpc_client = scheduler_rpc_client is None
        self._daemon_id = daemon_id or self._resolve_daemon_id(config.daemon_address)
        self._node_agent_client_factory = node_agent_client_factory or (
            lambda _daemon_address: _NoopNodeAgentClient()
        )
        bridge = SGLangInstanceOpsEngineAdapterBridge(
            scheduler_rpc_client=self._scheduler_rpc_client,
        )
        adapter = EngineAdapter(
            instance_id=config.instance_id,
            engine=config.engine,
            register_identity_transform=False,
        )
        adapter.register_artifact_fns(
            publish=bridge.publish,
            hydrate=bridge.hydrate,
            evict_local=bridge.evict_local,
        )
        self._executor = NodeAgentExecutor(
            daemon_id=self._daemon_id,
            daemon_address=config.daemon_address,
            instance_id=config.instance_id,
            version=version,
            engine_adapter=adapter,
            client_factory=self._node_agent_client_factory,
        )
        self._server: grpc.Server | None = None
        self._bound_endpoint: str | None = None

    @staticmethod
    def _resolve_daemon_id(daemon_address: str) -> str:
        daemon_client = DaemonCtl(daemon_address)
        try:
            status = daemon_client.get_worker_status()
        finally:
            daemon_client.close()
        daemon_id = str(status.daemon_id).strip()
        if not daemon_id:
            raise RuntimeError("Tensorcast instance-agent requires daemon_id")
        return daemon_id

    def start(self) -> None:
        if self._server is not None:
            return
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        add_servicer_to_server(server, self._executor)
        bound_port = server.add_insecure_port(self._config.execution_endpoint)
        if bound_port == 0:
            raise RuntimeError(
                "failed to bind Tensorcast instance-agent on "
                f"{self._config.execution_endpoint}"
            )
        server.start()
        self._server = server
        self._bound_endpoint = _replace_endpoint_port(
            self._config.execution_endpoint,
            bound_port,
        )
        logger.info(
            "Tensorcast instance-agent started instance_id=%s endpoint=%s",
            self._config.instance_id,
            self._bound_endpoint,
        )

    def stop(self) -> None:
        server = self._server
        self._server = None
        self._bound_endpoint = None
        if server is not None:
            server.stop(grace=1.0).wait()
        if self._owned_scheduler_rpc_client:
            self._scheduler_rpc_client.close()

    @property
    def bound_endpoint(self) -> str | None:
        return self._bound_endpoint


def maybe_build_tensorcast_instance_agent_config(
    *,
    server_args: ServerArgs,
    instance_ops_ipc_name: str | None,
) -> TensorcastInstanceAgentConfig | None:
    if server_args.node_rank != 0:
        return None
    if server_args.hicache_storage_backend != "tensorcast":
        return None
    if instance_ops_ipc_name is None or str(instance_ops_ipc_name).strip() == "":
        return None
    raw_extra_config = server_args.hicache_storage_backend_extra_config
    if not raw_extra_config:
        return None
    payload = (
        json.loads(raw_extra_config)
        if isinstance(raw_extra_config, str)
        else dict(raw_extra_config)
    )
    execution_endpoint = str(
        payload.get("instance_agent_execution_endpoint", "")
    ).strip()
    if not execution_endpoint:
        return None
    daemon_address = str(payload.get("daemon_address", "")).strip()
    if not daemon_address:
        logger.warning(
            "Tensorcast instance-agent skipped because daemon_address is missing from hicache_storage_backend_extra_config"
        )
        return None
    instance_id = f"{server_args.host}:{server_args.port}"
    engine = str(payload.get("engine", "sglang")).strip() or "sglang"
    return TensorcastInstanceAgentConfig(
        daemon_address=daemon_address,
        instance_id=instance_id,
        engine=engine,
        execution_endpoint=execution_endpoint,
        instance_ops_ipc_name=str(instance_ops_ipc_name),
    )


__all__ = [
    "INSTANCE_AGENT_ARTIFACT_MANIFEST_ALG",
    "INSTANCE_AGENT_EMBEDDED_MANIFEST_SCHEMA",
    "SGLangInstanceOpsEngineAdapterBridge",
    "TensorcastInstanceAgent",
    "TensorcastInstanceAgentConfig",
    "TensorcastInstanceOpsSchedulerRpcClient",
    "instance_publish_manifest_record_to_wire_manifest",
    "maybe_build_tensorcast_instance_agent_config",
    "wire_manifest_to_instance_publish_manifest_record",
]
