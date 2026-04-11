# ruff: noqa: E402

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import json

import grpc

from sglang.srt.tensorcast.instance_ops.instance_ops_types import (
    EvictLocalInstanceOpRequest,
    EvictLocalInstanceOpResult,
    HydrateInstanceOpRequest,
    InstanceOpStatus,
    PublishInstanceOpReqInput,
    PublishInstanceOpRespOutput,
)
from sglang.srt.tensorcast.request_bundle.request_bundle_types import (
    EngineOwnedManifestPayload,
    EngineOwnedManifestRecord,
    HydratePreparedResult,
    PageClosureCutoff,
    PreparedBundleLifecycleState,
    PreparedBundleRecord,
    PublishArtifactManifestEntry,
    PublishArtifactManifestRecord,
    PublishCompatibilityEnvelope,
    PublishManifestRecord,
    RankCoord,
    RankInstallRecord,
    RankInstallState,
    RankPublishClosureResult,
    RequestBundleLifecycleState,
    RequestBundleState,
    SourcePublishClosureResult,
)
from sglang.srt.tensorcast.instance_ops.instance_agent import (
    SGLangInstanceOpsEngineAdapterBridge,
    TensorcastInstanceAgent,
    TensorcastInstanceAgentConfig,
    TensorcastInstanceOpsSchedulerRpcClient,
    instance_publish_manifest_record_to_wire_manifest,
    wire_manifest_to_instance_publish_manifest_record,
)
from sglang.srt.tensorcast.instance_ops.instance_agent_service import (
    TensorcastInstanceAgentServiceHandle,
    _instance_agent_service_main,
)
from tensorcast.proto.node_agent.v1 import node_agent_pb2, node_agent_pb2_grpc
from tensorcast.proto.plan.v1 import plan_pb2
from tensorcast.api.context import CallContext


def _rank() -> RankCoord:
    return RankCoord(tp_rank=0, pp_rank=0)


def _sample_publish_manifest_record() -> PublishManifestRecord:
    rank = _rank()
    compatibility = PublishCompatibilityEnvelope(
        model_fingerprint="model-fingerprint",
        kv_layout_id="layout-v1",
        dtype="torch.float16",
        page_size=32,
        tp_size=1,
        pp_size=1,
        attention_arch="mha",
        required_ranks=(rank,),
    )
    engine_owned_payload = EngineOwnedManifestPayload(
        transfer_mode="prefill_closed_prompt_reuse",
        logical_request_id="rid-1",
        cutoff_token_count=64,
        frozen_last_page_index=1,
        tail_valid_tokens=0,
        prompt_token_digest="prompt-digest",
        compatibility=compatibility,
    )
    payload_sha256 = (
        __import__("hashlib")
        .sha256(
            json.dumps(
                engine_owned_payload.model_dump(mode="json", by_alias=True),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        .hexdigest()
    )
    artifact_manifest = PublishArtifactManifestRecord(
        artifact_manifest_digest="artifact-manifest-digest",
        logical_request_id="rid-1",
        snapshot_seq=1,
        required_ranks=(rank,),
        entries=(
            PublishArtifactManifestEntry(
                rank=rank,
                logical_page_index=0,
                page_hash="page-hash-0",
                artifact_id="artifact-0",
            ),
            PublishArtifactManifestEntry(
                rank=rank,
                logical_page_index=1,
                page_hash="page-hash-1",
                artifact_id="artifact-1",
            ),
        ),
    )
    return PublishManifestRecord(
        publish_manifest_digest="publish-manifest-digest",
        artifact_manifest=artifact_manifest,
        engine_owned_manifest=EngineOwnedManifestRecord(
            created_at_ms=100,
            expires_at_ms=60_100,
            artifact_manifest_digest=artifact_manifest.artifact_manifest_digest,
            payload_sha256=payload_sha256,
            payload=engine_owned_payload,
        ),
        prompt_token_digest=engine_owned_payload.prompt_token_digest,
        cutoff_token_count=engine_owned_payload.cutoff_token_count,
        tail_valid_tokens=0,
    )


def _sample_hydrate_result() -> HydratePreparedResult:
    rank = _rank()
    return HydratePreparedResult(
        prepared_bundle=PreparedBundleRecord(
            logical_request_id="rid-1",
            target_instance_id="127.0.0.1:30001",
            publish_manifest_digest="publish-manifest-digest",
            artifact_manifest_digest="artifact-manifest-digest",
            engine_owned_manifest_sha256="engine-owned-manifest-sha256",
            prepared_hold_set_id=None,
            state=PreparedBundleLifecycleState.PREPARED,
            required_ranks=(rank,),
            rank_installs=(
                RankInstallRecord(
                    tp_rank=rank.tp_rank,
                    pp_rank=rank.pp_rank,
                    state=RankInstallState.READY,
                    hydrated_page_count=2,
                    runnable_prefix_tokens=64,
                    local_install_handle="install-handle",
                ),
            ),
            prepared_at_ms=200,
            created_at_ms=150,
            prompt_token_digest="prompt-digest",
            cutoff_token_count=64,
            tail_valid_tokens=0,
        ),
        hold_set=None,
        reused_existing=False,
    )


def _sample_publish_closure_result() -> SourcePublishClosureResult:
    rank = _rank()
    return SourcePublishClosureResult(
        request_state=RequestBundleState(
            logical_request_id="rid-1",
            instance_id="127.0.0.1:30000",
            engine_request_id="rid-1",
            full_prompt_token_count=64,
            model_fingerprint="model-fingerprint",
            kv_layout_id="layout-v1",
            tp_size=1,
            pp_size=1,
            state=RequestBundleLifecycleState.PUBLISHED,
            snapshot_seq=1,
            publish_op_id="publish-op-1",
            frozen_cutoff_token_count=64,
            frozen_last_page_index=1,
            retained_until_ms=60_100,
            bundle_digest="bundle-digest",
            latest_publish_manifest_digest="publish-manifest-digest",
            required_ranks=(rank,),
            created_at_ms=100,
            updated_at_ms=101,
        ),
        cutoff=PageClosureCutoff(
            requested_token_count=64,
            materialized_token_count=64,
            cutoff_token_count=64,
            page_aligned_token_count=64,
            frozen_last_page_index=1,
            tail_valid_tokens=0,
        ),
        publish_manifest=_sample_publish_manifest_record(),
        rank_results=(RankPublishClosureResult(rank=rank, page_outcomes=()),),
    )


class FakeSchedulerRpcClient:
    def __init__(self) -> None:
        self.publish_calls: list[dict[str, object]] = []
        self.hydrate_calls: list[dict[str, object]] = []
        self.evict_calls: list[dict[str, object]] = []
        self.publish_result = type(
            "PublishResultHolder",
            (),
            {},
        )

    def publish_engine_request_id(
        self,
        *,
        engine_request_id: str,
        ttl_ms: int | None,
        timeout_ms: int | None,
        requested_at_ms: int,
        publish_op_id: str,
    ):
        self.publish_calls.append(
            {
                "engine_request_id": engine_request_id,
                "ttl_ms": ttl_ms,
                "timeout_ms": timeout_ms,
                "requested_at_ms": requested_at_ms,
                "publish_op_id": publish_op_id,
            }
        )
        publish_manifest = _sample_publish_manifest_record()
        return type(
            "SourcePublishClosureResultHolder",
            (),
            {
                "publish_manifest": publish_manifest,
                "request_state": type(
                    "RequestStateHolder",
                    (),
                    {"engine_request_id": engine_request_id},
                )(),
            },
        )()

    def hydrate(
        self,
        *,
        request: HydrateInstanceOpRequest,
        publish_manifest: PublishManifestRecord,
    ) -> HydratePreparedResult:
        self.hydrate_calls.append(
            {
                "request": request,
                "publish_manifest": publish_manifest,
            }
        )
        return _sample_hydrate_result()

    def evict_local(
        self,
        *,
        request: EvictLocalInstanceOpRequest,
    ) -> EvictLocalInstanceOpResult:
        self.evict_calls.append({"request": request})
        return EvictLocalInstanceOpResult(
            status=InstanceOpStatus.SUCCESS,
            logical_request_id=request.logical_request_id,
            publish_manifest_digest=request.publish_manifest_digest,
            evicted_bundle_count=1,
            evicted_hold_set_count=0,
        )

    def close(self) -> None:
        return None


def test_instance_agent_publish_manifest_wire_roundtrip() -> None:
    local_publish_manifest = _sample_publish_manifest_record()

    wire_publish_manifest = instance_publish_manifest_record_to_wire_manifest(
        local_publish_manifest,
        engine_request_id="rid-1",
    )
    decoded_engine_request_id, decoded_local_publish_manifest = (
        wire_manifest_to_instance_publish_manifest_record(wire_publish_manifest)
    )

    assert decoded_engine_request_id == "rid-1"
    assert decoded_local_publish_manifest == local_publish_manifest


def test_instance_ops_engine_adapter_bridge_calls_scheduler_rpc() -> None:
    rpc_client = FakeSchedulerRpcClient()
    bridge = SGLangInstanceOpsEngineAdapterBridge(
        scheduler_rpc_client=rpc_client,
    )

    publish_result = bridge.publish("rid-1", 60_000, (), None)
    assert publish_result.publish_manifest is not None
    assert rpc_client.publish_calls[0]["engine_request_id"] == "rid-1"
    assert rpc_client.publish_calls[0]["timeout_ms"] is None

    hydrate_result = bridge.hydrate(
        None,
        publish_result.publish_manifest,
        None,
    )
    assert hydrate_result.manifest is not None
    assert rpc_client.hydrate_calls[0]["request"].logical_request_id == "rid-1"

    evict_result = bridge.evict_local("rid-1", None)
    assert evict_result.engine_request_id == "rid-1"
    assert rpc_client.evict_calls[0]["request"].logical_request_id == "rid-1"


def test_instance_ops_scheduler_client_publish_uses_typed_transport(
    monkeypatch,
) -> None:
    sent_objects: list[object] = []

    class FakeSocket:
        def send_pyobj(self, obj: object) -> None:
            sent_objects.append(obj)

        def recv_pyobj(self):
            return PublishInstanceOpRespOutput(
                status=InstanceOpStatus.SUCCESS,
                result=_sample_publish_closure_result(),
            )

        def close(self, linger: int = 0) -> None:
            _ = linger

    monkeypatch.setattr(
        "sglang.srt.tensorcast.instance_ops.instance_agent.get_zmq_socket",
        lambda _context, _socket_type, _endpoint, _bind: FakeSocket(),
    )

    client = TensorcastInstanceOpsSchedulerRpcClient(
        instance_ops_ipc_name="ipc://tensorcast-instance-ops",
    )
    try:
        result = client.publish_engine_request_id(
            engine_request_id="rid-1",
            ttl_ms=60_000,
            timeout_ms=45_000,
            requested_at_ms=123,
            publish_op_id="publish-op-1",
        )
    finally:
        client.close()

    assert isinstance(sent_objects[0], PublishInstanceOpReqInput)
    assert sent_objects[0].engine_request_id == "rid-1"
    assert sent_objects[0].publish_op_id == "publish-op-1"
    assert sent_objects[0].timeout_ms == 45_000
    assert result.request_state.engine_request_id == "rid-1"


def test_instance_ops_engine_adapter_bridge_propagates_call_deadline() -> None:
    rpc_client = FakeSchedulerRpcClient()
    bridge = SGLangInstanceOpsEngineAdapterBridge(
        scheduler_rpc_client=rpc_client,
    )

    bridge.publish(
        "rid-1",
        60_000,
        (),
        CallContext(deadline_ms=90_000),
    )

    assert rpc_client.publish_calls[0]["timeout_ms"] == 90_000


def test_instance_agent_serves_execute_plan_over_grpc() -> None:
    rpc_client = FakeSchedulerRpcClient()
    agent = TensorcastInstanceAgent(
        TensorcastInstanceAgentConfig(
            daemon_address="127.0.0.1:50052",
            instance_id="127.0.0.1:30000",
            engine="sglang",
            execution_endpoint="127.0.0.1:0",
            instance_ops_ipc_name="ipc://unused",
        ),
        scheduler_rpc_client=rpc_client,
        daemon_id="daemon-1",
    )
    agent.start()
    assert agent.bound_endpoint is not None
    channel = grpc.insecure_channel(agent.bound_endpoint)
    try:
        stub = node_agent_pb2_grpc.NodeAgentServiceStub(channel)
        request = node_agent_pb2.ExecutePlanRequest()
        request.plan.plan_id = "plan-1"
        request.plan.context.request_id = "req-1"

        publish_step = request.plan.steps.add()
        publish_step.step_id = "publish"
        publish_step.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
        publish_step.target.target_id = "127.0.0.1:30000"
        publish_step.action.publish.engine_request_id = "rid-1"
        publish_step.action.publish.ttl_ms = 60_000

        hydrate_step = request.plan.steps.add()
        hydrate_step.step_id = "hydrate"
        hydrate_step.depends_on.append("publish")
        hydrate_step.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
        hydrate_step.target.target_id = "127.0.0.1:30000"
        hydrate_step.action.hydrate.publish_manifest.CopyFrom(
            instance_publish_manifest_record_to_wire_manifest(
                _sample_publish_manifest_record(),
                engine_request_id="rid-1",
            ).to_proto()
        )

        evict_step = request.plan.steps.add()
        evict_step.step_id = "evict"
        evict_step.depends_on.append("hydrate")
        evict_step.target.target_type = plan_pb2.TARGET_TYPE_INSTANCE
        evict_step.target.target_id = "127.0.0.1:30000"
        evict_step.action.evict_local.engine_request_id = "rid-1"

        response = stub.ExecutePlan(request, timeout=5.0)
    finally:
        channel.close()
        agent.stop()

    assert response.ok is True
    assert [step.step_id for step in response.steps] == ["publish", "hydrate", "evict"]
    assert rpc_client.publish_calls[0]["engine_request_id"] == "rid-1"
    assert rpc_client.hydrate_calls[0]["request"].publish_manifest_digest == (
        "publish-manifest-digest"
    )
    assert rpc_client.evict_calls[0]["request"].logical_request_id == "rid-1"


class _FakeRecvConnection:
    def __init__(self, messages: list[object] | None = None) -> None:
        self.messages = list(messages or [])
        self.closed = False

    def poll(self, _timeout: float) -> bool:
        return bool(self.messages)

    def recv(self) -> object:
        return self.messages.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeSendConnection:
    def __init__(
        self,
        peer: _FakeRecvConnection | None = None,
        *,
        on_send=None,
    ) -> None:
        self.peer = peer
        self.on_send = on_send
        self.closed = False
        self.sent: list[object] = []

    def send(self, obj: object) -> None:
        self.sent.append(obj)
        if self.peer is not None:
            self.peer.messages.append(obj)
        if self.on_send is not None:
            self.on_send(obj)

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self) -> None:
        self.started = False
        self.alive = False
        self.terminated = False

    def start(self) -> None:
        self.started = True
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        _ = timeout

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False


class _FakeMultiprocessingContext:
    def __init__(self, startup_message: tuple[object, ...]) -> None:
        self.startup_message = startup_message
        self.process = _FakeProcess()
        self._pipe_count = 0
        self.stop_commands: list[object] = []

    def Pipe(self, duplex: bool = False):  # noqa: N802
        assert duplex is False
        self._pipe_count += 1
        if self._pipe_count == 1:
            recv = _FakeRecvConnection([self.startup_message])
            send = _FakeSendConnection()
            return recv, send
        recv = _FakeRecvConnection()
        send = _FakeSendConnection(
            on_send=lambda obj: (
                self.stop_commands.append(obj),
                setattr(self.process, "alive", False),
            ),
        )
        return recv, send

    def Process(self, **kwargs):  # noqa: N802
        _ = kwargs
        return self.process


def test_instance_agent_service_handle_starts_and_stops() -> None:
    config = TensorcastInstanceAgentConfig(
        daemon_address="127.0.0.1:50052",
        instance_id="127.0.0.1:30000",
        engine="sglang",
        execution_endpoint="127.0.0.1:34110",
        instance_ops_ipc_name="ipc://tensorcast-instance-ops",
    )
    fake_context = _FakeMultiprocessingContext(("ready", "127.0.0.1:34110"))
    handle = TensorcastInstanceAgentServiceHandle(
        config,
        _context_factory=lambda _name: fake_context,
    )

    endpoint = handle.start()

    assert endpoint == "127.0.0.1:34110"
    assert handle.bound_endpoint == "127.0.0.1:34110"
    assert fake_context.process.started is True

    handle.stop()

    assert fake_context.stop_commands == ["stop"]
    assert fake_context.process.is_alive() is False


def test_instance_agent_service_handle_surfaces_startup_error() -> None:
    config = TensorcastInstanceAgentConfig(
        daemon_address="127.0.0.1:50052",
        instance_id="127.0.0.1:30000",
        engine="sglang",
        execution_endpoint="127.0.0.1:34110",
        instance_ops_ipc_name="ipc://tensorcast-instance-ops",
    )
    fake_context = _FakeMultiprocessingContext(("error", "boom", "traceback"))
    handle = TensorcastInstanceAgentServiceHandle(
        config,
        _context_factory=lambda _name: fake_context,
    )

    try:
        handle.start()
        raise AssertionError("expected start() to fail")
    except RuntimeError as exc:
        assert "boom" in str(exc)

    assert fake_context.process.is_alive() is False


def test_instance_agent_service_main_starts_and_stops_agent(monkeypatch) -> None:
    events: list[object] = []

    class FakeAgent:
        def __init__(self, config: TensorcastInstanceAgentConfig) -> None:
            self.bound_endpoint = config.execution_endpoint
            events.append(("init", config.instance_id))

        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    monkeypatch.setattr(
        "sglang.srt.tensorcast.instance_ops.instance_agent_service.TensorcastInstanceAgent",
        FakeAgent,
    )
    status_send = _FakeSendConnection()
    stop_recv = _FakeRecvConnection(["stop"])
    config = TensorcastInstanceAgentConfig(
        daemon_address="127.0.0.1:50052",
        instance_id="127.0.0.1:30000",
        engine="sglang",
        execution_endpoint="127.0.0.1:34110",
        instance_ops_ipc_name="ipc://tensorcast-instance-ops",
    )

    _instance_agent_service_main(config, status_send, stop_recv)

    assert status_send.sent == [("ready", "127.0.0.1:34110")]
    assert events == [("init", "127.0.0.1:30000"), "start", "stop"]
