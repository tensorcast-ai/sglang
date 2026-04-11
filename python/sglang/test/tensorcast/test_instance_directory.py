# ruff: noqa: E402

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import time

from sglang.test.tensorcast.test_support import install_memory_pool_host_stub

install_memory_pool_host_stub()

from sglang.srt.tensorcast.instance_ops.instance_directory import (
    TensorcastInstanceDirectoryConfig,
    TensorcastInstanceDirectoryRegistration,
    resolve_instance_agent_execution_endpoint,
)
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2


class FakeGlobalStoreStub:
    def __init__(self, *, heartbeat_interval_ms: int = 5) -> None:
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self.register_requests: list[global_store_pb2.RegisterInstanceRequest] = []
        self.heartbeat_requests: list[global_store_pb2.InstanceHeartbeatRequest] = []
        self.unregister_requests: list[global_store_pb2.UnregisterInstanceRequest] = []

    def RegisterInstance(
        self,
        request: global_store_pb2.RegisterInstanceRequest,
    ) -> global_store_pb2.RegisterInstanceResponse:
        self.register_requests.append(request)
        return global_store_pb2.RegisterInstanceResponse(
            status=global_store_pb2.Status.STATUS_OK,
            instance_id=request.instance_id,
            heartbeat_interval_ms=self.heartbeat_interval_ms,
        )

    def InstanceHeartbeat(
        self,
        request: global_store_pb2.InstanceHeartbeatRequest,
    ) -> global_store_pb2.InstanceHeartbeatResponse:
        self.heartbeat_requests.append(request)
        return global_store_pb2.InstanceHeartbeatResponse(
            status=global_store_pb2.Status.STATUS_OK
        )

    def UnregisterInstance(
        self,
        request: global_store_pb2.UnregisterInstanceRequest,
    ) -> global_store_pb2.UnregisterInstanceResponse:
        self.unregister_requests.append(request)
        return global_store_pb2.UnregisterInstanceResponse(
            status=global_store_pb2.Status.STATUS_OK
        )


class FakeGlobalStoreHandle:
    def __init__(self, stub: FakeGlobalStoreStub) -> None:
        self.stub = stub
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeDaemonClient:
    def __init__(self) -> None:
        self.closed = False

    def get_worker_status(self) -> store_daemon_pb2.GetWorkerStatusResponse:
        return store_daemon_pb2.GetWorkerStatusResponse(
            daemon_id="daemon-a",
            worker_id="worker-a",
        )

    def close(self) -> None:
        self.closed = True


def _wait_for(predicate, *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for condition")


def test_instance_directory_registration_registers_heartbeats_and_unregisters() -> None:
    stub = FakeGlobalStoreStub()
    stub_handle = FakeGlobalStoreHandle(stub)
    daemon_client = FakeDaemonClient()
    registration = TensorcastInstanceDirectoryRegistration(
        TensorcastInstanceDirectoryConfig(
            global_store_address="127.0.0.1:50051",
            daemon_address="127.0.0.1:50052",
            instance_id="127.0.0.1:30000",
            engine="sglang",
            execution_endpoint="10.0.0.1:30000",
            heartbeat_interval_ms=50,
        ),
        global_store_stub_factory=lambda _address: stub_handle,
        daemon_client_factory=lambda _address: daemon_client,
    )

    registration.start()
    _wait_for(lambda: len(stub.heartbeat_requests) >= 1)
    registration.stop()

    assert len(stub.register_requests) == 1
    register_request = stub.register_requests[0]
    assert register_request.instance_id == "127.0.0.1:30000"
    assert register_request.daemon_id == "daemon-a"
    assert register_request.worker_id == "worker-a"
    assert register_request.execution_endpoint == "10.0.0.1:30000"
    assert register_request.execution_host_kind == "node_agent_grpc"
    assert len(stub.heartbeat_requests) >= 1
    assert all(
        heartbeat.instance_id == "127.0.0.1:30000"
        and heartbeat.worker_id == "worker-a"
        for heartbeat in stub.heartbeat_requests
    )
    assert len(stub.unregister_requests) == 1
    assert stub.unregister_requests[0].is_graceful_shutdown
    assert daemon_client.closed
    assert stub_handle.closed


def test_resolve_instance_agent_execution_endpoint_requires_explicit_endpoint_for_wildcard_host() -> None:
    assert (
        resolve_instance_agent_execution_endpoint(
            listen_host="0.0.0.0",
            listen_port=30000,
            configured_endpoint="",
        )
        is None
    )
    assert (
        resolve_instance_agent_execution_endpoint(
            listen_host="0.0.0.0",
            listen_port=30000,
            configured_endpoint="10.0.0.1:30000",
        )
        == "10.0.0.1:30000"
    )
    assert (
        resolve_instance_agent_execution_endpoint(
            listen_host="127.0.0.1",
            listen_port=30000,
            configured_endpoint="",
        )
        == "127.0.0.1:30000"
    )
