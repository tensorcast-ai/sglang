# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import logging
import threading
from typing import Protocol

import grpc

from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.global_store.composite_stub import GlobalStoreCompositeStub
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2

logger = logging.getLogger(__name__)

INSTANCE_AGENT_EXECUTION_HOST_KIND = "node_agent_grpc"
_WILDCARD_LISTEN_HOSTS = {"", "0.0.0.0", "::", "[::]"}


class GlobalStoreStubProtocol(Protocol):
    def RegisterInstance(
        self,
        request: global_store_pb2.RegisterInstanceRequest,
    ) -> global_store_pb2.RegisterInstanceResponse: ...

    def InstanceHeartbeat(
        self,
        request: global_store_pb2.InstanceHeartbeatRequest,
    ) -> global_store_pb2.InstanceHeartbeatResponse: ...

    def UnregisterInstance(
        self,
        request: global_store_pb2.UnregisterInstanceRequest,
    ) -> global_store_pb2.UnregisterInstanceResponse: ...


class DaemonStatusClientProtocol(Protocol):
    def get_worker_status(self) -> store_daemon_pb2.GetWorkerStatusResponse: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class TensorcastInstanceDirectoryConfig:
    global_store_address: str
    daemon_address: str
    instance_id: str
    engine: str
    execution_endpoint: str
    heartbeat_interval_ms: int = 10_000
    signals_endpoint: str = ""
    execution_host_kind: str = INSTANCE_AGENT_EXECUTION_HOST_KIND
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _GlobalStoreStubHandle:
    stub: GlobalStoreStubProtocol
    close: Callable[[], None]


@dataclass(frozen=True)
class _DaemonDirectoryIdentity:
    daemon_id: str
    worker_id: str | None


def resolve_instance_agent_execution_endpoint(
    *,
    listen_host: str,
    listen_port: int,
    configured_endpoint: str,
) -> str | None:
    explicit_endpoint = str(configured_endpoint).strip()
    if explicit_endpoint:
        return explicit_endpoint
    resolved_host = str(listen_host).strip()
    if resolved_host in _WILDCARD_LISTEN_HOSTS:
        return None
    return f"{resolved_host}:{int(listen_port)}"


def _capability_flag(flag: int) -> int:
    return 1 << int(flag)


def _build_capability_flags(*, signals_endpoint: str) -> int:
    capability_flags = _capability_flag(
        global_store_pb2.INSTANCE_CAPABILITY_FLAG_NODE_AGENT_ENABLED
    )
    if str(signals_endpoint).strip():
        capability_flags |= _capability_flag(
            global_store_pb2.INSTANCE_CAPABILITY_FLAG_EXECUTION_SIGNALS_ENABLED
        )
    return capability_flags


def _resolve_daemon_directory_identity(
    daemon_client: DaemonStatusClientProtocol,
) -> _DaemonDirectoryIdentity:
    status = daemon_client.get_worker_status()
    daemon_id = str(status.daemon_id).strip()
    if not daemon_id:
        raise RuntimeError("Tensorcast daemon directory registration requires daemon_id")
    worker_id = str(status.worker_id).strip() or None
    return _DaemonDirectoryIdentity(daemon_id=daemon_id, worker_id=worker_id)


def _default_global_store_stub_factory(
    address: str,
) -> _GlobalStoreStubHandle:
    channel = grpc.insecure_channel(address)
    return _GlobalStoreStubHandle(
        stub=GlobalStoreCompositeStub(channel),
        close=channel.close,
    )


class TensorcastInstanceDirectoryRegistration:
    def __init__(
        self,
        config: TensorcastInstanceDirectoryConfig,
        *,
        global_store_stub_factory: Callable[[str], _GlobalStoreStubHandle]
        | None = None,
        daemon_client_factory: Callable[[str], DaemonStatusClientProtocol] | None = None,
    ) -> None:
        self._config = config
        self._global_store_stub_factory = (
            global_store_stub_factory or _default_global_store_stub_factory
        )
        self._daemon_client_factory = daemon_client_factory or DaemonCtl
        self._stub_handle: _GlobalStoreStubHandle | None = None
        self._daemon_client: DaemonStatusClientProtocol | None = None
        self._daemon_identity: _DaemonDirectoryIdentity | None = None
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_interval_ms = max(1, int(config.heartbeat_interval_ms))
        self._registered = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._registered:
                return
            stub_handle = self._global_store_stub_factory(
                self._config.global_store_address
            )
            daemon_client = self._daemon_client_factory(self._config.daemon_address)
            try:
                daemon_identity = _resolve_daemon_directory_identity(daemon_client)
                register_request = global_store_pb2.RegisterInstanceRequest(
                    instance_id=self._config.instance_id,
                    daemon_id=daemon_identity.daemon_id,
                    engine=self._config.engine,
                    signals_endpoint=self._config.signals_endpoint,
                    labels=dict(self._config.labels),
                    capability_flags=_build_capability_flags(
                        signals_endpoint=self._config.signals_endpoint
                    ),
                    execution_endpoint=self._config.execution_endpoint,
                    execution_host_kind=self._config.execution_host_kind,
                )
                if daemon_identity.worker_id is not None:
                    register_request.worker_id = daemon_identity.worker_id
                response = stub_handle.stub.RegisterInstance(register_request)
                if response.status != global_store_pb2.Status.STATUS_OK:
                    raise RuntimeError(
                        "Tensorcast instance-directory RegisterInstance failed"
                    )
                if response.heartbeat_interval_ms > 0:
                    self._heartbeat_interval_ms = int(response.heartbeat_interval_ms)
            except Exception:
                daemon_client.close()
                stub_handle.close()
                raise

            self._stub_handle = stub_handle
            self._daemon_client = daemon_client
            self._daemon_identity = daemon_identity
            self._registered = True
            self._stop_event.clear()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=(
                    "tensorcast-instance-directory-heartbeat"
                    f"-{self._config.instance_id}"
                ),
                daemon=True,
            )
            self._heartbeat_thread.start()

    def stop(self) -> None:
        with self._lock:
            heartbeat_thread = self._heartbeat_thread
            self._heartbeat_thread = None
            self._stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2.0)

        unregister_stub: GlobalStoreStubProtocol | None = None
        daemon_client = None
        with self._lock:
            if self._registered and self._stub_handle is not None:
                unregister_stub = self._stub_handle.stub
            daemon_client = self._daemon_client
            stub_handle = self._stub_handle
            self._stub_handle = None
            self._daemon_client = None
            self._daemon_identity = None
            self._registered = False

        if unregister_stub is not None:
            try:
                response = unregister_stub.UnregisterInstance(
                    global_store_pb2.UnregisterInstanceRequest(
                        instance_id=self._config.instance_id,
                        is_graceful_shutdown=True,
                    )
                )
                if response.status != global_store_pb2.Status.STATUS_OK:
                    logger.warning(
                        "Tensorcast instance-directory UnregisterInstance returned non-OK status for instance_id=%s",
                        self._config.instance_id,
                    )
            except Exception:
                logger.exception(
                    "Tensorcast instance-directory unregister failed for instance_id=%s",
                    self._config.instance_id,
                )
        if daemon_client is not None:
            daemon_client.close()
        if stub_handle is not None:
            stub_handle.close()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_interval_ms / 1000.0):
            stub = self._stub_handle.stub if self._stub_handle is not None else None
            daemon_identity = self._daemon_identity
            if stub is None or daemon_identity is None:
                return
            try:
                heartbeat_request = global_store_pb2.InstanceHeartbeatRequest(
                    instance_id=self._config.instance_id,
                    capability_flags=_build_capability_flags(
                        signals_endpoint=self._config.signals_endpoint
                    ),
                )
                if daemon_identity.worker_id is not None:
                    heartbeat_request.worker_id = daemon_identity.worker_id
                response = stub.InstanceHeartbeat(heartbeat_request)
                if response.status != global_store_pb2.Status.STATUS_OK:
                    logger.warning(
                        "Tensorcast instance-directory heartbeat returned non-OK status for instance_id=%s",
                        self._config.instance_id,
                    )
            except Exception:
                logger.exception(
                    "Tensorcast instance-directory heartbeat failed for instance_id=%s",
                    self._config.instance_id,
                )


__all__ = [
    "INSTANCE_AGENT_EXECUTION_HOST_KIND",
    "TensorcastInstanceDirectoryConfig",
    "TensorcastInstanceDirectoryRegistration",
    "resolve_instance_agent_execution_endpoint",
]
