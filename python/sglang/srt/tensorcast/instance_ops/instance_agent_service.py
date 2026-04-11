# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

from contextlib import suppress
import logging
import multiprocessing
from multiprocessing.connection import Connection
import traceback
from typing import Callable

from sglang.srt.tensorcast.instance_ops.instance_agent import (
    TensorcastInstanceAgent,
    TensorcastInstanceAgentConfig,
)

logger = logging.getLogger(__name__)

_READY = "ready"
_ERROR = "error"
_STOP = "stop"


def _instance_agent_service_main(
    config: TensorcastInstanceAgentConfig,
    status_send: Connection,
    stop_recv: Connection,
) -> None:
    agent: TensorcastInstanceAgent | None = None
    try:
        agent = TensorcastInstanceAgent(config)
        agent.start()
        status_send.send((_READY, agent.bound_endpoint or ""))
        while True:
            try:
                if not stop_recv.poll(0.5):
                    continue
                command = stop_recv.recv()
            except EOFError:
                break
            if command == _STOP:
                break
    except Exception as exc:
        logger.exception("Tensorcast instance-agent service failed")
        with suppress(Exception):
            status_send.send((_ERROR, str(exc), traceback.format_exc()))
        raise
    finally:
        if agent is not None:
            with suppress(Exception):
                agent.stop()
        with suppress(Exception):
            stop_recv.close()
        with suppress(Exception):
            status_send.close()


class TensorcastInstanceAgentServiceHandle:
    def __init__(
        self,
        config: TensorcastInstanceAgentConfig,
        *,
        start_timeout_s: float = 30.0,
        mp_context_name: str = "spawn",
        _context_factory: Callable[[str], multiprocessing.context.BaseContext]
        | None = None,
        _process_target: Callable[
            [TensorcastInstanceAgentConfig, Connection, Connection], None
        ]
        | None = None,
    ) -> None:
        self._config = config
        self._start_timeout_s = start_timeout_s
        self._mp_context_name = mp_context_name
        self._context_factory = _context_factory or multiprocessing.get_context
        self._process_target = _process_target or _instance_agent_service_main
        self._process: multiprocessing.Process | None = None
        self._status_recv: Connection | None = None
        self._stop_send: Connection | None = None
        self._bound_endpoint: str | None = None

    @property
    def bound_endpoint(self) -> str | None:
        return self._bound_endpoint

    def start(self) -> str | None:
        if self._process is not None:
            return self._bound_endpoint
        ctx = self._context_factory(self._mp_context_name)
        status_recv, status_send = ctx.Pipe(duplex=False)
        stop_recv, stop_send = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=self._process_target,
            args=(self._config, status_send, stop_recv),
            name="sglang-tensorcast-instance-agent",
            daemon=True,
        )
        process.start()
        status_send.close()
        stop_recv.close()
        self._process = process
        self._status_recv = status_recv
        self._stop_send = stop_send
        try:
            self._bound_endpoint = self._wait_until_ready()
            return self._bound_endpoint
        except Exception:
            self.stop()
            raise

    def _wait_until_ready(self) -> str | None:
        assert self._process is not None
        assert self._status_recv is not None
        if not self._status_recv.poll(self._start_timeout_s):
            if not self._process.is_alive():
                raise RuntimeError(
                    "Tensorcast instance-agent service exited before reporting ready"
                )
            raise TimeoutError(
                "Timed out waiting for Tensorcast instance-agent service readiness"
            )
        message = self._status_recv.recv()
        if not isinstance(message, tuple) or not message:
            raise RuntimeError("Malformed Tensorcast instance-agent service status")
        tag = message[0]
        if tag == _READY:
            endpoint = str(message[1]).strip() if len(message) > 1 else ""
            return endpoint or None
        if tag == _ERROR:
            error_message = str(message[1]).strip() if len(message) > 1 else ""
            error_traceback = str(message[2]).strip() if len(message) > 2 else ""
            details = error_message or "unknown instance-agent startup failure"
            if error_traceback:
                raise RuntimeError(
                    f"{details}\nTensorcast instance-agent traceback:\n{error_traceback}"
                )
            raise RuntimeError(details)
        raise RuntimeError(f"Unknown Tensorcast instance-agent service status: {tag!r}")

    def stop(self) -> None:
        process = self._process
        stop_send = self._stop_send
        status_recv = self._status_recv
        self._process = None
        self._stop_send = None
        self._status_recv = None
        self._bound_endpoint = None
        if stop_send is not None:
            with suppress(BrokenPipeError, EOFError, OSError):
                stop_send.send(_STOP)
            with suppress(Exception):
                stop_send.close()
        if status_recv is not None:
            with suppress(Exception):
                status_recv.close()
        if process is None:
            return
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)


__all__ = [
    "TensorcastInstanceAgentServiceHandle",
]
