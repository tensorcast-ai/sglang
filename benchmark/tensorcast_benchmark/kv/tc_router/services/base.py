"""Service launcher abstractions.

A `ServiceLauncher` knows how to start exactly one kind of long-lived process
(SGLang instance, Tensorcast daemon, Mooncake master, sgl-model-gateway, ...)
on a `Worker` and report back a `Service` handle for later teardown.

Launchers are Provider-agnostic: they only call `Worker.run` /
`Worker.start_background` / `Worker.stop_background` from the resource
abstraction. They do not know about brainctl or any specific cluster CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Service:
    """Runtime handle for a launched service.

    `endpoints` carries the network-reachable URLs / addresses this service
    exposes (e.g. `{"serving_http": "http://10.0.0.1:30000"}`). Driver code
    consumes endpoints; it does not assume any field names.

    `pid` and `pid_path` / `log_path` are inside the worker (or on the shared
    mount the worker writes to). `Worker.stop_background(pid_path=...)` is
    the canonical teardown.
    """

    name: str
    worker_id: str
    endpoints: dict[str, str]
    pid: int
    pid_path: str
    log_path: str
    metadata: dict[str, str] = field(default_factory=dict)


class ServiceLauncher(Protocol):
    """Each concrete launcher implements `launch` and `wait_ready`.

    `launch(worker, spec)` runs the start command via `Worker.start_background`
    and returns the `Service` handle. `wait_ready(service, timeout_s)` polls
    until the service is responding (typically via HTTP / TCP / log scan).
    """

    async def launch(self, worker, spec) -> Service: ...

    async def wait_ready(self, service: Service, *, timeout_s: float) -> None: ...
