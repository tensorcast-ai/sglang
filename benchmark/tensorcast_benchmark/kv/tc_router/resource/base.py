"""Resource abstraction protocols and config models.

A `ResourceProvider` adapts a pre-acquired cluster (described by a YAML file
written by an out-of-band acquisition script) to a uniform `Worker` interface
that the rest of the benchmark consumes. The benchmark itself never acquires
or releases workers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# --- Protocols ---------------------------------------------------------------


@runtime_checkable
class RemoteProcess(Protocol):
    """A remote command's lifecycle handle.

    For the brainctl provider the underlying `brainctl exec` runs to completion
    before returning, so most fields are populated immediately and `wait` is a
    no-op. Long-lived background services are managed via
    `Worker.start_background` / `Worker.stop_background` (PID-file pattern)
    and do not produce a live `RemoteProcess`.
    """

    @property
    def pid(self) -> int | None: ...

    @property
    def returncode(self) -> int | None: ...

    @property
    def stdout(self) -> str: ...

    @property
    def stderr(self) -> str: ...

    async def wait(self) -> int: ...
    async def kill(self) -> None: ...


@runtime_checkable
class Worker(Protocol):
    """A pre-acquired worker host the benchmark may use."""

    @property
    def id(self) -> str: ...

    @property
    def address(self) -> str: ...

    @property
    def node(self) -> str: ...

    @property
    def gpu_indices(self) -> tuple[int, ...]: ...

    @property
    def scratch_dir(self) -> str: ...

    @property
    def base_env(self) -> dict[str, str]: ...

    async def run(
        self,
        cmd: list[str] | str,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_s: float | None = None,
        check: bool = True,
        as_user: bool = True,
    ) -> RemoteProcess: ...

    async def start_background(
        self,
        cmd: str,
        *,
        name: str,
        log_path: str,
        pid_path: str,
        env: dict[str, str] | None = None,
    ) -> int: ...

    async def stop_background(self, *, pid_path: str) -> None: ...

    async def read_file(
        self, remote_path: str, *, max_bytes: int | None = None
    ) -> bytes: ...

    async def put_file(self, local: str | Path, remote: str) -> None: ...

    async def get_file(self, remote: str, local: str | Path) -> None: ...


@runtime_checkable
class ResourceProvider(Protocol):
    """Adapts a cluster YAML describing pre-acquired workers to `Worker` handles."""

    @classmethod
    def from_cluster_config(cls, path: str | Path) -> "ResourceProvider": ...

    def workers(self) -> list[Worker]: ...

    async def health_check(self) -> None: ...


# --- Pydantic config models --------------------------------------------------


class MountConfig(BaseModel):
    """Shared filesystem mount visible at the same path on driver host and every worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    spec: str = ""


class WorkerConfig(BaseModel):
    """One pre-acquired worker entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    address: str = Field(min_length=1)
    node: str = Field(min_length=1)
    process_handle: str = Field(
        min_length=1,
        description="Cluster CLI's identifier (e.g., the brainctl process name).",
    )
    gpu_indices: tuple[int, ...]
    scratch_dir: str = Field(min_length=1)
    base_env: dict[str, str] = Field(default_factory=dict)


class ServicePlacement(BaseModel):
    """Which worker hosts singleton services."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    global_store_worker_id: str = Field(min_length=1)
    mooncake_master_worker_id: str = Field(min_length=1)


class ProviderConfig(BaseModel):
    """Provider dispatch + provider-specific settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str = Field(min_length=1)
    namespace: str = ""
    cli: str = ""
    user: str = ""


class DriverHostConfig(BaseModel):
    """Driver host (where run_benchmark.py runs) settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scratch_dir: str = Field(min_length=1)


class ClusterConfig(BaseModel):
    """Top-level cluster YAML model. See arch.md § 9.1."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderConfig
    driver_host: DriverHostConfig
    mount: MountConfig
    workers: tuple[WorkerConfig, ...]
    service_placement: ServicePlacement

    @model_validator(mode="after")
    def _validate_workers(self) -> "ClusterConfig":
        if not self.workers:
            raise ValueError("cluster.workers must contain at least one worker")
        ids = [w.id for w in self.workers]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate worker.id found: {ids}")
        addresses = [w.address for w in self.workers]
        if len(set(addresses)) != len(addresses):
            raise ValueError(f"duplicate worker.address found: {addresses}")
        nodes = [w.node for w in self.workers]
        if len(set(nodes)) != len(nodes):
            raise ValueError(
                f"duplicate worker.node found (distinct-host requirement violated): {nodes}"
            )
        for w in self.workers:
            if not w.base_env:
                raise ValueError(
                    f"worker {w.id!r}: base_env is empty; at least RDMA env vars are expected"
                )
        if self.service_placement.global_store_worker_id not in ids:
            raise ValueError(
                f"service_placement.global_store_worker_id="
                f"{self.service_placement.global_store_worker_id!r} not in workers"
            )
        if self.service_placement.mooncake_master_worker_id not in ids:
            raise ValueError(
                f"service_placement.mooncake_master_worker_id="
                f"{self.service_placement.mooncake_master_worker_id!r} not in workers"
            )
        return self


def load_cluster_config(path: str | Path) -> ClusterConfig:
    """Parse and validate a cluster YAML file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"cluster YAML at {path} is not a mapping")
    return ClusterConfig.model_validate(raw)
