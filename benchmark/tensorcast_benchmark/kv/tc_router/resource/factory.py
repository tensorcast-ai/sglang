"""Factory for building a `ResourceProvider` from a cluster YAML."""

from __future__ import annotations

from pathlib import Path

from tensorcast_benchmark.kv.tc_router.resource.base import (
    ResourceProvider,
    load_cluster_config,
)
from tensorcast_benchmark.kv.tc_router.resource.local import LocalProvider
from tensorcast_benchmark.kv.tc_router.resource.static import StaticProvider


_REGISTRY: dict[str, type] = {
    "local": LocalProvider,
    "static": StaticProvider,
}


def from_cluster_config(path: str | Path) -> ResourceProvider:
    """Load a cluster YAML and return a ResourceProvider matching its `provider.kind`."""
    cfg = load_cluster_config(path)
    kind = cfg.provider.kind
    if kind not in _REGISTRY:
        raise ValueError(
            f"unknown provider.kind={kind!r}. Registered providers: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[kind].from_cluster_config(path)


def registered_kinds() -> list[str]:
    """Return the list of provider kinds the factory knows about."""
    return sorted(_REGISTRY)
