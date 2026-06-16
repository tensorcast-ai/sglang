"""Factory for building a `ResourceProvider` from a cluster YAML.

Dispatch is based on `provider.kind` in the YAML. v1 registers only
`brainctl`. New clusters add a provider implementation under `resource/`
and register it here (see arch.md § 14.2).
"""

from __future__ import annotations

from pathlib import Path

from .base import ResourceProvider, load_cluster_config
from .brainctl import BrainctlProvider


_REGISTRY: dict[str, type] = {
    "brainctl": BrainctlProvider,
}


def from_cluster_config(path: str | Path) -> ResourceProvider:
    """Load a cluster YAML and return a ResourceProvider matching its `provider.kind`."""
    cfg = load_cluster_config(path)
    kind = cfg.provider.kind
    if kind not in _REGISTRY:
        raise ValueError(
            f"unknown provider.kind={kind!r}. "
            f"Registered providers: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[kind].from_cluster_config(path)


def registered_kinds() -> list[str]:
    """Return the list of provider kinds the factory knows about."""
    return sorted(_REGISTRY)
