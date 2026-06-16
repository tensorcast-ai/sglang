"""Reserved for a future SSH-based fallback provider.

The v1 deployment uses BrainctlProvider (resource/brainctl.py).
StaticProvider exists as a documented future fallback for clusters that
expose plain SSH instead of a custom CLI; see arch.md § 14.4.

This module deliberately raises NotImplementedError on use so accidental
imports during v1 development surface immediately.
"""

from __future__ import annotations

from pathlib import Path


class StaticProvider:
    """Not implemented in v1."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "StaticProvider is not implemented in v1. Use BrainctlProvider."
        )

    @classmethod
    def from_cluster_config(cls, path: str | Path) -> "StaticProvider":
        raise NotImplementedError(
            "StaticProvider is not implemented in v1. Use BrainctlProvider."
        )

    def workers(self) -> list:
        raise NotImplementedError

    async def health_check(self) -> None:
        raise NotImplementedError
