"""Router internal state dataclasses (per arch § 6.3 / § 6.4).

These types are shared between `tc_router` (the implementation) and the
workload driver / metrics layer. None of them hold async resources so
they can be freely passed across coroutines.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional


# Type aliases — kept as plain str to avoid noise across the codebase.
InstanceId = str
SessionId = str


@dataclass
class SessionState:
    """Per-session router state.

    Mutable; each `tc_router.generate(session_id, ...)` call updates this.
    """

    session_id: SessionId
    home_instance: InstanceId = ""
    last_active_ts: float = 0.0  # monotonic
    turn_count: int = 0
    last_prompt_tokens: int = 0
    last_engine_request_id: str = ""
    # Tensorcast-specific. Stored as opaque object so this module doesn't
    # need to import the Tensorcast SDK; tc_router populates it.
    last_published_manifest: Any | None = None
    pending_migration: bool = False


@dataclass(frozen=True)
class LoadSample:
    """One snapshot of an instance's load, read from `/v1/loads`.

    Fields mirror what `tot_experiment.sglang_client.get_load` returns,
    plus `timestamp_monotonic` for staleness checks against the policy
    tick.
    """

    instance_id: InstanceId
    num_waiting_reqs: int
    num_running_reqs: int
    token_usage: float
    utilization: float
    gen_throughput: float
    timestamp_monotonic: float

    @property
    def queue_depth(self) -> int:
        """Combined waiting + running. The default load metric for routing."""
        return self.num_waiting_reqs + self.num_running_reqs


@dataclass(frozen=True)
class MigrationDecision:
    """One migration proposal emitted by `Policy.should_rebalance`."""

    session_id: SessionId
    source_instance: InstanceId
    target_instance: InstanceId
    decided_by: str  # human-readable policy name, e.g. "ThresholdPolicy"


@dataclass
class MigrationFuture:
    """Tracks an in-flight migration.

    A request arriving for the migrating session can `await
    completion_event.wait()` to defer routing until the migration finishes,
    or supersede it (cancel + reroute) — that decision is policy-level.
    """

    session_id: SessionId
    source_instance: InstanceId
    target_instance: InstanceId
    started_monotonic: float
    completion_event: asyncio.Event = field(default_factory=asyncio.Event)
    success: Optional[bool] = None
    error: str = ""

    def mark_success(self) -> None:
        self.success = True
        self.completion_event.set()

    def mark_failure(self, error: str) -> None:
        self.success = False
        self.error = error
        self.completion_event.set()
