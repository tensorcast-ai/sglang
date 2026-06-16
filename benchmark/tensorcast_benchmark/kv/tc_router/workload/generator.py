"""Closed-loop within session, controlled active count workload driver.

See arch § 5.2 for the full contract. Summary:

- Maintain a steady-state of `c_target` active sessions.
- Each session = one trajectory replayed faithfully.
- Per assistant boundary `k`, send `messages[0:k]` + `tools` to the router.
- Discard the model's output; advance the prompt using the original
  `messages[k]` + subsequent tool/user messages.
- Sleep `LogNormal(...)` between consecutive turns of the same session.

The driver is router-agnostic — it accepts any object satisfying the
`Router` protocol from `router.interface`. Tests use a `MockRouter`.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Callable, Optional

from ..metrics.per_turn import TurnRecord
from ..router.interface import GenerateResult, Router
from .trajectory_pool import Trajectory, _turn_chars


logger = logging.getLogger(__name__)


@dataclass
class WorkloadOutcome:
    """Aggregate stats after a workload run."""

    total_turns: int
    successful_turns: int
    failed_turns: int
    distinct_sessions_started: int
    sessions_completed_full_replay: int
    wall_seconds_actual: float


class WorkloadDriver:
    def __init__(
        self,
        router: Router,
        pool: list[Trajectory],
        inter_turn_sampler: Callable[[], float],
        *,
        c_target: int,
        wall_seconds: float,
        warmup_seconds: float = 0.0,
        start_jitter_s: float = 2.0,
        max_new_tokens_clip: int = 512,
        record_sink: Optional[Callable[[TurnRecord], None]] = None,
        rng_seed: int = 0,
        chars_per_token: float = 3.6,
        supervisor_tick_s: float = 0.1,
    ) -> None:
        if c_target < 1:
            raise ValueError("c_target must be >= 1")
        if wall_seconds <= 0:
            raise ValueError("wall_seconds must be > 0")
        if not pool:
            raise ValueError("trajectory pool is empty")
        self._router = router
        self._pool: list[Trajectory] = list(pool)
        self._delay = inter_turn_sampler
        self._c_target = c_target
        self._wall_seconds = wall_seconds
        self._warmup_seconds = warmup_seconds
        self._start_jitter_s = start_jitter_s
        self._max_new_tokens_clip = max_new_tokens_clip
        self._sink = record_sink
        self._rng = random.Random(rng_seed)
        self._chars_per_token = chars_per_token
        self._supervisor_tick_s = supervisor_tick_s

        self._pool_cursor = 0
        self._pool_lock = asyncio.Lock()
        self._records: list[TurnRecord] = []
        self._completed_full_replay = 0
        self._started_sessions = 0

    @property
    def records(self) -> list[TurnRecord]:
        return list(self._records)

    @property
    def completed_full_replay(self) -> int:
        return self._completed_full_replay

    async def _next_trajectory(self) -> Optional[Trajectory]:
        async with self._pool_lock:
            if self._pool_cursor >= len(self._pool):
                return None
            traj = self._pool[self._pool_cursor]
            self._pool_cursor += 1
            return traj

    def _build_record(
        self,
        *,
        traj: Trajectory,
        turn_idx: int,
        prompt_messages: list[dict],
        max_new_tokens: int,
        rid: str,
        result: Optional[GenerateResult],
        latency_ms: float,
        success: bool,
        error_message: str,
    ) -> TurnRecord:
        return TurnRecord(
            ts=time.time(),
            session_id=traj.session_id,
            instance_id=traj.instance_id,
            turn_index=turn_idx,
            prompt_messages_count=len(prompt_messages),
            prompt_tokens=result.prompt_tokens if result is not None else 0,
            max_new_tokens=max_new_tokens,
            served_instance=result.served_instance if result is not None else "",
            ttft_ms=result.ttft_ms if result is not None else None,
            latency_ms=latency_ms,
            cached_tokens=result.cached_tokens if result is not None else 0,
            used_hydrated_bundle=(
                result.used_hydrated_bundle if result is not None else False
            ),
            was_just_migrated=(
                result.was_just_migrated if result is not None else False
            ),
            rid=rid,
            success=success,
            error_message=error_message,
        )

    async def _session_runner(
        self, traj: Trajectory, *, deadline: float
    ) -> None:
        # Tiny start jitter so sessions don't synchronize at supervisor tick.
        if self._start_jitter_s > 0:
            jitter = self._rng.uniform(0, self._start_jitter_s)
            await asyncio.sleep(jitter)

        tools = list(traj.tools) or None
        last_idx = len(traj.assistant_indices) - 1

        for turn_idx, assistant_pos in enumerate(traj.assistant_indices):
            if time.monotonic() >= deadline:
                return

            prompt_messages = list(traj.messages[:assistant_pos])
            original_assistant = traj.messages[assistant_pos]
            estimated = max(1, int(_turn_chars(original_assistant) / self._chars_per_token))
            max_new_tokens = min(estimated, self._max_new_tokens_clip)
            rid = f"tcrouter:{traj.session_id}:turn{turn_idx:03d}"

            t0 = time.monotonic()
            result: Optional[GenerateResult] = None
            success = False
            error_message = ""
            try:
                result = await self._router.generate(
                    session_id=traj.session_id,
                    messages=prompt_messages,
                    tools=tools,
                    sampling_params={
                        "max_tokens": max_new_tokens,
                        "temperature": 0.0,
                    },
                )
                success = bool(getattr(result, "success", True))
                error_message = getattr(result, "error_message", "") or ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                success = False
                error_message = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.monotonic() - t0) * 1000.0

            rec = self._build_record(
                traj=traj,
                turn_idx=turn_idx,
                prompt_messages=prompt_messages,
                max_new_tokens=max_new_tokens,
                rid=rid,
                result=result,
                latency_ms=latency_ms,
                success=success,
                error_message=error_message,
            )
            self._records.append(rec)
            if self._sink is not None:
                try:
                    self._sink(rec)
                except Exception:  # noqa: BLE001
                    logger.exception("record_sink raised; continuing")

            if turn_idx == last_idx:
                self._completed_full_replay += 1
                return

            # Inter-turn delay.
            delay = self._delay()
            remaining = deadline - time.monotonic()
            if delay >= remaining:
                return
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise

    async def run(self) -> WorkloadOutcome:
        start = time.monotonic()
        deadline = start + self._wall_seconds
        active: set[asyncio.Task] = set()

        async def supervisor() -> None:
            while time.monotonic() < deadline:
                # Reap finished tasks.
                for task in [t for t in active if t.done()]:
                    active.discard(task)
                    exc = task.exception() if not task.cancelled() else None
                    if exc is not None and not isinstance(exc, asyncio.CancelledError):
                        logger.exception("session task crashed", exc_info=exc)
                # Refill up to c_target.
                while len(active) < self._c_target and time.monotonic() < deadline:
                    traj = await self._next_trajectory()
                    if traj is None:
                        break  # pool exhausted; no new sessions until run end
                    self._started_sessions += 1
                    task = asyncio.create_task(
                        self._session_runner(traj, deadline=deadline),
                        name=f"session_{traj.session_id}",
                    )
                    active.add(task)
                await asyncio.sleep(self._supervisor_tick_s)

        sup_task = asyncio.create_task(supervisor(), name="workload_supervisor")
        try:
            await sup_task
        except asyncio.CancelledError:
            pass
        finally:
            # Sessions internally check deadline before each turn / sleep, so
            # they exit gracefully on their own. Just wait them out (no cancel).
            if active:
                await asyncio.gather(*active, return_exceptions=True)

        wall_actual = time.monotonic() - start
        successful = sum(1 for r in self._records if r.success)
        return WorkloadOutcome(
            total_turns=len(self._records),
            successful_turns=successful,
            failed_turns=len(self._records) - successful,
            distinct_sessions_started=self._started_sessions,
            sessions_completed_full_replay=self._completed_full_replay,
            wall_seconds_actual=wall_actual,
        )
