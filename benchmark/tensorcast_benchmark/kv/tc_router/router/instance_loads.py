"""Background poller that maintains a fresh `LoadSample` per instance.

Polls each instance's `/v1/loads` HTTP endpoint at `period_ms` cadence
and stores the latest `LoadSample` in an in-memory map. Concurrent reads
return a snapshot via `snapshot()` / `get(instance_id)`.

The poller intentionally runs from the driver host (not on a worker),
mirroring arch § 6.6 — Tensorcast directory does not expose serving-side
queue metrics, so the load signal lives outside it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Mapping, Optional

import aiohttp

from .state import InstanceId, LoadSample


logger = logging.getLogger(__name__)


class InstanceLoadPoller:
    """Periodically refreshes load metrics for a fixed set of instances."""

    def __init__(
        self,
        instance_endpoints: Mapping[InstanceId, str],
        *,
        period_ms: int = 250,
        request_timeout_s: float = 2.0,
    ) -> None:
        if not instance_endpoints:
            raise ValueError("instance_endpoints must contain at least one entry")
        if period_ms < 10:
            raise ValueError("period_ms too low; minimum 10ms")
        # Strip trailing slashes; we'll append `/v1/loads` later.
        self._endpoints: dict[InstanceId, str] = {
            iid: url.rstrip("/") for iid, url in instance_endpoints.items()
        }
        self._period_s = period_ms / 1000.0
        self._request_timeout = aiohttp.ClientTimeout(total=request_timeout_s)
        self._loads: dict[InstanceId, LoadSample] = {}
        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def period_s(self) -> float:
        return self._period_s

    @property
    def instance_ids(self) -> list[InstanceId]:
        return list(self._endpoints.keys())

    def get(self, instance_id: InstanceId) -> Optional[LoadSample]:
        """Return the latest sample for `instance_id`, or None if not yet polled."""
        return self._loads.get(instance_id)

    def snapshot(self) -> dict[InstanceId, LoadSample]:
        """Return a shallow copy of the current load map."""
        return dict(self._loads)

    async def start(self) -> None:
        """Start the background poll loop. Call `stop()` to shut down."""
        if self._task is not None:
            raise RuntimeError("poller already started")
        self._stopping.clear()
        # `trust_env=False` so corporate proxies don't intercept our internal
        # cluster traffic (mirrors what we did in services/sglang.py).
        self._session = aiohttp.ClientSession(trust_env=False)
        self._task = asyncio.create_task(self._run(), name="instance_load_poller")

    async def stop(self) -> None:
        """Stop the background loop and close the HTTP session."""
        if self._task is None:
            return
        self._stopping.set()
        try:
            await asyncio.wait_for(self._task, timeout=self._period_s + 1.0)
        except asyncio.TimeoutError:
            self._task.cancel()
            with _suppress_cancelled():
                await self._task
        finally:
            self._task = None
            if self._session is not None:
                await self._session.close()
                self._session = None

    async def _run(self) -> None:
        # First poll immediately so callers don't see an empty map for the
        # full first period.
        await self._poll_all_once()
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=self._period_s
                )
                # If we got here without timing out, stop was set.
                break
            except asyncio.TimeoutError:
                pass
            await self._poll_all_once()

    async def _poll_all_once(self) -> None:
        assert self._session is not None
        tasks = [
            asyncio.create_task(self._poll_one(iid, url))
            for iid, url in self._endpoints.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for iid, result in zip(self._endpoints, results):
            if isinstance(result, Exception):
                logger.debug("load poll failed for %s: %s", iid, result)
                continue
            if result is not None:
                self._loads[iid] = result

    async def _poll_one(
        self, instance_id: InstanceId, base_url: str
    ) -> Optional[LoadSample]:
        assert self._session is not None
        url = f"{base_url}/v1/loads"
        try:
            async with self._session.get(
                url, timeout=self._request_timeout, proxy=None
            ) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return None

        return _parse_loads_response(instance_id, payload)


def _parse_loads_response(
    instance_id: InstanceId, payload: dict
) -> LoadSample:
    """Aggregate the per-DP-rank `loads` array into a single LoadSample.

    Mirrors `tot_experiment.sglang_client.get_load` aggregation:
    `num_*` are summed; `token_usage` / `utilization` / `gen_throughput`
    are averaged across ranks.
    """
    items = payload.get("loads") or []
    if not items:
        return LoadSample(
            instance_id=instance_id,
            num_waiting_reqs=0,
            num_running_reqs=0,
            token_usage=0.0,
            utilization=0.0,
            gen_throughput=0.0,
            timestamp_monotonic=time.monotonic(),
        )
    n = len(items)
    return LoadSample(
        instance_id=instance_id,
        num_waiting_reqs=sum(int(i.get("num_waiting_reqs") or 0) for i in items),
        num_running_reqs=sum(int(i.get("num_running_reqs") or 0) for i in items),
        token_usage=sum(float(i.get("token_usage") or 0) for i in items) / n,
        utilization=sum(float(i.get("utilization") or 0) for i in items) / n,
        gen_throughput=sum(float(i.get("gen_throughput") or 0) for i in items) / n,
        timestamp_monotonic=time.monotonic(),
    )


class _suppress_cancelled:
    """Tiny context manager that swallows asyncio.CancelledError only."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is asyncio.CancelledError
