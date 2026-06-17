"""Star-shaped RDMA reachability smoke test.

For runs with `transport.use_rdma=true`, the driver invokes this smoke check
before launching real services to fail fast on broken IB networking.

Pattern: from `workers[0]` (`server`), run `ib_write_bw` / `ib_send_bw` against
each of the other workers in turn. Single-worker setups are no-ops.

This module mirrors the contract used by `kv/share_remote/arch.md § 6.2`.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RDMASmokeResult:
    server_worker_id: str
    client_worker_id: str
    ok: bool
    detail: str


async def smoke_star(
    workers: Sequence,
    *,
    server_index: int = 0,
    timeout_s: float = 60.0,
) -> list[RDMASmokeResult]:
    """Run a star-shaped RDMA smoke from `workers[server_index]` to each other worker.

    Returns a list of per-pair results. Empty list when `len(workers) <= 1`.
    Raises if any pair fails — caller may catch to log / proceed in TCP mode.
    """
    if len(workers) <= 1:
        return []

    server = workers[server_index]
    clients = [w for i, w in enumerate(workers) if i != server_index]
    results: list[RDMASmokeResult] = []
    for client in clients:
        result = await _smoke_pair(server, client, timeout_s=timeout_s)
        results.append(result)
        if not result.ok:
            raise RuntimeError(
                f"RDMA smoke FAILED: server={server.id} client={client.id} — {result.detail}"
            )
    return results


async def _smoke_pair(server, client, *, timeout_s: float) -> RDMASmokeResult:
    """Run a single ib_write_bw probe between two workers.

    The exact command shape mirrors `kv/share_remote/scripts/`. v1 keeps it
    minimal: 1 second of `ib_write_bw -d <hca> -F` on the server, then the
    client connects, exits 0 on success.
    """
    port = 18500 + (secrets.randbelow(500))
    server_hca = _first_hca(server)
    client_hca = _first_hca(client)
    if not server_hca or not client_hca:
        return RDMASmokeResult(
            server_worker_id=server.id,
            client_worker_id=client.id,
            ok=False,
            detail=(
                f"NCCL_IB_HCA not set on one side: "
                f"server={server_hca!r} client={client_hca!r}"
            ),
        )

    server_cmd = (
        f"ib_write_bw -p {port} -d {shlex_quote(server_hca)} -F "
        f"--report_gbits -D 5 >/tmp/ib_write_bw_server_{port}.log 2>&1"
    )
    client_cmd = (
        f"sleep 2; "
        f"ib_write_bw -p {port} -d {shlex_quote(client_hca)} -F "
        f"--report_gbits -D 3 {shlex_quote(server.address)} "
        f">/tmp/ib_write_bw_client_{port}.log 2>&1"
    )

    server_task = asyncio.create_task(
        server.run(server_cmd, timeout_s=timeout_s, check=False, as_user=False)
    )
    client_task = asyncio.create_task(
        client.run(client_cmd, timeout_s=timeout_s, check=False, as_user=False)
    )
    server_proc, client_proc = await asyncio.gather(
        server_task, client_task, return_exceptions=False
    )
    if client_proc.returncode == 0 and server_proc.returncode == 0:
        return RDMASmokeResult(
            server_worker_id=server.id,
            client_worker_id=client.id,
            ok=True,
            detail=f"port={port} hca={server_hca}/{client_hca}",
        )
    return RDMASmokeResult(
        server_worker_id=server.id,
        client_worker_id=client.id,
        ok=False,
        detail=(
            f"ib_write_bw failed: server rc={server_proc.returncode}, "
            f"client rc={client_proc.returncode}; check /tmp/ib_write_bw_*_{port}.log "
            f"on each worker"
        ),
    )


def _first_hca(worker) -> str:
    """Return the first HCA from a comma-separated NCCL_IB_HCA, or empty string."""
    raw = (worker.base_env or {}).get("NCCL_IB_HCA", "").strip()
    if not raw:
        return ""
    return raw.split(",")[0].strip()


def shlex_quote(s: str) -> str:
    """Local re-export to keep the file self-contained."""
    import shlex as _shlex

    return _shlex.quote(s)
