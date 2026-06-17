"""Live validation script for Phase 1 BrainctlProvider.

Runs all 5 plan §1.3 live gates against a real cluster YAML:

  1. provider.health_check() passes
  2. worker.run(['echo', 'hello']) returns stdout 'hello'
  3. worker.run(['env']) includes NCCL_IB_HCA from base_env
  4. worker.start_background returns PID, stop_background succeeds, PID gone
  5. put_file -> read_file round-trips a small payload via shared mount

Usage:
  python -m tensorcast_benchmark.kv.tc_router.tools.live_check_resource \\
      configs/cluster_brainctl_single_h800.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
import time
from pathlib import Path

from tensorcast_benchmark.kv.tc_router.resource import factory


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class GateFailed(RuntimeError):
    pass


def _ok(label: str, detail: str = "") -> None:
    print(f"  {GREEN}PASS{RESET}  {label}{(' — ' + detail) if detail else ''}")


def _fail(label: str, detail: str) -> None:
    print(f"  {RED}FAIL{RESET}  {label} — {detail}")


def _info(s: str) -> None:
    print(f"  {YELLOW}info{RESET}  {s}")


async def gate_health_check(provider) -> None:
    print("\n[gate 1] provider.health_check()")
    t0 = time.monotonic()
    await provider.health_check()
    dt = time.monotonic() - t0
    _ok("health_check passed", f"{dt:.1f}s")


async def gate_run_echo(worker) -> None:
    print(f"\n[gate 2] worker.run(['echo', 'hello']) on {worker.id}")
    proc = await worker.run(["echo", "hello"], timeout_s=30.0)
    out = proc.stdout.strip()
    if proc.returncode != 0 or out != "hello":
        raise GateFailed(f"unexpected: rc={proc.returncode}, stdout={out!r}, stderr={proc.stderr!r}")
    _ok("echo hello", f"stdout={out!r}, rc={proc.returncode}")


async def gate_env_injection(worker) -> None:
    print(f"\n[gate 3] worker.run(['env']) on {worker.id} — verify NCCL_IB_HCA injection")
    proc = await worker.run(["env"], timeout_s=30.0)
    if proc.returncode != 0:
        raise GateFailed(f"env failed: rc={proc.returncode}, stderr={proc.stderr!r}")
    expected_hca = worker.base_env["NCCL_IB_HCA"]
    expected_gid = worker.base_env["NCCL_IB_GID_INDEX"]
    expected_master = worker.base_env["MASTER_ADDR"]
    lines = proc.stdout.splitlines()
    has_hca = any(line == f"NCCL_IB_HCA={expected_hca}" for line in lines)
    has_gid = any(line == f"NCCL_IB_GID_INDEX={expected_gid}" for line in lines)
    has_master = any(line == f"MASTER_ADDR={expected_master}" for line in lines)
    if not (has_hca and has_gid and has_master):
        sample = [ln for ln in lines if ln.startswith(("NCCL_", "MASTER_"))]
        raise GateFailed(
            f"env vars missing: hca={has_hca} gid={has_gid} master={has_master}; "
            f"saw: {sample}"
        )
    _ok("base_env injected", f"NCCL_IB_HCA={expected_hca!r}, NCCL_IB_GID_INDEX={expected_gid}, MASTER_ADDR={expected_master}")


async def gate_background(worker, scratch_dir: str) -> None:
    print(f"\n[gate 4] worker.start_background / stop_background on {worker.id}")
    suffix = secrets.token_hex(4)
    log_path = f"{scratch_dir}/live_check/bg_{suffix}.log"
    pid_path = f"{scratch_dir}/live_check/bg_{suffix}.pid"

    pid = await worker.start_background(
        "echo START $(date -Is); sleep 30; echo END $(date -Is)",
        name=f"live_check_{suffix}",
        log_path=log_path,
        pid_path=pid_path,
    )
    _info(f"started PID={pid} log={log_path} pid_path={pid_path}")

    # Verify the PID is alive.
    proc = await worker.run(
        f"sleep 1; if kill -0 {pid} >/dev/null 2>&1; then echo ALIVE; else echo DEAD; fi",
        timeout_s=10.0,
    )
    state = proc.stdout.strip()
    if state != "ALIVE":
        raise GateFailed(f"PID {pid} not alive: stdout={proc.stdout!r}")
    _ok("background process alive after start", f"PID={pid}")

    # Verify log file got written.
    log_proc = await worker.run(["cat", log_path], timeout_s=10.0)
    if not log_proc.stdout.startswith("START"):
        raise GateFailed(f"log file unexpected: {log_proc.stdout!r}")
    _ok("background log file populated", log_proc.stdout.splitlines()[0])

    # Stop and verify.
    await worker.stop_background(pid_path=pid_path)
    proc2 = await worker.run(
        f"if kill -0 {pid} >/dev/null 2>&1; then echo STILL_ALIVE; else echo DEAD; fi",
        timeout_s=10.0,
    )
    state2 = proc2.stdout.strip()
    if state2 != "DEAD":
        raise GateFailed(f"PID {pid} still alive after stop: stdout={proc2.stdout!r}")
    _ok("background process killed and PID reaped", f"PID={pid}")

    # Verify PID file removed.
    proc3 = await worker.run(
        f"if [[ -f {pid_path} ]]; then echo EXISTS; else echo GONE; fi",
        timeout_s=10.0,
    )
    if proc3.stdout.strip() != "GONE":
        raise GateFailed(f"pid file not cleaned up: {pid_path}")
    _ok("PID file removed", pid_path)


async def gate_file_roundtrip(worker, scratch_dir: str) -> None:
    print(f"\n[gate 5] put_file → read_file via shared mount on {worker.id}")
    suffix = secrets.token_hex(4)
    payload = f"hello-{suffix}-{time.time_ns()}".encode()
    local_src = Path(f"/tmp/live_check_src_{suffix}.bin")
    remote_path = f"{scratch_dir}/live_check/payload_{suffix}.bin"
    local_dst = Path(f"/tmp/live_check_dst_{suffix}.bin")

    local_src.write_bytes(payload)
    try:
        await worker.put_file(local_src, remote_path)
        _info(f"put_file OK ({len(payload)} bytes -> {remote_path})")

        readback = await worker.read_file(remote_path)
        if readback != payload:
            raise GateFailed(f"read_file mismatch: got {readback[:32]!r}, expected {payload[:32]!r}")
        _ok("read_file roundtrip", f"{len(payload)} bytes match")

        await worker.get_file(remote_path, local_dst)
        if local_dst.read_bytes() != payload:
            raise GateFailed("get_file content mismatch")
        _ok("get_file roundtrip", f"{local_dst}")

        # Cleanup remote
        await worker.run(f"rm -f {remote_path}", timeout_s=10.0)
    finally:
        local_src.unlink(missing_ok=True)
        local_dst.unlink(missing_ok=True)


async def main_async(cluster_yaml: Path) -> int:
    print(f"[live_check] loading cluster YAML: {cluster_yaml}")
    provider = factory.from_cluster_config(cluster_yaml)
    print(f"[live_check] provider: {type(provider).__name__}")
    workers = provider.workers()
    print(f"[live_check] workers: {[w.id for w in workers]}")
    if not workers:
        print(f"{RED}no workers in cluster YAML{RESET}", file=sys.stderr)
        return 1

    failures: list[tuple[str, str]] = []

    try:
        await gate_health_check(provider)
    except Exception as exc:  # noqa: BLE001
        _fail("gate_health_check", str(exc))
        failures.append(("gate 1 health_check", str(exc)))

    # Run worker-level gates against the first worker.
    worker = workers[0]
    for gate_fn in (gate_run_echo, gate_env_injection):
        try:
            await gate_fn(worker)
        except Exception as exc:  # noqa: BLE001
            _fail(gate_fn.__name__, str(exc))
            failures.append((gate_fn.__name__, str(exc)))

    for gate_fn in (gate_background, gate_file_roundtrip):
        try:
            await gate_fn(worker, scratch_dir=worker.scratch_dir)
        except Exception as exc:  # noqa: BLE001
            _fail(gate_fn.__name__, str(exc))
            failures.append((gate_fn.__name__, str(exc)))

    print()
    if failures:
        print(f"{RED}=== {len(failures)} gate(s) failed ==={RESET}")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        return 2
    print(f"{GREEN}=== all live gates passed ==={RESET}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cluster_yaml", type=Path)
    args = p.parse_args()
    return asyncio.run(main_async(args.cluster_yaml))


if __name__ == "__main__":
    raise SystemExit(main())
