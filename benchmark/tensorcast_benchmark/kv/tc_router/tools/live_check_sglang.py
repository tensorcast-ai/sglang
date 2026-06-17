"""Live validation script for Phase 2 SGLang launcher.

Launches one SGLang serving instance on a real worker described by a cluster
YAML, hits its `/health` and `/v1/models` endpoints, sends one short
`/v1/chat/completions` request, and verifies (a) content is non-empty and
(b) `tool_calls` is empty / absent (the arch §5.2.3 guardrail).

Usage:
  python -m tensorcast_benchmark.kv.tc_router.tools.live_check_sglang \\
      configs/cluster_brainctl_single_h800.yaml \\
      --model-path hf/Qwen3-32B \\
      --tp-size 2 \\
      --port 30001
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import aiohttp

from tensorcast_benchmark.kv.tc_router.resource import factory
from tensorcast_benchmark.kv.tc_router.services.sglang import (
    SGLangLauncher,
    SGLangLaunchSpec,
)


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _ok(label: str, detail: str = "") -> None:
    print(f"  {GREEN}PASS{RESET}  {label}{(' — ' + detail) if detail else ''}")


def _fail(label: str, detail: str) -> None:
    print(f"  {RED}FAIL{RESET}  {label} — {detail}")


def _info(s: str) -> None:
    print(f"  {YELLOW}info{RESET}  {s}")


async def gate_models(serving_url: str, expected_model: str | None) -> str:
    print("\n[gate B] GET /v1/models")
    async with aiohttp.ClientSession(trust_env=False) as session:
        async with session.get(
            f"{serving_url}/v1/models",
            timeout=aiohttp.ClientTimeout(total=10.0),
            proxy=None,
        ) as resp:
            payload = await resp.json()
    models = payload.get("data") or []
    if not models:
        raise RuntimeError(f"no models listed; raw response: {payload!r}")
    served_id = models[0].get("id", "")
    if expected_model:
        # Match suffix, since SGLang reports the model_path as the id.
        if expected_model not in served_id:
            raise RuntimeError(
                f"served model {served_id!r} does not contain expected substring {expected_model!r}"
            )
    _ok("model listed", f"id={served_id!r}")
    return served_id


async def gate_chat_completions(
    serving_url: str, *, model_id: str, question: str, max_tokens: int
) -> dict:
    print("\n[gate C] POST /v1/chat/completions (tool_calls must be empty)")
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    async with aiohttp.ClientSession(trust_env=False) as session:
        async with session.post(
            f"{serving_url}/v1/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=120.0),
            proxy=None,
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
            payload = await resp.json()

    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"no choices in response; payload keys={list(payload)}")
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    if not content:
        raise RuntimeError(f"empty content; message={msg!r}")
    if tool_calls:
        raise RuntimeError(
            "tool_calls is non-empty — SGLang appears to have a tool parser "
            f"enabled (arch § 5.2.3 violation): {tool_calls!r}"
        )
    _ok(
        "chat completion content non-empty, tool_calls absent",
        f"content[:80]={content[:80]!r}",
    )
    return payload


async def run_live_check(
    cluster_yaml: Path,
    *,
    model_path: str,
    tp_size: int,
    port: int,
    mem_fraction_static: float,
    page_size: int,
    ready_timeout_s: float,
    expected_model_substring: str | None,
    keep: bool,
) -> int:
    print(f"[live_check_sglang] cluster: {cluster_yaml}")
    provider = factory.from_cluster_config(cluster_yaml)
    workers = provider.workers()
    if not workers:
        print(f"{RED}cluster YAML has no workers{RESET}", file=sys.stderr)
        return 1
    worker = workers[0]
    print(f"[live_check_sglang] worker: {worker.id} ({worker.address})")

    spec = SGLangLaunchSpec(
        model_path=model_path,
        host=worker.address,
        port=port,
        tp_size=tp_size,
        mem_fraction_static=mem_fraction_static,
        page_size=page_size,
    )
    launcher = SGLangLauncher()
    service = None
    failures: list[tuple[str, str]] = []
    try:
        # Quick health check on the resource layer first, then launch.
        await provider.health_check()
        _ok("provider.health_check", "cluster ok")

        print(f"\n[gate A] launch SGLang ({tp_size}xGPU, port={port})")
        t_launch = time.monotonic()
        service = await launcher.launch(worker, spec)
        _info(
            f"PID={service.pid}  log={service.log_path}  pid_path={service.pid_path}  "
            f"endpoint={service.endpoints['serving_http']}"
        )
        try:
            await launcher.wait_ready(
                service, timeout_s=ready_timeout_s, poll_interval_s=3.0
            )
        except TimeoutError as exc:
            # Pull last bit of log for diagnostics.
            try:
                tail = (await worker.read_file(service.log_path))[-3000:].decode(
                    "utf-8", errors="replace"
                )
            except Exception:  # noqa: BLE001
                tail = "(could not read log)"
            raise RuntimeError(f"{exc}\n--- last 3 KB of {service.log_path} ---\n{tail}")
        dt = time.monotonic() - t_launch
        _ok("/health 200", f"{dt:.1f}s to ready")

        served_id = await gate_models(
            service.endpoints["serving_http"], expected_model_substring
        )

        await gate_chat_completions(
            service.endpoints["serving_http"],
            model_id=served_id,
            question="In one short sentence, name the capital of France.",
            max_tokens=32,
        )

    except Exception as exc:  # noqa: BLE001
        _fail("live_check_sglang", str(exc))
        failures.append(("live_check_sglang", str(exc)))
    finally:
        if service is not None and not keep:
            print(f"\n[teardown] stopping {service.name}")
            try:
                await launcher.stop(worker, service)
                _ok("service stopped", service.pid_path)
            except Exception as exc:  # noqa: BLE001
                _fail("service stop", str(exc))
                failures.append(("teardown", str(exc)))
        elif keep and service is not None:
            _info(f"keeping service alive (pid_path={service.pid_path}); stop manually with worker.stop_background")
        await launcher.aclose()

    print()
    if failures:
        print(f"{RED}=== {len(failures)} gate(s) failed ==={RESET}")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        return 2
    print(f"{GREEN}=== all live gates passed ==={RESET}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cluster_yaml", type=Path)
    p.add_argument(
        "--model-path",
        default="hf/Qwen3-32B",
    )
    p.add_argument("--tp-size", type=int, default=2)
    p.add_argument("--port", type=int, default=30001)
    p.add_argument("--mem-fraction-static", type=float, default=0.85)
    p.add_argument("--page-size", type=int, default=32)
    p.add_argument(
        "--ready-timeout-s",
        type=float,
        default=900.0,
        help="how long to wait for /health 200 after launch (TP=2 32B cold start ~5-10min)",
    )
    p.add_argument(
        "--expected-model-substring",
        default="Qwen3-32B",
        help="assert this string appears in the model id reported by /v1/models",
    )
    p.add_argument(
        "--keep",
        action="store_true",
        help="leave the SGLang service running after the gate (for manual inspection)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(
        run_live_check(
            args.cluster_yaml,
            model_path=args.model_path,
            tp_size=args.tp_size,
            port=args.port,
            mem_fraction_static=args.mem_fraction_static,
            page_size=args.page_size,
            ready_timeout_s=args.ready_timeout_s,
            expected_model_substring=args.expected_model_substring,
            keep=args.keep,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
