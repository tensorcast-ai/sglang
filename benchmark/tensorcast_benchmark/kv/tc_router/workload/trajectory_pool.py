"""Trajectory pool builder.

Loads SWE-Gym/OpenHands-Sampled-Trajectories parquet shards, applies the
filter declared in `benchmark.yaml`, and returns a deterministic, shuffled
list of `Trajectory` records.

Profile mode:

  python -m tensorcast_benchmark.kv.tc_router.workload.trajectory_pool \\
      --dataset-path /data/datasets/OpenHands-Sampled-Trajectories --report

prints distribution stats so we can sanity-check arch § 5.1.3 numbers
without reaching for a notebook.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Per arch § 5.1.3 footnote: o200k_base chars/token observed at ~3.6 across
# the SWE-Gym sample. We use this for filter pre-screening; benchmark-time
# logic should re-tokenize with the deployment model's tokenizer.
DEFAULT_CHARS_PER_TOKEN: float = 3.6


@dataclass(frozen=True)
class Trajectory:
    """One replayable session.

    `messages` is the original trajectory payload, preserved as-is so that
    SGLang's chat template + tool serialization see exactly what the
    original agent produced. `assistant_indices` tells the workload driver
    where to cut (one LLM call per assistant boundary, faithful replay).
    """

    session_id: str
    instance_id: str
    messages: tuple[dict, ...]
    tools: tuple[dict, ...]
    assistant_indices: tuple[int, ...]
    total_chars: int
    estimated_tokens: int
    resolved: bool


# --- helpers -----------------------------------------------------------------


def _strip_nulls_deep(obj):
    """Recursively remove `None` values from nested dicts / lists.

    pyarrow's `to_pylist()` materializes every nested-struct field declared
    in the parquet schema even when a particular row didn't populate it,
    leaving `None` values scattered through the messages / tools payload.
    SGLang's OpenAI-compat layer validates `tools[].function.parameters`
    against the JSON Schema metaschema, which rejects `None` as a property
    value (e.g. `properties.file_text: null` for a tool that doesn't use
    that field). We strip nulls at load time so each Trajectory carries
    clean OpenAI-compatible payloads.

    Empty strings, `0`, and `False` are preserved — only `None` is dropped.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        out: dict = {}
        for key, value in obj.items():
            cleaned = _strip_nulls_deep(value)
            if cleaned is not None:
                out[key] = cleaned
        return out
    if isinstance(obj, (list, tuple)):
        out_list: list = []
        for value in obj:
            cleaned = _strip_nulls_deep(value)
            if cleaned is not None:
                out_list.append(cleaned)
        return out_list
    return obj


def _canonicalize_tool(tool: dict) -> dict | None:
    """Make sure a single tool entry conforms to OpenAI's request shape.

    SGLang's request validator requires `tools[i].type` and
    `tools[i].function.parameters` (a JSON Schema). Missing `parameters`
    after null-stripping is rewritten to a minimal `{"type": "object",
    "properties": {}}` so the request still passes deserialization.
    Returns `None` if the tool is too malformed to recover.
    """
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    if not isinstance(function, dict):
        return None
    if not function.get("name"):
        return None
    if not isinstance(function.get("parameters"), dict):
        function = {**function, "parameters": {"type": "object", "properties": {}}}
    return {"type": tool.get("type", "function"), "function": function}


def _canonicalize_tools(tools: list) -> list[dict]:
    """Apply `_canonicalize_tool` to each entry; drop unrecoverable ones."""
    out: list[dict] = []
    for t in tools or ():
        c = _canonicalize_tool(t)
        if c is not None:
            out.append(c)
    return out


def _turn_chars(message: dict) -> int:
    """Approximate byte contribution of one message (content + tool_calls)."""
    n = len(message.get("content") or "")
    for tc in message.get("tool_calls") or ():
        f = tc.get("function") or {}
        n += len(f.get("name") or "") + len(f.get("arguments") or "")
    return n


def _trajectory_total_chars(messages) -> int:
    return sum(_turn_chars(m) for m in messages)


def _assistant_indices(messages) -> tuple[int, ...]:
    return tuple(i for i, m in enumerate(messages) if m.get("role") == "assistant")


def _find_parquet_shards(dataset_path: str | Path) -> list[Path]:
    base = Path(dataset_path)
    if base.is_file() and base.suffix == ".parquet":
        return [base]
    if not base.exists():
        raise FileNotFoundError(f"dataset_path does not exist: {base}")
    candidates = sorted((base / "data").glob("*.parquet"))
    if not candidates:
        candidates = sorted(base.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no parquet shards under {base}")
    return candidates


# --- public API --------------------------------------------------------------


def load_pool(
    dataset_path: str | Path,
    *,
    min_turns: int = 8,
    min_total_tokens: int = 8000,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    seed: int = 0,
    require_at_least_one_assistant: bool = True,
) -> list[Trajectory]:
    """Load all parquet shards, filter, deterministically shuffle, and return.

    Filter rules (arch § 5.1.4):
      - `len(messages) >= min_turns`
      - `total_chars / chars_per_token >= min_total_tokens`
      - at least one `assistant` boundary (otherwise nothing to replay)
    """
    import pyarrow.parquet as pq

    shards = _find_parquet_shards(dataset_path)
    out: list[Trajectory] = []
    for shard in shards:
        table = pq.read_table(
            str(shard),
            columns=["instance_id", "run_id", "resolved", "messages", "tools"],
        )
        for row in table.to_pylist():
            messages_raw = row.get("messages") or []
            n_turns = len(messages_raw)
            if n_turns < min_turns:
                continue
            total_chars = _trajectory_total_chars(messages_raw)
            est_tokens = int(total_chars / chars_per_token)
            if est_tokens < min_total_tokens:
                continue
            assistant_idx = _assistant_indices(messages_raw)
            if require_at_least_one_assistant and not assistant_idx:
                continue
            # Strip pyarrow's flattening-introduced `None` values from
            # messages/tools so SGLang's JSON Schema validators accept the
            # payloads at request time (see _strip_nulls_deep docstring).
            messages_clean = _strip_nulls_deep(messages_raw) or []
            tools_clean = _canonicalize_tools(_strip_nulls_deep(row.get("tools") or []) or [])
            out.append(
                Trajectory(
                    session_id=str(row.get("run_id") or ""),
                    instance_id=str(row.get("instance_id") or ""),
                    messages=tuple(messages_clean),
                    tools=tuple(tools_clean),
                    assistant_indices=assistant_idx,
                    total_chars=total_chars,
                    estimated_tokens=est_tokens,
                    resolved=bool(row.get("resolved")),
                )
            )

    rng = random.Random(seed)
    rng.shuffle(out)
    return out


# --- profile CLI -------------------------------------------------------------


def _quantile(xs: list[int | float], p: float) -> float:
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(p * (len(xs) - 1))))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--min-turns", type=int, default=8)
    parser.add_argument("--min-total-tokens", type=int, default=8000)
    parser.add_argument("--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report",
        action="store_true",
        help="print distribution summary",
    )
    args = parser.parse_args()

    pool = load_pool(
        args.dataset_path,
        min_turns=args.min_turns,
        min_total_tokens=args.min_total_tokens,
        chars_per_token=args.chars_per_token,
        seed=args.seed,
    )

    if not args.report:
        for traj in pool[:50]:
            print(f"{traj.session_id}\t{traj.instance_id}\tturns={len(traj.messages)}\testimated_tokens={traj.estimated_tokens}")
        if len(pool) > 50:
            print(f"... ({len(pool) - 50} more)")
        return 0

    if not pool:
        print(f"pool is EMPTY after filter (min_turns={args.min_turns}, min_total_tokens={args.min_total_tokens})")
        return 1

    n_turns = [len(t.messages) for t in pool]
    n_tokens = [t.estimated_tokens for t in pool]
    n_assistant = [len(t.assistant_indices) for t in pool]
    resolved_count = sum(1 for t in pool if t.resolved)

    print(f"dataset       : {args.dataset_path}")
    print(f"filter        : turns >= {args.min_turns}, tokens >= {args.min_total_tokens} (chars/token = {args.chars_per_token})")
    print(f"pool size     : {len(pool)}")
    print(f"resolved      : {resolved_count} ({resolved_count*100/len(pool):.1f}%)")
    print()
    print(f"{'metric':<25} {'min':>8} {'p25':>8} {'median':>8} {'p75':>8} {'p95':>8} {'max':>8}")
    for label, xs in [
        ("turn count", n_turns),
        ("estimated tokens", n_tokens),
        ("assistant calls/session", n_assistant),
    ]:
        print(
            f"{label:<25} "
            f"{_quantile(xs, 0.0):>8.0f} "
            f"{_quantile(xs, 0.25):>8.0f} "
            f"{_quantile(xs, 0.5):>8.0f} "
            f"{_quantile(xs, 0.75):>8.0f} "
            f"{_quantile(xs, 0.95):>8.0f} "
            f"{_quantile(xs, 1.0):>8.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
