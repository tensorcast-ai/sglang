"""Per-cell run-summary aggregator (arch § 10.3).

Reads `turns.jsonl` (always present after a run) plus optional
`migrations.jsonl` (only emitted by `tc_router`) and produces a single
`RunSummary` row. The driver concatenates all rows into `summary.csv`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, ConfigDict


class RunSummary(BaseModel):
    """One row in `summary.csv`. Schema mirrors arch § 10.3 verbatim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: str
    c_target: int
    trial: int
    inter_turn_delay_preset: str
    transport_mode: str  # "rdma" | "tcp"

    ttft_p50_ms: Optional[float]
    ttft_p95_ms: Optional[float]
    ttft_p99_ms: Optional[float]
    ttft_mean_ms: Optional[float]

    cached_token_ratio_mean: Optional[float]

    total_turns_completed: int
    total_requests_failed: int

    migration_count: int = 0
    migration_utilization: Optional[float] = None
    mean_publish_latency_ms: Optional[float] = None
    mean_hydrate_latency_ms: Optional[float] = None


# --- helpers ----------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _quantile(xs: list[float], p: float) -> Optional[float]:
    if not xs:
        return None
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    s = sorted(xs)
    if p == 0.0:
        return s[0]
    if p == 1.0:
        return s[-1]
    # Linear interpolation between order statistics, equivalent to
    # numpy.percentile(method='linear').
    n = len(s)
    h = p * (n - 1)
    lo = int(h)
    hi = min(lo + 1, n - 1)
    frac = h - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _mean(xs: list[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


# --- public API -------------------------------------------------------------


def aggregate_cell(
    *,
    turns_path: Path,
    migrations_path: Optional[Path],
    config: str,
    c_target: int,
    trial: int,
    inter_turn_delay_preset: str,
    transport_mode: str,
) -> RunSummary:
    """Aggregate one `(config, c_target, trial)` cell into a `RunSummary`.

    `migrations_path` is optional — gateway baselines don't emit one.
    Both files are JSONL with the schemas declared in arch § 10.1 / § 10.2.
    """
    turns = _read_jsonl(Path(turns_path))
    successful = [t for t in turns if t.get("success") is True]
    failed_count = len(turns) - len(successful)

    ttft_values = [
        float(t["ttft_ms"]) for t in successful
        if t.get("ttft_ms") is not None
    ]
    cached_ratios: list[float] = []
    for t in successful:
        prompt_tokens = int(t.get("prompt_tokens") or 0)
        cached_tokens = int(t.get("cached_tokens") or 0)
        if prompt_tokens > 0:
            cached_ratios.append(cached_tokens / prompt_tokens)

    migrations: list[dict] = []
    if migrations_path is not None:
        migrations = _read_jsonl(Path(migrations_path))

    consumed = sum(
        1 for m in migrations
        if m.get("consumed_by_turn_rid") and not m.get("wasted")
    )
    publish_latencies = [
        float(m["publish_latency_ms"]) for m in migrations
        if m.get("publish_latency_ms") is not None
    ]
    hydrate_latencies = [
        float(m["hydrate_latency_ms"]) for m in migrations
        if m.get("hydrate_latency_ms") is not None
    ]
    migration_utilization: Optional[float]
    migration_utilization = (consumed / len(migrations)) if migrations else None

    return RunSummary(
        config=config,
        c_target=c_target,
        trial=trial,
        inter_turn_delay_preset=inter_turn_delay_preset,
        transport_mode=transport_mode,
        ttft_p50_ms=_quantile(ttft_values, 0.50),
        ttft_p95_ms=_quantile(ttft_values, 0.95),
        ttft_p99_ms=_quantile(ttft_values, 0.99),
        ttft_mean_ms=_mean(ttft_values),
        cached_token_ratio_mean=_mean(cached_ratios),
        total_turns_completed=len(turns),
        total_requests_failed=failed_count,
        migration_count=len(migrations),
        migration_utilization=migration_utilization,
        mean_publish_latency_ms=_mean(publish_latencies),
        mean_hydrate_latency_ms=_mean(hydrate_latencies),
    )


def write_summary_csv(rows: Iterable[RunSummary], path: Path) -> None:
    """Write the canonical `summary.csv` file. Field order matches arch § 10.3."""
    rows = list(rows)
    if not rows:
        # Still create an empty file with header so downstream tools see it.
        rows = []
    fieldnames = list(RunSummary.model_fields.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())
