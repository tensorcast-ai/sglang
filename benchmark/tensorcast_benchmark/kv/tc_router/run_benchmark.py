#!/usr/bin/env python3
"""tc_router benchmark entry point.

Usage:

  python -m tensorcast_benchmark.kv.tc_router.run_benchmark \\
      --cluster configs/cluster_brainctl_single_h800.yaml \\
      --bench   configs/benchmark_baseline_smoke.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .driver.benchmark_loop import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cluster", required=True, type=Path)
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=None,
        help="Override the outputs/ directory; defaults to tc_router/outputs/.",
    )
    parser.add_argument(
        "--config-filter",
        default=None,
        help="Comma-separated subset of `bench.configs[].kind` to run. "
             "Default: all configs in bench.yaml that this Phase supports.",
    )
    parser.add_argument(
        "--sglang-ready-timeout-s",
        type=float,
        default=1500.0,
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config_filter: set[str] | None = None
    if args.config_filter:
        config_filter = {s.strip() for s in args.config_filter.split(",") if s.strip()}

    run_dir = asyncio.run(
        run_benchmark(
            cluster_yaml=args.cluster,
            bench_yaml=args.bench,
            outputs_root=args.outputs_root,
            config_filter=config_filter,
            sglang_ready_timeout_s=args.sglang_ready_timeout_s,
        )
    )
    print(f"\nDONE. run_dir = {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
