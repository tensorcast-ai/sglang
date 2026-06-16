"""Inter-turn delay sampler, with the three named presets from arch § 5.3.1.

Usage as a sanity-check CLI:

  python -m tensorcast_benchmark.kv.tc_router.workload.inter_turn_delay \\
      --preset agent_medium --n 10000 --report
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Preset(str, Enum):
    AGENT_FAST = "agent_fast"
    AGENT_MEDIUM = "agent_medium"
    AGENT_SLOW = "agent_slow"
    CUSTOM = "custom"


# (mu, sigma) of the underlying normal distribution.
# Values come from arch § 5.3.1 — DO NOT EDIT without updating the doc.
PRESET_PARAMS: dict[Preset, tuple[float, float]] = {
    Preset.AGENT_FAST: (2.1, 0.6),
    Preset.AGENT_MEDIUM: (3.0, 0.8),
    Preset.AGENT_SLOW: (4.1, 1.0),
}


@dataclass(frozen=True)
class DelayParams:
    mu: float
    sigma: float

    @classmethod
    def from_preset(
        cls,
        preset: Preset,
        *,
        custom_mu: Optional[float] = None,
        custom_sigma: Optional[float] = None,
    ) -> "DelayParams":
        if preset == Preset.CUSTOM:
            if custom_mu is None or custom_sigma is None:
                raise ValueError(
                    "preset=custom requires both custom_mu and custom_sigma"
                )
            return cls(mu=float(custom_mu), sigma=float(custom_sigma))
        if preset not in PRESET_PARAMS:
            raise ValueError(f"unknown preset: {preset!r}")
        mu, sigma = PRESET_PARAMS[preset]
        return cls(mu=mu, sigma=sigma)

    def median(self) -> float:
        return math.exp(self.mu)

    def mean(self) -> float:
        return math.exp(self.mu + self.sigma**2 / 2)

    def quantile(self, p: float) -> float:
        if not 0.0 < p < 1.0:
            raise ValueError("p must be in (0, 1)")
        z = statistics.NormalDist(0.0, 1.0).inv_cdf(p)
        return math.exp(self.mu + self.sigma * z)


class LogNormalSampler:
    """Callable returning seconds drawn from `LogNormal(mu, sigma)`."""

    def __init__(self, params: DelayParams, *, seed: Optional[int] = None) -> None:
        self._params = params
        self._rng = random.Random(seed)

    @property
    def params(self) -> DelayParams:
        return self._params

    def __call__(self) -> float:
        return self._rng.lognormvariate(self._params.mu, self._params.sigma)


def p90_seconds(
    preset: Preset,
    *,
    custom_mu: Optional[float] = None,
    custom_sigma: Optional[float] = None,
) -> float:
    """Return the 90th-percentile seconds of the named preset.

    Used in `ThresholdPolicy.should_consider_session_for_migration` per arch
    § 6.4 (the policy considers a session "still active" if it was last
    active within p90 of the inter-turn delay distribution).
    """
    return DelayParams.from_preset(
        preset, custom_mu=custom_mu, custom_sigma=custom_sigma
    ).quantile(0.90)


# --- CLI ---------------------------------------------------------------------


def _quantile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    idx = max(0, min(len(xs) - 1, int(p * (len(xs) - 1))))
    return xs[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        default="agent_medium",
        choices=[p.value for p in Preset if p != Preset.CUSTOM] + ["custom"],
    )
    parser.add_argument("--custom-mu", type=float, default=None)
    parser.add_argument("--custom-sigma", type=float, default=None)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    preset = Preset(args.preset)
    params = DelayParams.from_preset(
        preset, custom_mu=args.custom_mu, custom_sigma=args.custom_sigma
    )
    sampler = LogNormalSampler(params, seed=args.seed)
    samples = [sampler() for _ in range(args.n)]

    if not args.report:
        for v in samples:
            print(f"{v:.3f}")
        return 0

    theo_median = params.median()
    theo_mean = params.mean()
    theo_p5 = params.quantile(0.05)
    theo_p90 = params.quantile(0.90)
    theo_p95 = params.quantile(0.95)
    theo_p99 = params.quantile(0.99)

    emp_median = _quantile(samples, 0.50)
    emp_mean = sum(samples) / len(samples)
    emp_p5 = _quantile(samples, 0.05)
    emp_p90 = _quantile(samples, 0.90)
    emp_p95 = _quantile(samples, 0.95)
    emp_p99 = _quantile(samples, 0.99)

    print(f"preset={preset.value} mu={params.mu} sigma={params.sigma} n={args.n} seed={args.seed}")
    print()
    print(f"{'metric':<10} {'theoretical':>14} {'empirical':>14} {'rel_err':>8}")
    for label, theo, emp in [
        ("p5", theo_p5, emp_p5),
        ("median", theo_median, emp_median),
        ("mean", theo_mean, emp_mean),
        ("p90", theo_p90, emp_p90),
        ("p95", theo_p95, emp_p95),
        ("p99", theo_p99, emp_p99),
    ]:
        rel_err = abs(emp - theo) / theo if theo else 0.0
        print(f"{label:<10} {theo:>14.2f} {emp:>14.2f} {rel_err*100:>7.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
