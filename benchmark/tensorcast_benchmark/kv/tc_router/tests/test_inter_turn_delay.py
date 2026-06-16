"""Inter-turn delay sampler quantile sanity tests."""

from __future__ import annotations

import math
import statistics

import pytest

from tensorcast_benchmark.kv.tc_router.workload.inter_turn_delay import (
    DelayParams,
    LogNormalSampler,
    PRESET_PARAMS,
    Preset,
    p90_seconds,
)


def test_preset_params_match_arch_table() -> None:
    assert PRESET_PARAMS[Preset.AGENT_FAST] == (2.1, 0.6)
    assert PRESET_PARAMS[Preset.AGENT_MEDIUM] == (3.0, 0.8)
    assert PRESET_PARAMS[Preset.AGENT_SLOW] == (4.1, 1.0)


def test_delay_params_theoretical_quantiles_agent_medium() -> None:
    p = DelayParams.from_preset(Preset.AGENT_MEDIUM)
    # arch § 5.3.1 table:
    assert math.isclose(p.median(), 20.085, rel_tol=0.001)
    assert math.isclose(p.mean(), 27.66, rel_tol=0.005)
    assert math.isclose(p.quantile(0.05), 5.40, rel_tol=0.005)
    assert math.isclose(p.quantile(0.95), 74.80, rel_tol=0.005)


def test_p90_helper_matches_quantile() -> None:
    assert math.isclose(
        p90_seconds(Preset.AGENT_MEDIUM),
        DelayParams.from_preset(Preset.AGENT_MEDIUM).quantile(0.90),
        rel_tol=1e-12,
    )


def test_custom_preset_requires_both_params() -> None:
    with pytest.raises(ValueError, match="custom"):
        DelayParams.from_preset(Preset.CUSTOM)
    with pytest.raises(ValueError, match="custom"):
        DelayParams.from_preset(Preset.CUSTOM, custom_mu=1.0)


def test_custom_preset_uses_supplied_params() -> None:
    p = DelayParams.from_preset(Preset.CUSTOM, custom_mu=1.5, custom_sigma=0.3)
    assert (p.mu, p.sigma) == (1.5, 0.3)


def test_sampler_seed_is_deterministic() -> None:
    p = DelayParams.from_preset(Preset.AGENT_MEDIUM)
    a = LogNormalSampler(p, seed=42)
    b = LogNormalSampler(p, seed=42)
    seq_a = [a() for _ in range(10)]
    seq_b = [b() for _ in range(10)]
    assert seq_a == seq_b


def test_sampler_varying_seed_diverges() -> None:
    p = DelayParams.from_preset(Preset.AGENT_MEDIUM)
    a = LogNormalSampler(p, seed=1)
    b = LogNormalSampler(p, seed=2)
    assert [a() for _ in range(5)] != [b() for _ in range(5)]


# --- arch-table empirical sanity (the plan validation gate) -----------------


@pytest.mark.parametrize(
    "preset,expected_median,med_tol,expected_p95,p95_tol",
    [
        # Validation gate per plan § 3: 10000 samples from agent_medium,
        # P50 in [18, 22]s, P95 in [70, 80]s. Extend to all three presets.
        (Preset.AGENT_FAST, 8.17, 0.5, 21.95, 1.5),
        (Preset.AGENT_MEDIUM, 20.09, 1.0, 74.80, 4.0),
        (Preset.AGENT_SLOW, 60.34, 4.0, 311.0, 25.0),
    ],
)
def test_empirical_quantiles_within_tolerance(
    preset: Preset,
    expected_median: float,
    med_tol: float,
    expected_p95: float,
    p95_tol: float,
) -> None:
    sampler = LogNormalSampler(DelayParams.from_preset(preset), seed=42)
    samples = sorted(sampler() for _ in range(10000))
    emp_median = statistics.median(samples)
    emp_p95 = samples[int(0.95 * (len(samples) - 1))]
    assert abs(emp_median - expected_median) < med_tol, (
        f"{preset.value} median: empirical={emp_median:.2f} expected~{expected_median:.2f}"
    )
    assert abs(emp_p95 - expected_p95) < p95_tol, (
        f"{preset.value} P95: empirical={emp_p95:.2f} expected~{expected_p95:.2f}"
    )
