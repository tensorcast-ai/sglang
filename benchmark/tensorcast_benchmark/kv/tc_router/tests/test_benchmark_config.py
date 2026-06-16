"""Tests for benchmark.yaml config parsing."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from tensorcast_benchmark.kv.tc_router.driver.config import (
    BenchmarkConfig,
    load_benchmark_yaml,
)


GOOD: dict = {
    "run_id": "smoke",
    "model": {"path": "/p/Qwen3-32B", "tp_size": 2},
    "instances": {"count": 3, "base_port": 30001, "mem_fraction_static": 0.85},
    "transport": {"use_rdma": True},
    "workload": {
        "dataset_path": "/data/x",
        "pool_filter": {"min_turns": 8, "min_total_tokens": 8000},
        "inter_turn_delay": {"preset": "agent_medium"},
        "max_new_tokens_clip": 256,
        "start_jitter_s": 1.0,
        "wall_seconds": 60,
        "warmup_seconds": 0,
        "trials": 1,
        "c_target_sweep": [3, 6],
    },
    "configs": [
        {"kind": "gw_load_aware"},
        {"kind": "gw_cache_aware"},
    ],
}


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "bench.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_load_good_yaml(tmp_path: Path) -> None:
    cfg = load_benchmark_yaml(_write(tmp_path, GOOD))
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.model.tp_size == 2
    assert cfg.instances.count == 3
    assert cfg.workload.c_target_sweep == (3, 6)
    assert cfg.workload.inter_turn_delay.preset == "agent_medium"
    assert {c.kind for c in cfg.configs} == {"gw_load_aware", "gw_cache_aware"}


def test_custom_preset_requires_params(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD)
    bad["workload"]["inter_turn_delay"] = {"preset": "custom"}
    with pytest.raises(Exception, match="custom"):
        load_benchmark_yaml(_write(tmp_path, bad))


def test_empty_configs_rejected(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD)
    bad["configs"] = []
    with pytest.raises(Exception):
        load_benchmark_yaml(_write(tmp_path, bad))


def test_invalid_config_kind(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD)
    bad["configs"] = [{"kind": "bogus_router"}]
    with pytest.raises(Exception):
        load_benchmark_yaml(_write(tmp_path, bad))


def test_extra_field_rejected(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD)
    bad["unexpected"] = "x"
    with pytest.raises(Exception):
        load_benchmark_yaml(_write(tmp_path, bad))


def test_shipped_smoke_yaml_parses() -> None:
    """The shipped smoke YAML must parse cleanly."""
    p = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "benchmark_baseline_smoke.yaml"
    )
    cfg = load_benchmark_yaml(p)
    assert cfg.instances.count == 3
    assert cfg.model.tp_size == 2
    assert cfg.model.path.endswith("/Qwen3-32B")
