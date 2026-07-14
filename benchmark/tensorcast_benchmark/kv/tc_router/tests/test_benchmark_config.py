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
        "warmup_counts": 0,
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
    assert cfg.instances.sglang_log_level is None
    assert cfg.workload.inter_turn_delay.preset == "agent_medium"
    assert cfg.workload.warmup_counts == 0
    assert {c.kind for c in cfg.configs} == {"gw_load_aware", "gw_cache_aware"}


def test_warmup_counts_rejects_negative_values(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD)
    bad["workload"]["warmup_counts"] = -1
    with pytest.raises(Exception):
        load_benchmark_yaml(_write(tmp_path, bad))


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


def test_mooncake_config_parses_without_prefetch_threshold(tmp_path: Path) -> None:
    data = copy.deepcopy(GOOD)
    data["configs"] = [{"kind": "gw_load_aware_mooncake"}]
    data["mooncake"] = {
        "http_metadata_server_port": 62300,
        "master_port": 62301,
        "global_segment_size": "64gb",
        "eviction_high_watermark_ratio": 0.9,
        "device_name": "",
        "clear_storage_between_cells": True,
    }

    cfg = load_benchmark_yaml(_write(tmp_path, data))

    assert cfg.configs[0].kind == "gw_load_aware_mooncake"
    assert cfg.mooncake.http_metadata_server_port == 62300
    assert cfg.mooncake.master_port == 62301
    assert cfg.mooncake.global_segment_size == "64gb"
    assert cfg.mooncake.clear_storage_between_cells is True


def test_mooncake_config_rejects_prefetch_threshold(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD)
    bad["configs"] = [{"kind": "gw_load_aware_mooncake"}]
    bad["mooncake"] = {
        "http_metadata_server_port": 62300,
        "master_port": 62301,
        "global_segment_size": "64gb",
        "eviction_high_watermark_ratio": 0.9,
        "device_name": "",
        "clear_storage_between_cells": True,
        "prefetch_threshold": 1,
    }

    with pytest.raises(Exception, match="prefetch_threshold"):
        load_benchmark_yaml(_write(tmp_path, bad))


def test_mooncake_config_rejects_duplicate_ports(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD)
    bad["configs"] = [{"kind": "gw_load_aware_mooncake"}]
    bad["mooncake"] = {
        "http_metadata_server_port": 62300,
        "master_port": 62300,
    }

    with pytest.raises(Exception, match="must be distinct"):
        load_benchmark_yaml(_write(tmp_path, bad))


def test_tensorcast_config_parses_allocator_mode(tmp_path: Path) -> None:
    data = copy.deepcopy(GOOD)
    data["configs"] = [{"kind": "tc_router"}]
    data["tensorcast"] = {
        "global_store_port": 61050,
        "daemon_port": 61053,
        "daemon_p2p_port": 61090,
        "instance_agent_base_port": 61400,
        "instance_agent_start_timeout_s": 180,
        "daemon_stable_bytes": "16GB",
        "clear_storage_between_cells": True,
        "hicache_mem_layout": "page_blob_direct",
        "hicache_io_backend": "direct",
        "host_allocator_enabled": True,
        "host_allocator_region_ttl_ms": 600000,
        "host_allocator_region_name_prefix": "tc_router_sglang_host_pool",
    }

    cfg = load_benchmark_yaml(_write(tmp_path, data))

    assert cfg.configs[0].kind == "tc_router"
    assert cfg.tensorcast.host_allocator_enabled is True
    assert cfg.tensorcast.hicache_mem_layout == "page_blob_direct"
    assert cfg.tensorcast.hicache_io_backend == "direct"
    assert cfg.tensorcast.instance_agent_start_timeout_s == 180


def test_tensorcast_allocator_requires_page_blob_direct(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD)
    bad["configs"] = [{"kind": "tc_router"}]
    bad["tensorcast"] = {
        "host_allocator_enabled": True,
        "hicache_mem_layout": "page_first_direct",
        "hicache_io_backend": "direct",
    }

    with pytest.raises(Exception, match="page_blob_direct"):
        load_benchmark_yaml(_write(tmp_path, bad))


def test_tensorcast_allocator_requires_direct_io(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD)
    bad["configs"] = [{"kind": "tc_router"}]
    bad["tensorcast"] = {
        "host_allocator_enabled": True,
        "hicache_mem_layout": "page_blob_direct",
        "hicache_io_backend": "kernel",
    }

    with pytest.raises(Exception, match="direct"):
        load_benchmark_yaml(_write(tmp_path, bad))


def test_tensorcast_instance_agent_port_overflow_rejected(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD)
    bad["configs"] = [{"kind": "tc_router"}]
    bad["instances"]["count"] = 3
    bad["tensorcast"] = {"instance_agent_base_port": 65534}

    with pytest.raises(Exception, match="instance_agent_base_port"):
        load_benchmark_yaml(_write(tmp_path, bad))


def test_tensorcast_port_collision_rejected_for_tc_router(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD)
    bad["configs"] = [{"kind": "tc_router"}]
    bad["tensorcast"] = {"global_store_port": bad["instances"]["base_port"]}

    with pytest.raises(Exception, match="port collision"):
        load_benchmark_yaml(_write(tmp_path, bad))


def test_tensorcast_default_ports_not_consumed_by_gateway_only(tmp_path: Path) -> None:
    data = copy.deepcopy(GOOD)
    data["configs"] = [{"kind": "gw_load_aware"}]
    data["gateway"] = {"host": "127.0.0.1", "port": 61050}

    cfg = load_benchmark_yaml(_write(tmp_path, data))

    assert cfg.configs[0].kind == "gw_load_aware"
    assert cfg.gateway.port == 61050


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


def test_shipped_local_tc_router_smoke_yaml_parses() -> None:
    """The local single-node tc_router smoke YAML must parse cleanly."""
    p = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "benchmark_local_tc_router_smoke.yaml"
    )
    cfg = load_benchmark_yaml(p)
    assert cfg.transport.use_rdma is False
    assert cfg.model.path == "/mnt/data/models/Qwen3-32B"
    assert cfg.model.tp_size == 2
    assert cfg.instances.count == 3
    assert cfg.workload.max_new_tokens_clip == 256
    assert cfg.workload.wall_seconds == 60
    assert (
        cfg.workload.dataset_path == "/mnt/data/dataset/OpenHands-Sampled-Trajectories"
    )
    assert cfg.configs[0].kind == "tc_router"
    assert cfg.configs[0].policy == {"kind": "never_rebalance", "seed": 0}


def test_shipped_local_baseline_tp1_smoke_yaml_parses() -> None:
    """The local TP=1 gateway baseline smoke YAML must parse cleanly."""
    p = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "benchmark_local_baseline_tp1_smoke.yaml"
    )
    cfg = load_benchmark_yaml(p)
    assert cfg.model.tp_size == 1
    assert cfg.instances.count == 3
    assert cfg.transport.use_rdma is False
    assert {c.kind for c in cfg.configs} == {"gw_load_aware", "gw_cache_aware"}


def test_shipped_local_baseline_tp1_8inst_debug_yaml_parses() -> None:
    """The local TP=1 8-instance debug baseline YAML must parse cleanly."""
    p = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "benchmark_local_baseline_tp1_8inst_debug.yaml"
    )
    cfg = load_benchmark_yaml(p)
    assert cfg.model.tp_size == 1
    assert cfg.instances.count == 8
    assert cfg.instances.sglang_log_level == "debug"
    assert cfg.workload.c_target_sweep == (3, 6)


def test_shipped_static_baseline_8inst_tp2_yaml_parses() -> None:
    """The mixed local+SSH static gateway baseline YAML must parse cleanly."""
    p = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "benchmark_static_baseline_8inst_tp2.yaml"
    )
    cfg = load_benchmark_yaml(p)
    assert cfg.run_id == "static-baseline-8inst-tp2"
    assert cfg.model.path == "/mnt/data/models/Qwen3-32B"
    assert cfg.model.tp_size == 2
    assert cfg.instances.count == 8
    assert cfg.instances.base_port == 62101
    assert cfg.instances.sglang_log_level == "debug"
    assert cfg.transport.use_rdma is False
    assert {c.kind for c in cfg.configs} == {"gw_load_aware", "gw_cache_aware"}
    assert (
        cfg.workload.dataset_path == "/mnt/data/dataset/OpenHands-Sampled-Trajectories"
    )


def test_shipped_static_cache_aware_cacheonly_yaml_parses() -> None:
    """The cache-aware-only static baseline YAML must parse cleanly."""
    p = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "benchmark_static_cache_aware_cacheonly_4inst_tp2_c4_8_16_32_64_wall300_warmup10.yaml"
    )
    cfg = load_benchmark_yaml(p)
    assert cfg.run_id == (
        "static-cache-aware-cacheonly-4inst-tp2-c4-8-16-32-64-wall300-warmup10"
    )
    assert cfg.model.path == "/mnt/data/models/Qwen3-32B"
    assert cfg.model.tp_size == 2
    assert cfg.instances.count == 4
    assert len(cfg.configs) == 1
    assert cfg.configs[0].kind == "gw_cache_aware"
    assert cfg.configs[0].policy == {
        "cache_threshold": 0.0,
        "balance_abs_threshold": 1_000_000,
        "balance_rel_threshold": 1.5,
    }
    assert cfg.workload.c_target_sweep == (4, 8, 16, 32, 64)


def test_shipped_static_tc_router_8inst_tp2_yaml_parses() -> None:
    """The mixed local+SSH static tc_router smoke YAML must parse cleanly."""
    p = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "benchmark_static_tc_router_8inst_tp2.yaml"
    )
    cfg = load_benchmark_yaml(p)
    assert cfg.run_id == "static-tc-router-8inst-tp2"
    assert cfg.model.path == "/mnt/data/models/Qwen3-32B"
    assert cfg.model.tp_size == 2
    assert cfg.instances.count == 8
    assert cfg.instances.base_port == 62101
    assert cfg.instances.sglang_log_level == "debug"
    assert cfg.transport.use_rdma is False
    assert len(cfg.configs) == 1
    assert cfg.configs[0].kind == "tc_router"
    assert cfg.configs[0].policy == {"kind": "never_rebalance", "seed": 0}
    assert (
        cfg.workload.dataset_path == "/mnt/data/dataset/OpenHands-Sampled-Trajectories"
    )


def test_shipped_static_tc_router_migration_smoke_yaml_parses() -> None:
    p = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "benchmark_static_tc_router_migration_smoke_4inst_tp2.yaml"
    )
    cfg = load_benchmark_yaml(p)
    assert cfg.run_id == "static-tc-router-migration-smoke-4inst-tp2"
    assert cfg.configs[0].kind == "tc_router"
    assert cfg.configs[0].policy["kind"] == "migrate_once_after_turn"
    assert cfg.model.path == "/mnt/data/models/Qwen3-32B"
    assert cfg.model.tp_size == 2
    assert cfg.instances.count == 4
    assert cfg.transport.use_rdma is True
    assert cfg.tensorcast.host_allocator_enabled is True
    assert cfg.tensorcast.hicache_mem_layout == "page_blob_direct"
    assert cfg.tensorcast.hicache_io_backend == "direct"


def test_shipped_static_mooncake_4inst_tp2_yaml_parses() -> None:
    """The mixed local+SSH static Mooncake baseline YAML must parse cleanly."""
    p = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "benchmark_static_mooncake_4inst_tp2_c4_8_16_32_64_wall600_warmup30.yaml"
    )
    cfg = load_benchmark_yaml(p)
    assert cfg.run_id == "static-mooncake-4inst-tp2-c4-8-16-32-64-wall600-warmup30"
    assert cfg.model.path == "/mnt/data/models/Qwen3-32B"
    assert cfg.model.tp_size == 2
    assert cfg.instances.count == 4
    assert cfg.instances.base_port == 62101
    assert cfg.transport.use_rdma is False
    assert len(cfg.configs) == 1
    assert cfg.configs[0].kind == "gw_load_aware_mooncake"
    assert cfg.mooncake.http_metadata_server_port == 62300
    assert cfg.mooncake.master_port == 62301
    assert cfg.mooncake.clear_storage_between_cells is True
