"""Cluster-config YAML parsing and validation tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from tensorcast_benchmark.kv.tc_router.resource.base import (
    ClusterConfig,
    load_cluster_config,
)


GOOD_YAML: dict = {
    "provider": {
        "kind": "brainctl",
        "namespace": "shai-core",
        "cli": "brainctl",
        "user": "alice",
    },
    "driver_host": {"scratch_dir": "/mnt/jfs/scratch"},
    "mount": {"path": "/mnt/jfs"},
    "workers": [
        {
            "id": "worker_a",
            "address": "10.0.0.1",
            "node": "node-1",
            "process_handle": "rjob-001",
            "gpu_indices": [0, 1],
            "scratch_dir": "/mnt/jfs/worker_a",
            "base_env": {"NCCL_IB_HCA": "mlx5_2", "MASTER_ADDR": "10.0.0.1"},
        },
        {
            "id": "worker_b",
            "address": "10.0.0.2",
            "node": "node-2",
            "process_handle": "rjob-002",
            "gpu_indices": [0, 1],
            "scratch_dir": "/mnt/jfs/worker_b",
            "base_env": {"NCCL_IB_HCA": "mlx5_2", "MASTER_ADDR": "10.0.0.1"},
        },
    ],
    "service_placement": {
        "global_store_worker_id": "worker_a",
        "mooncake_master_worker_id": "worker_a",
    },
}


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cluster.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_good_cluster(tmp_path: Path) -> None:
    cfg = load_cluster_config(_write_yaml(tmp_path, GOOD_YAML))
    assert cfg.provider.kind == "brainctl"
    assert cfg.provider.user == "alice"
    assert len(cfg.workers) == 2
    assert cfg.workers[0].id == "worker_a"
    assert cfg.workers[0].gpu_indices == (0, 1)
    assert cfg.workers[0].base_env["NCCL_IB_HCA"] == "mlx5_2"
    assert cfg.mount.path == "/mnt/jfs"
    assert cfg.service_placement.global_store_worker_id == "worker_a"


def test_reject_duplicate_id(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD_YAML)
    bad["workers"][1]["id"] = "worker_a"
    with pytest.raises(Exception, match="duplicate worker.id"):
        load_cluster_config(_write_yaml(tmp_path, bad))


def test_reject_duplicate_address(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD_YAML)
    bad["workers"][1]["address"] = "10.0.0.1"
    with pytest.raises(Exception, match="duplicate worker.address"):
        load_cluster_config(_write_yaml(tmp_path, bad))


def test_reject_duplicate_node(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD_YAML)
    bad["workers"][1]["node"] = "node-1"
    with pytest.raises(Exception, match="duplicate worker.node"):
        load_cluster_config(_write_yaml(tmp_path, bad))


def test_reject_empty_base_env(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD_YAML)
    bad["workers"][0]["base_env"] = {}
    with pytest.raises(Exception, match="base_env is empty"):
        load_cluster_config(_write_yaml(tmp_path, bad))


def test_reject_invalid_global_store_placement(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD_YAML)
    bad["service_placement"]["global_store_worker_id"] = "worker_z"
    with pytest.raises(Exception, match="global_store_worker_id.*not in workers"):
        load_cluster_config(_write_yaml(tmp_path, bad))


def test_reject_invalid_mooncake_placement(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD_YAML)
    bad["service_placement"]["mooncake_master_worker_id"] = "worker_z"
    with pytest.raises(Exception, match="mooncake_master_worker_id.*not in workers"):
        load_cluster_config(_write_yaml(tmp_path, bad))


def test_reject_missing_node(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD_YAML)
    del bad["workers"][0]["node"]
    with pytest.raises(Exception):
        load_cluster_config(_write_yaml(tmp_path, bad))


def test_reject_zero_workers(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD_YAML)
    bad["workers"] = []
    bad["service_placement"] = {
        "global_store_worker_id": "x",
        "mooncake_master_worker_id": "x",
    }
    with pytest.raises(Exception):
        load_cluster_config(_write_yaml(tmp_path, bad))


def test_reject_extra_field(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD_YAML)
    bad["unexpected_field"] = "oops"
    with pytest.raises(Exception):
        load_cluster_config(_write_yaml(tmp_path, bad))


def test_example_cluster_yaml_loads() -> None:
    """The shipped example YAML must parse cleanly."""
    example = (
        Path(__file__).parent.parent
        / "configs"
        / "cluster_brainctl_example.yaml"
    )
    cfg = load_cluster_config(example)
    assert isinstance(cfg, ClusterConfig)
    assert len(cfg.workers) == 3
    assert cfg.provider.kind == "brainctl"
