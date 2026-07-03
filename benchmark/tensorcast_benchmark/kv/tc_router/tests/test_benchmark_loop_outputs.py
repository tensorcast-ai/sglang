"""Tests for run-directory scoped benchmark artifacts."""

from __future__ import annotations

from pathlib import Path

import yaml

from tensorcast_benchmark.kv.tc_router.driver.benchmark_loop import (
    _cluster_config_for_run,
    _derive_nccl_port,
    _save_resolved_configs,
)
from tensorcast_benchmark.kv.tc_router.driver.config import load_benchmark_yaml
from tensorcast_benchmark.kv.tc_router.resource.base import load_cluster_config


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _local_cluster() -> dict:
    return {
        "provider": {"kind": "local"},
        "driver_host": {"scratch_dir": "/old/driver"},
        "mount": {"path": "/old/mount", "spec": "local"},
        "workers": [
            {
                "id": "local_test",
                "address": "127.0.0.1",
                "node": "local_test",
                "process_handle": "local",
                "gpu_indices": [0, 1],
                "scratch_dir": "/old/worker",
                "base_env": {},
                "env_path_prepend": {"LD_LIBRARY_PATH": ["/compat"]},
            }
        ],
        "service_placement": {
            "global_store_worker_id": "local_test",
            "mooncake_master_worker_id": "local_test",
        },
    }


def _static_cluster() -> dict:
    return {
        "provider": {"kind": "static"},
        "driver_host": {"scratch_dir": "/mnt/data/tc_router_unit/old_driver"},
        "mount": {"path": "/mnt/data", "spec": "shared"},
        "workers": [
            {
                "id": "local_static",
                "address": "127.0.0.1",
                "node": "local_static",
                "process_handle": "local",
                "execution": "local",
                "gpu_indices": [0, 1],
                "scratch_dir": "/mnt/data/tc_router_unit/old_worker",
                "base_env": {},
                "env_path_prepend": {"LD_LIBRARY_PATH": ["/compat"]},
            }
        ],
        "service_placement": {
            "global_store_worker_id": "local_static",
            "mooncake_master_worker_id": "local_static",
        },
    }


def _benchmark() -> dict:
    return {
        "run_id": "unit",
        "model": {"path": "/models/qwen", "tp_size": 1},
        "instances": {"count": 1, "base_port": 43001},
        "transport": {"use_rdma": False},
        "workload": {
            "dataset_path": "/data/workload",
            "pool_filter": {"min_turns": 1, "min_total_tokens": 1},
            "inter_turn_delay": {"preset": "agent_medium"},
            "max_new_tokens_clip": 16,
            "start_jitter_s": 0.0,
            "wall_seconds": 1,
            "warmup_seconds": 0,
            "trials": 1,
            "c_target_sweep": [1],
        },
        "configs": [{"kind": "gw_load_aware"}],
    }


def test_local_cluster_scratch_paths_are_scoped_to_run_dir(tmp_path: Path) -> None:
    cluster_yaml = _write_yaml(tmp_path / "cluster.yaml", _local_cluster())
    run_dir = tmp_path / "outputs" / "run"

    cfg = _cluster_config_for_run(cluster_yaml, run_dir)

    assert cfg.driver_host.scratch_dir == str(run_dir / "scratch" / "driver")
    assert cfg.mount.path == str(run_dir / "mount")
    assert cfg.workers[0].scratch_dir == str(
        run_dir / "scratch" / "workers" / "local_test"
    )
    assert cfg.workers[0].env_path_prepend["LD_LIBRARY_PATH"] == ("/compat",)


def test_static_cluster_scratch_paths_are_scoped_to_run_dir(tmp_path: Path) -> None:
    cluster_yaml = _write_yaml(tmp_path / "static_cluster.yaml", _static_cluster())
    run_dir = Path("/mnt/data/tc_router_unit_outputs/run")

    cfg = _cluster_config_for_run(cluster_yaml, run_dir)

    assert cfg.driver_host.scratch_dir == str(run_dir / "scratch" / "driver")
    assert cfg.mount.path == "/mnt/data"
    assert cfg.workers[0].scratch_dir == str(
        run_dir / "scratch" / "workers" / "local_static"
    )


def test_save_resolved_configs_writes_effective_cluster_yaml(tmp_path: Path) -> None:
    cluster_yaml = _write_yaml(tmp_path / "cluster_input.yaml", _local_cluster())
    bench_yaml = _write_yaml(tmp_path / "benchmark_input.yaml", _benchmark())
    run_dir = tmp_path / "outputs" / "run"
    run_dir.mkdir(parents=True)
    bench_cfg = load_benchmark_yaml(bench_yaml)
    cluster_cfg = _cluster_config_for_run(cluster_yaml, run_dir)

    _save_resolved_configs(
        run_dir,
        cluster_yaml=cluster_yaml,
        bench_yaml=bench_yaml,
        bench_cfg=bench_cfg,
        cluster_cfg=cluster_cfg,
    )

    effective = load_cluster_config(run_dir / "cluster.yaml")
    assert (run_dir / "cluster_input.yaml").is_file()
    assert (run_dir / "benchmark.yaml").is_file()
    assert (run_dir / "benchmark_resolved.yaml").is_file()
    assert effective.workers[0].scratch_dir.startswith(str(run_dir))


def test_derive_nccl_port_uses_nearby_non_serving_port() -> None:
    assert _derive_nccl_port(65101) == 65201
    assert _derive_nccl_port(65500) == 65400
