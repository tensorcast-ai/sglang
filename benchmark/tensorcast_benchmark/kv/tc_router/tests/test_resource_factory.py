"""Provider factory dispatch tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from tensorcast_benchmark.kv.tc_router.resource import factory
from tensorcast_benchmark.kv.tc_router.resource.local import LocalProvider

# Reuse the GOOD_YAML fixture from test_resource_base via a local copy.
GOOD_YAML: dict = {
    "provider": {
        "kind": "local",
    },
    "driver_host": {"scratch_dir": "/tmp/tc_router/scratch"},
    "mount": {"path": "/tmp/tc_router"},
    "workers": [
        {
            "id": "worker_a",
            "address": "127.0.0.1",
            "node": "node-1",
            "process_handle": "local-a",
            "gpu_indices": [0, 1],
            "scratch_dir": "/tmp/tc_router/worker_a",
            "base_env": {},
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


def test_factory_dispatches_to_local(tmp_path: Path) -> None:
    provider = factory.from_cluster_config(_write_yaml(tmp_path, GOOD_YAML))
    assert isinstance(provider, LocalProvider)
    workers = provider.workers()
    assert len(workers) == 1
    w = workers[0]
    assert w.id == "worker_a"
    assert w.address == "127.0.0.1"
    assert w.node == "node-1"
    assert w.process_handle == "local-a"
    assert w.base_env == {}


def test_factory_rejects_unknown_kind(tmp_path: Path) -> None:
    bad = copy.deepcopy(GOOD_YAML)
    bad["provider"]["kind"] = "imaginary_cluster_cli"
    bad["workers"][0]["base_env"] = {"REQUIRED_FOR_NON_LOCAL": "1"}
    with pytest.raises(ValueError, match="unknown provider.kind"):
        factory.from_cluster_config(_write_yaml(tmp_path, bad))


def test_factory_registered_kinds_include_local_and_static() -> None:
    assert factory.registered_kinds() == ["local", "static"]


def test_local_provider_workers_idempotent(tmp_path: Path) -> None:
    """Calling workers() twice returns equivalent lists (cached)."""
    provider = factory.from_cluster_config(_write_yaml(tmp_path, GOOD_YAML))
    a = provider.workers()
    b = provider.workers()
    assert [w.id for w in a] == [w.id for w in b]
