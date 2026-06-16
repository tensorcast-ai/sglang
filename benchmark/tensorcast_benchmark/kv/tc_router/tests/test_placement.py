"""Tests for driver/placement.py."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tensorcast_benchmark.kv.tc_router.driver.placement import (
    InstanceAssignment,
    plan_instance_placement,
)


@dataclass
class _FakeWorker:
    id: str
    address: str
    gpu_indices: tuple[int, ...]


def test_one_instance_per_worker_each_uses_first_gpus() -> None:
    """If each worker has exactly tp_size GPUs, one instance lands per worker."""
    workers = [
        _FakeWorker("w0", "10.0.0.1", (0, 1)),
        _FakeWorker("w1", "10.0.0.2", (0, 1)),
        _FakeWorker("w2", "10.0.0.3", (0, 1)),
    ]
    plan = plan_instance_placement(workers, instances_count=3, tp_size=2, base_port=30001)
    assert len(plan) == 3
    assert [a.worker.id for a in plan] == ["w0", "w1", "w2"]
    assert [a.port for a in plan] == [30001, 30002, 30003]
    assert plan[0].gpu_indices == (0, 1)
    assert plan[1].gpu_indices == (0, 1)
    assert plan[2].gpu_indices == (0, 1)


def test_three_instances_on_single_worker_uses_disjoint_gpu_windows() -> None:
    """Phase 5 smoke layout: 3 instances × TP=2 on one 8-GPU worker."""
    w = _FakeWorker("w0", "10.0.0.1", tuple(range(8)))
    plan = plan_instance_placement([w], instances_count=3, tp_size=2, base_port=30001)
    assert len(plan) == 3
    assert all(a.worker.id == "w0" for a in plan)
    assert [a.port for a in plan] == [30001, 30002, 30003]
    # Disjoint windows.
    assert plan[0].gpu_indices == (0, 1)
    assert plan[1].gpu_indices == (2, 3)
    assert plan[2].gpu_indices == (4, 5)


def test_instance_id_uses_address_and_port() -> None:
    w = _FakeWorker("w0", "10.0.0.5", (0, 1))
    plan = plan_instance_placement([w], instances_count=1, tp_size=2, base_port=30050)
    assert plan[0].instance_id == "10.0.0.5:30050"


def test_packs_then_overflows_to_next_worker() -> None:
    workers = [
        _FakeWorker("w0", "10.0.0.1", tuple(range(4))),  # fits 2 × TP=2
        _FakeWorker("w1", "10.0.0.2", tuple(range(4))),
    ]
    plan = plan_instance_placement(workers, instances_count=3, tp_size=2, base_port=30001)
    assert [a.worker.id for a in plan] == ["w0", "w0", "w1"]
    assert [a.gpu_indices for a in plan] == [(0, 1), (2, 3), (0, 1)]


def test_insufficient_gpus_raises() -> None:
    workers = [_FakeWorker("w0", "10.0.0.1", (0, 1))]
    with pytest.raises(ValueError, match="insufficient GPUs"):
        plan_instance_placement(workers, instances_count=2, tp_size=2)


def test_empty_workers_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plan_instance_placement([], instances_count=1, tp_size=1)


def test_zero_count_rejected() -> None:
    w = _FakeWorker("w0", "10.0.0.1", (0, 1))
    with pytest.raises(ValueError, match=r"instances_count"):
        plan_instance_placement([w], instances_count=0, tp_size=1)
