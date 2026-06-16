"""Compute SGLang instance → (worker, port, GPU subset) placement.

Greedy-pack: walk workers in order; on each worker, allocate as many
non-overlapping `tp_size`-GPU windows as fit. Stop once `instances_count`
assignments are produced. Fail if total GPUs are insufficient.

This handles both:
  - one-instance-per-worker  (3 instances × 3 workers × 2 GPUs = 6 GPUs)
  - many-instances-per-worker (3 instances × 1 worker × 6 GPUs = 6 GPUs,
    Phase 5 smoke layout)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class InstanceAssignment:
    """One SGLang instance's planned placement on a Worker."""

    worker: object  # avoid type cycle on `Worker` Protocol
    port: int
    gpu_indices: tuple[int, ...]

    @property
    def instance_id(self) -> str:
        # Matches the SGLang launcher's endpoint shape.
        return f"{getattr(self.worker, 'address')}:{self.port}"


def plan_instance_placement(
    workers: Sequence,
    *,
    instances_count: int,
    tp_size: int,
    base_port: int = 30001,
) -> list[InstanceAssignment]:
    """Greedy pack `instances_count` SGLang instances across `workers`.

    Each instance consumes `tp_size` consecutive GPU indices on a worker;
    multiple instances on one worker get disjoint windows
    (`gpu_indices[0:tp]`, `gpu_indices[tp:2*tp]`, ...).

    Ports are assigned sequentially starting from `base_port`.
    """
    if instances_count < 1:
        raise ValueError("instances_count must be >= 1")
    if tp_size < 1:
        raise ValueError("tp_size must be >= 1")
    if not workers:
        raise ValueError("workers must be non-empty")

    assignments: list[InstanceAssignment] = []
    next_port = base_port
    for worker in workers:
        gpu_indices = tuple(getattr(worker, "gpu_indices") or ())
        max_per_worker = len(gpu_indices) // tp_size
        for window_idx in range(max_per_worker):
            if len(assignments) >= instances_count:
                break
            slice_start = window_idx * tp_size
            slice_end = slice_start + tp_size
            window = tuple(gpu_indices[slice_start:slice_end])
            assignments.append(
                InstanceAssignment(
                    worker=worker,
                    port=next_port,
                    gpu_indices=window,
                )
            )
            next_port += 1
        if len(assignments) >= instances_count:
            break

    if len(assignments) < instances_count:
        total_gpus = sum(len(getattr(w, "gpu_indices") or ()) for w in workers)
        raise ValueError(
            f"insufficient GPUs to place {instances_count} instances at TP={tp_size}: "
            f"need {instances_count * tp_size}, available {total_gpus} "
            f"(workers: {[getattr(w, 'id') for w in workers]})"
        )
    return assignments
