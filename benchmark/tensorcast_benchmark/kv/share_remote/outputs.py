from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from tensorcast_benchmark.kv.share_remote.models import (
    ShareRemotePaths,
    ShareRemoteRunSummary,
)


RUN_SUMMARY_CSV_FIELDS = [
    "run_id",
    "backend",
    "transport_use_rdma",
    "worker_count",
    "service_host_worker_index",
    "model_path",
    "tp_size",
    "prompt_count",
    "avg_prompt_length",
    "successful_prompt_groups",
    "failed_prompt_groups",
    "mean_ttft_by_position",
    "median_ttft_by_position",
    "p95_ttft_by_position",
    "mean_cached_tokens_by_position",
    "mean_improvement_vs_first_ms_by_position",
    "log_dir",
    "results_json_path",
    "worker_processes",
    "worker_hosts",
    "worker_ips",
    "worker_nodes",
    "observation",
]


def _normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def discover_workspace_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".venv" / "bin" / "python").is_file():
            return candidate
    raise RuntimeError(f"Could not find workspace root from {start}")


def discover_sglang_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "benchmark").is_dir() and (candidate / "python").is_dir():
            return candidate
    raise RuntimeError(f"Could not find sglang root from {start}")


def create_run_id(backend: str, tp_size: int, worker_count: int, prompt_count: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        f"{timestamp}_{backend}_tp{tp_size}_workers{worker_count}_prompts{prompt_count}"
    )


def build_paths(benchmark_root: Path, run_id: str) -> ShareRemotePaths:
    benchmark_root = benchmark_root.resolve()
    kv_root = benchmark_root.parent
    sglang_root = discover_sglang_root(benchmark_root)
    workspace_root = discover_workspace_root(benchmark_root)
    outputs_dir = benchmark_root / "outputs"
    run_dir = outputs_dir / run_id
    logs_dir = run_dir / "logs"
    generated_configs_dir = run_dir / "generated_configs"
    uv_bin = workspace_root / ".venv" / "bin" / "uv"
    if not uv_bin.exists():
        uv_bin = Path.home() / ".local" / "bin" / "uv"
    return ShareRemotePaths(
        benchmark_root=benchmark_root,
        benchmark_name=benchmark_root.name,
        kv_root=kv_root,
        sglang_root=sglang_root,
        workspace_root=workspace_root,
        outputs_dir=outputs_dir,
        run_dir=run_dir,
        logs_dir=logs_dir,
        generated_configs_dir=generated_configs_dir,
        results_json_path=run_dir / "prompt_results.jsonl",
        summary_json_path=run_dir / "summary.json",
        csv_path=outputs_dir / "benchmark_results.csv",
        orchestrator_log_path=logs_dir / "orchestrator.log",
        config_copy_path=run_dir / "config.yaml",
        resolved_config_path=run_dir / "resolved_config.yaml",
        driver_config_path=generated_configs_dir / "driver_config.yaml",
        worker_inventory_path=run_dir / "worker_inventory.json",
        venv_python=workspace_root / ".venv" / "bin" / "python",
        uv_bin=uv_bin,
        mooncake_master_bin=workspace_root / ".venv" / "bin" / "mooncake_master",
        scripts_dir=benchmark_root / "scripts",
    )


def prepare_paths(paths: ShareRemotePaths) -> None:
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.generated_configs_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2, sort_keys=True)
        file.write("\n")


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            file.write("\n")


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected YAML object at {path}, got {type(payload).__name__}")
    return payload


def dump_yaml(path: Path, payload: object) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def worker_log_dir(paths: ShareRemotePaths, worker_index: int) -> Path:
    return paths.logs_dir / f"worker_{worker_index:02d}"


def worker_log_path(paths: ShareRemotePaths, worker_index: int, name: str) -> Path:
    return worker_log_dir(paths, worker_index) / f"{name}.log"


def _normalize_existing_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        existing_header = reader.fieldnames or []
        if not existing_header or existing_header == RUN_SUMMARY_CSV_FIELDS:
            return
        existing_fields = set(existing_header)
        expected_fields = set(RUN_SUMMARY_CSV_FIELDS)
        if not existing_fields.issubset(expected_fields):
            raise RuntimeError(
                f"Unexpected CSV header in {path}: {existing_header} != {RUN_SUMMARY_CSV_FIELDS}"
            )
        rows = [
            {field: row.get(field, "") for field in RUN_SUMMARY_CSV_FIELDS}
            for row in reader
        ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: Path, summary: ShareRemoteRunSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    if file_exists:
        _normalize_existing_csv(path)
    row = {
        "run_id": summary.run_id,
        "backend": summary.backend,
        "transport_use_rdma": summary.transport_use_rdma,
        "worker_count": summary.worker_count,
        "service_host_worker_index": summary.service_host_worker_index,
        "model_path": summary.model_path,
        "tp_size": summary.tp_size,
        "prompt_count": summary.prompt_count,
        "avg_prompt_length": summary.avg_prompt_length,
        "successful_prompt_groups": summary.successful_prompt_groups,
        "failed_prompt_groups": summary.failed_prompt_groups,
        "mean_ttft_by_position": json.dumps(summary.mean_ttft_by_position),
        "median_ttft_by_position": json.dumps(summary.median_ttft_by_position),
        "p95_ttft_by_position": json.dumps(summary.p95_ttft_by_position),
        "mean_cached_tokens_by_position": json.dumps(
            summary.mean_cached_tokens_by_position
        ),
        "mean_improvement_vs_first_ms_by_position": json.dumps(
            summary.mean_improvement_vs_first_ms_by_position
        ),
        "log_dir": summary.log_dir,
        "results_json_path": summary.results_json_path,
        "worker_processes": json.dumps(summary.worker_processes),
        "worker_hosts": json.dumps(summary.worker_hosts),
        "worker_ips": json.dumps(summary.worker_ips),
        "worker_nodes": json.dumps(summary.worker_nodes),
        "observation": summary.observation,
    }
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_SUMMARY_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
