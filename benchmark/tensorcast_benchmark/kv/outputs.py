from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from tensorcast_benchmark.kv.models import BenchmarkPaths, RunSummary


RUN_SUMMARY_CSV_FIELDS = [
    "run_id",
    "backend",
    "tensorcast_daemon_mode",
    "model_path",
    "tp_size",
    "prompt_count",
    "avg_prompt_length",
    "success_pairs",
    "failed_pairs",
    "mean_instance_a_ttft_ms",
    "mean_instance_b_ttft_ms",
    "median_instance_a_ttft_ms",
    "median_instance_b_ttft_ms",
    "p95_instance_a_ttft_ms",
    "p95_instance_b_ttft_ms",
    "mean_ttft_improvement_ms",
    "median_ttft_improvement_ms",
    "p95_ttft_improvement_ms",
    "mean_ttft_speedup_ratio",
    "mean_source_publication_drain_ms",
    "median_source_publication_drain_ms",
    "p95_source_publication_drain_ms",
    "mean_source_publication_wait_ms",
    "source_publication_drain_timeout_count",
    "log_dir",
    "results_json_path",
    "worker_process",
    "worker_host",
    "worker_ip",
    "worker_node",
    "observation",
]


def _validate_run_summary_csv_fields() -> None:
    expected_fields = set(RunSummary.model_fields.keys())
    actual_fields = set(RUN_SUMMARY_CSV_FIELDS)
    if expected_fields != actual_fields:
        raise RuntimeError(
            f"RUN_SUMMARY_CSV_FIELDS mismatch: expected {expected_fields}, got {actual_fields}"
        )


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


def create_run_id(backend: str, tp_size: int, prompt_count: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}_{backend}_tp{tp_size}_pairs{prompt_count}"


def build_paths(benchmark_root: Path, run_id: str) -> BenchmarkPaths:
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
    return BenchmarkPaths(
        benchmark_root=benchmark_root,
        benchmark_name=benchmark_root.name,
        kv_root=kv_root,
        sglang_root=sglang_root,
        workspace_root=workspace_root,
        outputs_dir=outputs_dir,
        run_dir=run_dir,
        logs_dir=logs_dir,
        generated_configs_dir=generated_configs_dir,
        results_json_path=run_dir / "pair_results.jsonl",
        summary_json_path=run_dir / "summary.json",
        csv_path=outputs_dir / "benchmark_results.csv",
        orchestrator_log_path=logs_dir / "orchestrator.log",
        venv_python=workspace_root / ".venv" / "bin" / "python",
        uv_bin=uv_bin,
        mooncake_master_bin=workspace_root / ".venv" / "bin" / "mooncake_master",
        scripts_dir=benchmark_root / "scripts",
    )


def prepare_paths(paths: BenchmarkPaths) -> None:
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


def append_csv_row(path: Path, row: dict[str, object]) -> None:
    _validate_run_summary_csv_fields()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    if file_exists:
        _normalize_existing_csv(path)
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_SUMMARY_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
