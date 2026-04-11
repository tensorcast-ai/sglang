from __future__ import annotations

import csv
from pathlib import Path

from tensorcast_benchmark.kv.request_transfer.models import RequestTransferRunSummary


RUN_SUMMARY_CSV_FIELDS = [
    "run_id",
    "topology_mode",
    "model_path",
    "prompt_count",
    "avg_prompt_length",
    "successful_prompts",
    "failed_prompts",
    "publish_success_count",
    "hydrate_success_count",
    "prepared_bundle_verified_count",
    "mean_source_ttft_ms",
    "mean_target_ttft_ms",
    "mean_publish_latency_ms",
    "mean_hydrate_latency_ms",
    "mean_target_cached_tokens",
    "warmup_enabled",
    "warmup_success_count",
    "log_dir",
    "results_json_path",
    "worker_process_a",
    "worker_process_b",
    "worker_host_a",
    "worker_host_b",
    "worker_ip_a",
    "worker_ip_b",
    "worker_node_a",
    "worker_node_b",
    "observation",
]


def _validate_run_summary_csv_fields() -> None:
    expected_fields = set(RequestTransferRunSummary.model_fields.keys())
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
