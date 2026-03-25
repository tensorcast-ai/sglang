from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample LongBench jsonl files and summarize record schema."
    )
    parser.add_argument(
        "--dataset-dir",
        default="benchmark/tensorcast_benchmark/kv/dataset/LongBench",
    )
    parser.add_argument("--samples-per-file", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--output-json", default="")
    return parser


@dataclass(frozen=True)
class StringStats:
    count: int
    min_length: int
    median_length: float
    max_length: int


def classify_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def summarize_lengths(lengths: list[int]) -> StringStats | None:
    if not lengths:
        return None
    return StringStats(
        count=len(lengths),
        min_length=min(lengths),
        median_length=statistics.median(lengths),
        max_length=max(lengths),
    )


def sample_jsonl_schema(path: Path, samples_per_file: int) -> dict[str, Any]:
    field_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    string_lengths: dict[str, list[int]] = defaultdict(list)
    list_lengths: dict[str, list[int]] = defaultdict(list)
    dict_key_counts: dict[str, list[int]] = defaultdict(list)
    observed_key_sets: list[list[str]] = []
    sampled_rows = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if sampled_rows >= samples_per_file:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sampled_rows += 1
            keys = sorted(row.keys())
            observed_key_sets.append(keys)
            for key, value in row.items():
                value_type = classify_value(value)
                field_type_counts[key][value_type] += 1
                if isinstance(value, str):
                    string_lengths[key].append(len(value))
                elif isinstance(value, list):
                    list_lengths[key].append(len(value))
                elif isinstance(value, dict):
                    dict_key_counts[key].append(len(value))

    string_stats = {
        key: summarize_lengths(lengths).__dict__
        for key, lengths in string_lengths.items()
        if summarize_lengths(lengths) is not None
    }
    list_stats = {
        key: summarize_lengths(lengths).__dict__
        for key, lengths in list_lengths.items()
        if summarize_lengths(lengths) is not None
    }
    dict_stats = {
        key: summarize_lengths(lengths).__dict__
        for key, lengths in dict_key_counts.items()
        if summarize_lengths(lengths) is not None
    }

    return {
        "file": path.name,
        "sampled_rows": sampled_rows,
        "distinct_key_sets": observed_key_sets,
        "field_types": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(field_type_counts.items())
        },
        "string_length_stats": dict(sorted(string_stats.items())),
        "list_length_stats": dict(sorted(list_stats.items())),
        "dict_key_count_stats": dict(sorted(dict_stats.items())),
    }


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    files = sorted(dataset_dir.glob("*.jsonl"))
    if args.max_files > 0:
        files = files[: args.max_files]

    summaries = [
        sample_jsonl_schema(path, samples_per_file=args.samples_per_file)
        for path in files
    ]

    payload = {
        "dataset_dir": str(dataset_dir),
        "samples_per_file": args.samples_per_file,
        "files": summaries,
    }

    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
