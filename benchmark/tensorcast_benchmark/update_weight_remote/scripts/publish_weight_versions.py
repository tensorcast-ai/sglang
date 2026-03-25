#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

import tensorcast as tc
from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish multiple Tensorcast weight versions.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--weight-version-start", type=int, required=True)
    parser.add_argument("--num-versions", type=int, required=True)
    parser.add_argument("--daemon-address", required=True)
    parser.add_argument("--key-template", required=True)
    parser.add_argument("--history-path", default="")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load_tensors(model_path: str) -> dict[str, torch.Tensor]:
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        raise RuntimeError(f"model_path is not a directory: {model_dir}")
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise RuntimeError(f"no .safetensors files found in: {model_dir}")
    tensors: dict[str, torch.Tensor] = {}
    for safetensor_file in safetensor_files:
        file_tensors = load_file(str(safetensor_file), device="cpu")
        for name, tensor in file_tensors.items():
            tensors[name] = tensor
    return tensors


def choose_mutation_target(tensors: dict[str, torch.Tensor]) -> str:
    candidates = [
        name
        for name, tensor in tensors.items()
        if tensor.numel() > 0 and not torch.is_complex(tensor)
    ]
    if not candidates:
        raise RuntimeError("no mutable tensor candidate found for version stamping")
    return min(candidates, key=lambda name: tensors[name].numel() * tensors[name].element_size())


def build_version_tensor(base_tensor: torch.Tensor, version: int) -> torch.Tensor:
    if base_tensor.dtype == torch.bool:
        return torch.full_like(base_tensor, bool(version % 2))
    if torch.is_floating_point(base_tensor):
        return torch.full_like(base_tensor, float(version))
    if base_tensor.dtype.is_signed or base_tensor.dtype.is_floating_point:
        return torch.full_like(base_tensor, int(version))
    return torch.full_like(base_tensor, int(max(version, 0)))


def main() -> None:
    args = parse_args()
    versions = list(range(args.weight_version_start, args.weight_version_start + args.num_versions))
    tensors = load_tensors(args.model_path)
    target_name = choose_mutation_target(tensors)
    base_tensor = tensors[target_name].clone()

    publisher_config = WeightPublisherConfig(
        model_name=args.model_name,
        key_template=args.key_template,
        trigger_reload=False,
        wait_persistence=True,
        keep_last=max(len(versions) + 1, 2),
        policy="pinned",
        overflow_policy="reject",
        history_path=args.history_path or None,
    )

    tc.init(mode="connect", address=args.daemon_address)
    try:
        publisher = WeightPublisher(publisher_config)
        artifacts: list[dict[str, object]] = []
        for version in versions:
            tensors[target_name] = build_version_tensor(base_tensor, version)
            artifact_id = str(publisher.publish(tensors=tensors, version=version))
            artifacts.append(
                {
                    "weight_version": version,
                    "artifact_id": artifact_id,
                    "artifact_key": args.key_template.format(
                        model_name=args.model_name,
                        version=version,
                        weight_version=version,
                    ),
                }
            )
            print(json.dumps(artifacts[-1]), flush=True)
    finally:
        tc.shutdown()

    payload = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "mutation_tensor_name": target_name,
        "versions": artifacts,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
