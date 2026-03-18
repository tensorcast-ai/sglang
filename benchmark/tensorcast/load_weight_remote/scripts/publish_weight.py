#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file

import tensorcast as tc
from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish model weights into Tensorcast.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--weight-version", type=int, required=True)
    parser.add_argument("--daemon-address", required=True)
    parser.add_argument("--key-template", required=True)
    parser.add_argument("--history-path", default="")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_path)
    if not model_dir.is_dir():
        raise RuntimeError(f"model_path is not a directory: {model_dir}")

    publisher_config = WeightPublisherConfig(
        model_name=args.model_name,
        key_template=args.key_template,
        trigger_reload=False,
        wait_persistence=True,
        keep_last=2,
        policy="pinned",
        overflow_policy="reject",
        history_path=args.history_path or None,
    )

    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise RuntimeError(f"no .safetensors files found in: {model_dir}")

    tc.init(mode="connect", address=args.daemon_address)
    try:
        publisher = WeightPublisher(publisher_config)
        tensors: dict[str, object] = {}
        for safetensor_file in safetensor_files:
            tensors.update(load_file(str(safetensor_file), device="cpu"))
        artifact_id = str(publisher.publish(tensors=tensors, version=args.weight_version))
    finally:
        tc.shutdown()

    payload = {
        "artifact_id": artifact_id,
        "model_name": args.model_name,
        "weight_version": args.weight_version,
        "daemon_address": args.daemon_address,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
