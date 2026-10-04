#!/usr/bin/env python3
"""Validate the official NTU60 xsub joint DSTE checkpoint safely on CPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from model.DSTE import Downstream
from options.options_downstream import opts_ntu_60_cross_subject


EXPECTED_SIZE = 1_140_267_334
EXPECTED_SHA256 = "59a678cb28f2474eb035f6a42833b68a247854404ed1a46dc08964f43aaf3064"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_data_parallel_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "module."
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--skip-forward", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.checkpoint.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    size = path.stat().st_size
    checksum = sha256sum(path)
    if size != EXPECTED_SIZE:
        raise RuntimeError(f"size mismatch: expected {EXPECTED_SIZE}, got {size}")
    if checksum != EXPECTED_SHA256:
        raise RuntimeError(f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {checksum}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError as exc:
        raise RuntimeError(
            "This validator requires a PyTorch version that supports weights_only=True; "
            "it will not fall back to unsafe pickle loading."
        ) from exc

    expected_top_keys = {"epoch", "optimizer", "state_dict"}
    top_keys = set(checkpoint)
    if top_keys != expected_top_keys:
        raise RuntimeError(f"unexpected top-level keys: {sorted(top_keys)}")
    if checkpoint["epoch"] != 451:
        raise RuntimeError(f"unexpected epoch: {checkpoint['epoch']}")

    raw_state_dict = checkpoint["state_dict"]
    if not isinstance(raw_state_dict, dict) or not raw_state_dict:
        raise RuntimeError("state_dict is missing or empty")
    if not all(torch.is_tensor(value) for value in raw_state_dict.values()):
        raise RuntimeError("state_dict contains non-tensor values")
    state_dict = remove_data_parallel_prefix(raw_state_dict)
    del checkpoint, raw_state_dict
    gc.collect()

    opts = opts_ntu_60_cross_subject()
    model = Downstream(**opts.encoder_args)
    load_result = model.load_state_dict(state_dict, strict=False)

    missing = set(load_result.missing_keys)
    unexpected = set(load_result.unexpected_keys)
    if missing != {"fc.weight", "fc.bias"}:
        raise RuntimeError(f"unexpected missing keys: {sorted(missing)}")
    unexpected_backbone = sorted(key for key in unexpected if key.startswith("backbone."))
    if unexpected_backbone:
        raise RuntimeError(f"unexpected backbone keys: {unexpected_backbone}")

    forward_result: dict[str, object] = {"executed": False}
    if not args.skip_forward:
        model.eval()
        joint_temporal = torch.zeros(1, 64, 150)
        joint_spatial = torch.zeros(1, 50, 192)
        unused = torch.zeros(1, 1, 1)
        with torch.inference_mode():
            output = model(
                joint_temporal,
                joint_spatial,
                unused,
                unused,
                unused,
                unused,
            )
        if output.shape != (1, 60):
            raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise RuntimeError("forward output contains non-finite values")
        forward_result = {
            "executed": True,
            "joint_temporal_shape": list(joint_temporal.shape),
            "joint_spatial_shape": list(joint_spatial.shape),
            "output_shape": list(output.shape),
            "finite": True,
        }

    report = {
        "status": "structural_validation_passed",
        "checkpoint": str(path),
        "size_bytes": size,
        "sha256": checksum,
        "epoch": 451,
        "state_dict_entries": len(state_dict),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "missing_keys": sorted(missing),
        "unexpected_non_backbone_keys": sorted(unexpected),
        "forward": forward_result,
        "performance_validation": "pending_ntu60_xsub_validation_data",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
