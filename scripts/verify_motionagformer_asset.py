#!/usr/bin/env python3
"""Validate official MotionAGFormer-B asset with restricted checkpoint loading.

CPU-only strict state loading and one synthetic 243-frame forward pass.
Does not train, enable unrestricted pickle loading, rename or change the file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import pickletools
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/MotionAGFormer"
PINNED_COMMIT = "4756fd1eb7cc73f0e991f091ff2280e030ab85f3"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    path = args.checkpoint.resolve()
    if path.stat().st_size != 141930389:
        raise RuntimeError("official download length mismatch")
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError("upstream revision changed")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    # Audit pickle references without executing them. The official training
    # checkpoint stores a NumPy float64 metric alongside tensors/optimizer data.
    allowed_pickle_globals = {
        "torch._utils _rebuild_tensor_v2", "torch FloatStorage", "torch LongStorage",
        "collections OrderedDict", "numpy.core.multiarray scalar", "numpy dtype", "_codecs encode",
    }
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(names) != 1 or archive.getinfo(names[0]).file_size > 10 * 1024**2:
            raise RuntimeError("unexpected pickle metadata")
        globals_seen = set()
        for opcode, argument, _ in pickletools.genops(archive.read(names[0])):
            if opcode.name == "STACK_GLOBAL":
                raise RuntimeError("unaudited dynamic global")
            if opcode.name == "GLOBAL":
                globals_seen.add(argument)
        if not globals_seen <= allowed_pickle_globals:
            raise RuntimeError("unaudited pickle global")
    safe_numpy_metadata = [np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(safe_numpy_metadata):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # Pinned upstream train.py saves/loads the network under checkpoint['model'].
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("expected official model state")
    state = checkpoint["model"]
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise RuntimeError("invalid tensor state")
    # Remove only the known DataParallel prefix, with no ignored/missing tensors.
    clean = {key.removeprefix("module."): value for key, value in state.items()}
    if len(clean) != len(state):
        raise RuntimeError("duplicate keys after prefix normalization")
    if not all(torch.isfinite(value).all() for value in clean.values()):
        raise RuntimeError("nonfinite downloaded state")
    config_path = UPSTREAM / "configs/h36m/MotionAGFormer-base.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(UPSTREAM))
    from utils.learning import load_model
    model = load_model(SimpleNamespace(**config)).eval()
    model.load_state_dict(clean, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        output = model(torch.zeros(1, 243, 17, 3))
    if tuple(output.shape) != (1, 243, 17, 3) or not torch.isfinite(output).all():
        raise RuntimeError("synthetic forward check failed")
    report = {
        "passed": True, "checkpoint": str(path), "bytes": path.stat().st_size,
        "sha256": file_hash(path), "official_source": "https://github.com/TaatiTeam/MotionAGFormer",
        "official_file_id": "1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP", "upstream_commit": commit,
        "config_sha256": file_hash(config_path), "checkpoint_keys": list(checkpoint),
        "model_state_tensor_count": len(clean), "parameters": sum(p.numel() for p in model.parameters()),
        "strict_load": True, "weights_only": True, "synthetic_shape": list(output.shape),
        "pickle_globals_static_audit": sorted(globals_seen),
        "numpy_metadata_allowlist": ["scalar", "dtype", "dtype[float64]"],
        "synthetic_finite": True, "device": "cpu",
        "scope": "asset compatibility only; no H36M accuracy or SAFER experiment performed",
        "publisher_checksum_verified": False,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
