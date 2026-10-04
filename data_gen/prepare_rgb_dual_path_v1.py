#!/usr/bin/env python3
"""Prepare proxy NTU25 and causal global features from upstream pose assets.

This recovered stage intentionally starts after detector/ViTPose/
MotionAGFormer because their original external repositories and checkpoints
were not included in the surviving archive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_gen.h36m17_to_ntu25 import convert
from fall_pipeline.common.global_motion import build_global_131


def starts(frame_count: int, window: int = 64, stride: int = 8) -> list[int]:
    if frame_count < window:
        return []
    output = list(range(0, frame_count - window + 1, stride))
    final = frame_count - window
    if output[-1] != final:
        output.append(final)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ: pose3d_h36m[T,17,3], global_channels[T,12]")
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-torso", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source = np.load(args.input)
    h36m = source["pose3d_h36m"]
    global_channels = source["global_channels"]
    if len(h36m) != len(global_channels):
        raise ValueError("pose and global timeline lengths differ")
    ntu25, proxy_mask = convert(h36m, args.reference_torso)
    valid = source["global_valid"] if "global_valid" in source else None
    window_starts = starts(len(h36m))
    global_features = np.stack([
        build_global_131(
            global_channels[start:start + 64],
            None if valid is None else valid[start:start + 64],
        ) for start in window_starts
    ]) if window_starts else np.empty((0, 131), dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ntu25=ntu25,
        proxy_joint_mask=proxy_mask,
        window_starts=np.asarray(window_starts, dtype=np.int64),
        window_endpoints=np.asarray(window_starts, dtype=np.int64) + 63,
        global_features=global_features,
    )
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps({
        "source": str(args.input), "frames": len(h36m), "windows": len(window_starts),
        "future_frames_seen": 0, "global_feature_dim": 131, "native_ntu25": False,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
