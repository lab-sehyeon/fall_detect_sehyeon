#!/usr/bin/env python3
"""Read-only train/val audit of exactly zero pose windows; no inference."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/fall_processed/SAFER-Activities/clean3d_v1_reconstructed"
report = {"scope": "train/validation only, no holdout access or performance tuning", "splits": {}}
for split in ("train", "val"):
    data = np.load(DATA / split / "data_joint.npy", mmap_mode="r", allow_pickle=False)
    zero_indices = []
    nonfinite = 0
    for begin in range(0, len(data), 256):
        block = np.asarray(data[begin:begin + 256])
        zero = ~np.any(block != 0, axis=(1, 2, 3, 4))
        zero_indices.extend((np.flatnonzero(zero) + begin).tolist())
        nonfinite += int(np.count_nonzero(~np.isfinite(block).all(axis=(1, 2, 3, 4))))
    report["splits"][split] = {"windows": len(data), "all_zero_windows": len(zero_indices),
                                "first_zero_indices": zero_indices[:20], "nonfinite_windows": nonfinite}
print(json.dumps(report, indent=2))
