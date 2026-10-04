#!/usr/bin/env python3
"""Replay saved endpoint scores through the recovered D0/D1 decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .rgb_recovery_common import D0_LEGACY, D1_DECOUPLED, decode_recovery_endpoints, run_length_encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, help="NPZ with endpoint_frames, g2_logits, s0g0_logits")
    parser.add_argument("output", type=Path)
    parser.add_argument("--fall-alert-mode", choices=(D0_LEGACY, D1_DECOUPLED), default=D0_LEGACY)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source = np.load(args.scores)
    result = decode_recovery_endpoints(
        source["endpoint_frames"], source["g2_logits"], source["s0g0_logits"],
        frame_count=args.frame_count, fall_alert_mode=args.fall_alert_mode,
    )
    serializable = {
        "fall_alert_mode": result["fall_alert_mode"],
        "contract": result["contract"],
        "events": result["events"],
        "diagnostics": result["diagnostics"],
        "fall_detected": result["fall_detected"],
        "state_timeline_rle": run_length_encode(result["frame_state"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
