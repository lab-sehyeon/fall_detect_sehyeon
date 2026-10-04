#!/usr/bin/env python3
"""Audit files recovered from the surviving 2026-08-27 project record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED = (
    "model/DSTE.py", "model/STTR.py", "feeder/feeder_downstream.py",
    "data_gen/ntu_gendata.py", "data_gen/h36m17_to_ntu25.py",
    "data_gen/motionagformer_overlap.py", "data_gen/external_dataset_eligibility.py",
    "fall_pipeline/common/features.py", "fall_pipeline/common/global_motion.py",
    "fall_pipeline/common/integrity.py", "fall_pipeline/joint/models.py",
    "fall_pipeline/safer/context.py",
    "fall_pipeline/external/rgb_recovery_common.py",
    "fall_pipeline/external/replay_rgb_fall_recovery_decoder_v1.py",
    "configs/rgb_dual_path_inference_v1.json",
    "configs/rgb_fall_recovery_v1.json", "configs/decoder_latch_fix_v1.json",
    "docs/code_directory_guide.md", "docs/recovery_status_2026-09-02.md",
)

IRREPLACEABLE_ASSETS = (
    "checkpoint/adl_baseline/pretrained_encoder_reference.pth.tar",
    "checkpoint/adl_baseline/best_adl_head.pth",
    "checkpoint/fall/FINAL_JOINT_FALL_V3/final_j1_model.pth",
    "data/NTU-RGB-D-60-AGCN-UMURL/xsub/train_data_joint.npy",
    "data/fall_processed/SAFER-Activities",
    "data/fall_processed/FU-Kinect-Fall",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    missing_code = [name for name in REQUIRED if not (args.root / name).is_file()]
    missing_assets = [name for name in IRREPLACEABLE_ASSETS if not (args.root / name).exists()]
    report = {
        "reconstructed_core_complete": not missing_code,
        "required_files": len(REQUIRED),
        "missing_code": missing_code,
        "external_assets_available": not missing_assets,
        "missing_irreplaceable_assets": missing_assets,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("reconstructed core:", "PASS" if not missing_code else "FAIL")
        print("external data/checkpoints:", "AVAILABLE" if not missing_assets else "MISSING")
        for name in missing_code + missing_assets:
            print(" -", name)
    return 1 if missing_code else 0


if __name__ == "__main__":
    sys.exit(main())
