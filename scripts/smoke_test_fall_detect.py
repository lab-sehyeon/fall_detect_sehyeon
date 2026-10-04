#!/usr/bin/env python3
"""CPU smoke test for the reconstructed fall_detect environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import mmcv
import mmpose
import numpy as np
import torch

from data_gen.external_dataset_eligibility import assess_clip
from data_gen.h36m17_to_ntu25 import convert
from data_gen.motionagformer_overlap import overlap_add, window_starts
from fall_pipeline.common.features import dense_dste_feature, fuse_overlapping_logits
from fall_pipeline.common.global_motion import build_global_131
from fall_pipeline.external.rgb_recovery_common import (
    D0_LEGACY,
    D1_DECOUPLED,
    decode_recovery_endpoints,
)
from fall_pipeline.joint.models import JointFallModel
from fall_pipeline.safer.context import posterior_dynamics
from model.DSTE import Downstream


def _third_party_smoke() -> None:
    checks = (
        (
            ROOT / "third_party" / "MotionAGFormer",
            "import torch; from model.MotionAGFormer import MotionAGFormer; "
            "m=MotionAGFormer(n_layers=1,dim_in=3,dim_feat=16,dim_rep=16,"
            "num_heads=4,n_frames=9); y=m(torch.randn(1,9,17,3)); "
            "assert tuple(y.shape)==(1,9,17,3)",
        ),
        (
            ROOT / "third_party" / "LaDy",
            "from libs.models.LaDy import Model; from libs.transformer import GaussianSmoothing",
        ),
        (
            ROOT,
            "import torch; from mmpose.models.backbones.vit import ViT; import mmcv; "
            "m=ViT(img_size=64,patch_size=16,embed_dim=32,depth=1,num_heads=4); "
            "y=m(torch.randn(1,3,64,64)); assert tuple(y.shape)==(1,32,4,4); "
            "assert mmcv.__version__=='1.5.0'",
        ),
    )
    for cwd, command in checks:
        subprocess.run([sys.executable, "-c", command], cwd=cwd, check=True)


def _recovered_pipeline_smoke() -> None:
    rng = np.random.default_rng(7)
    pose, proxy = convert(rng.normal(size=(12, 17, 3)).astype(np.float32))
    assert pose.shape == (12, 25, 3) and int(proxy.sum()) == 8

    starts = window_starts(400)
    merged = overlap_add(
        [np.full((243, 17, 3), index, dtype=np.float32) for index in range(len(starts))],
        starts,
        400,
    )
    assert merged.shape == (400, 17, 3) and np.isfinite(merged).all()
    assert assess_clip(128, [{"fall_start": 3.0}])["eligible"]

    global_feature = build_global_131(rng.normal(size=(64, 12)).astype(np.float32))
    assert global_feature.shape == (131,)

    temporal = torch.randn(2, 64, 32)
    spatial = torch.randn(2, 50, 32)
    dense = dense_dste_feature(temporal, spatial)
    assert dense.shape == (2, 64, 64)
    joint = JointFallModel(feature_dim=64, bottleneck_dim=16)
    assert joint(torch.randn(2, 64), "safer").shape == (2, 4)
    assert joint(torch.randn(2, 64), "fu").shape == (2, 2)
    assert posterior_dynamics(torch.randn(2, 16)).shape == (2, 35)

    logits, valid = fuse_overlapping_logits(
        torch.tensor([0, 4]), torch.randn(2, 8, 4), sequence_length=12
    )
    assert logits.shape == (12, 4) and bool(valid.all())

    model = Downstream(
        t_input_size=150,
        s_input_size=192,
        hidden_size=32,
        num_head=4,
        num_layer=2,
        num_class=4,
    ).eval()
    jt, js = torch.randn(2, 64, 150), torch.randn(2, 50, 192)
    with torch.inference_mode():
        clip_logits = model(jt, js, jt, js, jt, js)
        dense_logits = model(jt, js, jt, js, jt, js, detect=True)
    assert clip_logits.shape == (2, 4)
    assert dense_logits.shape == (2, 64, 4)

    frames = np.arange(8) * 8 + 63
    g2 = np.asarray([0, 1, 1, 0, 0, 0, 1, 0])
    s0 = np.asarray([0, 1, 2, 3, 0, 0, 0, 0])
    legacy = decode_recovery_endpoints(frames, g2, s0, fall_alert_mode=D0_LEGACY)
    decoupled = decode_recovery_endpoints(frames, g2, s0, fall_alert_mode=D1_DECOUPLED)
    np.testing.assert_array_equal(legacy["endpoint_state"], decoupled["endpoint_state"])
    assert decoupled["diagnostics"]["fall_alerts"] >= legacy["diagnostics"]["fall_alerts"]


def main() -> None:
    import chumpy
    import scipy
    import torchvision

    _third_party_smoke()
    _recovered_pipeline_smoke()
    print(
        "fall_detect smoke: PASS | "
        f"python={sys.version.split()[0]} torch={torch.__version__} "
        f"cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()} "
        f"numpy={np.__version__} scipy={scipy.__version__} opencv={cv2.__version__} "
        f"torchvision={torchvision.__version__} mmcv={mmcv.__version__} "
        f"mmpose={mmpose.__version__} chumpy={chumpy.__version__}"
    )


if __name__ == "__main__":
    main()
