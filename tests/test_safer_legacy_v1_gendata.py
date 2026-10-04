import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data_gen.preflight_safer_legacy_v1 import EXPECTED_SPLITS
from data_gen.safer_legacy_v1_candidates import (
    CANDIDATES,
    convert_window,
    convert_windows,
    sequence_context,
)
from data_gen.safer_legacy_v1_gendata import (
    clean_window_starts,
    derive_four_class,
    load_freeze_manifest,
    materialize_split,
)


def annotation(frames=72):
    rng = np.random.default_rng(21)
    pose = rng.normal(size=(1, frames, 17, 3)).astype(np.float32)
    labels = (np.arange(frames) % 16).astype(np.int64)
    return {
        "frame_dir": "day_normal_p01_cam2_d1",
        "total_frames": frames,
        "keypoint_3d": pose,
        "labels": labels,
    }


class SaferLegacyV1GenerationTest(unittest.TestCase):
    def test_four_class_derivation(self):
        coarse = np.asarray([0, 9, 10, 11, 12, 13, 15], dtype=np.int64)
        np.testing.assert_array_equal(
            derive_four_class(coarse),
            np.asarray([0, 0, 1, 2, 3, 0, 0], dtype=np.int64),
        )

    def test_clean_windows_exclude_nonfinite_overlap(self):
        value = annotation(frames=80)
        self.assertEqual(clean_window_starts(value).tolist(), [0, 8, 16])
        value["keypoint_3d"][0, 70, 0, 0] = np.nan
        self.assertEqual(clean_window_starts(value).tolist(), [0])

    def test_batch_conversion_matches_single_window_conversion(self):
        value = annotation(frames=80)
        pose = value["keypoint_3d"][0]
        starts = np.asarray([0, 8, 16], dtype=np.int64)
        candidate = CANDIDATES["legacy_coco_umurl_window_noscale"]
        context = sequence_context(pose, candidate)
        batched = convert_windows(pose, starts, candidate, context)
        expected = np.stack(
            [convert_window(pose[start : start + 64], candidate, context) for start in starts]
        )
        np.testing.assert_allclose(batched, expected, rtol=1e-6, atol=1e-6)

    def test_freeze_manifest_locks_candidate_and_recovery_limits(self):
        root = Path(__file__).resolve().parents[1]
        manifest, candidate, digest = load_freeze_manifest(
            root / "configs/safer_legacy_v1_structural_reconstruction_v1.json"
        )
        self.assertEqual(candidate.name, "legacy_coco_umurl_window_noscale")
        self.assertTrue(manifest["recovery_limits"]["experimental_compatibility_reconstruction"])
        self.assertEqual(len(digest), 64)

    def test_materialized_split_schema_and_integrity(self):
        value = annotation(frames=72)
        starts = np.asarray([0, 8], dtype=np.int64)
        candidate = CANDIDATES["legacy_coco_umurl_window_noscale"]
        expected = {"sequences": 1, "candidate_windows": 2, "clean_windows": 2}
        with tempfile.TemporaryDirectory() as temporary:
            split_root = Path(temporary) / "train"
            with patch.dict(EXPECTED_SPLITS, {"train": expected}):
                manifest = materialize_split(
                    split="train",
                    annotations=[value],
                    plans=[starts],
                    split_root=split_root,
                    candidate=candidate,
                    batch_windows=2,
                    full_clean_windows=2,
                    resume=False,
                )
            self.assertTrue(manifest["integrity"]["passed"])
            data = np.load(split_root / "data_joint.npy", mmap_mode="r")
            self.assertEqual(data.shape, (2, 3, 64, 25, 2))
            self.assertEqual(np.count_nonzero(data[..., 1]), 0)
            self.assertTrue(np.isfinite(data).all())
            dense = np.load(split_root / "dense_derived_labels.npy")
            center = np.load(split_root / "center_derived_labels.npy")
            np.testing.assert_array_equal(center, dense[:, 32])
            with (split_root / "label.pkl").open("rb") as stream:
                names, labels = pickle.load(stream)
            self.assertEqual(names, [
                "day_normal_p01_cam2_d1__f0000000",
                "day_normal_p01_cam2_d1__f0000008",
            ])
            self.assertEqual(labels, center.tolist())
            progress = json.loads((split_root / "progress.json").read_text())
            self.assertEqual(progress["status"], "completed")


if __name__ == "__main__":
    unittest.main()
