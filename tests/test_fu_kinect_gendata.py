from pathlib import Path
import unittest

import numpy as np

from data_gen.fu_kinect_gendata import (
    RawClip,
    exclusion_policy,
    map_kinect20_to_ntu25,
    normalize_official_umurl,
    repair_zero_frames,
)


def clip(relpath: str, subject: int, repeat: int, value: float) -> RawClip:
    raw = np.full((3, 60), value, dtype=np.float64)
    from data_gen.fu_kinect_gendata import matrix_sha256

    return RawClip(
        path=Path(relpath),
        relpath=relpath,
        action="falling",
        subject=subject,
        repeat=repeat,
        raw=raw,
        content_sha256=matrix_sha256(raw),
    )


class FuKinectGenerationTest(unittest.TestCase):
    def test_duplicate_exclusion_is_subject_safe(self):
        same_a = clip("falling/1/a_1.mat", 1, 1, 1.0)
        same_b = clip("falling/1/a_2.mat", 1, 2, 1.0)
        cross_a = clip("falling/2/a_1.mat", 2, 1, 2.0)
        cross_b = clip("falling/3/a_1.mat", 3, 1, 2.0)
        unexpected = clip("falling/4/a_41.mat", 4, 41, 3.0)
        excluded = exclusion_policy([same_a, same_b, cross_a, cross_b, unexpected])
        self.assertNotIn(same_a.relpath, excluded)
        self.assertEqual(
            excluded[same_b.relpath]["reason"],
            "exact_duplicate_within_subject_keep_first",
        )
        self.assertIn(cross_a.relpath, excluded)
        self.assertIn(cross_b.relpath, excluded)
        self.assertEqual(
            excluded[unexpected.relpath]["reason"],
            "unexpected_repetition_outside_1_to_8",
        )

    def test_zero_repair_mapping_and_normalization(self):
        pose = np.zeros((5, 20, 3), dtype=np.float64)
        base = np.arange(60, dtype=np.float64).reshape(20, 3) + 1.0
        pose[1] = base
        pose[3] = base + 2.0
        pose[4] = base + 3.0
        repaired, report = repair_zero_frames(pose.reshape(5, 60))
        self.assertEqual(repaired.shape, (4, 20, 3))
        self.assertEqual(report["trimmed_leading_zero_frames"], 1)
        self.assertEqual(report["interpolated_internal_zero_frames"], [1])
        np.testing.assert_allclose(repaired[1], (repaired[0] + repaired[2]) / 2)

        mapped = map_kinect20_to_ntu25(repaired)
        np.testing.assert_array_equal(mapped[:, :20], repaired)
        np.testing.assert_array_equal(mapped[:, 21], repaired[:, 7])
        np.testing.assert_array_equal(mapped[:, 24], repaired[:, 11])
        normalized = normalize_official_umurl(mapped)
        np.testing.assert_allclose(normalized[:, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(
            (normalized[0, 8] - normalized[0, 4])[1:], 0.0, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
