import unittest

import numpy as np

from data_gen.safer_legacy_v1_candidates import (
    CANDIDATES,
    alignment_matrix,
    convert_window,
    map_coco_midpoint_proxy,
    sequence_context,
)


class SaferLegacyV1CandidatesTest(unittest.TestCase):
    def test_coco_midpoint_mapping(self):
        pose = np.arange(17 * 3, dtype=np.float32).reshape(1, 17, 3)
        mapped = map_coco_midpoint_proxy(pose)
        self.assertEqual(mapped.shape, (1, 25, 3))
        np.testing.assert_allclose(mapped[0, 0], (pose[0, 11] + pose[0, 12]) / 2)
        np.testing.assert_allclose(mapped[0, 20], (pose[0, 5] + pose[0, 6]) / 2)
        np.testing.assert_array_equal(mapped[0, 21], pose[0, 9])
        np.testing.assert_array_equal(mapped[0, 24], pose[0, 10])

    def test_umurl_alignment_reaches_positive_x(self):
        shoulder = np.asarray((0.3, -0.7, 0.2), dtype=np.float32)
        rotation = alignment_matrix(shoulder, "umurl_full_3d")
        aligned = rotation @ shoulder
        self.assertGreater(aligned[0], 0)
        self.assertLess(float(np.abs(aligned[1:]).max()), 1e-6)

    def test_candidate_centers_spine_mid_and_aligns_shoulder(self):
        rng = np.random.default_rng(4)
        pose = rng.normal(size=(64, 17, 3)).astype(np.float32)
        candidate = CANDIDATES["legacy_coco_umurl_window_noscale"]
        context = sequence_context(pose, candidate)
        output = convert_window(pose, candidate, context)
        self.assertLess(float(np.abs(output[:, 1]).max()), 1e-6)
        shoulder = output[0, 8] - output[0, 4]
        self.assertGreater(shoulder[0], 0)
        self.assertLess(float(np.abs(shoulder[1:]).max()), 1e-5)

    def test_unit_torso_uses_sequence_median(self):
        pose = np.ones((64, 17, 3), dtype=np.float32)
        pose[:, 5] = (0, 4, 0)
        pose[:, 6] = (2, 4, 0)
        pose[:, 11] = (0, 0, 0)
        pose[:, 12] = (2, 0, 0)
        candidate = CANDIDATES["legacy_coco_umurl_window_unit_torso"]
        context = sequence_context(pose, candidate)
        self.assertAlmostEqual(float(context["scale"]), 0.25)

    def test_ntu_torso_targets_half_unit(self):
        pose = np.ones((64, 17, 3), dtype=np.float32)
        pose[:, 5] = (0, 4, 0)
        pose[:, 6] = (2, 4, 0)
        pose[:, 11] = (0, 0, 0)
        pose[:, 12] = (2, 0, 0)
        candidate = CANDIDATES["legacy_coco_umurl_window_ntu_torso"]
        context = sequence_context(pose, candidate)
        self.assertAlmostEqual(float(context["scale"]), 0.125)

    def test_legacy_generic_z_alignment_reaches_positive_z(self):
        rng = np.random.default_rng(8)
        pose = rng.normal(size=(64, 17, 3)).astype(np.float32)
        candidate = CANDIDATES["legacy_coco_generic_z11_5_window_noscale"]
        output = convert_window(pose, candidate, sequence_context(pose, candidate))
        vector = output[0, 5] - output[0, 11]
        self.assertGreater(vector[2], 0)
        self.assertLess(float(np.abs(vector[:2]).max()), 1e-5)

    def test_center_only_does_not_rotate(self):
        rng = np.random.default_rng(9)
        pose = rng.normal(size=(64, 17, 3)).astype(np.float32)
        candidate = CANDIDATES["legacy_coco_center_only_noscale"]
        output = convert_window(pose, candidate, sequence_context(pose, candidate))
        mapped = map_coco_midpoint_proxy(pose)
        expected = mapped - mapped[:, 1:2]
        np.testing.assert_allclose(output, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
