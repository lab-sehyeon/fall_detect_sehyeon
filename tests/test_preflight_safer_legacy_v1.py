import unittest

import numpy as np

from data_gen.preflight_safer_legacy_v1 import (
    annotation_window_stats,
    matching_subject_subsets,
    subject_id,
    window_starts,
)


class SaferLegacyV1PreflightTest(unittest.TestCase):
    def test_subject_id(self):
        self.assertEqual(subject_id("apr_25_2023_fall_p005_d01.mp4"), 5)
        self.assertEqual(subject_id("day_normal_p06_cam8"), 6)

    def test_window_starts_does_not_append_tail(self):
        np.testing.assert_array_equal(window_starts(63), np.asarray([], dtype=np.int64))
        np.testing.assert_array_equal(window_starts(64), np.asarray([0]))
        np.testing.assert_array_equal(window_starts(65), np.asarray([0]))
        np.testing.assert_array_equal(window_starts(80), np.asarray([0, 8, 16]))

    def test_nonfinite_frame_rejects_each_overlapping_window(self):
        pose = np.zeros((1, 80, 17, 3), dtype=np.float32)
        pose[0, 70, 0, 0] = np.nan
        report = annotation_window_stats({
            "frame_dir": "sample_p002_d01.mp4",
            "total_frames": 80,
            "keypoint_3d": pose,
        })
        self.assertEqual(report["candidate_windows"], 3)
        self.assertEqual(report["clean_windows"], 1)
        self.assertEqual(report["excluded_windows"], 2)
        self.assertEqual(report["bad_frames"], 1)

    def test_matching_subject_subsets_checks_both_counts(self):
        per_subject = {
            1: {"candidate_windows": 10, "clean_windows": 9},
            2: {"candidate_windows": 20, "clean_windows": 18},
            3: {"candidate_windows": 30, "clean_windows": 29},
        }
        self.assertEqual(matching_subject_subsets(per_subject, 30, 27), [(1, 2)])


if __name__ == "__main__":
    unittest.main()
