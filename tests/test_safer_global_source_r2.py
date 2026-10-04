import unittest
import numpy as np
from scripts.audit_safer_global_source_r2 import geometry_counts


class GlobalSourceTests(unittest.TestCase):
    def test_corner_xywh_without_filtering_zero_nonfinite(self):
        box = np.array([[10, 20, 5, 8], [0, 0, 0, 0], [np.nan, 0, 4, 4]])
        xy = np.tile([12, 24], (3, 17, 1))
        score = np.ones((3, 17))
        stats = geometry_counts(box, xy, score, 100, 100)
        self.assertEqual(stats['frames'], 3)
        self.assertEqual(stats['bbox_nonfinite_frames'], 1)
        self.assertEqual(stats['bbox_zero_frames'], 1)
        self.assertEqual(stats['xywh_plausible_frames'], 1)
        self.assertEqual(stats['xyxy_plausible_frames'], 0)
        self.assertEqual(stats['xywh_inside_confident_joints'], 17)
        self.assertEqual(stats['confident_joints'], 51)

    def test_threshold_and_bad_scores_explicit(self):
        box = np.array([[0, 0, 10, 10]])
        xy = np.full((1, 17, 2), 5.)
        score = np.full((1, 17), .199)
        score[0, 0], score[0, 1] = .2, np.nan
        stats = geometry_counts(box, xy, score, 10, 10)
        self.assertEqual(stats['confident_joints'], 1)
        self.assertEqual(stats['confidence_nonfinite_frames'], 1)
        self.assertEqual(stats['xywh_inside_confident_joints'], 1)
        with self.assertRaisesRegex(RuntimeError, 'shape'):
            geometry_counts(box, xy[:0], score, 10, 10)


if __name__ == '__main__':
    unittest.main()
