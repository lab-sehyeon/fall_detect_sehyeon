import unittest
import numpy as np

from fall_pipeline.fu.zs_reconstruction import starts_for, aligned_indices, prepare, aggregate, clip_outputs, metrics


class FuZsTest(unittest.TestCase):
    def test_starts(self):
        self.assertEqual(starts_for(35), [0])
        self.assertEqual(starts_for(64), [0])
        self.assertEqual(starts_for(65), [0, 1])
        self.assertEqual(starts_for(72), [0, 8])
        self.assertEqual(starts_for(73), [0, 8, 9])
        with self.assertRaises(RuntimeError):
            starts_for(0)

    def test_aligned_sampling(self):
        self.assertEqual(aligned_indices(7).tolist(), [0, 1, 2, 4, 5, 6])
        for length in range(1, 301):
            index = aligned_indices(length)
            expected = np.floor(np.arange(len(index)) * 6 / 5 + .5).astype(np.int64)
            np.testing.assert_array_equal(index, expected)
            self.assertLessEqual(index[-1], length - 1)
            self.assertEqual(len(index), (length - 1) * 5 // 6 + 1)

    def test_last_pose_padding(self):
        data = np.zeros((1, 3, 300, 25, 2), dtype=np.float32)
        data[:, :, 34] = 2
        windows, plan = prepare(data, np.array([35]), "zs1_native30")
        self.assertEqual(windows.shape, (1, 3, 64, 25, 2))
        self.assertTrue(np.all(windows[0, :, 34:] == 2))
        self.assertEqual(plan["lengths"].tolist(), [35])

    def test_padding_prediction_excluded(self):
        logits = np.zeros((1, 64, 4), dtype=np.float32)
        logits[:, :35, 0] = 2
        logits[:, 35:, 1] = 100
        plan = {"owners": np.array([0]), "starts": np.array([0]), "lengths": np.array([35])}
        timeline, counts, offsets = aggregate(logits, plan)
        decisions, _ = clip_outputs(timeline, offsets)
        self.assertFalse(decisions[0])
        self.assertEqual(len(timeline), 35)
        self.assertTrue(np.all(counts == 1))

    def test_overlap_is_logits_not_probability(self):
        logits = np.zeros((2, 64, 4), dtype=np.float32)
        logits[0, :, 0], logits[1, :, 1] = 10, 4
        plan = {"owners": np.array([0, 0]), "starts": np.array([0, 1]), "lengths": np.array([65])}
        timeline, counts, _ = aggregate(logits, plan)
        np.testing.assert_array_equal(timeline[1], [5, 2, 0, 0])
        self.assertEqual(counts.tolist(), [1] + [2] * 63 + [1])

    def test_independent_aggregation_and_scores(self):
        rng = np.random.default_rng(7)
        logits = rng.standard_normal((3, 64, 4)).astype(np.float32)
        plan = {"owners": np.array([0, 1, 1]), "starts": np.array([0, 0, 1]), "lengths": np.array([35, 65])}
        first = aggregate(logits, plan)
        second = aggregate(logits, plan, independent=True)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        d1, s1 = clip_outputs(first[0], first[2])
        d2, s2 = clip_outputs(second[0], second[2], independent=True)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_allclose(s1, s2, rtol=0, atol=1e-12)

    def test_clip_metrics(self):
        result = metrics(np.array([1, 0, 0]), np.array([5, 4, 0]), np.array([True, True, False]), np.array([.9, .8, .2]))
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(result["lying_fp"], 1)
        self.assertEqual(result["auprc"], 1)

    def test_whole_clip_constant_resize(self):
        data = np.full((1, 3, 300, 25, 2), 3, dtype=np.float32)
        windows, plan = prepare(data, np.array([35]), "zs0_resize64")
        np.testing.assert_allclose(windows, 3, atol=1e-6)
        self.assertEqual(plan["lengths"].tolist(), [64])

    def test_nonfinite_rejected(self):
        plan = {"owners": np.array([0]), "starts": np.array([0]), "lengths": np.array([64])}
        logits = np.zeros((1, 64, 4), dtype=np.float32)
        logits[0, 0, 0] = np.nan
        with self.assertRaises(RuntimeError):
            aggregate(logits, plan)


if __name__ == "__main__":
    unittest.main()
