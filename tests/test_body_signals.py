import unittest
import numpy as np
from fall_pipeline.primitives.body_signals import body_signals, descriptor, fold_standardize


class BodySignalsTests(unittest.TestCase):
    def setUp(self):
        self.pose = np.random.default_rng(0).normal(size=(9, 25, 3))

    def test_shape_finite(self):
        x = body_signals(self.pose, 30)
        self.assertEqual(x.shape, (9, 12))
        self.assertEqual(descriptor(x).shape, (156,))
        self.assertTrue(np.isfinite(x).all())

    def test_rigid_scale_translation_invariance(self):
        rotation, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
        translated = self.pose * 2.7 @ rotation + np.arange(9)[:, None, None] * np.array([3, -4, 7])
        np.testing.assert_allclose(body_signals(self.pose, 30), body_signals(translated, 30), atol=2e-6)

    def test_invalid_and_derivative_boundary(self):
        self.pose[4] = 0
        x = body_signals(self.pose, 30)
        self.assertTrue((x[4] == 0).all())
        self.assertTrue((x[5, 9:11] == 0).all())
        self.assertTrue((body_signals(np.zeros((64, 25, 3)), 25) == 0).all())

    def test_constant_and_one_frame(self):
        p = np.repeat(self.pose[:1], 9, 0)
        self.assertTrue(np.allclose(body_signals(p, 25)[:, 9:], 0, atol=1e-5))
        x = descriptor(body_signals(p[:1], 25)).reshape(12, 13)
        self.assertTrue((x[:, [1, 11, 12]] == 0).all())

    def test_fps_changes_derivatives_only(self):
        a, b = body_signals(self.pose, 30), body_signals(self.pose, 15)
        np.testing.assert_allclose(a[:, 9:11], b[:, 9:11] * 2)
        np.testing.assert_array_equal(a[:, :9], b[:, :9])
        np.testing.assert_array_equal(a[:, 11], b[:, 11])

    def test_descriptor_order(self):
        x = np.tile(np.arange(3)[:, None], (1, 12))
        expected = [1, np.sqrt(2/3), 0, 2, .2, .5, 1, 1.5, 1.8, 0, 2, 2, 1]
        np.testing.assert_allclose(descriptor(x).reshape(12, 13), np.tile(expected, (12, 1)))

    def test_scaler_train_only(self):
        x = np.array([[0., 2], [2, 2], [100, -30]])
        y, mean, scale = fold_standardize(x, [0, 1])
        np.testing.assert_array_equal(mean, [1, 2])
        np.testing.assert_array_equal(scale, [1, 1e-6])
        self.assertEqual(y[2, 0], 99)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            body_signals(np.zeros((0, 25, 3)), 30)
        with self.assertRaises(ValueError):
            descriptor(np.full((3, 12), np.nan))
