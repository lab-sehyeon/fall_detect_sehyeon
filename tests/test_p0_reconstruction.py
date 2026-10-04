import unittest
import numpy as np
from fall_pipeline.primitives.p0_reconstruction import bootstrap, features_for


class P0Tests(unittest.TestCase):
    def test_feature_dimensions_and_baseline_unchanged(self):
        rng = np.random.default_rng(0)
        ts, primitives = rng.normal(size=(10, 2048)).astype('f'), rng.normal(size=(10, 156)).astype('f')
        result, _, _ = features_for(ts, primitives, np.arange(10) % 5, 0)
        self.assertIs(result['d0_ts'], ts)
        self.assertEqual(result['d0_ts_primitive'].shape, (10, 2204))
        np.testing.assert_array_equal(result['d0_ts_primitive'][:, :2048], ts)

    def test_bootstrap_identical_zero(self):
        subjects = np.repeat(np.arange(10), 2)
        labels = np.tile([0, 1], 10)
        report, delta = bootstrap(labels, subjects, subjects % 5, labels, labels, 100)
        self.assertEqual(report['mean_fold_f1_delta_ci95'], [0., 0.])
        self.assertTrue((delta == 0).all())

    def test_bootstrap_pairing_improvement_and_determinism(self):
        subjects = np.repeat(np.arange(10), 2)
        labels = np.tile([0, 1], 10)
        args = (labels, subjects, subjects % 5, np.zeros(20), labels, 100)
        report, delta = bootstrap(*args)
        self.assertEqual(report['mean_fold_f1_delta_ci95'], [1., 1.])
        np.testing.assert_array_equal(delta, bootstrap(*args)[1])
