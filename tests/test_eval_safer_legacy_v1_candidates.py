import unittest

import numpy as np

from fall_pipeline.common.eval_safer_legacy_v1_candidates import (
    HISTORICAL_VALIDATION_PERCENT,
    NTU_FALL_INDEX,
    WindowRef,
    candidate_decision,
    f0a_metrics,
    historical_distance,
)


class F0ACandidateEvaluationTest(unittest.TestCase):
    def test_metrics_use_a043_argmax_and_coarse_fall_label(self):
        labels = np.asarray([10, 10, 11, 12, 0], dtype=np.int64)
        logits = np.zeros((5, 60), dtype=np.float32)
        logits[:, 0] = 1.0
        logits[[0, 2], NTU_FALL_INDEX] = 2.0
        metrics = f0a_metrics(labels, logits)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 1))
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["fall_vs_lie_samples"], 4)
        self.assertIsNotNone(metrics["auprc"])
        self.assertIsNotNone(metrics["fall_vs_lie_auprc"])

    def test_historical_distance_is_percentage_point_error(self):
        metrics = {
            "f1": HISTORICAL_VALIDATION_PERCENT["f1"] / 100.0,
            "auprc": HISTORICAL_VALIDATION_PERCENT["auprc"] / 100.0,
            "fall_vs_lie_auprc": (
                HISTORICAL_VALIDATION_PERCENT["fall_vs_lie_auprc"] / 100.0
            ),
        }
        distance = historical_distance(metrics)
        self.assertAlmostEqual(distance["sum_absolute_percentage_point_error"], 0.0)

    def test_window_reference_is_stable_value(self):
        self.assertEqual(WindowRef(3, 16), WindowRef(3, 16))

    def test_closest_is_not_selected_with_wrong_adl_head_epoch(self):
        result = {
            "candidate": {"name": "candidate_a"},
            "historical_distance": {
                "absolute_percentage_point_error": {
                    "f1": 0.1, "auprc": 0.2, "fall_vs_lie_auprc": 0.3,
                },
                "sum_absolute_percentage_point_error": 0.6,
            },
        }
        decision = candidate_decision([result], True, 155)
        self.assertEqual(decision["closest_candidate"], "candidate_a")
        self.assertIsNone(decision["selected_candidate"])
        self.assertFalse(decision["selection_gate"]["passed"])

    def test_matching_epoch_and_metrics_can_select(self):
        result = {
            "candidate": {"name": "candidate_a"},
            "historical_distance": {
                "absolute_percentage_point_error": {
                    "f1": 0.1, "auprc": 0.2, "fall_vs_lie_auprc": 0.3,
                },
                "sum_absolute_percentage_point_error": 0.6,
            },
        }
        decision = candidate_decision([result], True, 150)
        self.assertEqual(decision["selected_candidate"], "candidate_a")
        self.assertTrue(decision["selection_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
