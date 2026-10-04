import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fall_pipeline.safer.train_f0b_safer_heads import (
    CANDIDATE_ORDER,
    atomic_json_dump,
    candidate_is_better,
    classification_metrics,
    create_optimizers,
    initialize_heads,
    load_recovery_config,
    selection_key_for_json,
    train_epoch,
)


class F0BSaferHeadTrainingTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config_path = root / "configs/f0b_safer_v1_recovery_sgd_v1.json"
        self.config, _ = load_recovery_config(self.config_path)

    def test_preregistered_configuration_is_exact(self):
        optimization = self.config["optimization"]
        self.assertEqual(optimization["optimizer"], "SGD")
        self.assertEqual(optimization["epochs"], 50)
        self.assertEqual(optimization["batch_size"], 512)
        self.assertEqual(optimization["learning_rate"], 0.006)
        self.assertIsNone(optimization["class_weight"])
        self.assertFalse(self.config["input"]["test_ood_access"])

    def test_changed_configuration_is_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["optimization"]["learning_rate"] = 0.001
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                load_recovery_config(path)

    def test_metrics_and_selection_order(self):
        labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        logits = np.full((8, 4), -3.0, dtype=np.float32)
        logits[np.arange(8), labels] = 3.0
        metrics = classification_metrics(labels, logits)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["fall_f1"], 1.0)
        self.assertAlmostEqual(metrics["conditional_fall_vs_lie_auprc"], 1.0)
        worse_primary = copy.deepcopy(metrics)
        worse_primary["macro_f1"] = 0.9
        better_secondary = copy.deepcopy(metrics)
        better_secondary["conditional_fall_vs_lie_auprc"] = 0.5
        baseline = copy.deepcopy(metrics)
        baseline["conditional_fall_vs_lie_auprc"] = 0.4
        self.assertFalse(candidate_is_better(worse_primary, metrics))
        self.assertTrue(candidate_is_better(better_secondary, baseline))
        self.assertFalse(candidate_is_better(metrics, metrics))

    def test_missing_secondary_selection_metric_is_strict_json_null(self):
        metrics = {
            "macro_f1": 0.25,
            "conditional_fall_vs_lie_auprc": None,
        }
        payload = selection_key_for_json(metrics)
        self.assertEqual(payload, [0.25, None])
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)

    def test_atomic_json_writer_rejects_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            with self.assertRaises(ValueError):
                atomic_json_dump(path, {"selection_key": [0.25, float("-inf")]})
            self.assertFalse(path.exists())

    def test_both_heads_train_on_the_same_cpu_minibatches(self):
        rng = np.random.default_rng(9)
        count = 8
        arrays = {
            "temporal_features.npy": rng.normal(size=(count, 1024)).astype(np.float32),
            "spatial_features.npy": rng.normal(size=(count, 1024)).astype(np.float32),
            "center_derived_labels.npy": np.arange(count, dtype=np.int64) % 4,
        }
        split = {"count": count, "arrays": arrays}
        device = torch.device("cpu")
        torch.manual_seed(0)
        heads = initialize_heads(self.config, device)
        optimizers = create_optimizers(heads, self.config)
        before = {
            name: head.weight.detach().clone() for name, head in heads.items()
        }
        losses = train_epoch(
            heads=heads,
            optimizers=optimizers,
            split=split,
            permutation=torch.arange(count),
            batch_size=4,
            device=device,
            max_batches=0,
        )
        self.assertEqual(tuple(losses), CANDIDATE_ORDER)
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        for name, head in heads.items():
            self.assertFalse(torch.equal(before[name], head.weight))


if __name__ == "__main__":
    unittest.main()
