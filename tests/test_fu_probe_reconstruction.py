import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fall_pipeline.fu.probe_reconstruction import pooled_parts, clip_pool, representations, fit_fold, selection_key, CONFIG
from fall_pipeline.fu import probe_reconstruction as probe


class FuProbeTest(unittest.TestCase):
    def test_padding_excluded_from_temporal_and_hidden(self):
        temporal = torch.tensor([[[1.], [3.], [100.]]])
        hidden = torch.tensor([[[2.], [4.], [90.]]])
        spatial = torch.tensor([[[1.], [5.]]])
        parts = pooled_parts(temporal, spatial, hidden, [2])
        self.assertEqual(parts["t"].item(), 3)
        self.assertEqual(parts["s"].item(), 5)
        self.assertEqual(parts["h"].item(), 3)

    def test_clip_pool(self):
        pooled = clip_pool(np.array([[1., 2.], [3., 4.], [7., 8.]], np.float32), np.array([0, 0, 1]), 2)
        np.testing.assert_array_equal(pooled, [[2, 3], [7, 8]])
        with self.assertRaises(RuntimeError):
            clip_pool(np.ones((1, 2)), np.array([0]), 2)

    def test_concat(self):
        parts = {"t": np.ones((3, 2)), "s": np.ones((3, 4)), "h": np.ones((3, 5))}
        features = representations(parts)
        self.assertEqual(features["d2_concat"].shape, (3, 11))

    def test_selection_earliest_tie(self):
        rows = [{"epoch": i, "f1": .5, "auprc": .6, "accuracy": .7} for i in [1, 2]]
        self.assertEqual(max(rows, key=selection_key)["epoch"], 1)

    def test_tiny_training_and_leak_guard(self):
        torch.set_num_threads(2)
        config = json.loads(CONFIG.read_text())
        config["training"]["epochs"] = 2
        rng = np.random.default_rng(4)
        x = rng.standard_normal((20, 4)).astype(np.float32)
        labels = np.tile([0, 1], 10)
        actions = np.where(labels == 1, 5, 4)
        folds = np.repeat(np.arange(5), 4)
        subjects = folds + 1
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "fold"
            result = fit_fold(x, labels, actions, folds, subjects, 0, config, dest, torch.device("cpu"))
            self.assertIn(result["epoch"], [1, 2])
            self.assertEqual(result["metrics"]["clips"], 4)
            self.assertEqual(fit_fold(x, labels, actions, folds, subjects, 0, config, dest, torch.device("cpu")), result)
            with self.assertRaises(RuntimeError):
                fit_fold(x, labels, actions, folds, np.ones(20), 0, config, Path(directory) / "leak", torch.device("cpu"))

    def test_epoch_resume_matches_uninterrupted(self):
        torch.set_num_threads(2)
        config = json.loads(CONFIG.read_text())
        config["training"]["epochs"] = 2
        x = np.random.default_rng(8).standard_normal((20, 4)).astype(np.float32)
        labels = np.tile([0, 1], 10)
        actions = np.where(labels == 1, 5, 4)
        folds = np.repeat(np.arange(5), 4)
        subjects = folds + 1
        actual_save = probe.save_torch
        def interrupt(path, value):
            actual_save(path, value)
            if path.name == "last.pt" and value["epoch"] == 1:
                raise RuntimeError("intentional interruption")
        with tempfile.TemporaryDirectory() as directory:
            resumed, uninterrupted = Path(directory) / "resumed", Path(directory) / "full"
            args = (x, labels, actions, folds, subjects, 0, config)
            with patch.object(probe, "save_torch", interrupt), self.assertRaisesRegex(RuntimeError, "intentional"):
                fit_fold(*args, resumed, torch.device("cpu"))
            first = fit_fold(*args, resumed, torch.device("cpu"))
            second = fit_fold(*args, uninterrupted, torch.device("cpu"))
            self.assertEqual(first["metrics"], second["metrics"])
            a = torch.load(resumed / "last.pt", weights_only=True)["state_dict"]
            b = torch.load(uninterrupted / "last.pt", weights_only=True)["state_dict"]
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))


if __name__ == "__main__":
    unittest.main()
