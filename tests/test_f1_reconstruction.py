import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from fall_pipeline.safer.f1_core import (
    ResidualTemporalAdapter, Timeline, candidate_input, dense_backbone_features,
    make_candidates, runs, segment_counts, selection_key, sqrt_weights,
)
from fall_pipeline.safer.run_f1_reconstruction import (
    CONFIG, CONFIG_HASH, config_load, initialize, optimizer_for, train_step,
)
from fall_pipeline.common.integrity import hash_named_tensors, sha256_file
from data_gen.safer_legacy_v1_gendata import derive_four_class
from fall_pipeline.safer import run_f1_reconstruction as runner
from fall_pipeline.safer.audit_f1_reconstruction import audit
from fall_pipeline.safer.f1_progress_docs import publish


class DummyDSTE(torch.nn.Module):
    def backbone(self, temporal, spatial):
        batch = len(temporal)
        return (torch.arange(64).float().view(1, 64, 1).expand(batch, 64, 1024),
                torch.arange(50).float().view(1, 50, 1).expand(batch, 50, 1024))


def small_timeline():
    labels = np.resize(np.array([0, 10, 11, 12, 9], dtype=np.int64), 80)
    rows = [{"sequence_index": 0, "total_frames": 80, "output_start": 0, "output_stop": 2}]
    return Timeline(rows, np.array([0, 0]), np.array([0, 8]), np.stack([labels[:64], labels[8:72]]))


class F1ReconstructionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_config_is_fixed_and_separates_reconstruction(self):
        config = config_load()
        self.assertEqual(sha256_file(CONFIG), CONFIG_HASH)
        self.assertFalse(config["historical_exact_reproduction"])
        self.assertEqual(config["training"]["epochs"], 20)
        self.assertEqual(config["execution"]["physical_cuda_visible_devices"], "0")
        self.assertFalse(config["evaluation"]["event_used_for_selection"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.json"
            config["training"]["epochs"] = 15
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(RuntimeError, "preregistered config changed"):
                config_load(path)

    def test_spatial_mask_excludes_empty_person_and_zero_joint(self):
        batch = torch.zeros(2, 3, 64, 25, 2)
        batch[:, :, :, 1:, 0] = 1
        temporal, context = dense_backbone_features(DummyDSTE(), batch)
        self.assertEqual(tuple(temporal.shape), (2, 64, 1024))
        self.assertTrue(torch.equal(temporal[0, :, 0], torch.arange(64).float()))
        self.assertTrue(torch.equal(context, torch.full_like(context, 12.5)))

    def test_empty_context_is_zero_and_nonfinite_skeleton_rejected(self):
        temporal, context = dense_backbone_features(DummyDSTE(), torch.zeros(1, 3, 64, 25, 2))
        self.assertTrue(torch.isfinite(temporal).all())
        self.assertEqual(torch.count_nonzero(context), 0)
        bad = torch.ones(1, 3, 64, 25, 2)
        bad[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            dense_backbone_features(DummyDSTE(), bad)

    def test_representation_broadcast_and_order(self):
        t, s = torch.randn(2, 64, 1024), torch.randn(2, 1024)
        self.assertIs(candidate_input("temporal_only__plain", t, s), t)
        joined = candidate_input("temporal_spatial__sqrt", t, s)
        self.assertTrue(torch.equal(joined[:, :, :1024], t))
        self.assertTrue(torch.equal(joined[:, 33, 1024:], s))

    def test_residual_initially_identity_and_hidden_contract(self):
        model = ResidualTemporalAdapter(8, hidden_dim=16).eval()
        value = torch.randn(3, 64, 8)
        expected = model.gelu(model.projection(model.input_norm(value)))
        self.assertTrue(torch.equal(model.hidden(value), expected))
        self.assertEqual(tuple(model(value).shape), (3, 64, 4))

    def test_loss_pair_initializations_equal(self):
        config = copy.deepcopy(config_load())
        config["model"]["hidden_dim"] = 8
        models = make_candidates(config, torch.device("cpu"))
        for representation in ("temporal_only", "temporal_spatial"):
            a = hash_named_tensors(models[representation + "__plain"].state_dict().items())
            b = hash_named_tensors(models[representation + "__sqrt"].state_dict().items())
            self.assertEqual(a, b)

    def test_all_four_candidates_train_and_weight_unit(self):
        config = copy.deepcopy(config_load())
        config["model"]["hidden_dim"] = 8
        models = make_candidates(config, torch.device("cpu"))
        before = {name: hash_named_tensors(m.state_dict().items()) for name, m in models.items()}
        optimizers = {name: optimizer_for(m, config) for name, m in models.items()}
        weights = sqrt_weights([100, 25, 4, 1])
        expected = np.array([0.1, 0.2, 0.5, 1.0])
        np.testing.assert_allclose(weights.numpy(), expected / expected.mean(), atol=1e-7)
        losses = train_step(models, optimizers, torch.randn(3, 64, 1024), torch.randn(3, 1024),
                            torch.arange(192).reshape(3, 64) % 4, weights, config)
        self.assertEqual(len(losses), 4)
        for name, model in models.items():
            self.assertTrue(np.isfinite(losses[name]))
            self.assertNotEqual(before[name], hash_named_tensors(model.state_dict().items()))
        with self.assertRaisesRegex(RuntimeError, "all four"):
            sqrt_weights([1, 2, 0, 4])

    def test_raw_logit_mean_before_softmax_and_uncovered_exclusion(self):
        timeline = small_timeline()
        self.assertEqual(int(timeline.covered.sum()), 72)
        self.assertEqual(int(timeline.counts[9]), 2)
        logits = np.zeros((2, 64, 4), dtype=np.float32)
        logits[0, :, 1], logits[1, :, 1] = 8, -2
        totals = timeline.empty()
        timeline.add(totals, 0, logits[:1])
        timeline.add(totals, 1, logits[1:])
        mean = timeline.mean(totals)
        self.assertEqual(mean.shape, (72, 4))
        self.assertEqual(mean[9, 1], 3)
        self.assertEqual(mean[0, 1], 8)
        self.assertEqual(mean[70, 1], -2)
        metric = timeline.metrics(mean)
        self.assertEqual(metric["uncovered_frames"], 8)
        self.assertEqual(metric["samples"], 72)

    def test_perfect_timeline_metrics_and_unstable_definition(self):
        timeline = small_timeline()
        logits = np.full((72, 4), -5.0, np.float32)
        logits[np.arange(72), timeline.labels] = 5
        metrics = timeline.metrics(logits)
        self.assertEqual(metrics["macro_f1"], 1)
        self.assertEqual(metrics["fall_vs_unstable_auprc"], 1)
        self.assertEqual(metrics["reconstructed_event_iou_0p1"]["f1"], 1)

    def test_bad_overlap_and_negative_start_rejected(self):
        row = [{"sequence_index": 0, "total_frames": 80, "output_start": 0, "output_stop": 2}]
        labels = np.zeros((2, 64), dtype=np.int64)
        labels[1, 0] = 10
        with self.assertRaisesRegex(RuntimeError, "inconsistency"):
            Timeline(row, [0, 0], [0, 8], labels)
        with self.assertRaisesRegex(RuntimeError, "bounds"):
            Timeline(row, [0, 0], [-1, 8], labels)

    def test_matching_one_to_one_and_coverage_gaps(self):
        self.assertEqual(segment_counts([(0, 8), (8, 10)], [(0, 10)]), (1, 1, 0))
        self.assertEqual(segment_counts([], [(0, 10)]), (0, 0, 1))
        self.assertEqual(segment_counts([(20, 30)], [(0, 10)]), (0, 1, 1))
        self.assertEqual(runs([True, True, False, True]), [(0, 2), (3, 4)])

    def test_selection_priority_and_missing_secondary(self):
        base = {"macro_f1": .5, "fall_vs_unstable_auprc": .8, "lying_down_f1": .4}
        self.assertGreater(selection_key({**base, "macro_f1": .6, "fall_vs_unstable_auprc": .1}), selection_key(base))
        self.assertGreater(selection_key({**base, "fall_vs_unstable_auprc": .9}), selection_key(base))
        self.assertGreater(selection_key({**base, "lying_down_f1": .5}), selection_key(base))
        self.assertLess(selection_key({**base, "fall_vs_unstable_auprc": None}), selection_key(base))

    def test_resume_rejects_other_contract_and_nonempty_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            initialize(root, {"config": "one"}, False)
            initialize(root, {"config": "one"}, True)
            with self.assertRaisesRegex(RuntimeError, "nonempty"):
                initialize(root, {"config": "one"}, False)
            with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
                initialize(root, {"config": "two"}, True)

    def test_synthetic_full_pipeline_and_exact_epoch_resume(self):
        config = copy.deepcopy(config_load())
        config["training"].update(epochs=2, batch_size=2)
        config["evaluation"]["batch_size"] = 2
        config["execution"].update(extraction_batch_size=2, checkpoint_every_batches=1)
        config["input"]["windows"] = {s: 4 for s in ("train", "val", "test", "ood")}
        labels = np.resize(np.array([0, 10, 11, 12, 9], dtype=np.int64), 88)
        data = np.zeros((4, 3, 64, 25, 2), dtype=np.float32)
        data[:, :, :, 1:, 0] = 1
        data[-1] = 0  # Keep an actual empty window in the end-to-end regression.
        arrays = {"data_joint.npy": data, "sequence_index.npy": np.zeros(4, dtype=np.int64),
                  "window_start.npy": np.array([0, 8, 16, 24], dtype=np.int64),
                  "dense_coarse_labels.npy": np.stack([labels[i:i+64] for i in (0, 8, 16, 24)])}
        arrays["dense_derived_labels.npy"] = derive_four_class(arrays["dense_coarse_labels.npy"])
        manifest = {"payload": {"data_joint.npy": {"array_payload_sha256": runner.payload_digest(data)}}}
        def fake_source(cfg, split, include_data=True):
            row = {"sequence_index": 0, "total_frames": 88, "output_start": 0, "output_stop": 4,
                   "subject": ["train", "val", "test", "ood"].index(split) + 1}
            return Path("unused"), manifest, arrays, [row]
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(runner, "source", side_effect=fake_source), \
                patch.object(runner, "backbone", side_effect=lambda *args: DummyDSTE().eval()), \
                patch.object(runner, "assert_contract"), patch.object(runner, "disk_gate", return_value=10**12), \
                patch.object(runner, "publish"), patch.object(runner, "config_load", return_value=config), \
                patch.object(torch.cuda, "get_rng_state", side_effect=lambda *args: torch.get_rng_state()), \
                patch.object(torch.cuda, "set_rng_state"), patch.object(runner, "log"):
            complete, resumed = Path(temporary) / "complete", Path(temporary) / "resumed"
            for root in (complete, resumed):
                initialize(root, {"synthetic": True}, False)
                runner.extract(config, root, torch.device("cpu"))
            runner.train(config, complete, torch.device("cpu"))
            original_step = runner.train_step
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("synthetic interruption at epoch two")
                return original_step(*args, **kwargs)
            with patch.object(runner, "train_step", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    runner.train(config, resumed, torch.device("cpu"))
            self.assertEqual(torch.load(resumed / "training/last.pt", weights_only=True)["epoch"], 1)
            runner.train(config, resumed, torch.device("cpu"))
            full_state = torch.load(complete / "training/last.pt", weights_only=True)
            resumed_state = torch.load(resumed / "training/last.pt", weights_only=True)
            self.assertEqual(full_state["best"], resumed_state["best"])
            for name in config["candidates"]:
                self.assertEqual(hash_named_tensors(full_state["models"][name].items()),
                                 hash_named_tensors(resumed_state["models"][name].items()))
            runner.evaluate(config, complete, torch.device("cpu"))
            report = runner.read(complete / "final_report.json")
            self.assertEqual(report["status"], "completed")
            self.assertTrue(report["frozen_backbone_adl_exact"])
            self.assertEqual(report["results"]["test"]["windows"], 4)
            independent = audit(complete)
            self.assertTrue(independent["passed"])
            self.assertEqual(independent["epochs_verified"], 2)
            # Completed held-outs reuse validated artifacts, never re-infer.
            with patch.object(runner, "dense_backbone_features", side_effect=RuntimeError("must not infer again")):
                runner.evaluate(config, complete, torch.device("cpu"))

    def test_generated_documents_match_and_shared_is_sanitized(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root = project / "run"
            root.mkdir()
            publish(root, config_load(), "extract", project_root=project)
            shared = (project / "docs/shared/2026-09-20_f1_run_shared.md").read_text()
            internal = (project / "docs/internal/2026-09-20_f1_run_internal.md").read_text()
            self.assertIn("DOC-20260920-f1-run-R1", shared)
            self.assertIn("DOC-20260920-f1-run-R1", internal)
            self.assertNotIn(str(project), shared)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", shared)
            self.assertNotIn("checkpoint/", shared)
            publish(root, config_load(), "train", project_root=project)
            self.assertIn("DOC-20260920-f1-run-R2", (project / "docs/shared/2026-09-20_f1_run_shared.md").read_text())


if __name__ == "__main__":
    unittest.main()
