import copy
import unittest

import numpy as np
import torch

from fall_pipeline.safer import s0f_core as f

MODEL = {'hidden': 8, 'dilations': [1, 2, 4], 'kernel': 3, 'dropout': 0.1, 'classes': 4}


class S0FTests(unittest.TestCase):
    def test_targets_three_events_and_center(self):
        y = np.r_[np.repeat(10, 8), np.repeat(12, 8), np.repeat(7, 8), np.repeat(6, 8), np.repeat(4, 8)]
        target = f.targets(y)
        np.testing.assert_array_equal(np.flatnonzero(target == 1), np.arange(6, 11))
        np.testing.assert_array_equal(np.flatnonzero(target == 3), np.arange(14, 19))
        np.testing.assert_array_equal(np.flatnonzero(target == 2), np.arange(30, 35))

    def test_collision_nearest_then_earlier(self):
        np.testing.assert_array_equal(f.targets(np.array([10, 12, 12, 7, 7])), [1, 1, 1, 3, 3])
        np.testing.assert_array_equal(f.targets(np.array([10, 0, 12])), [0, 0, 0])
        with self.assertRaises(RuntimeError):
            f.targets(np.array([10, 12]), 4)

    def test_sequence_reset_no_cross_boundary(self):
        self.assertEqual(sum(map(len, f.events(np.array([10])).values())), 0)
        self.assertEqual(sum(map(len, f.events(np.array([12])).values())), 0)

    def test_dynamics_formula_and_prefix(self):
        x = np.arange(16*11, dtype=np.float32).reshape(11, 16)/100
        features = f.dynamics(x)
        self.assertEqual(features.shape, (11, 35))
        np.testing.assert_array_equal(features[0, 16:32], 0)
        np.testing.assert_allclose(features[:, :16].sum(1), 1, atol=1e-6)
        np.testing.assert_array_equal(features[1:, 16:32], features[1:, :16]-features[:-1, :16])
        np.testing.assert_array_equal(features[:5], f.dynamics(x[:5]))
        uniform = f.dynamics(np.zeros((1, 16), np.float32))
        self.assertAlmostEqual(float(uniform[0, 32]), np.log(16), places=6)
        self.assertAlmostEqual(float(uniform[0, 33]), 1/16)
        self.assertEqual(uniform[0, 34], 0)

    def test_dynamics_bad_shape(self):
        for bad in (np.zeros((4, 15)), np.full((2, 16), np.nan), np.zeros((0, 16))):
            with self.assertRaises(RuntimeError):
                f.dynamics(bad)

    def test_head_causal_chunk_and_backward(self):
        torch.manual_seed(0)
        head = f.TransitionHead(MODEL)
        self.assertEqual(head.context, 14)
        self.assertTrue(f.prefix_audit(head, torch.device('cpu'))['same_shape_future_perturbation_exact'])
        head.train()
        x = torch.randn(3, 64, 35)
        head(x).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_train_rng_continuation(self):
        torch.manual_seed(12)
        head = f.TransitionHead(MODEL).train()
        x = torch.randn(3, 64, 35)
        before = torch.get_rng_state()
        first = head(x)
        torch.set_rng_state(before)
        torch.testing.assert_close(first, head(x), rtol=0, atol=0)

    def test_balanced_sampler_determinism(self):
        groups = [np.array([i]) for i in range(4)]
        x = f.balanced_order(groups, 1001, 11)
        counts = np.bincount(x, minlength=4)
        self.assertLessEqual(counts.max()-counts.min(), 1)
        np.testing.assert_array_equal(x, f.balanced_order(groups, 1001, 11))
        self.assertFalse(np.array_equal(x, f.balanced_order(groups, 1001, 12)))

    def test_window_groups_overlap_and_tail(self):
        rows = [{'target': np.zeros(130, np.int64)} for _ in range(4)]
        for k in (1, 2, 3):
            rows[k]['target'][32] = k
        rows[1]['target'][34] = 2
        windows, groups = f.window_groups(rows)
        self.assertEqual(len(windows), 4*9)
        self.assertTrue(set(groups[1]) & set(groups[2]))
        self.assertFalse(set(groups[0]) & set(np.concatenate(groups[1:])))
        self.assertTrue(all(len(g) for g in groups))

    def test_capped_weights(self):
        np.testing.assert_array_equal(f.weights([10000, 1, 100, 4], 20), [1, 20, 10, 20])
        np.testing.assert_array_equal(f.weights([10000, 1, 100, 4], 50), [1, 50, 10, 50])

    def test_onsets_and_matching_one_to_one(self):
        p = np.array([1, 1, 0, 1, 2, 2, 1])
        np.testing.assert_array_equal(f.onsets(p, 1), [0, 3, 6])
        self.assertEqual(f.match([8], [7, 9], 2), [1, 0, 1])
        self.assertEqual(f.match([1, 9, 20], [10], 12), [1, 2, 0])
        self.assertEqual(f.match([], [10], 12), [0, 0, 1])

    def test_matching_independent_two_pointer(self):
        rng = np.random.default_rng(2)
        for _ in range(100):
            p, g = [np.sort(rng.choice(100, 15, replace=False)) for _ in range(2)]
            # Equivalent interval matching reference, without masks or shared helper.
            remaining = list(p); tp = 0
            for t in g:
                while remaining and remaining[0] < t-4:
                    remaining.pop(0)
                if remaining and remaining[0] <= t+4:
                    tp += 1; remaining.pop(0)
            self.assertEqual(f.match(p, g, 4), [tp, len(p)-tp, len(g)-tp])

    def test_view_metrics_missing_classes_zero(self):
        metric = f.EventMetrics()
        y = np.array([10, 10, 12, 12])
        metric.add(np.array([0, 0, 1, 0]), y, 1)
        metric.add(np.array([0, 0, 0, 0]), y, 2)
        out = metric.result()
        self.assertAlmostEqual(out['macro_f1'], (2/3)/3)
        self.assertEqual(out['worst_view_macro_f1'], 0)
        self.assertEqual(out['frames'], 8)

    def test_selection_view_first_tie_epoch_and_candidate(self):
        m = lambda worst, overall: {'worst_view_macro_f1': worst, 'macro_f1': overall}
        history = [{'epoch': 1, 'checkpoint_sha256': 'a', 'metrics': {'cap20': m(.2, .8), 'cap50': m(.3, .4)}},
                   {'epoch': 2, 'checkpoint_sha256': 'b', 'metrics': {'cap20': m(.3, .4), 'cap50': m(.3, .4)}}]
        self.assertEqual(f.choose(history, ['cap20', 'cap50'])['candidate'], 'cap50')
        self.assertEqual(f.choose(history, ['cap20', 'cap50'])['epoch'], 1)

    def test_gate_all_required_and_camera_parse(self):
        cfg = {'macro_f1_min': .4, 'worst_view_macro_f1_min': .2, 'each_recall_min': .4, 'false_events_per_minute_max': 6}
        m = {'macro_f1': .4, 'worst_view_macro_f1': .2, 'per_class': [{'recall': .4}]*3, 'false_events_per_minute': 6}
        self.assertTrue(f.gate(m, cfg)['feasible_for_integration'])
        failed = copy.deepcopy(m); failed['per_class'] = [{'recall': .4}, {'recall': .399}, {'recall': .8}]
        self.assertFalse(f.gate(failed, cfg)['feasible_for_integration'])
        self.assertEqual(f.view_id('dec_07_2023_fall_p006_d04'), 4)
        with self.assertRaises(RuntimeError):
            f.view_id('unknown')


if __name__ == '__main__':
    unittest.main()
