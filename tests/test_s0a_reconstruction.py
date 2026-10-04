import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fall_pipeline.safer import s0a_core as core
from fall_pipeline.safer import run_s0a_reconstruction as run


def reference_edit(a, b):
    table = np.zeros((len(a) + 1, len(b) + 1), np.int64)
    table[:, 0] = np.arange(len(a) + 1)
    table[0] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            table[i, j] = min(table[i - 1, j] + 1, table[i, j - 1] + 1,
                              table[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return int(table[-1, -1])


def reference_segments(p, y, threshold):
    def spans(sequence):
        output = []
        for frame, label in enumerate(sequence):
            if not output or label != output[-1][0]:
                output.append([int(label), frame, frame + 1])
            else:
                output[-1][2] += 1
        return output
    ps, ys = spans(p), spans(y)
    matched = set()
    for label, start, end in ps:
        overlap = [max(0, min(end, ge) - max(start, gs)) / (max(end, ge) - min(start, gs))
                   if gl == label else 0. for gl, gs, ge in ys]
        if overlap:
            best = max(range(len(overlap)), key=overlap.__getitem__)
            if overlap[best] >= threshold and best not in matched:
                matched.add(best)
    return [len(matched), len(ps) - len(matched), len(ys) - len(matched)]


class S0ATests(unittest.TestCase):
    def test_edit_against_scalar_dynamic_program(self):
        rng = np.random.default_rng(14)
        for n in range(15):
            for m in range(15):
                a, b = rng.integers(0, 4, n), rng.integers(0, 4, m)
                self.assertEqual(core.edit_distance(a, b), reference_edit(a, b))

    def test_segment_matches_reference_and_class_zero_is_included(self):
        rng = np.random.default_rng(11)
        for _ in range(50):
            p, y = rng.integers(0, 4, (2, 35))
            for threshold in (.1, .25, .5):
                self.assertEqual(core.segment_counts(p, y, threshold), reference_segments(p, y, threshold))
        self.assertEqual(core.segment_counts([0, 0], [0, 0], .5), [1, 0, 0])
        self.assertEqual(core.segment_counts([], [], .5), [0, 0, 0])

    def test_overlap_independent_scatter_and_tail(self):
        rng = np.random.default_rng(1)
        starts = np.array([0, 8, 16])
        dense = rng.normal(size=(3, 64, 16)).astype(np.float32)
        total = np.zeros((85, 16), np.float64)
        core.add_logits(total, starts, dense)
        expected = np.zeros_like(total)
        for start, values in zip(starts, dense):
            for frame, row in enumerate(values):
                expected[start + frame] += row
        np.testing.assert_array_equal(total, expected)
        counts = core.coverage(85, starts)
        wanted = np.zeros(85, np.int64)
        for start in starts:
            wanted[start:start + 64] += 1
        np.testing.assert_array_equal(counts, wanted)
        self.assertEqual(int(np.count_nonzero(counts)), 80)
        with self.assertRaises(RuntimeError):
            core.coverage(10, np.array([0]))

    def test_direct_transitions_no_sequence_bridging(self):
        y = [10, 10, 12, 12, 7, 7]
        p = [10, 12, 12, 12, 7, 7]
        value = core.transition_counts(p, y, (10, 12), 1)
        self.assertEqual(value, {'tp': 1, 'fp': 0, 'fn': 0, 'delays_frames': [-1]})
        value = core.transition_counts([10, 11, 12], [10, 10, 12], (10, 12), 12)
        self.assertEqual(value['tp'], 0)
        self.assertEqual(value['fn'], 1)

    def test_metrics_and_absent_classes(self):
        metrics = core.Metrics([[10, 12]], 12)
        metrics.add([0, 0, 1, 1], [0, 0, 1, 1])
        metrics.add([1, 1], [1, 1])
        result = metrics.result()
        self.assertEqual(result['macro_f1'], 2 / 16)
        self.assertEqual(result['segment_f1_50'], 1)
        self.assertEqual(result['edit'], 1)
        self.assertEqual(result['switches'], 1)
        self.assertEqual(result['switches_per_minute'], 250)
        self.assertEqual(result['segments']['50']['tp'], 3)

    def test_compact_random_window_layout_dense_labels_and_frequency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'sequence_cache'
            (cache / 'train').mkdir(parents=True)
            rows, poses, labels = [], [], []
            for i in range(2):
                (cache / str(i)).mkdir()
                poses.append(np.arange(80 * 25 * 3, dtype=np.float32).reshape(80, 25, 3) + i * 10000)
                labels.append((np.arange(80) + i) % 16)
                np.save(cache / str(i) / 'ntu25.npy', poses[i])
                np.save(cache / str(i) / 'labels.npy', labels[i])
                rows.append({'uid': str(i), 'sequence_index': i, 'output_start': 3 * i, 'output_stop': 3 * i + 3})
            (cache / 'train/sequences.json').write_text(json.dumps(rows))
            np.save(cache / 'train/sequence_index.npy', [0, 0, 0, 1, 1, 1])
            np.save(cache / 'train/window_start.npy', [0, 8, 16, 0, 8, 16])
            data = core.CompactSplit(root, 'train')
            x, y = data.batch([5, 0, 3, 1])
            for pos, (sid, start) in enumerate([(1, 16), (0, 0), (1, 0), (0, 8)]):
                np.testing.assert_array_equal(x[pos, :, :, :, 0], poses[sid][start:start + 64].transpose(2, 0, 1))
                np.testing.assert_array_equal(y[pos], labels[sid][start:start + 64])
            self.assertFalse(x[:, :, :, :, 1].any())
            self.assertEqual(data.frequencies()['covered_frames'], 160)
            self.assertEqual(data.frequencies()['counts'], [10] * 16)

    def test_selection_ties_and_order(self):
        metric = {'macro_f1': .4, 'segment_f1_50': .2, 'edit': .1}
        history = [{'epoch': i, 'metrics': {'plain': metric, 'sqrt': metric}, 'checkpoint_sha256': str(i)} for i in (1, 2)]
        self.assertEqual(run.choose(history, ['plain', 'sqrt'])['candidate'], 'plain')
        self.assertEqual(run.choose(history, ['plain', 'sqrt'])['epoch'], 1)
        history[1]['metrics'] = {'plain': metric, 'sqrt': {**metric, 'edit': .11}}
        self.assertEqual(run.choose(history, ['plain', 'sqrt'])['candidate'], 'sqrt')

    def test_heads_same_init_and_optimizer_resume(self):
        config = run.io.read(run.CONFIG)
        heads, optimizers = run.make_heads(config, torch.device('cpu'))
        torch.testing.assert_close(heads['plain'].weight, heads['sqrt'].weight, rtol=0, atol=0)
        x = torch.ones(1, 2048)
        for name in heads:
            heads[name](x).sum().backward()
            optimizers[name].step()
        state = {'contract': {'x': 1}, 'epoch': 2, 'next': 128, 'history': []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run.save_state(root, state, heads, optimizers)
            run.save_state(root, state, heads, optimizers)
            restored, opts = run.make_heads(config, torch.device('cpu'))
            got = run.restore_state(root, {'x': 1}, restored, opts)
            self.assertEqual(got['next'], 128)
            for name in heads:
                torch.testing.assert_close(restored[name].weight, heads[name].weight, rtol=0, atol=0)
                for a, b in zip(opts[name].state.values(), optimizers[name].state.values()):
                    torch.testing.assert_close(a['momentum_buffer'], b['momentum_buffer'], rtol=0, atol=0)

    def test_pause_and_disk_fail_closed(self):
        config = run.io.read(run.CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'PAUSE_REQUESTED').touch()
            with self.assertRaises(run.io.PauseRequested):
                run.check_pause(root, config)
        with tempfile.TemporaryDirectory() as directory, patch.object(run.io, 'disk_gate', side_effect=RuntimeError('reserve')):
            with self.assertRaises(run.io.PauseRequested):
                run.check_pause(Path(directory), config)

    def test_holdout_verifier_only_opens_named_split(self):
        # A train audit must never invoke a broad all-split loader.
        import inspect
        source = inspect.getsource(core.verify_split)
        self.assertNotIn('verify_source(', source)
        self.assertNotIn("'test'", source)
        self.assertNotIn("'ood'", source)


if __name__ == '__main__':
    unittest.main()
