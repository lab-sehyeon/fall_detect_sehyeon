import copy
import tempfile
from pathlib import Path
import unittest

import torch

from fall_pipeline.safer import run_s0b_reconstruction as old
from fall_pipeline.safer import run_s0b_reconstruction_r2 as new
from fall_pipeline.safer import run_s0c_reconstruction_r2 as decoder
from scripts import migrate_s0b_mb512 as migration
from scripts import run_state_recovery_queue_r2 as queue


class Microbatch512Tests(unittest.TestCase):
    def test_only_approved_config_difference(self):
        before = old.io.read(old.CONFIG)
        after = new.io.read(new.CONFIG)
        migration.validate_change(before, after)
        after['training']['lr'] *= 2
        with self.assertRaises(RuntimeError):
            migration.validate_change(before, after)

    def test_effective_batch_cannot_change(self):
        before, after = old.io.read(old.CONFIG), new.io.read(new.CONFIG)
        after['training']['batch_size'] = 512
        with self.assertRaises(RuntimeError):
            migration.validate_change(before, after)

    def test_recursive_exact_state_audit(self):
        state = {'head': torch.randn(2, 3), 'opt': {0: {'moment': torch.ones(2)}},
                 'rng': torch.get_rng_state(), 'cursor': 217088, 'history': []}
        cloned = copy.deepcopy(state)
        self.assertTrue(migration.equal_tree(state, cloned))
        cloned['opt'][0]['moment'][0] = 0
        self.assertFalse(migration.equal_tree(state, cloned))

    def test_checkpoint_roundtrip_preserves_state(self):
        state = {'contract': {'revision': 1}, 'heads': {'tcn': {'weight': torch.randn(4, 3)}},
                 'optimizers': {'tcn': {'state': {0: {'step': torch.tensor(212.)}}}},
                 'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.arange(8, dtype=torch.uint8),
                 'epoch': 1, 'next': 217088}
        changed = copy.deepcopy(state)
        changed['contract'] = {'revision': 2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'resume.pt'
            new.io.save_tensor(path, changed)
            saved = torch.load(path, weights_only=True)
        saved['contract'] = state['contract']
        self.assertTrue(migration.equal_tree(saved, state))

    def test_downstream_only_parent_revision_changes(self):
        before = new.io.read(new.io.ROOT / 'configs/s0c_document_reconstruction_v1.json')
        after = new.io.read(decoder.CONFIG)
        for key in ('experiment_id', 'output', 's0b_config'):
            before[key] = after[key]
        self.assertEqual(before, after)
        self.assertIs(decoder.b, new)
        self.assertIs(queue.b, new)
        self.assertIs(queue.c, decoder)
        self.assertEqual(queue.BROOT, new.io.ROOT / new.io.read(new.CONFIG)['output'])
        self.assertIn('run_s0b_reconstruction_r2.py', ' '.join(new.CODE))


if __name__ == '__main__':
    unittest.main()
