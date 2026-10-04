import copy
import unittest
from fall_pipeline.safer import run_s0b_reconstruction_r2 as old
from fall_pipeline.safer import run_s0b_reconstruction_r3 as new
from scripts import migrate_s0b_dual as migration
from scripts import run_state_recovery_queue_r3 as queue


class DualContractTests(unittest.TestCase):
    def test_only_approved_changes(self):
        before=old.io.read(old.CONFIG);after=new.io.read(new.CONFIG)
        migration.validate_change(before,after)
        for key in ('lr','batch_size','epochs'):
            changed=copy.deepcopy(after);changed['training'][key]*=2
            with self.assertRaises(RuntimeError):migration.validate_change(before,changed)
        self.assertFalse(after['training']['amp']);self.assertFalse(after['training']['tf32'])
        self.assertEqual(after['evaluation'],before['evaluation'])

    def test_only_approved_gpu_indices(self):
        before=old.io.read(old.CONFIG);after=new.io.read(new.CONFIG)
        after['parallel']['physical_devices']=[0,2]
        with self.assertRaises(RuntimeError):migration.validate_change(before,after)

    def test_queue_points_to_r3(self):
        self.assertIs(queue.b,new)
        self.assertEqual(queue.BROOT,new.io.ROOT/new.io.read(new.CONFIG)['output'])
        self.assertEqual(new.io.read(queue.c.CONFIG)['s0b_config'],'configs/s0b_document_reconstruction_v3.json')


if __name__=='__main__':unittest.main()
