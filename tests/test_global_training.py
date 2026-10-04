from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from fall_pipeline.joint import global_motion_reconstruction as run


class GlobalTrainingTests(unittest.TestCase):
    def test_train_scaler_constant_and_no_holdout_dependency(self):
        train=np.array([[1.,3.,9.],[3.,7.,9.]])
        mean,scale=run.scaler_fit(train)
        np.testing.assert_array_equal(mean,[2.,5.,9.]);np.testing.assert_array_equal(scale,[1.,2.,1.])
        test=np.array([[1000.,1000.,1000.]])
        _=(test-mean)/scale
        np.testing.assert_array_equal(mean,train.mean(0))

    def test_one_epoch_three_heads_cpu_replay_and_lock(self):
        torch.set_num_threads(2);rng=np.random.default_rng(4)
        d=dict(x=rng.normal(size=(12,2048)).astype(np.float32),g=rng.normal(size=(12,131)).astype(np.float32),
               y=np.tile(np.arange(4),3))
        data=run.to_device(d,torch.device('cpu'));config=run.io.read(run.CONFIG)
        with tempfile.TemporaryDirectory() as folder,patch.object(run.data,'safety'),patch.object(run,'publish'):
            root=Path(folder);(root/'train_scaler.npz').write_bytes(b'test scaler')
            lock=run.fit(root,data,data,config,{},torch.device('cpu'),smoke=True)
            self.assertFalse(lock['validation_adoption_eligible'])
            self.assertFalse(lock['holdout_used_for_selection'])
            for name in run.VARIANTS:
                head,state=run.selected_head(root,lock,name,torch.device('cpu'))
                logits=run.predict(head,data,name,4)
                self.assertLess(run.audit_logits(d,name,state,logits,config),1e-4)
                self.assertEqual(logits.shape,(12,4))
            # Resuming a completed epoch must reproduce the same selection manifest.
            again=run.fit(root,data,data,config,{},torch.device('cpu'),smoke=True)
            self.assertEqual(again,lock)

    def test_holdout_cache_requires_lock_before_open(self):
        with self.assertRaisesRegex(RuntimeError,'holdout'):
            run.data.cache({},'ood')

    def test_class_weight_train_counts_only(self):
        weights=run.train_weights(torch.tensor([0,0,0,0,1,2,3]))
        self.assertLess(weights[0].item(),weights[1].item())
        self.assertAlmostEqual(weights.mean().item(),1.)


if __name__=='__main__':unittest.main()
