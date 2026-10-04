import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from fall_pipeline.joint import reconstruction_core as core
from fall_pipeline.joint import reconstruction_inputs as inputs
from fall_pipeline.joint import run_reconstruction as run
from fall_pipeline.safer import v2_controls_reconstruction as io


class JointTests(unittest.TestCase):
    def fixture(self):
        config=io.read(run.CONFIG);config['model'].update(feature_dim=16,bottleneck=4)
        config['training'].update(safer_batch_size=8,fu_batch_size=4,eval_batch_size=7,j0_epochs=3,j1_epochs=3)
        rng=np.random.default_rng(4)
        cpu={s:{'x':rng.normal(size=(n,16)).astype('f'),'y':np.arange(n,dtype=np.int64)%4}
             for s,n in (('train',32),('val',11))}
        subjects=np.arange(25)%5+1;y=np.arange(25,dtype=np.int64)%2
        fcpu={'x':rng.normal(size=(25,16)).astype('f'),'y':y,'actions':np.where(y,5,4),
              'folds':subjects-1,'subjects':subjects}
        device=torch.device('cpu')
        return config,cpu,fcpu,{s:inputs.to_device(v,device) for s,v in cpu.items()},inputs.to_device(fcpu,device)

    def test_nested_indices_subject_disjoint_and_outer_excluded(self):
        _,_,f,_,_=self.fixture()
        for k in range(5):
            train,val,outer=core.split_indices(f['folds'],f['subjects'],k,True)
            self.assertTrue((f['folds'][val]==(k+1)%5).all())
            self.assertTrue((f['folds'][outer]==k).all())
            self.assertFalse(set(train)&set(outer));self.assertEqual(len(train),15)

    def test_batch_pairs_cover_safer_once_and_only_fu_train(self):
        c,_,f,_,_=self.fixture();train=np.flatnonzero(f['folds']>1)
        batches=list(core.paired_batches(33,train,1,c))
        np.testing.assert_array_equal(np.sort(torch.cat([p[0] for p in batches]).numpy()),np.arange(33))
        self.assertTrue(all(len(fi)==4 and set(fi.tolist())<=set(train) for _,fi in batches))
        other=list(core.paired_batches(33,train,1,c))
        for a,b in zip(batches,other): self.assertTrue(torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]))

    def test_epoch0_exact_heads_copied_and_adapter_trainable(self):
        c,_,_,s,f=self.fixture();device=torch.device('cpu')
        j0=core.model_for('j0',c,device);j1=core.model_for('j1',c,device,io.cpu_state(j0))
        self.assertTrue(core.epoch0_exact(j0,j1,s['val'],f,np.arange(25),c)['passed'])
        self.assertNotEqual(j0.safer_head.weight.data_ptr(),j1.safer_head.weight.data_ptr())
        opt=core.optimizer_for(j1,'j1',c)
        core.train_epoch(j1,opt,s['train'],f,np.arange(25),1,'j1',c)
        self.assertTrue((j1.adapter.up.weight!=0).any())

    def test_holdout_loader_is_locked(self):
        with self.assertRaisesRegex(RuntimeError,'holdout arrays'):
            inputs.load_safer({},'test')

    def test_epoch_resume_exact_including_dropout(self):
        c,_,_,s,f=self.fixture();device=torch.device('cpu');train,val,_=core.split_indices(f['folds'],f['subjects'],0,True)
        initial=io.cpu_state(core.model_for('j0',c,device));contract={'test_only':True}
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,patch.object(run,'publish'):
            ra,rb=Path(a),Path(b);da=ra/'development/fold0/j1';db=rb/'development/fold0/j1'
            full,_=run.fit_phase(ra,da,'j1',s,f,train,val,c,device,contract,False,initial)
            with patch.object(io,'pause',side_effect=[None,io.PauseRequested('test')]):
                with self.assertRaises(io.PauseRequested): run.fit_phase(rb,db,'j1',s,f,train,val,c,device,contract,False,initial)
            resumed,_=run.fit_phase(rb,db,'j1',s,f,train,val,c,device,contract,False,initial)
            self.assertEqual(io.hash_named_tensors(full.state_dict().items()),io.hash_named_tensors(resumed.state_dict().items()))
            latest_a=torch.load(da/'latest.pt',weights_only=True);latest_b=torch.load(db/'latest.pt',weights_only=True)
            self.assertEqual(io.hash_named_tensors(latest_a['state_dict'].items()),io.hash_named_tensors(latest_b['state_dict'].items()))

    def test_outer_changes_do_not_affect_training_or_selection(self):
        c,_,_,s,f=self.fixture();train,val,outer=core.split_indices(f['folds'],f['subjects'],0,True)
        f2=copy.deepcopy(f);f2['x'][outer]=100;f2['y'][outer]=1-f2['y'][outer]
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,patch.object(run,'publish'):
            m1,r1=run.fit_phase(Path(a),Path(a)/'nested/fold0/j0','j0',s,f,train,val,c,torch.device('cpu'),{},True)
            m2,r2=run.fit_phase(Path(b),Path(b)/'nested/fold0/j0','j0',s,f2,train,val,c,torch.device('cpu'),{},True)
            self.assertEqual(r1['best']['metrics'],r2['best']['metrics'])
            self.assertEqual(io.hash_named_tensors(m1.state_dict().items()),io.hash_named_tensors(m2.state_dict().items()))

    def test_complete_tiny_development_locked_nested_final_audit(self):
        c,cpu,fcpu,s,f=self.fixture();device=torch.device('cpu');contract={'test_only':True}
        with tempfile.TemporaryDirectory() as folder,patch.object(run,'publish'),patch.object(inputs,'load_safer',return_value=cpu['val']):
            root=Path(folder)
            run.fit_folds(root,'development',s,f,c,device,contract,True)
            run.aggregate(root,'development',f,c,True);run.locked_safer(root,{},c,device,True)
            run.fit_folds(root,'nested',s,f,c,device,contract,True)
            nested=run.aggregate(root,'nested',f,c,True)
            final=run.final_fit(root,s,f,nested,c,device,contract,True)
            self.assertEqual(final['epochs']['j0'],int(np.median(nested['j0']['best_epochs'])))
            self.assertTrue(run.audit(root,cpu,fcpu,{},c,True)['passed'])

    def test_zero_median_final_is_exact_identity(self):
        c,_,_,s,f=self.fixture()
        with tempfile.TemporaryDirectory() as folder,patch.object(run,'publish'):
            root=Path(folder);nested={'j0':{'best_epochs':[1]*5},'j1':{'best_epochs':[0,0,0,1,2]}}
            io.save_json(root/'nested/report.json',nested)
            result=run.final_fit(root,s,f,nested,c,torch.device('cpu'),{},False)
            self.assertEqual(result['epochs']['j1'],0)
            np.testing.assert_array_equal(np.load(root/'final/j0/training_fu_logits.npy'),np.load(root/'final/j1/training_fu_logits.npy'))


if __name__=='__main__': unittest.main()
