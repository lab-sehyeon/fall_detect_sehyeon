import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data_gen import safer_v2_reconstruction as v2


class V2RunnerTests(unittest.TestCase):
    def fixture(self, root):
        (root/'sequences').mkdir()
        v2.save_json(root/'run_contract.json',{'test_only':True})
        items=[]
        rng=np.random.default_rng(0)
        for i,frames in enumerate((63,64,88,4170)):
            a={'uid':f'normal_{i:04d}','total_frames':frames,'frame_dir':f'in_lab/example_p01_{i}',
               'labels':np.resize(np.array([0,10,12,12,11,12,0],np.int64),frames)}
            items.append(a)
            ntu=rng.normal(size=(frames,25,3)).astype(np.float32);ntu[:,1]=0
            np.savez(root/'sequences'/f'{a["uid"]}.npz',ntu25=ntu,root_trajectory=rng.normal(size=(frames,3)).astype('f'))
        count=sum(len(v2.window_starts(a['total_frames'])) for a in items)
        return {'train':items},{'materialization':{'expected_windows':{'train':count}}}

    def test_writer_full_independent_roundtrip_and_idempotent_resume(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(v2,'publish'), patch.object(v2,'disk_gate'):
            root=Path(folder);items,config=self.fixture(root)
            first=v2.materialize(root,items,config)
            report=v2.audit_materialized(root,items,config)
            self.assertTrue(report['integrity']['passed'])
            self.assertEqual(report['splits']['train']['windows'],519)
            self.assertEqual(first,v2.materialize(root,items,config))
            self.assertFalse(first['train']['integrity'].get('passed',False))

    def test_resume_flushed_sequences(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(v2,'publish'), patch.object(v2,'disk_gate'):
            root=Path(folder);items,config=self.fixture(root)
            with patch.object(v2,'pause_gate',side_effect=[None,None,v2.PauseRequested('test')]):
                with self.assertRaises(v2.PauseRequested): v2.materialize(root,items,config)
            self.assertEqual(v2.read(root/'materialized/train/progress.json')['next_sequence'],2)
            v2.materialize(root,items,config)
            self.assertTrue(v2.audit_materialized(root,items,config)['research_usable'])

    def test_corruption_rejected_even_with_rewritten_writer_hash(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(v2,'publish'), patch.object(v2,'disk_gate'):
            root=Path(folder);items,config=self.fixture(root)
            v2.materialize(root,items,config)
            dest=root/'materialized/train';path=dest/'center_coarse_labels.npy'
            value=np.load(path,mmap_mode='r+');value[0]=15;value.flush()
            manifest=v2.read(dest/'split_manifest.json')
            manifest['payload'][path.name]=v2.file_array_hash(path,value)
            v2.save_json(dest/'split_manifest.json',manifest)
            with self.assertRaises(AssertionError): v2.audit_materialized(root,items,config)
            self.assertFalse((root/'materialized/materialization_manifest.json').exists())

    def test_payload_hash_excludes_npy_header(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'x.npy'
            value=np.lib.format.open_memmap(path,mode='w+',dtype=np.int64,shape=(3,7));value[:]=2;value.flush()
            hashes=v2.file_array_hash(path,value)
            self.assertEqual(hashes['array_payload_sha256'],v2.array_digest(value))
            self.assertEqual(hashes['file_sha256'],v2.sha256_file(path))

    def test_pause_and_forbidden_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);v2.pause_gate(root)
            (root/'PAUSE_REQUESTED').touch()
            with self.assertRaises(v2.PauseRequested): v2.pause_gate(root)
        with self.assertRaisesRegex(RuntimeError,'CAUCA'): v2.guard_path('/tmp/CaUcA/input')

    def test_dummy_lifting_keeps_all_frames_and_model_frozen(self):
        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.sizes=[]
            def forward(self,x):
                self.sizes.append(len(x));return x.clone()
        model=Dummy()
        for frames in (17,1200):
            rng=np.random.default_rng(0)
            xy=rng.uniform(0,100,size=(1,frames,17,2)).astype('f')
            a={'keypoint':xy,'keypoint_score':np.ones((1,frames,17),np.float32),
               'width':100,'height':100,'labels':np.zeros(frames,np.int64)}
            before=xy.copy()
            values,meta=v2.infer_sequence(a,model,torch.device('cpu'),{'ntu_proxy':{'reference_torso':.5}})
            self.assertEqual(values['h36m'].shape,(frames,17,3))
            self.assertTrue((values['h36m'][:,0]==0).all())
            self.assertTrue(np.isfinite(values['ntu25']).all())
            np.testing.assert_array_equal(a['keypoint'],before)
            self.assertEqual(meta['repairs'],{'frames':0,'values':0})
        self.assertEqual(model.sizes,[1,1,4,4,1,1])


if __name__=='__main__': unittest.main()
