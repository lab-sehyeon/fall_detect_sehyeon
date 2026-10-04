import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from fall_pipeline.safer import v2_controls_reconstruction as v2


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.fc=torch.nn.Linear(2048,60)
        self.requires_grad_(False)
    def backbone(self,jt,js):
        return jt.mean(-1,keepdim=True).expand(-1,-1,1024),js.mean(-1,keepdim=True).expand(-1,-1,1024)


class V2ControlsTests(unittest.TestCase):
    def source(self,count=27):
        coarse=np.resize(np.array([0,10,11,12],np.int64),count)
        lookup=np.array([0]*10+[1,2,3]+[0]*3,np.int64)
        return {'count':count,'data':np.random.default_rng(3).normal(size=(count,3,64,25,2)).astype('f'),
                'arrays':{'center_coarse_labels.npy':coarse,'center_derived_labels.npy':lookup[coarse],
                          'sequence_index.npy':np.zeros(count,np.int64),'window_start.npy':np.arange(count,dtype=np.int64)*8}}

    def config(self):
        cfg=v2.read(v2.CONFIG);cfg['execution']['extract_batch_size']=8;cfg['execution']['flush_batches']=2
        return cfg

    def test_holdout_requires_selection_before_creating_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaisesRegex(RuntimeError,'holdout before'):
                v2.extract(root,'test',self.source(),Dummy(),torch.device('cpu'),self.config(),'model')
            self.assertFalse((root/'cache').exists())

    def test_extract_exact_roundtrip_and_prefix_resume(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(v2,'publish'),patch.object(v2,'disk_gate'):
            root=Path(folder);source=self.source();model=Dummy();cfg=self.config()
            with patch.object(v2,'pause',side_effect=[None,v2.PauseRequested('test')]):
                with self.assertRaises(v2.PauseRequested): v2.extract(root,'train',source,model,torch.device('cpu'),cfg,'model')
            self.assertEqual(v2.read(root/'cache/train/progress.json')['next'],16)
            cached=v2.extract(root,'train',source,model,torch.device('cpu'),cfg,'model')
            self.assertEqual(cached['count'],27)
            np.testing.assert_array_equal(cached['arrays']['center_coarse_labels.npy'],source['arrays']['center_coarse_labels.npy'])
            v2.verify_cache(root/'cache/train',source,'model',None)

    def test_corrupt_cache_prefix_rejected(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(v2,'publish'),patch.object(v2,'disk_gate'):
            root=Path(folder);source=self.source();model=Dummy();cfg=self.config()
            with patch.object(v2,'pause',side_effect=[None,v2.PauseRequested('test')]):
                with self.assertRaises(v2.PauseRequested): v2.extract(root,'train',source,model,torch.device('cpu'),cfg,'model')
            a=np.load(root/'cache/train/temporal_features.npy',mmap_mode='r+');a[0,0]+=1;a.flush()
            with self.assertRaisesRegex(RuntimeError,'resume prefix'):
                v2.extract(root,'train',source,model,torch.device('cpu'),cfg,'model')

    def test_tiny_full_pipeline_and_safe_torch_resume(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(v2,'publish'),patch.object(v2,'disk_gate'):
            root=Path(folder);cfg=self.config();model=Dummy();cache={};device=torch.device('cpu')
            for split in ('train','val'): cache[split]=v2.extract(root,split,self.source(),model,device,cfg,'model')
            training=v2.read(v2.ROOT/cfg['matched_training_config'])
            contract={'torch':str(torch.__version__),'test_only':True}
            selection=v2.fit_heads(root,cache,training,device,True,contract)
            again=v2.fit_heads(root,cache,training,device,True,contract)
            self.assertEqual(selection,again)
            for split in ('test','ood'):
                cache[split]=v2.extract(root,split,self.source(),model,device,cfg,'model',v2.sha256_file(root/'selection.json'))
            v2.evaluate(root,cache,selection,device)
            report=v2.audit_outputs(root,cache,selection,cfg)
            self.assertTrue(report['passed'])
            self.assertFalse(selection['research_usable'])

    def test_independent_metric_corruption_and_missing_class(self):
        labels=np.array([0,0,1,1]);logits=np.zeros((4,4),np.float32)
        metrics=v2.head_ops.classification_metrics(labels,logits)
        v2.independent_metrics(labels,logits,metrics)
        wrong=copy.deepcopy(metrics);wrong['macro_f1']+=.1
        with self.assertRaises(RuntimeError): v2.independent_metrics(labels,logits,wrong)

    def test_config_matched_training_and_holdout_contract(self):
        cfg=v2.read(v2.CONFIG)
        self.assertEqual(v2.sha256_file(v2.ROOT/cfg['matched_training_config']),cfg['matched_training_config_sha256'])
        training=v2.read(v2.ROOT/cfg['matched_training_config'])
        self.assertEqual(training['optimization']['epochs'],cfg['f0b']['epochs'])
        self.assertEqual(training['optimization']['seed'],0)
        self.assertTrue(cfg['f0b']['selected_test_ood_only'])
        self.assertFalse(cfg['historical_exact_reproduction'])


if __name__=='__main__': unittest.main()
