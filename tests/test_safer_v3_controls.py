import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from data_gen.safer_v3_runtime import SequenceWindows
from fall_pipeline.safer import v3_controls_reconstruction as run

io=run.io


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.fc=torch.nn.Linear(2048,60);self.requires_grad_(False)
    def backbone(self,jt,js):
        return jt.mean(-1,keepdim=True).expand(-1,-1,1024),js.mean(-1,keepdim=True).expand(-1,-1,1024)


class V3ControlsTests(unittest.TestCase):
    def fixture(self,root):
        rng=np.random.default_rng(4);rows=[{'uid':'a'},{'uid':'b'}]
        for row in rows: io.save_array(root/row['uid']/'ntu25.npy',rng.normal(size=(200,25,3)).astype('f'))
        ids=np.repeat(np.arange(2,dtype=np.int64),12);starts=np.tile(np.arange(12,dtype=np.int64)*8,2)
        data=SequenceWindows(root,rows,ids,starts);coarse=np.resize(np.array([0,10,11,12],np.int64),24)
        return {'count':24,'data':data,'arrays':dict(zip(io.META,(coarse,np.array([0]*10+[1,2,3]+[0]*3)[coarse],ids,starts)))}

    def test_cross_sequence_and_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=self.fixture(root);value=source['data'][10:15]
            expected=[]
            for i in range(10,15):
                row=source['data'].rows[int(source['arrays']['sequence_index.npy'][i])]
                pose=np.load(root/row['uid']/'ntu25.npy');start=source['arrays']['window_start.npy'][i]
                item=np.zeros((3,64,25,2),np.float32);item[...,0]=pose[start:start+64].transpose(2,0,1);expected.append(item)
            np.testing.assert_array_equal(value,np.stack(expected));self.assertLessEqual(len(source['data'].cache),2)

    def test_matched_configuration_is_unchanged_except_source_output(self):
        spec,cfg=run.effective_config();base=io.read(io.ROOT/spec['base_config'])
        for key,value in base.items():
            if key not in ('data_root','output_dir'): self.assertEqual(cfg[key],value)

    def test_complete_tiny_runtime_controls(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(io,'publish'),patch.object(io,'disk_gate'):
            root=Path(folder);source=self.fixture(root/'sequence_cache');_,config=run.effective_config()
            config['execution'].update(extract_batch_size=8,flush_batches=2);model=Dummy();device=torch.device('cpu');cache={}
            for split in ('train','val'): cache[split]=io.extract(root,split,source,model,device,config,'test')
            selected=io.fit_heads(root,cache,io.read(io.ROOT/config['matched_training_config']),device,True,{'test_only':True})
            for split in ('test','ood'): cache[split]=io.extract(root,split,source,model,device,config,'test',io.sha256_file(root/'selection.json'))
            io.evaluate(root,cache,selected,device);self.assertTrue(io.audit_outputs(root,cache,selected,config)['passed'])


if __name__=='__main__': unittest.main()
