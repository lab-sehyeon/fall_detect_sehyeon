import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from fall_pipeline.joint import run_v3_reconstruction as run


class JointV3Tests(unittest.TestCase):
    def test_pins_reference_matched_method(self):
        spec=run.io.read(run.CONFIG)
        for name,key in (('base_config','base_config_sha256'),('controls_config','controls_config_sha256')):
            self.assertEqual(run.io.sha256_file(run.ROOT/spec[name]),spec[key])

    def test_v3_wrapper_tiny_complete_cpu_smoke(self):
        c=run.io.read(run.base.CONFIG);c['model'].update(feature_dim=16,bottleneck=4)
        c['training'].update(safer_batch_size=8,fu_batch_size=4,eval_batch_size=7)
        rng=np.random.default_rng(9);fu={'x':rng.normal(size=(25,16)).astype('f'),'y':np.arange(25,dtype=np.int64)%2,
          'actions':np.where(np.arange(25)%2,5,4),'subjects':np.arange(25)%5+1,'folds':np.arange(25)%5}
        data={'x':rng.normal(size=(24,16)).astype('f'),'y':np.arange(24,dtype=np.int64)%4}
        with tempfile.TemporaryDirectory() as folder:
            c['output_dir']=folder
            with patch.object(run,'config_and_sources',return_value=({},c,{})),patch.object(run,'contract_for',return_value={'test_only':True}),patch.object(run.io,'guard',side_effect=lambda p:Path(p)),patch.object(run.io,'disk_gate'),patch.object(run.io,'device_for',return_value=torch.device('cpu')),patch.object(run.inputs,'load_fu',return_value=fu),patch.object(run.inputs,'load_safer',return_value=data),patch.object(run,'publish'),patch.object(run.base,'publish'),patch('sys.argv',['run','--stage','smoke']):
                run.main()
            report=run.io.read(Path(folder)/'v3_smoke/final_report.json')
            self.assertTrue(report['passed']);self.assertFalse(report['research_usable'])


if __name__=='__main__': unittest.main()
