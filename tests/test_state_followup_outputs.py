import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fall_pipeline.safer import run_s0c_reconstruction as c
from fall_pipeline.safer import state_followup_core as core
from scripts import run_state_recovery_queue as queue


class StateOutputTests(unittest.TestCase):
    def test_candidate_commit_resume_and_tamper_rejection(self):
        ac=c.io.read(c.io.ROOT/'configs/s0a_document_reconstruction_v1.json')
        cfg=copy.deepcopy(c.io.read(c.CONFIG));cfg['execution']['reserve_gib']=0
        rng=np.random.default_rng(1)
        records=[{'uid':'x','truth':rng.integers(0,16,100),'logits':rng.normal(size=(100,16)),
                  'source_sha256':'synthetic','label_sha256':'synthetic'}]
        spec=core.decoder_grid(cfg)[2]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dest=root/'candidate'
            first=c.candidate_eval(root,dest,records,spec,ac,cfg)
            self.assertEqual(first,c.candidate_eval(root,dest,records,spec,ac,cfg))
            changed=copy.deepcopy(spec);changed['duration']=8
            with self.assertRaises(RuntimeError):c.candidate_eval(root,dest,records,changed,ac,cfg)
            np.save(dest/'x.npy',np.zeros(100,np.uint8))
            with self.assertRaises(RuntimeError):c.candidate_eval(root,dest,records,spec,ac,cfg)

    def test_all_candidates_roundtrip_and_latency_outputs(self):
        ac=c.io.read(c.io.ROOT/'configs/s0a_document_reconstruction_v1.json')
        cfg=copy.deepcopy(c.io.read(c.CONFIG));cfg['execution']['reserve_gib']=0
        x=np.full((80,16),-3.);x[:20,10]=3.;x[20:50,12]=3.;x[50:,7]=3.
        rec=[{'uid':'a','truth':x.argmax(1),'logits':x,'source_sha256':'synthetic','label_sha256':'synthetic'}]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for spec in core.decoder_grid(cfg):
                c.candidate_eval(root,root/spec['name'],rec,spec,ac,cfg)
            result=c.latency_report(root/'diagnostic',rec,root/'raw',cfg)
            self.assertEqual(result['10->12']['detected'],1)
            self.assertEqual(result['12->7']['delay_median_frames'],0.)
            self.assertTrue((root/'diagnostic/events.csv').exists())

    def test_current_process_identity_and_missing_pid(self):
        import os
        value=queue.identity(os.getpid())
        self.assertEqual(value['pid'],os.getpid());self.assertTrue(value['start_ticks'].isdigit())
        self.assertIsNone(queue.identity(2147483647))


if __name__=='__main__':unittest.main()
