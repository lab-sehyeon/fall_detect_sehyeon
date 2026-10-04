import copy
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import test_safer_v3_reconstruction as prior_tests
import test_safer_v3_controls as control_tests
import test_joint_v3_reconstruction as joint_tests
from data_gen import safer_v3_geometry as old
from data_gen import safer_v3_geometry_r3 as g
from data_gen import safer_v3_reconstruction_r3 as run
from fall_pipeline.safer import v3_controls_reconstruction_r3 as controls
from fall_pipeline.joint import run_v3_reconstruction_r3 as joint
from scripts import run_recovery_v3_queue_r3 as queue


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.c=run.io.read(run.CONFIG)
        self.xy=np.random.default_rng(27).normal(size=(729,17,2)).astype('f')
        self.h=np.concatenate((old.coco_xy_to_h36m(self.xy),np.ones((729,17,1))),axis=-1)
        self.h-=self.h[:,:1];self.starts=np.array([0,243,486])

    def test_all_valid_matches_original_metrics(self):
        before=old.measure(self.h,self.xy,self.starts,self.c);after=g.measure(self.h,self.xy,self.starts,self.c)
        for key,value in before.items(): np.testing.assert_array_equal(value,after[key])

    def test_zero_source_excludes_only_2d_evidence(self):
        original_h=self.h.copy();self.xy[243]=0
        before=old.measure(self.h,self.xy,self.starts,self.c);after=g.measure(self.h,self.xy,self.starts,self.c)
        self.assertEqual(after['source_invalid_frames'],1);self.assertEqual(len(after['nme']),728)
        self.assertEqual(after['degenerate_projection_frames'],0);self.assertEqual(after['unsupported_seam_count'],1)
        np.testing.assert_array_equal(after['seam_points'],[486])
        for key in ('velocity','cv'): np.testing.assert_array_equal(after[key],before[key])
        np.testing.assert_array_equal(self.h,original_h)

    def test_support_stencil_includes_both_endpoints(self):
        valid=np.ones(30,bool)
        for missing in (4,15):
            mask=valid.copy();mask[missing]=False
            self.assertFalse(g.seam_support(mask,[10],5)[0])
        for missing in (3,16):
            mask=valid.copy();mask[missing]=False
            self.assertTrue(g.seam_support(mask,[10],5)[0])
        valid[0]=False;self.assertFalse(g.seam_support(valid,[1],5)[0])

    def test_candidate_degeneracy_not_hidden_in_invalid_source(self):
        self.xy[200]=0;self.h[200]=0
        m=g.aggregate([g.measure(self.h,self.xy,self.starts,self.c)])
        self.assertEqual(m['degenerate_projection_frames'],1)
        self.assertFalse(g.gates(m,m,self.c)['checks']['nondegenerate'])

    def test_empty_support_fails_not_zero_metric(self):
        self.xy[:]=0;m=g.measure(self.h,self.xy,self.starts,self.c)
        with self.assertRaisesRegex(ValueError,'no eligible'): g.aggregate([m])

    def test_nonfinite_source_is_error(self):
        self.xy[12,2,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'): g.measure(self.h,self.xy,self.starts,self.c)

    def test_gate_detects_different_source_support(self):
        base=g.aggregate([g.measure(self.h,self.xy,self.starts,self.c)]);base.update(seam_rate=.8,nme_p95=1.,bone_cv_p95=1.,speed_p99=1.)
        candidate=copy.deepcopy(base);candidate['seam_rate']=.1
        self.assertTrue(g.gates(candidate,base,self.c)['passed'])
        candidate['source_valid_frames']-=1
        self.assertFalse(g.gates(candidate,base,self.c)['passed'])

    def test_mask_independent_of_model_and_confidence(self):
        self.xy[400]=0
        a=g.measure(self.h,self.xy,self.starts,self.c);b=g.measure(self.h*2,self.xy,self.starts,self.c)
        self.assertEqual(a['source_valid_frames'],b['source_valid_frames'])
        np.testing.assert_array_equal(a['seam_points'],b['seam_points'])

    def test_config_and_queue_isolation(self):
        prior=run.io.read(run.ROOT/'configs/safer_v3_document_reconstruction_v2.json')
        for key in ('pilot','full','execution','source_config_sha256','v2_final_sha256'): self.assertEqual(self.c[key],prior[key])
        for name,module,args,root,report in queue.STAGES:
            self.assertTrue(module.endswith('_r3'));self.assertTrue('_R3' in str(root) or '_r3' in str(root))
        self.assertEqual(joint.io.read(joint.CONFIG)['controls_config_sha256'],run.io.sha256_file(controls.CONFIG))
        self.assertIn('data_gen/safer_v3_geometry_r3.py',queue.fingerprint())

    def test_geometry_import_isolated_from_dste(self):
        p=subprocess.run([sys.executable,'-c','import sys; from data_gen import safer_v3_reconstruction_r3; assert "model" not in sys.modules'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_pilot_resume_and_full_cache(self):
        with patch.object(prior_tests,'run',run),patch.object(prior_tests.runtime.io,'guard',side_effect=lambda p:Path(p).resolve()):
            prior_tests.V3Tests().test_pilot_reuses_raw_and_resumes_without_policy_pass()
            with patch.object(g,'gates',return_value={'passed':True,'test_only':True}):
                prior_tests.V3Tests().test_full_compact_cache_and_runtime_slice_audit()


class ControlsR3Tests(control_tests.V3ControlsTests):
    def setUp(self): self.change=patch.object(control_tests,'run',controls);self.change.start()
    def tearDown(self): self.change.stop()


class JointR3Tests(joint_tests.JointV3Tests):
    def setUp(self): self.change=patch.object(joint_tests,'run',joint);self.change.start()
    def tearDown(self): self.change.stop()


if __name__=='__main__': unittest.main()
