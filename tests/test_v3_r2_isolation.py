import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import test_safer_v3_reconstruction as geometry_tests
import test_safer_v3_controls as control_tests
import test_joint_v3_reconstruction as joint_tests
from data_gen import safer_v3_reconstruction_r2 as geometry
from fall_pipeline.safer import v3_controls_reconstruction_r2 as controls
from fall_pipeline.joint import run_v3_reconstruction_r2 as joint
from scripts import run_recovery_v3_queue_r2 as queue


class GeometryR2Tests(geometry_tests.V3Tests):
    def setUp(self):
        self.change=patch.object(geometry_tests,'run',geometry);self.change.start()
        self.scope=patch.object(geometry_tests.runtime.io,'guard',side_effect=lambda p:Path(p).resolve());self.scope.start()
    def tearDown(self): self.scope.stop();self.change.stop()


class ControlsR2Tests(control_tests.V3ControlsTests):
    def setUp(self): self.change=patch.object(control_tests,'run',controls);self.change.start()
    def tearDown(self): self.change.stop()


class JointR2Tests(joint_tests.JointV3Tests):
    def setUp(self): self.change=patch.object(joint_tests,'run',joint);self.change.start()
    def tearDown(self): self.change.stop()


class IsolationTests(unittest.TestCase):
    def test_fresh_geometry_process_has_no_dste_model(self):
        p=subprocess.run([sys.executable,'-c','import sys; from data_gen import safer_v3_reconstruction_r2; assert "model" not in sys.modules'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_no_geometry_or_gate_change(self):
        old=geometry.io.read(geometry.ROOT/'configs/safer_v3_document_reconstruction_v1.json');new=geometry.io.read(geometry.CONFIG)
        for key in ('pilot','geometry','full','execution','source_config_sha256','v2_final_sha256'): self.assertEqual(old[key],new[key])

    def test_queue_all_destinations_are_new_revision(self):
        for name,module,args,root,report in queue.STAGES:
            self.assertTrue(module.endswith('_r2'))
            self.assertTrue('_R2' in str(root) or '_r2' in str(root))
        _,cc=controls.effective_config();self.assertEqual(queue.CONTROL,controls.ROOT/cc['output_dir'])
        jc=joint.io.read(joint.CONFIG);self.assertEqual(queue.JOINT,joint.ROOT/jc['output_dir'])


if __name__=='__main__': unittest.main()
