import unittest
import numpy as np
from data_gen.safer_v2_geometry import (starts_for, coco_xy_to_h36m, model_input, windows_at, flip_numpy,
                                      fuse_windows, normalize_ntu, lying_origin, H36M_TO_NTU)


class V2GeometryTests(unittest.TestCase):
    def test_regular_tail_and_short(self):
        np.testing.assert_array_equal(starts_for(486), [0,243])
        np.testing.assert_array_equal(starts_for(500), [0,243,257])
        np.testing.assert_array_equal(starts_for(100), [0])
        np.testing.assert_array_equal(starts_for(12638), list(range(0,12638-243+1,243)) + [12638-243])
        self.assertEqual(len(starts_for(12638)),53)

    def test_invalid_timeline(self):
        with self.assertRaises(ValueError): starts_for(0)
        with self.assertRaises(ValueError): starts_for(100, stride=244)

    def test_padding_is_last_not_resample(self):
        x=np.arange(100)[:,None]
        p=windows_at(x, np.array([0]))
        np.testing.assert_array_equal(p[0,:100,0],np.arange(100))
        self.assertTrue((p[0,100:] == 99).all())

    def test_mapping_anatomy(self):
        x=np.repeat(np.arange(17)[None,:,None],2,2).astype('f')
        y=coco_xy_to_h36m(x)
        expected=[11.5,12,14,16,11,13,15,8.5,5.5,0,1.5,5,7,9,6,8,10]
        np.testing.assert_array_equal(y[0,:,0],expected)

    def test_confidence_repair_and_official_order(self):
        p=np.zeros((2,17,2),np.float32);p[...,0]=500;p[...,1]=250
        s=np.tile(np.arange(17,dtype=float),(2,1));s[0,2]=np.nan;s[1,4]=np.inf
        out,repair=model_input(p,s,1000,500)
        self.assertEqual(repair,{'values':2,'frames':2})
        np.testing.assert_array_equal(out[...,:2],0)
        self.assertEqual(out[0,1,2],1)  # unchanged COCO score, not right-hip score12
        self.assertEqual(out[0,2,2],0)
        self.assertTrue(np.isnan(s[0,2]))

    def test_flip_involution(self):
        x=np.random.default_rng(0).normal(size=(4,243,17,3)).astype('f')
        np.testing.assert_array_equal(flip_numpy(flip_numpy(x)),x)

    def test_overlap_independent_and_real_timeline(self):
        for frames in [1,100,243,244,485,486,500,12638]:
            x=np.broadcast_to(np.arange(frames)[:,None,None],(frames,17,3)).astype('f').copy()
            starts=starts_for(frames);windows=windows_at(x,starts)
            output,count=fuse_windows(windows,starts,frames)
            other,other_count=fuse_windows(windows,starts,frames,True)
            np.testing.assert_array_equal(output,x)
            np.testing.assert_array_equal(other,output)
            np.testing.assert_array_equal(count,other_count)
            self.assertTrue(count.min() >= 1 and count.max() <= 2)

    def test_yaw_sign_center_and_scale(self):
        pose=np.random.default_rng(2).normal(size=(10,17,3)).astype('f')
        pose[:,0]=0
        ntu,root,meta=normalize_ntu(pose)
        self.assertTrue((ntu[:,1] == 0).all())
        shoulder=ntu[0,8]-ntu[0,4]
        self.assertGreater(shoulder[0],0)
        self.assertLess(abs(shoulder[2]),1e-6)
        self.assertAlmostEqual(float(np.median(np.linalg.norm(ntu[:,20]-ntu[:,0],axis=1))),.5,places=6)
        scaled=pose[:,H36M_TO_NTU]*meta['scale'];theta=meta['yaw']
        # Independent complex-plane yaw, z-axis imaginary part.
        xyz=scaled.copy();xz=(scaled[...,0]+1j*scaled[...,2])*np.exp(-1j*theta)
        xyz[...,0],xyz[...,2]=xz.real,xz.imag
        np.testing.assert_allclose(root,xyz[:,1],atol=1e-6)
        np.testing.assert_allclose(ntu,xyz-xyz[:,1:2],atol=1e-6)

    def test_zero_degenerate_preserved(self):
        ntu,root,meta=normalize_ntu(np.zeros((64,17,3),np.float32))
        self.assertTrue((ntu == 0).all() and (root == 0).all())
        self.assertEqual(meta['degenerate_torso_frames'],64)
        self.assertIsNone(meta['reference_frame'])

    def test_origin_no_gap_filling(self):
        y=np.array([12,10,12,12,0,12,11,12,12,3])
        np.testing.assert_array_equal(lying_origin(y),[-1,-1,0,0,-1,-1,-1,1,1,-1])
