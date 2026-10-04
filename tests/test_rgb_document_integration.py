import unittest
import numpy as np
from fall_pipeline.external import rgb_document_io as io
from fall_pipeline.common.global_reconstruction import raw_channels,causal_arrays,features_from_causal,reference_feature

class RGBIntegrationTests(unittest.TestCase):
    def test_cascade_preserves_all_base_and_never_calls_fallback(self):
        base=np.array([[0,0,2,2,.5],[3,3,5,5,.7]],np.float32)
        def fail():raise AssertionError('fallback called with B0 candidates')
        got,rescue=io.cascade(base,fail);np.testing.assert_array_equal(got,base);self.assertFalse(rescue)

    def test_cascade_empty_and_top1(self):
        x,r=io.cascade([],lambda:[]);self.assertEqual(x.shape,(0,5));self.assertFalse(r)
        x,r=io.cascade([],lambda:[[0,0,3,3,.1],[1,1,3,3,.2]])
        self.assertEqual(len(x),1);self.assertTrue(r);self.assertAlmostEqual(float(x[0,4]),.2)

    def test_track_missing_tie_and_geometry(self):
        self.assertIsNone(io.select_track([],None))
        c=np.array([[0,0,4,4,.3],[5,5,10,10,.9]],np.float32)
        np.testing.assert_array_equal(io.select_track(c,None),c[1])
        np.testing.assert_array_equal(io.select_track(c,c[0]),c[0])
        np.testing.assert_array_equal(io.select_track(np.stack((c[0],c[0])),None),c[0])
        with self.assertRaisesRegex(RuntimeError,'candidates'):io.select_track([[0,0,-1,1,.5]],None)

    def test_quality_no_confidence_rescue(self):
        b=np.tile([0,0,10,10,.9],(100,1));x=np.ones((100,17,2));s=np.ones((100,17))*.4
        spec=dict(bbox_coverage_min=.8,pose_coverage_min=.8,pelvis_median_min=.3)
        self.assertTrue(io.quality(b,x,s,spec)['passed'])
        b[:21]=0;self.assertFalse(io.quality(b,x,s,spec)['passed'])
        b[:21]=[0,0,10,10,.9];s[:,:]=.2;self.assertFalse(io.quality(b,x,s,spec)['passed'])

    def test_global_xywh_and_coordinate_scale_invariance(self):
        n=80;xy=np.ones((n,17,2),np.float32)*20;xy[:,5:7,1]=10
        saved=dict(xy=xy,scores=np.ones((n,17),np.float32)*.6,boxes=np.tile([0,0,60,50,.9],(n,1)),width=100,height=60)
        a=io.annotation(saved);x,m=raw_channels(a);np.testing.assert_allclose(a['bboxes'][0],[0,0,60,50])
        b={**a,'keypoint':a['keypoint']*2,'bboxes':a['bboxes']*2,'width':200,'height':120}
        y,v=raw_channels(b);np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(m,v)
        causal=causal_arrays(x,m);starts=np.array([0,8,16]);features=features_from_causal(causal,m,starts)
        for i,s in enumerate(starts):np.testing.assert_array_equal(features[i],reference_feature(causal,m,int(s)))

    def test_dste_windows_layout_and_zero_second_person(self):
        ntu=np.arange(80*25*3,dtype=np.float32).reshape(80,25,3);starts=np.array([0,8,16])
        x=io.windows(ntu,starts);self.assertEqual(x.shape,(3,3,64,25,2));self.assertTrue((x[...,1]==0).all())
        for i,s in enumerate(starts):np.testing.assert_array_equal(x[i,...,0].transpose(1,2,0),ntu[s:s+64])
        with self.assertRaisesRegex(RuntimeError,'bounds'):io.windows(ntu,np.array([17]))

if __name__=='__main__':unittest.main()
