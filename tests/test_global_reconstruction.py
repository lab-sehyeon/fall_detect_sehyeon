import unittest
import numpy as np
from fall_pipeline.common.global_reconstruction import raw_channels,causal_arrays,features_from_causal,reference_feature
from data_gen.rgb_alignment_reconstruction import clip_intervals,past_frame_indices


class GlobalReconstructionTests(unittest.TestCase):
    def test_raw_xywh_normalization_and_mask(self):
        xy=np.ones((1,70,17,2))*[100,80];xy[:,:,5:7,1]=40
        a=dict(keypoint=xy,keypoint_score=np.ones((1,70,17)),bboxes=np.tile([20,10,100,90],(70,1)),
               width=200,height=100,total_frames=70)
        x,m=raw_channels(a)
        np.testing.assert_allclose(x[0,:6],[.5,.8,.35,.55,.5,.9])
        np.testing.assert_allclose(x[0,8:11],[0,-.4,.4]);self.assertTrue(m.all())
        a['bboxes'][0]=0;a['keypoint_score'][0,1,11]=np.nan
        x,m=raw_channels(a);self.assertFalse(m[0].any());self.assertFalse(m[1,0]);self.assertTrue(np.isfinite(x).all())

    def test_global_feature_reference_and_no_future(self):
        rng=np.random.default_rng(19);v=rng.normal(size=(170,12)).astype(np.float32);m=rng.random((170,12))>.15
        a=causal_arrays(v,m);f=features_from_causal(a,m,[0,8,64,96])
        for index,start in enumerate([0,8,64,96]):
            np.testing.assert_allclose(f[index],reference_feature(a,m,start),atol=1e-6,rtol=1e-6)
        v[100:]+=10000;m[100:]=False;b=causal_arrays(v,m)
        for x,y in zip(a,b):np.testing.assert_array_equal(x[:100],y[:100])
        np.testing.assert_array_equal(f[:2],features_from_causal(b,m,[0,8]))

    def test_full_sequence_derivative_not_reset_at_window(self):
        v=np.repeat(np.arange(100,dtype=np.float32)[:,None],12,axis=1);m=np.ones_like(v,bool)
        a=causal_arrays(v,m);self.assertEqual(a[2][8,0],25)
        f=features_from_causal(a,m,[8])[0];self.assertEqual(f[60],25);self.assertEqual(f[62],25)
        self.assertEqual(f[-1],64)

    def test_missing_feature_finite_and_source_not_changed(self):
        v=np.zeros((70,12),np.float32);m=np.zeros_like(v,bool);original=v.copy()
        f=features_from_causal(causal_arrays(v,m),m,[0])[0]
        np.testing.assert_array_equal(f[:-1],0);self.assertEqual(f[-1],64)
        np.testing.assert_array_equal(v,original)
        with self.assertRaises(ValueError):features_from_causal(causal_arrays(v,m),m,[8])

    def test_boundary_clip_is_audited_and_keeps_source(self):
        rows=[dict(start='0',end='2',source_row=2,label='fall'),dict(start='2',end='2.04',source_row=3,label='other')]
        output,audit=clip_intervals(rows,2.01)
        self.assertEqual(output[-1]['end'],2.01);self.assertEqual(len(audit),1)
        self.assertEqual(rows[-1]['end'],'2.04')
        output,audit=clip_intervals(rows,1.9);self.assertEqual(len(output),1)
        self.assertTrue(audit[-1]['entirely_outside'])

    def test_sampling_only_past_and_short_video_kept(self):
        pts=np.arange(20)/30;times,indices=past_frame_indices(pts,20/30)
        self.assertTrue((pts[indices]<=times).all());self.assertTrue((times<20/30).all())
        self.assertEqual(indices[0],0);self.assertLess(len(times),64)
        with self.assertRaises(RuntimeError):past_frame_indices([0,0,.1],.2)


if __name__=='__main__':unittest.main()
