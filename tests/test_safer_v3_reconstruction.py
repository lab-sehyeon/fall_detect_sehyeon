import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from data_gen import safer_v3_geometry as g
from data_gen import safer_v3_reconstruction as run
from data_gen import safer_v2_geometry as v2
from data_gen import safer_v3_runtime as runtime


class V3Tests(unittest.TestCase):
    def config(self): return run.io.read(run.CONFIG)

    def test_weights_positive_symmetric(self):
        for kind in ('uniform','triangular','hann'):
            w=g.weights(kind);self.assertTrue((w>=.05).all());self.assertEqual(w[121],1)
            np.testing.assert_allclose(w,w[::-1],atol=1e-14)

    def test_fusion_independent_and_identity(self):
        rng=np.random.default_rng(4)
        for n in (1,242,243,244,729,1001):
            x=rng.normal(size=(n,17,3)).astype('f');x[:,0]=0
            for stride in (243,121,81):
                starts=v2.starts_for(n,stride=stride);raw=v2.windows_at(x,starts)
                for kind in ('uniform','triangular','hann'):
                    actual=g.fuse(raw,starts,n,kind);expected=g.fuse(raw,starts,n,kind,independent=True)
                    for a,b in zip(actual,expected): np.testing.assert_array_equal(a,b)
                    np.testing.assert_array_equal(actual[0],x)
                    self.assertGreaterEqual(actual[1].min(),1)

    def test_uncovered_rejected(self):
        with self.assertRaisesRegex(ValueError,'uncovered'): g.fuse(np.zeros((1,243,17,3),'f'),[0],300,'uniform')

    def test_label_blind_train_only_hash_order(self):
        c=self.config();records=[{'frame_dir':str(i),'split':'train' if i<20 else 'test','total_frames':1000,'labels':object()} for i in range(25)]
        a=g.pilot_records(records,c);b=g.pilot_records(list(reversed(records)),c)
        self.assertEqual([x['frame_dir'] for x in a],[x['frame_dir'] for x in b]);self.assertEqual(len(a),8)
        self.assertTrue(all(x['split']=='train' for x in a))
        class NoLabels(dict):
            def __getitem__(self,key):
                if key in ('labels','keypoint_3d'): raise AssertionError('forbidden field')
                return super().__getitem__(key)
        protected=[NoLabels(x) for x in records];self.assertEqual(len(g.pilot_records(protected,c)),8)

    def test_local_jump_detects_injected_seam(self):
        n=729;t=np.arange(n)[:,None,None];x=np.zeros((n,17,3));x[:,1:,0]=np.arange(n)[:,None]*.01
        stable=g.jump_ratios(x,[243]);x[243:,1:,0]+=5
        jump=g.jump_ratios(x,[243]);self.assertLess(stable[0],2);self.assertGreater(jump[0],3)
        np.testing.assert_array_equal(g.boundaries(np.array([0,243,486]),n),[243,486])

    def test_metrics_finite_root_and_degenerate_fail_closed(self):
        c=self.config();rng=np.random.default_rng(0);xy=rng.normal(size=(729,17,2)).astype('f')
        h=np.concatenate((v2.coco_xy_to_h36m(xy),np.ones((729,17,1))),axis=-1);h-=h[:,:1]
        m=g.measure(h,xy,np.array([0,243,486]),c);a=g.aggregate([m])
        self.assertTrue(a['root_exact_zero']);self.assertLess(a['nme_p95'],1e-7)
        zero=g.aggregate([g.measure(np.zeros_like(h),np.zeros_like(xy),np.array([0,243,486]),c)])
        self.assertFalse(g.gates(zero,zero,c)['passed'])

    def test_geometry_gates_and_ranking_no_posthoc(self):
        c=self.config();base={'seam_rate':.8,'nme_p95':1.,'bone_cv_p95':1.,'speed_p99':1.,'root_exact_zero':True,
                            'degenerate_projection_frames':0,'degenerate_bones':0}
        good={**base,'seam_rate':.1,'nme_p95':1.04,'bone_cv_p95':1.04,'speed_p99':.99}
        self.assertTrue(g.gates(good,base,c)['passed'])
        self.assertFalse(g.gates({**good,'speed_p99':1.001},base,c)['passed'])
        self.assertFalse(g.gates({**good,'root_exact_zero':False},base,c)['passed'])

    def test_overlap_diagnostics_exact_copies(self):
        x=np.random.default_rng(8).normal(size=(400,17,3)).astype('f');starts=np.array([0,121,157]);raw=v2.windows_at(x,starts)
        result=g.overlap_diagnostics(raw,starts)
        self.assertEqual(result['shape_rms']['p95'],0);self.assertAlmostEqual(result['bone_scale_ratio']['p95'],1)

    def test_pilot_reuses_raw_and_resumes_without_policy_pass(self):
        c=self.config();c['pilot']['count']=2;rng=np.random.default_rng(4);records=[]
        for i in range(2):
            xy=rng.normal(size=(729,17,2)).astype('f');h=np.concatenate((v2.coco_xy_to_h36m(xy),np.zeros((729,17,1),'f')),axis=-1);h-=h[:,:1]
            records.append({'uid':str(i),'frame_dir':str(i),'split':'train','total_frames':729,'keypoint':xy[None],'pose':h})
        def raw(a,stride,*args):
            starts=v2.starts_for(729,stride=stride);return v2.windows_at(a['pose'],starts),starts,{}
        def baseline(a,*args):
            values,starts,_=raw(a,243);return {'h36m':a['pose'],'window_predictions':values,'starts':starts}
        with tempfile.TemporaryDirectory() as folder,patch.object(run,'publish'),patch.object(run.v2,'load_model',return_value=torch.nn.Linear(1,1)),patch.object(run,'baseline',side_effect=baseline),patch.object(run,'raw_windows',side_effect=raw) as infer,patch.object(run.io,'disk_gate'):
            root=Path(folder);first=run.pilot(root,records,c,{}, {'test':True},torch.device('cpu'))
            self.assertEqual(infer.call_count,6);self.assertIsNone(first['selected']);self.assertTrue(first['baseline_equality_passed'])
            second=run.pilot(root,records,c,{}, {'test':True},torch.device('cpu'))
            self.assertEqual(first,second);self.assertEqual(infer.call_count,6)

    def test_full_compact_cache_and_runtime_slice_audit(self):
        c=self.config();c['v2_root']='v2';contract={'test_only':True};rng=np.random.default_rng(13)
        with tempfile.TemporaryDirectory() as folder,patch.object(run,'publish'),patch.object(run.io,'disk_gate'),patch.object(run,'ROOT',Path(folder)),patch.object(run.io,'guard',side_effect=lambda p:Path(p).resolve()):
            root=Path(folder)/'v3';(root/'sequences').mkdir(parents=True);records=[];splits={};n=300
            run.io.save_json(root/'pilot/report.json',{'test_only':True})
            policy={'name':'s121_triangular','stride':121,'kind':'triangular','pilot_report_sha256':run.io.sha256_file(root/'pilot/report.json')}
            run.io.save_json(root/'policy_lock.json',policy);run.io.save_json(root/'invariance.json',{'passed':True,'before':'test','after':'test'})
            for i,split in enumerate(('train','val','test','ood')):
                xy=rng.normal(size=(n,17,2)).astype('f');score=np.ones((n,17),'f')
                a={'uid':str(i),'frame_dir':f'x_p{i+1}_test','split':split,'total_frames':n,'keypoint':xy[None],
                   'keypoint_score':score[None],'width':1920,'height':1080,'labels':np.arange(n,dtype=np.int64)%16}
                records.append(a);splits[split]=[a];h=np.concatenate((v2.coco_xy_to_h36m(xy),rng.normal(size=(n,17,1)).astype('f')),axis=-1);h-=h[:,:1]
                starts=v2.starts_for(n,stride=121);raw=v2.windows_at(h,starts);fused,cover,div=g.fuse(raw,starts,n,'triangular')
                path=root/'sequences'/f'{i}.npz';run.save_npz(path,raw=raw,starts=starts,h36m=fused,coverage=cover,divisor=div)
                inp,repairs=v2.model_input(xy,score,1920,1080)
                run.io.save_json(path.with_suffix('.json'),{'model_hash':'test','policy':policy,'sha256':run.io.sha256_file(path),
                    'model_input_sha256':run.v2.array_digest(inp),'repairs':repairs})
                ws=run.v2.window_starts(n);coarse=a['labels'][ws+32];c['full']['expected_windows'][split]=len(ws)
                old=Path(folder)/'v2/materialized'/split
                for name,value in (('sequence_index.npy',np.zeros(len(ws),np.int64)),('window_start.npy',ws),('center_coarse_labels.npy',coarse),('center_derived_labels.npy',run.v2.derive_four_class(coarse))): run.io.save_array(old/name,value)
            def old_pose(a,config):
                with np.load(root/'sequences'/f'{a["uid"]}.npz') as saved: return {'h36m':saved['h36m'],'starts':saved['starts']}
            with patch.object(run,'policy_lock',return_value=policy),patch.object(run.v2,'sources',return_value=(records,splits,{})),patch.object(run,'baseline',side_effect=old_pose),patch.object(g,'gates',return_value={'passed':True,'test_only':True}):
                result=run.audit_and_cache(root,records,splits,c,{'ntu_proxy':{'reference_torso':.5}},contract)
            self.assertTrue(result['structural_integrity']);self.assertEqual(result['frames'],1200)
            self.assertTrue(runtime.verify_source(root)['passed'])
            source=runtime.source_split(root,'train',False);value=source['data'][0:3]
            self.assertEqual(value.shape,(3,3,64,25,2));self.assertTrue((value[...,1]==0).all())
            jt=value.transpose(0,2,4,3,1).reshape(3,64,150);js=value.transpose(0,4,3,2,1).reshape(3,50,192)
            self.assertEqual(jt.shape,(3,64,150));self.assertEqual(js.shape,(3,50,192));self.assertTrue((js[:,25:]==0).all())


if __name__=='__main__': unittest.main()
