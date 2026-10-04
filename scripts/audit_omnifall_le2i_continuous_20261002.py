"""Independent CSV GT, D1 decoder, feature and CPU head audit for38 videos."""
from pathlib import Path
import csv, os, sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fall_pipeline.external import rgb_document_io as io
from fall_pipeline.external import omnifall_le2i_continuous_20261002 as run
from fall_pipeline.external.rgb_recovery_common import decode_recovery_endpoints, D1_DECOUPLED
from fall_pipeline.common.global_reconstruction import raw_channels, causal_arrays, features_from_causal, reference_feature
from fall_pipeline.joint.models import JointFallModel
from data_gen.safer_v2_geometry import normalize_ntu
from data_gen.safer_v3_geometry import fuse
from data_gen.rgb_alignment_reconstruction import past_frame_indices


def independent_counts(times, intervals):
    # This official test set contains at most one fall per video. Do not use scorer matching code.
    io.require(len(intervals)<=1, 'independent audit supports at most one event per video')
    if not intervals: return 0,len(times),0
    start,end=intervals[0]
    tp=int(any(start-.5 <= t <= end+3. for t in times))
    return tp,len(times)-tp,1-tp


def validate_metrics(counts, reported):
    tp,fp,fn=map(int,counts)
    values=dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,
        recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    for k,v in values.items():io.require(abs(reported[k]-v)<1e-12,'independent metric mismatch: '+k)
    return values


def main():
    io.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit only')
    torch.set_num_threads(2)
    cfg=run.setup();output,plan=io.locked(cfg); published=io.read(output/'evaluation.json')
    truth={r['id']:r for r in io.read(output/'ground_truth.json')}
    official=list(csv.DictReader((run.ASSET/'metadata/labels/le2i.csv').read_text().splitlines()))
    paths={r['path'] for r in csv.DictReader((run.ASSET/'metadata/splits/cs/le2i/test.csv').read_text().splitlines())}
    io.require(len(plan)==38 and len(published['rows'])==38 and {r['path'] for r in plan}==paths,'scope')
    csv_gt={p:sorted((float(r['start']),float(r['end'])) for r in official if r['path']==p and int(r['label'])==1) for p in paths}
    io.require(sum(map(len,csv_gt.values()))==22,'GT22')
    prep={r['id']:r for r in io.read(run.PREP/'manifest.json')['records']}
    originals={r['path']:r for r in io.read(run.ASSET/'original_manifest.json')['files']}
    cached=io.read(output/'cache_provenance.json')
    for r in cached['records']:
        for name,digest in r['hashes'].items():
            io.require(io.sha(ROOT/r['source']/name)==digest==io.sha(output/r['id']/name),'cached artifacts changed')
    glob=io.read(ROOT/cfg['inference']['global_config']); selected=io.read(ROOT/glob['output_dir']/'selection.json')['best']
    joint=JointFallModel().eval().requires_grad_(False)
    joint.load_state_dict(torch.load(ROOT/glob['joint_root']/'final/final_j1.pt',map_location='cpu',weights_only=True),strict=True)
    heads={}
    for name,dim in [('G0',2048),('G1',131),('G2',2179)]:
        head=torch.nn.Linear(dim,4).eval().requires_grad_(False)
        head.load_state_dict(torch.load(ROOT/glob['output_dir']/name/f'epoch_{selected[name]["epoch"]:03d}.pt',map_location='cpu',weights_only=True),strict=True)
        heads[name]=head
    with np.load(ROOT/glob['output_dir']/'train_scaler.npz') as z:mean,scale=z['mean'],z['scale']
    rows=[];max_error=0.;windows=0
    for n,(item,reported) in enumerate(zip(plan,published['rows']),1):
        io.safety(cfg);sid=item['id'];dest=output/sid;p=prep[sid]
        io.require(sid==reported['id'],'order')
        for key in ['source','video','mapping']:
            io.require(io.sha(ROOT/p[key])==p[key+'_sha256'],'input changed: '+key)
        io.require(p['pixel_identical'] and p['source_sha256']==originals[item['path']]['sha256'],'official source identity')
        intervals=csv_gt[item['path']]
        stored=[(e['fall_start'],e['fall_end']) for e in truth[sid]['episodes']]
        io.require(len(intervals)==len(stored),'GT count')
        if intervals:np.testing.assert_allclose(intervals,stored,atol=5e-6,rtol=0)
        with np.load(ROOT/item['mapping']) as mapping,np.load(dest/'frontend.npz') as front:
            times,indices=past_frame_indices(mapping['source_timestamps'],item['source_duration_seconds'],25)
            np.testing.assert_array_equal(times,mapping['canonical_timestamps'])
            np.testing.assert_array_equal(indices,mapping['source_indices'])
            np.testing.assert_array_equal(front['source_indices'],indices)
            np.testing.assert_array_equal(front['timestamps'],times)
            io.require(np.all(mapping['source_timestamps'][indices]<=times),'future input')
            boxes,xy,scores=front['boxes'],front['xy'],front['scores']
            bv=np.isfinite(boxes).all(1)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])&(boxes[:,4]>0)
            pv=bv&np.isfinite(xy).all((1,2))&np.isfinite(scores).all(1)&(scores>0).any(1)
            pelvis=np.where(pv,np.minimum(scores[:,11],scores[:,12]),0)
            q=bool(bv.mean()>=.8 and pv.mean()>=.8 and np.median(pelvis)>=.3)
            annotation=io.annotation(front);previous=None
            for i in range(len(boxes)):
                candidates=front['candidates'][front['offsets'][i]:front['offsets'][i+1]]
                chosen=io.select_track(candidates,previous)
                np.testing.assert_array_equal(boxes[i],np.zeros(5,np.float32) if chosen is None else chosen)
                if chosen is not None:previous=chosen
                io.require(not front['rescue'][i] or len(candidates)<=1,'fallback top1')
        fm=io.stage_done(dest,'frontend')
        io.require(q==fm['quality']['passed']==reported['quality_passed'],'quality mismatch')
        io.require(fm['models_before']==fm['models_after'] and not fm['labels_used'] and not fm['training'],'frontend frozen')
        event_times={h:[] for h in ['G0','G2']}
        if q:
            lm=io.stage_done(dest,'lift');im=io.stage_done(dest,'inference')
            io.require(lm['model_before']==lm['model_after'] and im['models_before']==im['models_after'],'model mutation')
            with np.load(dest/'lift.npz') as lift,np.load(dest/'inference.npz') as prediction:
                starts=np.arange(0,item['frames']-63,8);windows+=len(starts)
                np.testing.assert_array_equal(starts,lift['window_starts'])
                np.testing.assert_array_equal(starts,prediction['window_starts'])
                np.testing.assert_array_equal(starts+63,prediction['window_endpoints'])
                h,coverage,_=fuse(lift['lifting_raw'],lift['lifting_starts'],item['frames'],'triangular',.05,True)
                np.testing.assert_array_equal(h,lift['h36m']);np.testing.assert_array_equal(coverage,lift['coverage'])
                ntu,_,_=normalize_ntu(h,.5);np.testing.assert_array_equal(ntu,lift['ntu25'])
                channels,mask=raw_channels(annotation);causal=causal_arrays(channels,mask)
                features=features_from_causal(causal,mask,starts);np.testing.assert_array_equal(features,lift['global131'])
                for i,s in enumerate(starts):np.testing.assert_array_equal(features[i],reference_feature(causal,mask,int(s)))
                standardized=((features.astype(float)-mean)/scale).astype(np.float32)
                np.testing.assert_array_equal(standardized,prediction['global_standardized'])
                with torch.inference_mode():
                    adapted=joint.adapter(torch.from_numpy(prediction['pooled'])).numpy()
                    np.testing.assert_allclose(adapted,prediction['adapted'],atol=1e-4,rtol=1e-5)
                    for name,x in [('G0',prediction['adapted']),('G1',standardized),('G2',np.column_stack((prediction['adapted'],standardized)))]:
                        logits=heads[name](torch.from_numpy(x)).numpy()
                        np.testing.assert_allclose(logits,prediction[name],atol=1e-4,rtol=1e-5)
                        np.testing.assert_array_equal(logits.argmax(1),prediction[name].argmax(1))
                        max_error=max(max_error,float(np.max(np.abs(logits-prediction[name]))))
                for name in event_times:
                    decoded=decode_recovery_endpoints(starts+63,prediction[name],np.zeros(len(starts)),fall_alert_mode=D1_DECOUPLED)
                    event_times[name]=[e['time'] for e in decoded['events'] if e['type']=='fall_trigger']
                    io.require(event_times[name]==[e['time'] for e in reported['heads'][name]['predictions']],'D1 mismatch')
        else:
            io.require(not (dest/'lift.npz').exists() and not (dest/'inference.npz').exists(),'quality bypass')
            io.require(io.read(dest/'quality_rejection.json')['classifier_run'] is False,'rejection receipt')
        row=dict(id=sid,scene=item['scene'],quality_passed=q,heads={})
        for name in event_times:
            row['heads'][name]=validate_metrics(independent_counts(event_times[name],intervals),reported['heads'][name])
        rows.append(row)
        if n%5==0:print('independent audit',n,38,flush=True)
    def validate_group(group,target,name):
        return validate_metrics([sum(r['heads'][name][k] for r in group) for k in ['tp','fp','fn']],target)
    verified={}
    for name in ['G0','G2']:
        ref=published['heads'][name]
        verified[name]=dict(primary=validate_group(rows,ref['primary'],name),
            quality_pass=validate_group([r for r in rows if r['quality_passed']],ref['quality_pass'],name))
        for scene,target in ref['per_scene'].items():validate_group([r for r in rows if r['scene']==scene],target,name)
    io.require(sum(r['quality_passed'] for r in rows)==published['quality_pass'],'quality denominator')
    io.locked(cfg)
    result=dict(passed=True,videos=38,fall_events=22,windows=windows,reused_videos=22,new_videos=16,
        all_original_avi_hashes_unchanged=True,cache_sources_unchanged=True,
        labels_reloaded_from_official_csv=True,full_feature_rebuild=True,independent_d1_replay=True,
        cpu_head_max_abs=max_error,cpu_argmax_identical=True,heads=verified,rows=rows,
        evaluation_sha256=io.sha(output/'evaluation.json'),audit_code_sha256=io.sha(Path(__file__)))
    io.save(output/'independent_audit.json',result)
    run.status('completed',videos=38,fall_events=22,heads=verified)


if __name__=='__main__':main()
