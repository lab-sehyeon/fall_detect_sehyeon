"""Independent CPU audit: original labels, timelines, geometry, heads and metrics."""
from pathlib import Path
from fractions import Fraction
import csv, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import run_le2i130_20261003 as run
from fall_pipeline.external import rgb_document_io as io
from data_gen.safer_v3_geometry import fuse
from data_gen.safer_v2_geometry import normalize_ntu


def metrics(tp,fp,fn):
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def ap(y,scores):
    tp=fp=0;recall=area=0.
    for value in sorted(set(scores),reverse=True):
        group=[label for label,s in zip(y,scores) if s==value];tp+=sum(group);fp+=len(group)-sum(group)
        next_recall=tp/sum(y);area+=(next_recall-recall)*tp/(tp+fp);recall=next_recall
    return area


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit only'
    torch.set_num_threads(2);cfg=run.setup();out,plan=run.locked(cfg)
    truth={r['id']:r for r in io.read(out/'ground_truth.json')}
    ev=io.read(out/'evaluation.json');reported={(r['model'],r['id']):r for r in ev['rows']}
    old=io.read(run.parent.OUT/'evaluation.json')
    old_rows={(r['model'],r['id']):r for r in old['rows'] if r['scope']=='le2i127'}
    assert len(plan)==len(truth)==130 and len(reported)==520
    joint=torch.load(ROOT/cfg['model']['j1']['path'],map_location='cpu',weights_only=True)
    w={k.removeprefix('adapter.'):v for k,v in joint.items() if k.startswith('adapter.')}
    head=torch.load(ROOT/cfg['model']['g0']['path'],map_location='cpu',weights_only=True)
    reference=io.read(ROOT/cfg['lifting_source_config'])['ntu_proxy']['reference_torso']
    independent=[];cases=[];max_logit_error=0.;head_windows=0
    for number,item in enumerate(plan,1):
        sid=item['id'];dest=out/sid;gt=truth[sid]
        annotation=ROOT/item['source_annotation'];assert io.sha(annotation)==item['annotation_sha256']
        scalars=[int(line.strip()) for line in annotation.read_text().splitlines() if re.fullmatch('[0-9]+',line.strip())]
        assert len(scalars)==2 and scalars==item['raw_header']
        media=io.read(dest/'source_media.json')
        source_meta=run.video_prep.probe(ROOT/media['raw_video'])['streams'][0]
        assert source_meta==media['metadata']
        fps=float(Fraction(source_meta['avg_frame_rate']))
        assert fps==item['source_fps'] and source_meta['avg_frame_rate']==item['source_fps_rational']
        expected=[] if scalars==[0,0] else [dict(fall_start=scalars[0]/fps,fall_end=scalars[1]/fps)]
        assert gt['episodes']==expected
        trace=io.read(dest/'source_trace.json');assert io.sha(dest/'source_trace.json')==item['trace_sha256']
        assert io.sha(ROOT/item['video'])==item['video_sha256']
        with np.load(ROOT/item['mapping']) as m,np.load(dest/'frontend.npz') as f:
            indices=m['source_indices'];timestamps=m['canonical_timestamps']
            rational=[Fraction(x-trace['start_pts'])*Fraction(trace['time_base_num'],trace['time_base_den']) for x in trace['pts']]
            cursor=0;expected_indices=[]
            for j in range(item['frames']):
                target=Fraction(j,25)
                while cursor+1<len(rational) and rational[cursor+1]<=target:cursor+=1
                assert rational[cursor]<=target;expected_indices.append(cursor)
            np.testing.assert_array_equal(indices,expected_indices)
            np.testing.assert_array_equal(timestamps,np.arange(item['frames'])/25)
            np.testing.assert_array_equal(f['source_indices'],indices);np.testing.assert_array_equal(f['timestamps'],timestamps)
            boxes,xy,scores=f['boxes'],f['xy'],f['scores'];previous=None
            for i,box in enumerate(boxes):
                candidates=f['candidates'][f['offsets'][i]:f['offsets'][i+1]]
                if not len(candidates):chosen=np.zeros(5,np.float32)
                else:
                    if previous is None:rank=candidates[:,4]
                    else:
                        overlap=np.maximum(0,np.minimum(previous[2:4],candidates[:,2:4])-np.maximum(previous[:2],candidates[:,:2])).prod(1)
                        union=np.maximum(0,previous[2:4]-previous[:2]).prod()+np.maximum(0,candidates[:,2:4]-candidates[:,:2]).prod(1)-overlap
                        rank=.7*np.divide(overlap,union,out=np.zeros_like(overlap,dtype=float),where=union>0)+.3*candidates[:,4]
                    chosen=candidates[int(np.argmax(rank))];previous=chosen
                np.testing.assert_array_equal(box,chosen)
            bv=np.isfinite(boxes).all(1)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])&(boxes[:,4]>0)
            pv=bv&np.isfinite(xy).all((1,2))&np.isfinite(scores).all(1)&(scores>0).any(1)
            own_pass=bool(bv.mean()>=.8 and pv.mean()>=.8 and np.median(np.where(pv,np.minimum(scores[:,11],scores[:,12]),0))>=.3 and len(xy)>=64)
            comparison_pass=bool(len(xy)>=64 and np.any(scores>0))
        front=io.stage_done(dest,'frontend');assert front['quality']['passed']==own_pass
        assert front['models_before']==front['models_after'] and not front['labels_used'] and not front['training']
        if own_pass:
            lift=io.stage_done(dest,'lift');inf=io.stage_done(dest,'inference')
            assert lift['model_before']==lift['model_after'] and inf['model_before']==inf['model_after']
            with np.load(dest/'lift.npz') as z,np.load(dest/'inference.npz') as prediction:
                h,cov,_=fuse(z['lifting_raw'],z['lifting_starts'],item['frames'],'triangular',.05,True)
                np.testing.assert_array_equal(z['h36m'],h);np.testing.assert_array_equal(z['coverage'],cov)
                ntu,_,_=normalize_ntu(h,reference);np.testing.assert_array_equal(ntu,z['ntu25'])
                with torch.inference_mode():
                    x=torch.from_numpy(prediction['pooled'])
                    hidden=F.layer_norm(x,(2048,),w['norm.weight'],w['norm.bias'],1e-5)
                    hidden=F.gelu(F.linear(hidden,w['down.weight'],w['down.bias']))
                    adapted=x+F.linear(hidden,w['up.weight'],w['up.bias'])
                    logits=F.linear(adapted,head['weight'],head['bias']).numpy()
                np.testing.assert_allclose(adapted.numpy(),prediction['adapted'],atol=1e-4,rtol=1e-5)
                np.testing.assert_allclose(logits,prediction['G0'],atol=1e-4,rtol=1e-5)
                np.testing.assert_array_equal(logits.argmax(1),prediction['G0'].argmax(1))
                max_logit_error=max(max_logit_error,float(np.abs(logits-prediction['G0']).max()));head_windows+=len(logits)
        else:assert not (dest/'inference.npz').exists() and not io.read(dest/'quality_rejection.json')['classifier_run']
        case=dict(video=item['path'],added_video=item['added_video'],source_start_frame=scalars[0],source_end_frame=scalars[1],source_fps=fps)
        for model in ['own','stgcnpp','msg3d','cnn1d']:
            passed=own_pass if model=='own' else comparison_pass;alarms=[];best=0.;positive=False
            stage='inference' if model=='own' else model
            if passed:
                meta=io.stage_done(dest,stage)
                if model!='own':assert meta['source_frontend_sha256']==io.sha(dest/'frontend.npz') and not meta['annotations_read'] and not meta['training']
                with np.load(dest/(stage+'.npz')) as z:
                    starts=z['window_starts'];ends=z['window_endpoints'];logits=z['G0' if model=='own' else 'logits'].astype(float)
                    np.testing.assert_array_equal(starts,np.arange(0,item['frames']-63,8));np.testing.assert_array_equal(ends,starts+63)
                    assert logits.shape==(len(starts),4 if model=='own' else 15) and np.isfinite(logits).all()
                    exp=np.exp(logits-logits.max(1,keepdims=True));prob=exp/exp.sum(1,keepdims=True)
                    if model!='own':
                        np.testing.assert_array_equal(z['input_starts'],starts+16)
                        np.testing.assert_allclose(z['probabilities'],prob,atol=1e-6,rtol=1e-5)
                index=1 if model=='own' else 9;prev=False
                for end,logit in zip(ends,logits):
                    current=int(logit.argmax())==index
                    if current and not prev:alarms.append(float(end/25))
                    prev=current
                positive=bool((logits.argmax(1)==index).any());best=float(prob[:,index].max())
            # Each video has at most one original annotated fall, independently matched.
            tp=int(bool(expected) and any(expected[0]['fall_start']-.5<=t<=expected[0]['fall_end']+3 for t in alarms))
            event=metrics(tp,len(alarms)-tp,len(expected)-tp);got=reported[(model,sid)]
            assert all(got['event'][k]==v for k,v in event.items())
            assert [r['time'] for r in got['predictions']]==alarms and got['processed']==passed
            assert bool(got['video_prediction'])==positive and abs(got['max_fall_probability']-best)<1e-12
            if not item['added_video']:
                previous=old_rows[(model,sid)]
                for k in ['predictions','event','processed','video_prediction','max_fall_probability','windows']:assert got[k]==previous[k],(sid,model,k)
            independent.append(dict(model=model,added=item['added_video'],truth=bool(expected),positive=positive,passed=passed,best=best,event=event))
            case.update({model+'_processed':passed,model+'_tp':event['tp'],model+'_fp':event['fp'],model+'_fn':event['fn'],model+'_alarms':';'.join(map(str,alarms))})
        cases.append(case)
        if number%25==0:print('AUDIT',number,130,flush=True)
    changes={}
    for model in ['own','stgcnpp','msg3d','cnn1d']:
        rows=[r for r in independent if r['model']==model];s=ev['summaries'][model]['le2i130']
        event=metrics(*(sum(r['event'][k] for r in rows) for k in ['tp','fp','fn']))
        assert s['event']==event and s['videos']==130 and s['fall_events']==s['positive_videos']==99
        assert s['processed']==sum(r['passed'] for r in rows)
        y=[r['truth'] for r in rows];p=[r['positive'] for r in rows]
        tp=sum(a and b for a,b in zip(y,p));fp=sum(not a and b for a,b in zip(y,p));fn=sum(a and not b for a,b in zip(y,p));tn=sum(not a and not b for a,b in zip(y,p))
        video=metrics(tp,fp,fn)
        assert all(s['video'][k]==v for k,v in video.items()) and s['video']['tn']==tn
        assert s['video']['accuracy']==(tp+tn)/130 and s['video']['specificity']==tn/(tn+fp)
        assert abs(s['video']['ap']-ap(y,[r['best'] for r in rows]))<1e-12
        added=metrics(*(sum(r['event'][k] for r in rows if r['added']) for k in ['tp','fp','fn']))
        previous=old['summaries'][model]['le2i127']['event']
        assert all(previous[k]+added[k]==event[k] for k in ['tp','fp','fn'])
        changes[model]=dict(previous127=previous,added3=added,full130=event,f1_delta_pp=(event['f1']-previous['f1'])*100)
    for r in io.read(out/'cache_reuse.json'):
        assert io.sha(ROOT/r['source'])==io.sha(ROOT/r['copy'])==r['sha256']
    done=io.read(out/'comparators_completed.json');assert done['passed']
    assert all(v['passed'] and v['model_before']==v['model_after'] for v in done['models'].values())
    run.locked(cfg)
    with (out/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    io.save(out/'comparison_127_to_130.json',changes)
    report=dict(passed=True,videos=130,fall_events=99,nonfall_videos=31,models=4,reused_prediction_videos=128,
                newly_inferred_videos=2,old127_predictions_metrics_unchanged=True,raw_annotation_values_verified=True,
                temporal_order_tracking_quality_verified=True,geometry_and_functional_head_verified=True,
                functional_head_windows=head_windows,max_logit_error=max_logit_error,metrics_independently_recomputed=True,
                evaluation_sha256=io.sha(out/'evaluation.json'),contract_sha256=io.sha(out/'contract.json'))
    io.save(out/'independent_audit.json',report);run.base.status('completed',audit=report)
    print('AUDIT PASS',report,flush=True)


if __name__=='__main__':main()
