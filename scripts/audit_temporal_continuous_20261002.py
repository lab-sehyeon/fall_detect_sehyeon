"""CPU independent timeline, tracking, geometry, functional head and metric audit."""
from pathlib import Path
from fractions import Fraction
import os,sys
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fall_pipeline.external import rgb_document_io as io
from fall_pipeline.external import temporal_continuous_20261002 as run
from data_gen.safer_v2_geometry import normalize_ntu
from data_gen.safer_v3_geometry import fuse


def metric(tp,fp,fn):
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def ap_score(labels,scores):
    # Descending score groups preserve sklearn's treatment of tied probabilities.
    values=sorted(set(scores),reverse=True);tp=fp=0;last_recall=0.;area=0.;positives=sum(labels)
    for value in values:
        group=[y for y,s in zip(labels,scores) if s==value]
        tp+=sum(group);fp+=len(group)-sum(group);recall=tp/positives
        area+=(recall-last_recall)*tp/(tp+fp);last_recall=recall
    return area


def main():
    io.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit only');torch.set_num_threads(2)
    cfg=run.setup();output,plan=io.locked(cfg);evaluation=io.read(output/'evaluation.json')
    joint=torch.load(ROOT/cfg['model']['j1']['path'],map_location='cpu',weights_only=True)
    w={k.removeprefix('adapter.'):v for k,v in joint.items() if k.startswith('adapter.')}
    head=torch.load(ROOT/cfg['model']['g0']['path'],map_location='cpu',weights_only=True)
    max_logit=max_adapter=0.;windows=0;records={}
    for number,item in enumerate(plan,1):
        dest=output/item['id'];fm=io.stage_done(dest,'frontend');trace=io.read(dest/'source_trace.json')
        io.require(io.sha(dest/'source_trace.json')==item['trace_sha256'],'trace hash')
        io.require(io.sha(ROOT/item['mapping'])==item['mapping_sha256'],'mapping hash')
        with np.load(ROOT/item['mapping']) as mapping,np.load(dest/'frontend.npz') as front:
            indices=mapping['source_indices'];times=mapping['canonical_timestamps']
            rational=[Fraction(p-trace['start_pts'])*Fraction(trace['time_base_num'],trace['time_base_den']) for p in trace['pts']]
            cursor=0;expected=[]
            for j in range(item['frames']):
                target=Fraction(j,25)
                while cursor+1<len(rational) and rational[cursor+1]<=target:cursor+=1
                io.require(rational[cursor]<=target,'future source time');expected.append(cursor)
            np.testing.assert_array_equal(indices,expected)
            np.testing.assert_array_equal(times,np.arange(item['frames'])/25)
            np.testing.assert_array_equal(front['source_indices'],indices);np.testing.assert_array_equal(front['timestamps'],times)
            boxes,xy,score=front['boxes'],front['xy'],front['scores'];previous=None
            for i in range(len(boxes)):
                c=front['candidates'][front['offsets'][i]:front['offsets'][i+1]]
                if not len(c):expected_box=np.zeros(5,np.float32)
                else:
                    if previous is None:scores=c[:,4]
                    else:
                        wh=np.maximum(0,np.minimum(previous[2:4],c[:,2:4])-np.maximum(previous[:2],c[:,:2]));inter=wh.prod(1)
                        union=np.maximum(0,previous[2:4]-previous[:2]).prod()+np.maximum(0,c[:,2:4]-c[:,:2]).prod(1)-inter
                        overlap=np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0)
                        scores=.7*overlap+.3*c[:,4]
                    expected_box=c[int(np.argmax(scores))];previous=expected_box
                np.testing.assert_array_equal(boxes[i],expected_box)
            bv=np.isfinite(boxes).all(1)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])&(boxes[:,4]>0)
            pv=bv&np.isfinite(xy).all((1,2))&np.isfinite(score).all(1)&(score>0).any(1)
            pelvis=np.where(pv,np.minimum(score[:,11],score[:,12]),0)
            passed=bool(bv.mean()>=.8 and pv.mean()>=.8 and np.median(pelvis)>=.3 and len(boxes)>=64)
        io.require(passed==fm['quality']['passed'],'quality mismatch')
        io.require(fm['models_before']==fm['models_after'] and not fm['labels_used'] and not fm['training'],'frontend mutated')
        times=[];positive=False;prob=0.
        if passed:
            lm=io.stage_done(dest,'lift');im=io.stage_done(dest,'inference')
            io.require(lm['model_before']==lm['model_after'] and im['model_before']==im['model_after'],'model mutation')
            with np.load(dest/'lift.npz') as lift,np.load(dest/'inference.npz') as prediction:
                starts=np.arange(0,item['frames']-63,8);windows+=len(starts)
                for actual in [lift['window_starts'],prediction['window_starts']]:np.testing.assert_array_equal(actual,starts)
                np.testing.assert_array_equal(prediction['window_endpoints'],starts+63)
                h,coverage,_=fuse(lift['lifting_raw'],lift['lifting_starts'],item['frames'],'triangular',.05,True)
                np.testing.assert_array_equal(lift['h36m'],h);np.testing.assert_array_equal(lift['coverage'],coverage)
                ntu,_,_=normalize_ntu(h,.5);np.testing.assert_array_equal(ntu,lift['ntu25'])
                with torch.inference_mode():
                    x=torch.from_numpy(prediction['pooled'])
                    hidden=F.layer_norm(x,(2048,),w['norm.weight'],w['norm.bias'],1e-5)
                    hidden=F.gelu(F.linear(hidden,w['down.weight'],w['down.bias']))
                    adapted=x+F.linear(hidden,w['up.weight'],w['up.bias'])
                    logits=F.linear(adapted,head['weight'],head['bias']).numpy()
                np.testing.assert_allclose(adapted.numpy(),prediction['adapted'],atol=1e-4,rtol=1e-5)
                np.testing.assert_allclose(logits,prediction['G0'],atol=1e-4,rtol=1e-5)
                np.testing.assert_array_equal(logits.argmax(1),prediction['G0'].argmax(1))
                max_logit=max(max_logit,float(np.abs(logits-prediction['G0']).max()))
                max_adapter=max(max_adapter,float(np.abs(adapted.numpy()-prediction['adapted']).max()))
                labels=prediction['G0'].argmax(1);previous=False
                for end,label in zip(starts+63,labels):
                    now=label==1
                    if now and not previous:times.append(float(end/25))
                    previous=now
                positive=bool((labels==1).any());z=prediction['G0'].astype(float);ex=np.exp(z-z.max(1,keepdims=True))
                prob=float((ex[:,1]/ex.sum(1)).max())
        else:
            io.require(not (dest/'inference.npz').exists(),'quality bypass')
            io.require(io.read(dest/'quality_rejection.json')['classifier_run'] is False,'missing rejection')
        records[item['id']]=dict(quality_passed=passed,times=times,positive=positive,probability=prob)
        if number%20==0:print('audit',number,len(plan),flush=True)
    rows=[]
    truth=io.read(output/'ground_truth.json')
    # Reload the parent ground truth independently; do not trust only the new scorer input.
    for scope,parent_name in run.SOURCES.items():
        parent=ROOT/'data/fall_processed/RGB'/parent_name
        source_truth=io.read(parent/'ground_truth.json');source_plan=io.read(parent/'plan.json')
        selected=[r for r in truth if r['scope']==scope]
        if scope.startswith('le2i'):
            expected={'le2i__'+r['id']:r['episodes'] for r in source_truth}
            io.require({r['id']:r['episodes'] for r in selected}==expected,'inherited Le2i GT differs')
        else:
            labels={r['id']:r['label'] for r in source_truth};by_path={}
            for r in source_plan:
                by_path.setdefault(r['path'],[])
                if labels[r['id']]==1:by_path[r['path']].append((r['start'],r['end']))
            io.require({r['path']:[(e['fall_start'],e['fall_end']) for e in r['episodes']] for r in selected}==by_path,'official source GT differs')
    for gt,reported in zip(truth,evaluation['rows']):
        io.require(gt['id']==reported['id'] and gt['scope']==reported['scope'],'score order')
        record=records[gt['id']];events=gt['episodes'];io.require(len(events)<=1,'audit event scope')
        tp=int(bool(events) and any(events[0]['fall_start']-.5<=t<=events[0]['fall_end']+3 for t in record['times']))
        expected=metric(tp,len(record['times'])-tp,len(events)-tp)
        for key,val in expected.items():io.require(abs(reported['event'][key]-val)<1e-12,'event metric differs')
        io.require([x['time'] for x in reported['predictions']]==record['times'],'edge times differ')
        io.require(record['quality_passed']==reported['quality_passed'],'quality score differs')
        io.require(record['positive']==reported['video_prediction'],'video decision differs')
        io.require(abs(record['probability']-reported['max_fall_probability'])<1e-12,'probability differs')
        rows.append(dict(gt,**record,event=expected))
    for scope,target in evaluation['datasets'].items():
        selected=[r for r in rows if r['scope']==scope];y=[bool(r['episodes']) for r in selected];p=[r['positive'] for r in selected]
        event=metric(*(sum(r['event'][k] for r in selected) for k in ['tp','fp','fn']))
        io.require(event==target['event'],'aggregate event differs')
        tp=sum(a and b for a,b in zip(y,p));fp=sum(not a and b for a,b in zip(y,p));fn=sum(a and not b for a,b in zip(y,p));tn=sum(not a and not b for a,b in zip(y,p))
        video=dict(metric(tp,fp,fn),tn=tn,accuracy=(tp+tn)/len(y),specificity=tn/(tn+fp),ap=ap_score(y,[r['probability'] for r in selected]))
        for key,val in video.items():io.require(abs(target['video'][key]-val)<1e-12,'video aggregate differs: '+key)
        io.require(target['quality_pass']==sum(r['quality_passed'] for r in selected),'quality aggregate differs')
    for path,digest in io.read(output/'source_snapshot.json').items():io.require(io.sha(ROOT/path)==digest,'original changed: '+path)
    io.locked(cfg)
    io.save(output/'independent_audit.json',dict(passed=True,videos=len(plan),memberships=len(rows),windows=windows,
        rational_timeline_verified=True,tracking_replayed=True,quality_recomputed=True,geometry_rebuilt=True,
        functional_cpu_head_verified=True,cpu_argmax_identical=True,max_logit_abs=max_logit,max_adapter_abs=max_adapter,
        original_sources_unchanged=True,event_and_video_metrics_verified=True,
        evaluation_sha256=io.sha(output/'evaluation.json'),audit_code_sha256=io.sha(Path(__file__))))
    run.status('completed',datasets=evaluation['datasets'],windows=windows)

if __name__=='__main__':main()
