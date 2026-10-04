"""Independent multi-event, denominator, temporal and raw-probability audit."""
from pathlib import Path
import argparse,csv,json,os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fall_pipeline.external import rgb_document_io as io

def counts(times,episodes):
    remaining=sorted(enumerate(episodes),key=lambda x:(x[1]['fall_start'],x[0]));tp=0
    for t in sorted(times):
        for pos,(index,e) in enumerate(remaining):
            if e['fall_start']-.5<=t<=e['fall_end']+3:
                tp+=1;remaining.pop(pos);break
    return tp,len(times)-tp,len(episodes)-tp

def ap(labels,scores):
    if not sum(labels):return None
    tp=fp=0;last=0.;area=0.
    for s in sorted(set(scores),reverse=True):
        group=[y for y,x in zip(labels,scores) if x==s];tp+=sum(group);fp+=len(group)-sum(group);recall=tp/sum(labels)
        area+=(recall-last)*tp/(tp+fp);last=recall
    return area

def main(dataset):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit only'
    out=ROOT/'data/fall_processed/RGB/three_by_three_20261003_r1'/dataset
    saved=io.read(out/'contract.json')
    for key in ['code_sha256','frozen_files','experiment_files']:
        for p,wanted in saved[key].items():assert io.sha(ROOT/p)==wanted,p
    done=io.read(out/'comparators_completed.json');assert done['passed']
    for v in done['models'].values():assert v['model_before']==v['model_after'] and v['passed']
    plan=io.read(out/'plan.json');truth=io.read(out/'ground_truth.json');evaluation=io.read(out/'evaluation.json')
    meta=ROOT/'data/source_archives/OmniFall/candidate_metadata_20261002_r1'
    paths={r['path'] for r in csv.DictReader((meta/f'splits/cs/{dataset}/test.csv').open())}
    assert {r['path'] for r in plan}==paths and len(plan)==len(truth)==len(paths)
    labels=list(csv.DictReader((meta/f'labels/{dataset}.csv').open()));byid={r['id']:r for r in truth}
    report={(r['model'],r['id']):r for r in evaluation['rows']};assert len(report)==len(plan)*4
    cases=[];independent=[]
    for row in plan:
        dest=out/row['id'];gt=byid[row['id']];expected=[dict(fall_start=float(r['start']),fall_end=float(r['end'])) for r in labels if r['path']==row['path'] and int(r['label'])==1]
        assert gt['episodes']==expected
        trace=io.read(dest/'source_trace.json')
        with np.load(ROOT/row['mapping']) as z:indices=z['source_indices'];times=z['canonical_timestamps']
        np.testing.assert_array_equal(times,np.arange(row['frames'])/25)
        ticks=(np.array(trace['pts'])-trace['start_pts'])*trace['time_base_num']*25;targets=np.arange(row['frames'])*trace['time_base_den']
        assert (ticks[indices]<=targets).all() and (np.diff(indices)>=0).all()
        has_next=indices+1<len(ticks);assert (ticks[indices[has_next]+1]>targets[has_next]).all()
        with np.load(dest/'frontend.npz') as z:np.testing.assert_array_equal(z['source_indices'],indices);np.testing.assert_array_equal(z['timestamps'],times)
        front=io.stage_done(dest,'frontend');assert front['models_before']==front['models_after']
        case=dict(dataset=dataset,video=row['path'],fall_events=len(expected))
        for model in ['own','stgcnpp','msg3d','cnn1d']:
            own=model=='own';passed=front['quality']['passed'] if own else io.stage_done(dest,model)['passed'];alarm=[];score=0.;positive=False
            if passed:
                stage='inference' if own else model;m=io.stage_done(dest,stage)
                if own:assert m['model_before']==m['model_after']
                else:assert m['source_frontend_sha256']==io.sha(dest/'frontend.npz')
                with np.load(dest/(stage+'.npz')) as z:
                    logits=z['G0' if own else 'logits'].astype(float);starts=z['window_starts'];ends=z['window_endpoints']
                    np.testing.assert_array_equal(starts,np.arange(0,row['frames']-63,8));np.testing.assert_array_equal(ends,starts+63)
                    ex=np.exp(logits-logits.max(1,keepdims=True));prob=ex/ex.sum(1,keepdims=True)
                    if not own:np.testing.assert_allclose(prob,z['probabilities'],atol=1e-6,rtol=1e-5);np.testing.assert_array_equal(z['input_starts'],starts+16)
                fall_index=1 if own else 9;prev=False
                for end,logit in zip(ends,logits):
                    current=int(np.argmax(logit))==fall_index
                    if current and not prev:alarm.append(float(end/25))
                    prev=current
                positive=bool((logits.argmax(1)==fall_index).any());score=float(prob[:,fall_index].max())
            elif own:assert not (dest/'inference.npz').exists()
            got=report[(model,row['id'])];c=counts(alarm,expected)
            assert c==tuple(got['event'][k] for k in ['tp','fp','fn'])
            assert alarm==[p['time'] for p in got['predictions']] and bool(got['video_prediction'])==positive
            assert abs(got['max_fall_probability']-score)<1e-10 and got['processed']==passed
            independent.append(dict(model=model,truth=bool(expected),positive=positive,score=score,passed=passed,counts=c))
            case.update({model+'_processed':int(passed),model+'_tp':c[0],model+'_fp':c[1],model+'_fn':c[2],model+'_alarms':';'.join(map(str,alarm))})
        cases.append(case)
    for model in ['own','stgcnpp','msg3d','cnn1d']:
        rows=[r for r in independent if r['model']==model];s=evaluation['summaries'][model][dataset]
        y=[r['truth'] for r in rows];p=[r['positive'] for r in rows];tp=sum(a and b for a,b in zip(y,p));fp=sum(not a and b for a,b in zip(y,p));fn=sum(a and not b for a,b in zip(y,p));tn=sum(not a and not b for a,b in zip(y,p))
        assert s['videos']==len(rows) and s['processed']==sum(r['passed'] for r in rows) and s['positive_videos']==sum(y)
        for unit,c in [('event',np.array([r['counts'] for r in rows]).sum(0)),('video',(tp,fp,fn))]:
            assert list(c)==[s[unit][k] for k in ['tp','fp','fn']]
            a,b,c=c;precision=a/(a+b) if a+b else 0.;recall=a/(a+c) if a+c else 0.;f1=2*a/(2*a+b+c) if 2*a+b+c else 0.
            for k,n in [('precision',precision),('recall',recall),('f1',f1)]:assert abs(s[unit][k]-n)<1e-12
        assert s['video']['tn']==tn
        expect=ap(y,[r['score'] for r in rows]);assert expect is None and s['video']['ap'] is None or abs(expect-s['video']['ap'])<1e-12
        assert s['video']['specificity']==(tn/(tn+fp) if tn+fp else None)
    io.save(out/'independent_audit.json',dict(passed=True,videos=len(plan),models=4,evaluation_sha256=io.sha(out/'evaluation.json'),multi_event_matching=True,unique_test_video_denominator=True,temporal_order=True))
    with (out/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    print('AUDIT PASS',dataset,len(plan),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['edf','occu','OOPS']);args=parser.parse_args();main(args.dataset)
