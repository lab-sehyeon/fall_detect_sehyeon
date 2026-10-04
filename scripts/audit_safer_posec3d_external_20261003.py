"""Independent counting, probability, coverage and immutable-input audit."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fall_pipeline.external import rgb_document_io as io

OUT=ROOT/'data/fall_processed/RGB/safer_posec3d_external_20261003_r1'
PARENT=ROOT/'data/fall_processed/RGB/temporal_continuous_20261002_r1'
SCOPES=['le2i38','gmdcsa37','cauca19']


def ap(labels,scores):
    tp=fp=0;previous_recall=0.;area=0.
    for s in sorted(set(scores),reverse=True):
        ys=[y for y,x in zip(labels,scores) if x==s]
        tp+=sum(ys);fp+=len(ys)-sum(ys);recall=tp/sum(labels)
        area+=(recall-previous_recall)*tp/(tp+fp);previous_recall=recall
    return area


def event_counts(times,episodes):
    # Each source video in these fixed scopes has <=1 labelled fall.
    assert len(episodes)<=1
    if episodes:
        e=episodes[0]
        matched=int(any(e['fall_start']-.5<=t<=max(e['fall_start'],e['fall_end'])+3 for t in times))
    else:matched=0
    return matched,len(times)-matched,len(episodes)-matched


def read_prediction(model,row):
    if model=='own':
        dest=PARENT/row['id'];passed=io.read(dest/'frontend.json')['quality']['passed']
        if not passed:return dict(processed=False,times=[],positive=False,score=0.,windows=0)
        with np.load(dest/'inference.npz') as p:
            logits=p['G0'].astype(float);indices=p['window_endpoints'];start=p['window_starts']
        ex=np.exp(logits-logits.max(1,keepdims=True));prob=ex/ex.sum(1,keepdims=True)
        flags=logits.argmax(1)==1;scores=prob[:,1]
    else:
        dest=OUT/row['id'];m=io.read(dest/'prediction.json');passed=m['passed']
        assert io.sha(dest/'prediction.npz')==m['payload_sha256']
        assert io.sha(PARENT/row['id']/'frontend.npz')==m['source_frontend_sha256']
        assert m['training'] is False and m['annotations_read'] is False
        with np.load(dest/'prediction.npz') as p:
            prob=p['probabilities'];v=p['view_logits'].astype(float);indices=p['window_endpoints'];start=p['window_starts']
            np.testing.assert_array_equal(p['input_starts'],start+16)
        assert prob.shape==(len(start),15) and v.shape==(len(start),2,15)
        if passed:
            ex=np.exp(v-v.max(2,keepdims=True));expected=(ex/ex.sum(2,keepdims=True)).mean(1)
            np.testing.assert_allclose(prob,expected,atol=1e-6,rtol=1e-5)
            np.testing.assert_array_equal(prob.argmax(1),expected.argmax(1))
            np.testing.assert_allclose(prob.sum(1),1,atol=1e-6)
        else:assert not np.count_nonzero(prob)
        flags=prob.argmax(1)==9;scores=prob[:,9]
    expected=np.arange(0,row['frames']-63,8)
    np.testing.assert_array_equal(start,expected);np.testing.assert_array_equal(indices,expected+63)
    times=[];previous=False
    for index,flag in zip(indices,flags):
        if flag and not previous:times.append(float(index/25))
        previous=bool(flag)
    return dict(processed=bool(passed),times=times,positive=bool(np.any(flags)),score=float(max(scores)) if len(scores) else 0.,windows=len(start))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit only'
    contract=io.read(OUT/'contract.json')
    for p,wanted in contract['files'].items():assert io.sha(ROOT/p)==wanted,p
    own_audit=io.read(PARENT/'independent_audit.json');assert own_audit['passed']
    assert io.sha(PARENT/'evaluation.json')==own_audit['evaluation_sha256']
    done=io.read(OUT/'inference_completed.json');assert done['passed'] and done['model_before']==done['model_after']
    evaluation=io.read(OUT/'evaluation.json');truth=io.read(OUT/'ground_truth.json');plan=io.read(OUT/'plan.json')
    assert len(truth)==len(plan)==94
    inherited=[x for x in io.read(PARENT/'ground_truth.json') if x['scope'] in SCOPES]
    assert truth==inherited
    items={x['id']:x for x in plan};reported={(r['model'],r['id']):r for r in evaluation['rows']}
    assert len(reported)==188
    independent={};cases=[];summary_rows=[]
    for number,gt in enumerate(truth,1):
        case=dict(dataset=gt['scope'],video=gt['id'],ground_truth_fall=int(bool(gt['episodes'])))
        for model in ['own','posec3d']:
            pred=read_prediction(model,items[gt['id']]);counts=event_counts(pred['times'],gt['episodes'])
            r=reported[(model,gt['id'])]
            assert counts==tuple(r['event'][x] for x in ['tp','fp','fn'])
            assert pred['times']==[x['time'] for x in r['predictions']]
            assert pred['positive']==bool(r['video_prediction']) and pred['processed']==r['processed']
            assert abs(pred['score']-r['max_fall_probability'])<1e-12
            independent[(model,gt['id'])]=dict(pred,counts=counts,truth=bool(gt['episodes']),scope=gt['scope'])
            case.update({model+'_processed':int(pred['processed']),model+'_video_prediction':int(pred['positive']),
                         model+'_event_tp':counts[0],model+'_event_fp':counts[1],model+'_event_fn':counts[2],
                         model+'_alarm_times':';'.join(f'{t:.2f}' for t in pred['times'])})
        cases.append(case)
        if number%20==0:print('independent audit',number,94,flush=True)
    for model in ['own','posec3d']:
        for scope in SCOPES:
            rs=[r for (m,_),r in independent.items() if m==model and r['scope']==scope]
            y=[r['truth'] for r in rs];p=[r['positive'] for r in rs]
            tp=sum(a and b for a,b in zip(y,p));fp=sum(not a and b for a,b in zip(y,p));fn=sum(a and not b for a,b in zip(y,p));tn=sum(not a and not b for a,b in zip(y,p))
            ev=np.array([r['counts'] for r in rs]).sum(0).tolist();summary=evaluation['summaries'][model][scope]
            assert len(rs)==summary['videos'] and sum(y)==summary['positive']
            assert sum(r['processed'] for r in rs)==summary['processed']
            for unit,counts in [('event',ev),('video',[tp,fp,fn])]:
                s=summary[unit];et,ef,en=counts
                for k,n in zip(['tp','fp','fn'],counts):assert n==s[k]
                precision=et/(et+ef) if et+ef else 0;recall=et/(et+en) if et+en else 0
                f1=2*et/(2*et+ef+en) if 2*et+ef+en else 0
                for k,n in [('precision',precision),('recall',recall),('f1',f1)]:assert abs(n-s[k])<1e-12
                summary_rows.append(dict(model=model,dataset=scope,unit=unit,videos=len(rs),positive=sum(y),processed=summary['processed'],
                    tp=et,fp=ef,fn=en,tn=tn if unit=='video' else '',precision_pct=100*precision,recall_pct=100*recall,f1_pct=100*f1,
                    ap_pct=100*s['ap'] if unit=='video' else ''))
            vs=summary['video'];assert tn==vs['tn']
            assert abs(ap(y,[r['score'] for r in rs])-vs['ap'])<1e-12
            assert abs((tp+tn)/len(rs)-vs['accuracy'])<1e-12
            assert abs(tn/(tn+fp)-vs['specificity'])<1e-12
            if model=='own':
                old=io.read(PARENT/'evaluation.json')['datasets'][scope]
                for unit in ['event','video']:assert old[unit]==summary[unit]
    # Verify deterministic frame order directly from official sampler audit saved before inference.
    synthetic=io.read(OUT/'synthetic_validation.json');assert synthetic['passed']
    sample=np.array(synthetic['official_sample_indices']);assert sample.shape==(1,48)
    np.testing.assert_array_equal(sample[0],np.arange(48))
    for name,rows in [('results',summary_rows),('cases',cases)]:
        path=ROOT/'docs/shared'/f'2026-10-03_safer_posec3d_{name}.csv'
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(passed=True,videos=94,prediction_rows=188,comparator_windows=sum(r['windows'] for (m,_),r in independent.items() if m=='posec3d'),
        exact_same_video_manifest=True,immutable_inputs_and_assets=True,source_gt_unchanged=True,
        source_training_exclusion='documented dataset-level sources; not a sample-level foundation-pretraining proof',
        multiclip_softmax_independently_verified=True,probability_argmax_identical=True,chronological_sampler=True,
        original_own_metrics_identical=True,all_failures_retained=True,event_video_ap_independently_recounted=True,
        complete_backbone_independent_implementation=False,evaluation_sha256=io.sha(OUT/'evaluation.json'),
        audit_code_sha256=io.sha(Path(__file__)))
    io.save(OUT/'independent_audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
