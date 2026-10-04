"""Independently recompute SVM decisions, chronological alarms and both metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys,time,hashlib
from pathlib import Path
import numpy as np,joblib
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score
import evaluate_hfd_reproduction_20261004 as run
ROOT=run.ROOT;OUT=run.OUT
def manual_events(alarms,episodes):
    used=set();hit=0
    for a in sorted(alarms,key=lambda a:a['time']):
        viable=[(e['fall_start'],i) for i,e in enumerate(episodes) if i not in used and e['fall_start']-.5<=a['time']<=max(e['fall_start'],e['fall_end'])+3]
        if viable:used.add(min(viable)[1]);hit+=1
    return dict(tp=hit,fp=len(alarms)-hit,fn=len(episodes)-hit)
def audit():
    run.check_contract();clf=joblib.load(OUT/'svm.joblib');rows=run.read(OUT/'predictions.json');plan=run.read(OUT/'plan.json');truth={x['id']:x for x in run.read(OUT/'ground_truth.json')}
    assert len(rows)==len(plan)==230 and len({r['id'] for r in rows})==230
    validation=run.read(OUT/'source_validation.json');source=np.load(OUT/'source_features.npz');fold_scores=[]
    assert source['X'].shape==(360,4096) and np.bincount(source['y']).tolist()==[201,159]
    for fold in validation['folds']:
        tr,te=fold['train_indices'],fold['test_indices'];assert len(tr)==251 and len(te)==109 and not set(tr)&set(te)
        f=float(f1_score(source['y'][te],fold['predictions'],zero_division=0));assert np.isclose(f,fold['fall']['f1']);fold_scores.append(f)
    assert np.isclose(np.mean(fold_scores),validation['mean_fall_f1'])
    summary=run.read(OUT/'evaluation.json')['summaries'];byid={x['id']:x for x in rows};maxerr=0.;chunks=0;frames=0;aggregate={}
    for item in plan:
        row=byid[item['id']];assert row['processed'],row
        dest=OUT/item['id'];assert run.sha(dest/'features.npz')==row['payload_sha256']
        z=np.load(dest/'features.npz');X=z['features'];manual=X.astype(np.float64)@clf.coef_[0]+clf.intercept_[0];err=float(np.max(np.abs(manual-z['decision'])));maxerr=max(maxerr,err)
        assert np.allclose(manual,z['decision'],atol=1e-7,rtol=1e-7);pred=(manual>0).astype(int);assert np.array_equal(pred,z['predictions'])
        trace=run.read(ROOT/item['trace']);n=len(trace['pts']);end=np.arange(n//16)*16+15;ts=np.array([(trace['pts'][int(i)]-trace['start_pts'])*trace['time_base_num']/trace['time_base_den'] for i in end])
        assert np.array_equal(end,z['end_frames']) and np.array_equal(ts,z['times']) and np.all(np.diff(ts)>0)
        alarms=[];previous=0
        for j,p in enumerate(pred):
            if p and not previous:alarms.append(dict(frame=int(end[j]),time=float(ts[j])))
            previous=p
        assert alarms==row['predictions'];e=manual_events(alarms,truth[item['id']]['episodes']);assert all(e[k]==row['event'][k] for k in e)
        assert row['video_prediction']==int(pred.any());chunks+=len(pred);frames+=n
    for model,rr in [('hfd_reproduction',rows),('own',run.read(OUT/'own_predictions.json'))]:
        aggregate[model]={}
        for scope in ['le2i130','cauca100']:
            subset=[r for r in rr if r['scope']==scope];ct={k:0 for k in ['tp','fp','fn']}
            for r in subset:
                e=manual_events(r['predictions'],truth[r['id']]['episodes'])
                for k,v in e.items():ct[k]+=v
            expected=summary[model][scope];assert all(ct[k]==expected['event'][k] for k in ct)
            y=[int(bool(truth[r['id']]['episodes'])) for r in subset];p=[r['video_prediction'] for r in subset];cm=confusion_matrix(y,p,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel());f1=float(f1_score(y,p,zero_division=0));acc=float(accuracy_score(y,p));ef1=2*ct['tp']/(2*ct['tp']+ct['fp']+ct['fn']) if ct['tp'] else 0.
            assert np.isclose(f1,expected['video']['f1']) and np.isclose(acc,expected['video']['accuracy']) and np.isclose(ef1,expected['event']['f1'])
            aggregate[model][scope]=dict(event=dict(**ct,f1=ef1),video=dict(tp=tp,fp=fp,fn=fn,tn=tn,f1=f1,accuracy=acc))
    # Re-extract three whole videos through the untouched official source function.
    ns=run.official_namespace();replays=[]
    for item in [plan[0],plan[129],plan[-1]]:
        base=OUT/'audit_links'/item['id'];path=base/'Fall'/'cam1'/'audit.mp4';path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.symlink_to(ROOT/item['video'])
        X,y=ns['extractFeatures'](str(run.WEIGHTS),str(base),'',False);cached=np.load(OUT/item['id']/'features.npz')['features'];diff=float(np.max(np.abs(X-cached)));assert np.allclose(X,cached,atol=1e-5,rtol=1e-5)
        replays.append(dict(id=item['id'],chunks=len(X),max_feature_difference=diff))
    run.save(OUT/'independent_audit.json',dict(passed=True,videos=230,frames=frames,chunks=chunks,source_validation_f1=float(np.mean(fold_scores)),linear_svm_max_error=maxerr,official_feature_replays=replays,summaries=aggregate))
    print('audit passed',frames,'frames',chunks,'chunks',flush=True)
if __name__=='__main__':audit()
