"""Independent full-scope URFD metric and trace validation."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score,average_precision_score
from fall_pipeline.external import rgb_document_io as io

def main():
    root=ROOT/'data/fall_processed/RGB/external_urfd_20261002_r1'
    plan=io.read(root/'plan.json');published=io.read(root/'evaluation.json')
    assert len(plan)==70 and len({p['id'] for p in plan})==70
    expected={f'{kind}-{n:02d}':int(kind=='fall') for kind,count in [('adl',40),('fall',30)] for n in range(1,count+1)}
    assert set(expected)=={p['id'] for p in plan}
    rows=[];trace={};y=[];pred=[];scores=[];beforeafter=True
    for item,reported in zip(plan,published['rows']):
        sid=item['id'];assert reported['id']==sid and reported['label']==expected[sid]
        dest=root/sid;front=io.stage_done(dest,'frontend');trace[sid]={'frontend':io.sha(dest/'frontend.json')}
        classified=False;decision=0;prob=0.;windows=0
        if 'preprocessing_rejection' in item:
            assert sid=='adl-37' and not front['quality']['passed']
            assert not (dest/'inference.npz').exists()
        else:
            with np.load(ROOT/item['mapping']) as m,np.load(dest/'frontend.npz') as f:
                np.testing.assert_array_equal(f['source_indices'],m['source_indices'])
                np.testing.assert_array_equal(f['timestamps'],m['canonical_timestamps'])
                assert np.all(m['source_timestamps'][m['source_indices']]<=m['canonical_timestamps'])
                boxes=f['boxes'];xy=f['xy'];sc=f['scores']
                valid=np.isfinite(boxes).all(1)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])&(boxes[:,4]>0)
                pose=valid&np.isfinite(xy).all((1,2))&np.isfinite(sc).all(1)&(sc>0).any(1)
                quality=bool(valid.mean()>=.8 and pose.mean()>=.8 and np.median(np.where(pose,np.minimum(sc[:,11],sc[:,12]),0))>=.3 and len(xy)>=64)
                assert quality==front['quality']['passed']
            assert front['models_before']==front['models_after']
            if quality:
                lift=io.stage_done(dest,'lift');inference=io.stage_done(dest,'inference')
                assert lift['model_before']==lift['model_after'] and inference['models_before']==inference['models_after']
                with np.load(dest/'inference.npz') as z,np.load(dest/'lift.npz') as l:
                    starts=np.arange(0,item['frames']-63,8)
                    np.testing.assert_array_equal(z['window_starts'],starts)
                    np.testing.assert_array_equal(l['window_starts'],starts)
                    logits=z['G0'].astype(float);decision=int(any(np.argmax(v)==1 for v in logits));windows=len(logits)
                    probs=[1/sum(np.exp(v-v[1])) for v in logits]
                    prob=float(max(probs));classified=True
                trace[sid].update(lift=io.sha(dest/'lift.json'),inference=io.sha(dest/'inference.json'))
            else:assert not (dest/'inference.npz').exists()
        assert decision==reported['prediction'] and abs(prob-reported['max_fall_probability'])<1e-12
        y.append(expected[sid]);pred.append(decision);scores.append(prob)
        reason=('timestamp_missing' if 'preprocessing_rejection' in item else 'shorter_than64' if item['frames']<64 else 'pose_quality' if not classified else 'classified')
        rows.append(dict(id=sid,label=expected[sid],prediction=decision,reason=reason,windows=windows))
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel())
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    metrics=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=float(precision),recall=float(recall),f1=float(f1),
                 accuracy=float(accuracy_score(y,pred)),ap=float(average_precision_score(y,scores)))
    for key,value in metrics.items():assert abs(value-published['metrics'][key])<1e-12
    coverage={key:dict(total=sum(r['reason']==key for r in rows),positive=sum(r['reason']==key and r['label']==1 for r in rows),negative=sum(r['reason']==key and r['label']==0 for r in rows)) for key in ['classified','timestamp_missing','shorter_than64','pose_quality']}
    audit=dict(passed=True,metrics=metrics,coverage=coverage,rows=rows,stage_metadata_sha256=trace,
               full_denominator=70,gt_derived_independently_from_official_categories=True,
               note='TN after rejection means system no alarm on a negative, not successful classifier inference',
               evaluation_sha256=io.sha(root/'evaluation.json'),audit_code_sha256=io.sha(Path(__file__)))
    io.save(ROOT/'docs/internal/2026-10-02_evidence_audit/urfd_independent_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['rows','stage_metadata_sha256']},indent=2))

if __name__=='__main__':main()
