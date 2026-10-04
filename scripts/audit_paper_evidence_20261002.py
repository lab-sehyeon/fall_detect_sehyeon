"""Independent CPU replay and leakage checks on existing FU nested results."""
from pathlib import Path
import hashlib
import json
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/internal/2026-10-02_evidence_audit'
torch.set_num_threads(1)


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def forward(x,w,phase):
    # Independent functional implementation: no import of training/model code.
    x=torch.from_numpy(np.asarray(x,dtype=np.float32))
    with torch.inference_mode():
        if phase=='j1':
            z=F.layer_norm(x,(2048,),w['adapter.norm.weight'],w['adapter.norm.bias'],1e-5)
            z=F.linear(z,w['adapter.down.weight'],w['adapter.down.bias'])
            z=F.gelu(z,approximate='none')
            x=x+F.linear(z,w['adapter.up.weight'],w['adapter.up.bias'])
        return F.linear(x,w['fu_head.weight'],w['fu_head.bias']).numpy()
def f1(y,p):
    tp=np.sum((y==1)&(p==1));fp=np.sum((y==0)&(p==1));fn=np.sum((y==1)&(p==0))
    return float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.
def selection(row):
    m=row['metrics'];return ((m['safer']['macro_f1']+m['fu']['f1'])/2,m['fu']['auprc'],m['safer']['conditional_fall_vs_lie_auprc'])


def main():
    OUT.mkdir(exist_ok=True)
    cfg=read(ROOT/'configs/joint_document_reconstruction_v1.json')
    fu=ROOT/cfg['fu']['root'];cache=ROOT/cfg['fu']['features']
    assert sha(cache)==cfg['fu']['features_sha256']
    with np.load(cache,allow_pickle=False) as z:x=np.concatenate((z['t'],z['s']),1)
    y=np.load(fu/'labels.npy');actions=np.load(fu/'action_ids.npy');subjects=np.load(fu/'subjects.npy');folds=np.load(fu/'fold_ids.npy')
    assert x.shape==(993,2048) and np.isfinite(x).all()
    assert np.array_equal(y,(actions==5).astype(y.dtype))
    assert np.array_equal(folds,(subjects-1)%5)
    root=ROOT/'checkpoint/fall/JOINT_V3_DOCUMENT_RECONSTRUCTION_20260922_R3/v3/nested'
    results=[];predictions={phase:np.full(993,-1) for phase in ['j0','j1']}
    for k in range(5):
        d=root/f'fold{k}';outer=np.load(d/'outer_indices.npy');lock=read(d/'selection_lock.json')
        assert np.array_equal(outer,np.flatnonzero(folds==k))
        for phase in ['j0','j1']:
            p=d/phase;res=read(p/'result.json');history=read(p/'history.json')
            assert sha(p/'result.json')==lock[phase+'_result_sha256']
            assert sha(p/'history.json')==res['history_sha256']
            assert max(history,key=selection)==res['best']
            train=np.array(res['train_indices']);val=np.array(res['val_indices'])
            assert np.array_equal(train,np.flatnonzero((folds!=k)&(folds!=(k+1)%5)))
            assert np.array_equal(val,np.flatnonzero(folds==(k+1)%5))
            sets=[set(subjects[ix].tolist()) for ix in [train,val,outer]]
            assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
            epoch=res['best']['epoch'];weight=p/f'epoch_{epoch:03d}.pt'
            assert sha(weight)==res['best']['weight_sha256']
            w=torch.load(weight,map_location='cpu',weights_only=True)
            replay=forward(x[outer],w,phase);saved=np.load(d/f'{phase}_outer_logits.npy')
            assert sha(d/f'{phase}_outer_logits.npy')==read(d/'outer_result.json')['results'][phase]['logits_sha256']
            np.testing.assert_allclose(replay,saved,atol=1e-4,rtol=1e-5)
            assert np.array_equal(replay.argmax(1),saved.argmax(1))
            replay_val=forward(x[val],w,phase);saved_val=np.load(p/f'fu_{epoch:03d}.npy')
            np.testing.assert_allclose(replay_val,saved_val,atol=1e-4,rtol=1e-5)
            predictions[phase][outer]=replay.argmax(1)
            results.append(dict(fold=k,phase=phase,epoch=epoch,weight_sha256=sha(weight),
                train_subjects=sorted(sets[0]),validation_subjects=sorted(sets[1]),outer_subjects=sorted(sets[2]),
                max_abs_error=float(np.abs(replay-saved).max()),argmax_exact=True,
                inner_selection_recomputed=True,f1=f1(y[outer],replay.argmax(1))))
        print('replayed fold',k,flush=True)
    for phase in predictions:
        idx=np.load(root/f'{phase}_oof_indices.npy');z=np.load(root/f'{phase}_oof_logits.npy')
        assert np.array_equal(np.sort(idx),np.arange(993))
        assert np.array_equal(predictions[phase][idx],z.argmax(1))
    # Exact, unpadded clip fingerprints: cross-subject duplicates can invalidate OOF.
    data=np.load(fu/'data_joint.npy',mmap_mode='r');lengths=np.load(fu/'num_frame.npy')
    fingerprints={};duplicate_groups=[]
    for n,l in enumerate(lengths):
        h=hashlib.sha256(np.ascontiguousarray(data[n,:,:int(l)]).tobytes()).hexdigest()
        fingerprints.setdefault(h,[]).append(n)
    duplicate_groups=[v for v in fingerprints.values() if len(v)>1]
    cross_subject=[v for v in duplicate_groups if len(set(subjects[v]))>1]
    # Conditional paired subject bootstrap; does not estimate training-seed variation.
    rng=np.random.default_rng(20261002);ids=np.unique(subjects);groups={s:np.flatnonzero(subjects==s) for s in ids}
    delta=[]
    for _ in range(2000):
        idx=np.concatenate([groups[s] for s in rng.choice(ids,len(ids),replace=True)])
        delta.append(f1(y[idx],predictions['j1'][idx])-f1(y[idx],predictions['j0'][idx]))
    paired=dict(observed_difference=f1(y,predictions['j1'])-f1(y,predictions['j0']),
        percentile95=list(map(float,np.quantile(delta,[.025,.975]))),resamples=2000,seed=20261002,
        unit='subject; paired fixed OOF predictions',posthoc=True,
        limitation='conditional diagnostic, not training-seed uncertainty or causal adapter estimate')
    report=dict(passed=not cross_subject,feature_sha256=sha(cache),fold_replays=results,
        max_abs_error=max(x['max_abs_error'] for x in results),exact_duplicate_groups=duplicate_groups,
        cross_subject_duplicate_groups=cross_subject,paired_bootstrap=paired,
        f1={p:f1(y,predictions[p]) for p in predictions},
        boundaries=['Does not prove intent or rule out all data leakage',
                    'Replays weights from cached features; encoder extraction checked separately',
                    'Selection timestamps and no prior holdout exposure cannot be proven from self-authored logs',
                    'Historical Le2i raw predictions are unavailable'])
    (OUT/'fu_checkpoint_replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='fold_replays'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
