"""Audit original author outputs and replay fixed representative videos."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,pickle,re,sys,hashlib
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,roc_auc_score
from verify_privacy_x3d_20261004 import load,ROOT,REPO
import mmcv
from mmaction.datasets import build_dataset
OUT=ROOT/'data/fall_processed/RGB/privacy_x3d_external_20261004_r1'
HFD=ROOT/'data/fall_processed/RGB/hfd_reproduction_20261004_r1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit():
    contract=read(OUT/'contract.json')
    for p,h in contract['files'].items():assert sha(ROOT/p)==h,p
    result=read(OUT/'evaluation.json');manifest=read(OUT/'input_manifest.json');truth=read(OUT/'ground_truth.json')
    assert read(OUT/'plan.json')==read(HFD/'plan.json') and truth==read(HFD/'ground_truth.json')
    assert read(OUT/'own_predictions.json')==read(HFD/'own_predictions.json')
    assert len(result['rows'])==230 and len({r['id'] for r in result['rows']})==230
    gt={r['id']:int(bool(r['episodes'])) for r in truth};summaries={};replays=[]
    model,cfg,checkpoint=load()
    assert all(torch.equal(model.state_dict()[k].cpu(),v.cpu()) for k,v in checkpoint['state_dict'].items())
    for scope in ['le2i130','cauca100']:
        entries=[r for r in manifest if r['scope']==scope];rows=[r for r in result['rows'] if r['scope']==scope]
        with (OUT/(scope+'_results.pkl')).open('rb') as f:logits=np.asarray(pickle.load(f))
        y=np.array([gt[r['id']] for r in entries]);assert logits.shape==(len(y),2) and np.isfinite(logits).all()
        assert [r['id'] for r in rows]==[r['id'] for r in entries]
        stored=np.array([r['logits'] for r in rows]);assert np.array_equal(logits,stored)
        # Binary argmax of averaged logits independently reproduces softmax>0.5.
        pred=(logits[:,1]>logits[:,0]).astype(int);assert pred.tolist()==[r['video_prediction'] for r in rows]
        assert [gt[r['id']] for r in rows]==[r['ground_truth'] for r in rows]
        tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
        f1=float(f1_score(y,pred,zero_division=0));acc=float(accuracy_score(y,pred))
        summary=result['summaries'][scope]
        for k,v in dict(tp=tp,fp=fp,fn=fn,tn=tn,f1=f1,accuracy=acc).items():assert np.isclose(v,summary[k]),(scope,k,v)
        # Validate all author rounded output fields, including its round-before-F1 rule.
        p32=torch.softmax(torch.tensor(logits,dtype=torch.float32),dim=-1)[:,1].numpy()
        shifted=logits.astype(np.float64)-logits.max(axis=1,keepdims=True);ex=np.exp(shifted);prob=ex[:,1]/ex.sum(axis=1)
        assert np.max(np.abs(prob-p32))<1e-6
        precision=round(tp/(tp+fp),4) if tp+fp else 0.;recall=round(tp/(tp+fn),4)
        expected=dict(accuracy=round(acc,4),precision=precision,recall=recall,F1_score=round(2*precision*recall/(precision+recall),4) if precision+recall else 0.,auc=round(roc_auc_score(y,p32),4))
        log=(OUT/(scope+'_author_metrics.log')).read_text()
        assert summary['author_metrics_exit']==0,log
        for key,value in expected.items():
            match=re.search(r'\b'+key+r':?\s+([\d.]+)',log);assert match and float(match[1])==value,(key,value,log)
        summaries[scope]=dict(tp=tp,fp=fp,fn=fn,tn=tn,f1=f1,accuracy=acc,author_rounded=expected)
        # Fixed first/last plus closest decision boundary; no parameter changes.
        indices=sorted({0,len(entries)-1,int(np.argmin(np.abs(logits[:,1]-logits[:,0])))})
        testcfg=mmcv.Config.fromfile(str(ROOT/'configs'/('privacy_x3d_external_20261004_r1_'+scope+'.py')))
        dataset=build_dataset(testcfg.data.test,dict(test_mode=True))
        for i in indices:
            data=dataset[i];imgs=data['imgs'];assert hashlib.sha256(imgs.numpy().tobytes()).hexdigest()==entries[i]['input_tensor_sha256']
            with torch.no_grad():actual=model(imgs.unsqueeze(0).cuda(),domain_label=torch.ones(1,device='cuda'),return_loss=False)[0]
            difference=float(np.max(np.abs(actual-logits[i])))
            assert np.allclose(actual,logits[i],atol=1e-5,rtol=1e-5),(entries[i]['id'],difference)
            assert int(actual[1]>actual[0])==int(pred[i])
            replays.append(dict(id=entries[i]['id'],scope=scope,max_logit_difference=difference,input_tensor_hash_match=True))
        print('audited',scope,summaries[scope],flush=True)
    # Official source files, config, images and weights remain unchanged after replay.
    for p,h in contract['files'].items():assert sha(ROOT/p)==h,p
    report=dict(passed=True,videos=230,strict_tensors=len(checkpoint['state_dict']),all_contract_files_verified=len(contract['files']),
      same230_video_list_and_ground_truth=True,all_decisions_recomputed=True,author_metrics_verified=True,
      replays=replays,summaries=summaries,official_results_sha256={scope:sha(OUT/(scope+'_results.pkl')) for scope in summaries},
      evaluation_sha256=sha(OUT/'evaluation.json'),no_training_or_target_tuning=True)
    (OUT/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('audit passed',flush=True)
if __name__=='__main__':audit()
