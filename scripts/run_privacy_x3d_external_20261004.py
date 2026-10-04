"""Run untouched author test/evaluation programs after explicit UDA-RGB approval."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,json,subprocess,sys,hashlib,pickle,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'third_party/original_privacy_x3d_20261004'
OUT=ROOT/'data/fall_processed/RGB/privacy_x3d_external_20261004_r1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def run():
    p=argparse.ArgumentParser();p.add_argument('--approval-receipt',type=Path,required=True);args=p.parse_args()
    approval=read(args.approval_receipt)
    assert approval['scope']=='Original Privacy X3D UDA checkpoint on RGB external videos; not source-only RGB'
    assert approval['user_confirmed'] is True and approval['user_reply']
    contract=read(OUT/'prepared_contract.json')
    for name,h in contract['files'].items():assert sha(ROOT/name)==h,name
    files=contract['files']|{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str(args.approval_receipt.resolve().relative_to(ROOT)):sha(args.approval_receipt)}
    frozen=dict(contract,files=files,status='approved; frozen before inference')
    if (OUT/'contract.json').exists():assert read(OUT/'contract.json')==frozen
    else:save(OUT/'contract.json',frozen)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(REPO),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    summaries={};rows=[]
    plan=read(OUT/'input_manifest.json')
    for scope in ['le2i130','cauca100']:
        out=OUT/(scope+'_results.pkl');cfg=ROOT/'configs'/('privacy_x3d_external_20261004_r1_'+scope+'.py')
        assert not out.exists(),'Preserve prior inference; use new revision if necessary'
        cmd=[sys.executable,str(REPO/'tools/test.py'),str(cfg),str(REPO/'checkpoints/best_top1_acc_epoch_239.pth'),'--out',str(out)]
        start=time.monotonic()
        with (OUT/(scope+'_author_test.log')).open('w') as log:subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        # Execute the author's scoring program as well; preserve rounding differences.
        cmd=[sys.executable,str(REPO/'own_scipt/evaluation/evaluation.py'),str(out),str(OUT/(scope+'_labels.txt'))]
        with (OUT/(scope+'_author_metrics.log')).open('w') as log:r=subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        with out.open('rb') as f:logits=np.asarray(pickle.load(f))
        subset=[x for x in plan if x['scope']==scope];assert logits.shape==(len(subset),2) and np.isfinite(logits).all()
        prob=torch.softmax(torch.tensor(logits,dtype=torch.float32),dim=-1)[:,1].numpy();pred=prob>.5;y=np.array([x['label'] for x in subset]);tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tn=int(((y==0)&~pred).sum())
        summaries[scope]=dict(videos=len(subset),processed=len(subset),tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/(2*tp+fp+fn) if tp else 0.,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn),accuracy=(tp+tn)/len(subset),author_metrics_exit=r.returncode,seconds=time.monotonic()-start)
        for item,logit,pv,label in zip(subset,logits,prob,pred):rows.append(dict(id=item['id'],scope=scope,model='privacy_x3d_uda_rgb',processed=True,video_prediction=int(label),fall_probability=float(pv),logits=logit.tolist(),ground_truth=item['label']))
        print(scope,summaries[scope],flush=True)
    for name,h in files.items():assert sha(ROOT/name)==h,name
    save(OUT/'evaluation.json',dict(unit='whole video only; no event timestamps',summaries=summaries,rows=rows,contract_sha256=sha(OUT/'contract.json')))
if __name__=='__main__':run()
