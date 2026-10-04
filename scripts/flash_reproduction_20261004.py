"""Source-only FLASH reproduction. Original network, explicitly corrected data IO.

Training never opens target videos, labels, or predictions. See frozen config.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'third_party/original_flash_20261004'
OUT = ROOT / 'data/fall_processed/FLASH/flash_reproduction_20261004_r1'
EVIDENCE = ROOT / 'docs/internal/2026-10-04_flash_reproduction_evidence'
CONFIG = ROOT / 'configs/flash_reproduction_20261004_r1.json'
sys.path.insert(0, str(REPO))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(p)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def status(stage, **kwargs):
    row = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **kwargs)
    save(OUT / 'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def prepare():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from HGCN.hypergraph import Hypergraph
    from HGCN.data_processing import FallDataLoader
    import joblib

    require(not (OUT / 'source_contract.json').exists(), 'source already frozen')
    OUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    manifest, xs, ys, files = [], [], [], {}
    previous = read(EVIDENCE.parent / '2026-10-04_fall_specialized_baselines_evidence/source_archive_audit.json')
    hashes = {r['file']: r['sha256'] for r in previous}
    for archive in sorted((ROOT / 'data/source_archives/ImpactUPFall_20261004_r1').glob('*.zip')):
        rel = str(archive.relative_to(ROOT))
        require(sha(archive) == hashes[rel], 'source archive changed')
        files[rel] = hashes[rel]
        with zipfile.ZipFile(archive) as z:
            for member in sorted(z.namelist()):
                if not member.endswith('.csv'):
                    continue
                raw = z.read(member)
                df = pd.read_csv(io.BytesIO(raw))
                require(df.shape == (100, 100), 'source shape')
                expected = [f'Joint{j}_{c}' for j in range(1, 34) for c in ('X', 'Y', 'Z')]
                require(df.columns[:99].tolist() == expected, 'source coordinate header')
                a = df.to_numpy(dtype=np.float32)
                require(np.isfinite(a).all() and set(np.unique(a[:, -1])) <= {0, 1}, 'source values')
                xs.append(a[:, :99].reshape(100, 33, 3))
                ys.append(a[:, -1])
                manifest.append(dict(id=Path(member).stem, archive=rel, member=member,
                                     sha256=hashlib.sha256(raw).hexdigest()))
    x, y = np.stack(xs), np.stack(ys)
    require(len(x) == 82 and len({m['id'] for m in manifest}) == 82, 'source membership')
    idx = np.arange(len(x))
    train_val, test = train_test_split(idx, test_size=.1, random_state=42, shuffle=True, stratify=y[:, -1])
    train, val = train_test_split(train_val, test_size=.111, random_state=42, shuffle=True,
                                 stratify=y[train_val, -1])
    sets = {'train': train, 'validation': val, 'test': test}
    scaler = StandardScaler().fit(x[train].reshape(-1, 99))
    h = Hypergraph(num_node=33).H_norm.numpy()
    normalized = scaler.transform(x.reshape(-1, 99)).astype(np.float32).reshape(x.shape)
    transformed = np.einsum('ij,btjc->btic', h, normalized).astype(np.float32)
    # Preserve official three augmentations; use the SAME time indices for labels.
    np.random.seed(42)
    loader = object.__new__(FallDataLoader)
    aug_x, aug_y, aug_records = [], [], []
    for i in train:
        temporal_index = loader.augment_motion_sequence_fixed_length(np.arange(100)[:, None]).ravel()
        require(np.all(np.diff(temporal_index) >= 0), 'augmentation reverses time')
        aug_x.extend([transformed[i, temporal_index], loader.rotate_skeleton(transformed[i]),
                      loader.scale_skeleton(transformed[i])])
        aug_y.extend([y[i, temporal_index], y[i], y[i]])
        aug_records.append(dict(source=int(i), temporal_indices=temporal_index.tolist()))
    np.savez_compressed(OUT / 'source.npz', x=x, y=y, transformed=transformed,
                        train=train, validation=val, test=test,
                        train_x=np.concatenate([transformed[train], np.asarray(aug_x, np.float32)]),
                        train_y=np.concatenate([y[train], np.asarray(aug_y, np.float32)]),
                        mean=scaler.mean_, scale=scaler.scale_, graph=h)
    joblib.dump(scaler, OUT / 'sc1.pkl')
    for split, inds in sets.items():
        for i in inds:
            manifest[int(i)]['split'] = split
    save(OUT / 'source_manifest.json', manifest)
    save(OUT / 'augmentation_manifest.json', aug_records)
    config = dict(experiment_id='FLASH_REPRODUCTION_20261004_R1',
        provenance='author official network; source-only retraining with documented data-loader corrections',
        author_commit='04a1215314472085aea68009159c5f9237e9e7cf',
        paper='https://arxiv.org/html/2607.25791v1',
        source='https://zenodo.org/records/12773013', seed=42,
        input={'frames':100,'joints':33,'channels':3,'columns':'all 99 CSV coordinates in stored order; no remapping',
               'normalization':'train frames only StandardScaler; frozen for all other inputs',
               'graph_preprocessing':'official HGCN.hypergraph.Hypergraph H_norm (26 hyperedges), preserved',
               'model_graph':'official HGCN.hypergraph_ablation.HypergraphAblation full (6 hyperedges), preserved'},
        split={'method':'official two-stage sklearn random sequence split, applied BEFORE augmentation',
               'test_size':.1,'validation_of_remaining':.111,'counts':{k:len(v) for k,v in sets.items()},
               'subject_independent':False,'pairwise_camera_independent':False},
        optimizer={'name':'AdamW','lr':1e-4,'weight_decay':1e-5,'betas':[.9,.999],'eps':1e-8},
        training={'max_epochs':300,'batch_size':32,'microbatch_size':'set by source-only memory probe before training',
                  'loss':'BCEWithLogitsLoss mean','gradient_clip':.5,'early_stopping_patience':15,
                  'checkpoint':'minimum validation BCE','threshold_logit':0.,'precision':'float32',
                  'train_drop_last':True,'validation_test_drop_last':False},
        scheduler={'name':'ReduceLROnPlateau','factor':.5,'patience':7,'min_lr':1e-6},
        corrections=['coordinate slices j:j+3 replaced by all XYZ columns (equivalent 3*j:3*j+3)',
                     'split original sequences before augmentation; no augmented source in validation/test',
                     'fit scaler on source training only','warp time labels together with skeleton',
                     'keep partial evaluation batches'],
        limitations=['public source82 sequences all include impact, no independent ADL clip',
                     'published split membership not available; new recorded seed42 split',
                     'official graph indices retained, their anatomical naming not independently verified',
                     'original network 74966881 parameters differs from paper table37.1M; no architecture alteration',
                     'source reproduction metrics not directly identical to paper protocol'],
        device='physical GPU0 only',target_used_for_training=False)
    save(CONFIG, config)
    source_report = dict(sequences=len(x), frames=int(y.size), labels={str(k):int((y==k).sum()) for k in (0,1)},
        counts={k:len(v) for k,v in sets.items()}, training_after_augmentation=len(train)*4,
        overlaps={a+'_'+b:len(set(sets[a])&set(sets[b])) for a,b in [('train','validation'),('train','test'),('validation','test')]},
        scaler_fitted_on_frames=len(train)*100,columns_used=99,temporal_order_preserved=True)
    save(EVIDENCE/'source_preparation.json',source_report)
    for p in [Path(__file__), CONFIG, OUT/'source.npz', OUT/'sc1.pkl', OUT/'source_manifest.json',
              OUT/'augmentation_manifest.json', *REPO.glob('HGCN/*.py'), REPO/'train.py']:
        files[str(p.relative_to(ROOT))] = sha(p)
    save(OUT/'source_contract.json',dict(files=files,created=datetime.now(timezone.utc).isoformat()))
    status('source_prepared', **source_report)


def check_contract():
    for p,digest in read(OUT/'source_contract.json')['files'].items():
        require(sha(ROOT/p)==digest,'source contract changed: '+p)


def setup():
    import numpy as np
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','GPU0 restriction')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'GPU0 not available')
    torch.set_num_threads(4)
    np.random.seed(42); random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark=True
    from HGCN.hmamba import HyperMamba
    return HyperMamba(device='cuda').cuda()


def probe():
    import numpy as np
    import torch
    check_contract()
    require(not (OUT/'best.pt').exists(),'probe after training refused')
    m=setup()
    with np.load(OUT/'source.npz') as z:
        x=torch.tensor(z['train_x'][:4],device='cuda');y=torch.tensor(z['train_y'][:4],device='cuda')
    optimizer=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-5)
    torch.cuda.reset_peak_memory_stats();t=time.monotonic()
    m.train();p=m(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(p,y)
    require(torch.isfinite(loss).item(),'nonfinite source probe')
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),.5);optimizer.step()
    torch.cuda.synchronize()
    result=dict(batch_size=4,seconds=time.monotonic()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                parameters=sum(p.numel() for p in m.parameters()),loss=float(loss.detach()),passed=True,
                purpose='source-only memory/correctness probe; discard temporary weights; restart seed42 for training')
    save(EVIDENCE/'memory_probe.json',result);print(json.dumps(result),flush=True)


def frame_metrics(y,logits):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y).reshape(-1).astype(bool); p=np.asarray(logits).reshape(-1)>0
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,accuracy=(tp+tn)/len(y),precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                auc=float(roc_auc_score(y,np.asarray(logits).reshape(-1))) if len(np.unique(y))==2 else None)


def train():
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset,DataLoader
    check_contract();require(not (OUT/'best.pt').exists(),'existing training not overwritten')
    probe_result=read(EVIDENCE/'memory_probe.json');require(probe_result['passed'],'no verified microbatch')
    micro=4
    save(OUT/'execution_contract.json',dict(source_contract_sha256=sha(OUT/'source_contract.json'),microbatch=micro,
         effective_batch=32,precision='fp32',gradient_accumulation='micro loss weighted by micro/effective batch; clip and step once',
         torch=torch.__version__,gpu=torch.cuda.get_device_name(0),created=datetime.now(timezone.utc).isoformat()))
    model=setup()
    with np.load(OUT/'source.npz') as z:
        train_x=torch.tensor(z['train_x']);train_y=torch.tensor(z['train_y'])
        source_x=torch.tensor(z['transformed']);source_y=torch.tensor(z['y']);val=z['validation'];test=z['test']
    loader=DataLoader(TensorDataset(train_x,train_y),batch_size=32,shuffle=True,drop_last=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=7,min_lr=1e-6)
    def evaluate(indices):
        model.eval();pred=[];loss_sum=0.
        with torch.no_grad():
            for start in range(0,len(indices),micro):
                ii=indices[start:start+micro];yy=source_y[ii].cuda();out=model(source_x[ii].cuda())
                require(torch.isfinite(out).all().item(),'nonfinite source evaluation')
                loss_sum+=float(torch.nn.functional.binary_cross_entropy_with_logits(out,yy,reduction='sum'))
                pred.append(out.cpu().numpy())
        return np.concatenate(pred),loss_sum/(len(indices)*100)
    best=float('inf');best_epoch=None;patience=0;history=[];started=time.monotonic()
    for epoch in range(1,301):
        et=time.monotonic();model.train();total_loss=0.;correct=0;count=0
        for bx,by in loader:
            optimizer.zero_grad(set_to_none=True)
            for start in range(0,len(bx),micro):
                xx=bx[start:start+micro].cuda();yy=by[start:start+micro].cuda()
                logits=model(xx);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,yy)
                require(torch.isfinite(loss).item(),'nonfinite training loss')
                (loss*(len(xx)/len(bx))).backward()
                total_loss+=float(loss.detach())*yy.numel();correct+=int(((logits>0)==yy.bool()).sum());count+=yy.numel()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            require(torch.isfinite(grad).item(),'nonfinite gradient')
            optimizer.step()
        vp,vl=evaluate(val);scheduler.step(vl)
        row=dict(epoch=epoch,train_loss=total_loss/count,train_accuracy=correct/count,val_loss=vl,
                 validation=frame_metrics(source_y[val].numpy(),vp),lr=optimizer.param_groups[0]['lr'],
                 seconds=time.monotonic()-et)
        history.append(row)
        if vl<best:
            best=vl;best_epoch=epoch;patience=0
            payload=dict(model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                         epoch=epoch,validation_loss=vl,source_contract_sha256=sha(OUT/'source_contract.json'),
                         execution_contract_sha256=sha(OUT/'execution_contract.json'),author_commit=read(CONFIG)['author_commit'])
            torch.save(payload,OUT/'best.tmp.pt');(OUT/'best.tmp.pt').replace(OUT/'best.pt')
        else:
            patience+=1
        save(OUT/'history.json',history)
        status('training',**row,best_epoch=best_epoch,patience_used=patience,elapsed_seconds=time.monotonic()-started)
        if patience>=15:
            break
    # Selection is over. The source test is read once with the frozen selected model.
    cp=torch.load(OUT/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(cp['model_state_dict'],strict=True)
    vp,vl=evaluate(val);tp,tl=evaluate(test)
    np.savez_compressed(OUT/'source_predictions.npz',validation_indices=val,validation_logits=vp,
                        validation_labels=source_y[val].numpy(),test_indices=test,test_logits=tp,test_labels=source_y[test].numpy())
    result=dict(passed=True,epochs=len(history),best_epoch=best_epoch,validation_loss=vl,test_loss=tl,
                validation=frame_metrics(source_y[val].numpy(),vp),test=frame_metrics(source_y[test].numpy(),tp),
                checkpoint_sha256=sha(OUT/'best.pt'),scaler_sha256=sha(OUT/'sc1.pkl'),
                source_predictions_sha256=sha(OUT/'source_predictions.npz'),seconds=time.monotonic()-started,
                target_data_opened=False,original_paper_reproduction_claim=False)
    save(OUT/'source_evaluation.json',result);check_contract();status('source_training_complete',**result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','probe','train'])
    args=parser.parse_args();globals()[args.stage]()
