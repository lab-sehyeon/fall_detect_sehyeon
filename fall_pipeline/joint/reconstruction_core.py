"""Explicit document-reconstruction J0/J1 math, sampling and metrics."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from fall_pipeline.joint.models import JointFallModel
from fall_pipeline.safer.train_f0b_safer_heads import classification_metrics


def model_for(stage,config,device,initial=None):
    torch.manual_seed(config['training']['seed'])
    model=JointFallModel(config['model']['feature_dim'],config['model']['bottleneck'],config['model']['dropout'])
    if stage=='j0':
        model.adapter=nn.Identity()
        for head in (model.safer_head,model.fu_head): nn.init.normal_(head.weight,0,.01);nn.init.zeros_(head.bias)
    else:
        if initial is None: raise ValueError('J1 requires locked J0')
        for name in ('safer_head','fu_head'):
            getattr(model,name).load_state_dict({k.removeprefix(name+'.'):v for k,v in initial.items() if k.startswith(name+'.')},strict=True)
    return model.to(device)


def optimizer_for(model,stage,config):
    p=config['training']
    groups=[{'params':list(model.safer_head.parameters())+list(model.fu_head.parameters()),'lr':p['j0_lr'] if stage=='j0' else p['head_lr']}]
    if stage=='j1': groups.insert(0,{'params':model.adapter.parameters(),'lr':p['adapter_lr']})
    optimizer=torch.optim.AdamW(groups,betas=tuple(p['betas']),eps=p['eps'],weight_decay=p['weight_decay'])
    expected={id(v) for v in model.parameters() if v.requires_grad}
    actual={id(v) for g in optimizer.param_groups for v in g['params']}
    if expected!=actual: raise RuntimeError('optimizer scope')
    return optimizer


def weights(labels,classes):
    count=torch.bincount(labels,minlength=classes).float().clamp_min(1)
    value=count.rsqrt();return value/value.mean()


def split_indices(folds,subjects,k,nested):
    val_fold=(k+1)%5 if nested else k
    train=np.flatnonzero((folds!=k)&(folds!=val_fold));val=np.flatnonzero(folds==val_fold)
    outer=np.flatnonzero(folds==k) if nested else np.empty(0,np.int64)
    groups=[set(subjects[x]) for x in (train,val,outer)]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)): raise RuntimeError('subject leakage')
    if len(train)+len(val)+len(outer)!=len(folds): raise RuntimeError('fold coverage')
    return train,val,outer


def paired_batches(safer_count,fu_train,epoch,config):
    p=config['training'];sg=torch.Generator().manual_seed(p['seed']+2*epoch)
    fg=torch.Generator().manual_seed(p['seed']+2*epoch+1)
    order=torch.randperm(safer_count,generator=sg);fu_train=torch.as_tensor(fu_train,dtype=torch.long)
    if not len(fu_train): raise ValueError('empty FU train')
    fu_order=torch.empty(0,dtype=torch.long);cursor=0
    for start in range(0,safer_count,p['safer_batch_size']):
        chunks=[];remaining=p['fu_batch_size']
        while remaining:
            if cursor==len(fu_order): fu_order=fu_train[torch.randperm(len(fu_train),generator=fg)];cursor=0
            take=min(remaining,len(fu_order)-cursor);chunks.append(fu_order[cursor:cursor+take]);cursor+=take;remaining-=take
        yield order[start:start+p['safer_batch_size']],torch.cat(chunks)


def train_epoch(model,optimizer,safer,fu,fu_train,epoch,stage,config):
    p=config['training'];factor=p['gamma']**sum(epoch-1>=m for m in p[stage+'_schedule'])
    bases=[p['j0_lr']] if stage=='j0' else [p['adapter_lr'],p['head_lr']]
    for group,base in zip(optimizer.param_groups,bases): group['lr']=base*factor
    model.train();sy=safer['y'];fy=fu['y'];device=sy.device
    sw=weights(sy,4);fw=weights(fy[fu_train],2);total=np.zeros(2);steps=0
    for si,fi in paired_batches(len(sy),fu_train,epoch,config):
        si=si.to(device);fi=fi.to(device);optimizer.zero_grad(set_to_none=True)
        sl=nn.functional.cross_entropy(model(safer['x'][si],'safer'),sy[si],weight=sw)
        fl=nn.functional.cross_entropy(model(fu['x'][fi],'fu'),fy[fi],weight=fw)
        (sl+fl).backward();optimizer.step()
        total+=np.array([sl.item(),fl.item()]);steps+=1
    if not steps: raise RuntimeError('no training steps')
    return {'safer':float(total[0]/steps),'fu':float(total[1]/steps),'steps':steps}


def predict(model,x,dataset,batch):
    model.eval();parts=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch): parts.append(model(x[start:start+batch],dataset).cpu().numpy())
    return np.concatenate(parts)


def binary_metrics(labels,actions,logits):
    labels=np.asarray(labels);logits=np.asarray(logits)
    if logits.shape!=(len(labels),2) or not np.isfinite(logits).all(): raise ValueError('binary logits')
    pred=logits.argmax(1);score=torch.softmax(torch.from_numpy(logits).double(),1).numpy()[:,1]
    cm=confusion_matrix(labels,pred,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel());lying=np.asarray(actions)==4
    return {'count':len(labels),'f1':float(f1_score(labels,pred,zero_division=0)),
            'auprc':float(average_precision_score(labels,score)) if len(np.unique(labels))==2 else None,
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,'lying_fp':int(np.sum((pred==1)&lying)),
            'lying_count':int(lying.sum()),'lying_fpr':float(np.mean(pred[lying]==1)) if lying.any() else None}


def evaluation(model,safer_val,fu,fu_val,config):
    batch=config['training']['eval_batch_size']
    sl=predict(model,safer_val['x'],'safer',batch);fl=predict(model,fu['x'][fu_val],'fu',batch)
    sm=classification_metrics(safer_val['y'].cpu().numpy(),sl)
    fm=binary_metrics(fu['y'][fu_val].cpu().numpy(),fu['actions'][fu_val],fl)
    return {'safer':sm,'fu':fm},sl,fl


def selection_key(row):
    metrics=row['metrics'];s=metrics['safer'];f=metrics['fu']
    return ((s['macro_f1']+f['f1'])/2, f['auprc'] if f['auprc'] is not None else -float('inf'),
            s['conditional_fall_vs_lie_auprc'] if s['conditional_fall_vs_lie_auprc'] is not None else -float('inf'))


def epoch0_exact(j0,j1,safer_val,fu,indices,config):
    batch=config['training']['eval_batch_size']
    for dataset,x in (('safer',safer_val['x']),('fu',fu['x'][indices])):
        a=predict(j0,x,dataset,batch);b=predict(j1,x,dataset,batch)
        if not np.array_equal(a,b): raise RuntimeError('J1 epoch0 not exact J0')
    return {'passed':True,'scope':'all SAFER validation and designated FU selection clips; outer fold unopened'}
