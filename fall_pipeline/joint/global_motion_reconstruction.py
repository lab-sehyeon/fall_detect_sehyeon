"""Document-based matched G0/G1/G2 training with frozen reconstructed final J1-V3."""
import argparse
from datetime import datetime,timezone
import fcntl
import gc
import os
from pathlib import Path
import sys
import numpy as np
import torch

from data_gen import global_motion_reconstruction as data
from fall_pipeline.joint import reconstruction_inputs as inputs
from fall_pipeline.joint.models import JointFallModel
from fall_pipeline.safer import train_f0b_safer_heads as metrics
from fall_pipeline.safer import v2_controls_reconstruction as io

ROOT,CONFIG=data.ROOT,data.CONFIG
VARIANTS=('G0','G1','G2')
CODE=data.CODE+['fall_pipeline/joint/global_motion_reconstruction.py','fall_pipeline/joint/models.py',
    'fall_pipeline/joint/reconstruction_inputs.py','fall_pipeline/safer/train_f0b_safer_heads.py',
    'fall_pipeline/safer/v2_controls_reconstruction.py','fall_pipeline/common/integrity.py']


def publish(root,stage,**fields):
    state=dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),**fields)
    io.save_json(root/'status.json',state);print(io.json.dumps(state,ensure_ascii=False),flush=True)
    if root.name.endswith('_smoke'):return
    status='completed' if stage=='completed' else 'paused' if stage in ('failed','paused') else 'in_progress'
    header=f'- 문서 ID: `DOC-20260928-global-motion-run-R1`\n- 기준일: 2026-09-28\n- 상태: `{status}`\n'
    words={'raw':'전역 움직임 원자료 생성·검산','raw_completed':'497개 원자료 검사 완료',
           'cache':'전체 시간축 특징 생성·검산','source':'입력 동일성 검사','representation':'고정 J1 특징 추출',
           'train':'G0/G1/G2 동일 조건 학습','selection':'validation 선택 고정',
           'evaluate':'고정 모델 평가','audit':'독립 검산','completed':'학습·평가·검산 완료',
           'failed':'검증 실패로 중단','paused':'안전 경계에서 일시 중단'}
    internal=f'# Global 모델 실행 기록\n\n{header}\n```json\n{io.json.dumps(state,indent=2,ensure_ascii=False)}\n```\n\nroot: `{root}`\n\n[계약](2026-09-28_global_motion_training_internal.md)\n'
    shared=f'# Global 모델 학습 진행\n\n{header}\n현재 단계: {words.get(stage,stage)}.\n'
    if 'epoch' in fields:shared+=f'완료 epoch: {fields["epoch"]}/50.\n'
    if 'split' in fields:shared+=f'대상 split: {fields["split"]}.\n'
    if 'sequences' in fields:shared+=f'확인 sequence: {fields["sequences"]}/{fields["total"]}.\n'
    shared+='\n[연구 방법](2026-09-28_global_motion_training_shared.md). 원본과 동일한 수치의 복원이나 외부 배포 성능을 뜻하지 않는다.\n'
    for kind,text in [('internal',internal),('shared',shared)]:
        path=ROOT/f'docs/{kind}/2026-09-28_global_motion_run_{kind}.md';temp=path.with_suffix('.tmp')
        temp.write_text(text);os.replace(temp,path)


def source_contract(config,smoke):
    alignment=ROOT/'data/source_archives/rgb_alignment_20260928_r1/final_report.json'
    aligned=io.read(alignment)
    io.require(aligned['passed'] and aligned['oops_videos']==818 and aligned['safer_ood_sequences']==30,
               'data alignment audit required')
    checks={ROOT/config['controls_root']/'final_report.json':config['controls_report_sha256'],
            ROOT/config['joint_root']/'final_report.json':config['joint_report_sha256'],
            ROOT/config['joint_root']/'final/final_j1.pt':config['j1_weight_sha256'],
            ROOT/config['sequence_root']/'manifest.json':config['sequence_manifest_sha256']}
    old=io.read(ROOT/'configs/safer_v2_controls_document_reconstruction_v1.json')
    checks[ROOT/old['encoder']]=old['encoder_sha256']
    checks[ROOT/old['adl_root']/'best_adl_head.pth']=old['adl_head_sha256']
    for path,digest in checks.items():io.require(io.sha256_file(path)==digest,'frozen source changed: '+path.name)
    for file in (ROOT/config['controls_root']/'final_report.json',ROOT/config['joint_root']/'final_report.json'):
        io.require(io.read(file)['passed'],'parent not validated')
    sources={}
    for split in config['split_counts']:
        path=ROOT/config['controls_root']/'cache'/split/'manifest.json'
        sources[split]=dict(path=str(path.parent),manifest=str(path),sha256=io.sha256_file(path))
    prepared=dict(lineage='v3',data_root=str(ROOT/config['sequence_root']),sources=sources,counts=config['split_counts'])
    contract=dict(config_sha256=io.sha256_file(CONFIG),code_sha256={p:io.sha256_file(ROOT/p) for p in CODE},
                  frozen_files={str(p.relative_to(ROOT)):h for p,h in checks.items()},
                  source_manifests={s:v['sha256'] for s,v in sources.items()},smoke=smoke,
                  alignment_report_sha256=io.sha256_file(alignment),
                  torch=str(torch.__version__),numpy=str(np.__version__),historical_exact_reproduction=False)
    return contract,prepared


def scaler_fit(training):
    x=np.asarray(training,np.float64);mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1
    return mean,scale


def adapt_source(config,prepared,split,device,limit=None,locked=False):
    d=inputs.load_safer(prepared,split,locked=locked,limit=limit)
    model=JointFallModel().eval().requires_grad_(False)
    state=torch.load(ROOT/config['joint_root']/'final/final_j1.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    x=d['x'];batch=1024;checks={}
    for begin in sorted(set([0,(len(x)//2//batch)*batch,((len(x)-1)//batch)*batch])):
        with torch.no_grad():checks[begin]=model.adapter(torch.from_numpy(x[begin:begin+batch].copy())).numpy()
    model=model.to(device)
    maximum=0.
    with torch.no_grad():
        for begin in range(0,len(x),batch):
            data.safety(config)
            value=model.adapter(torch.from_numpy(x[begin:begin+batch]).to(device)).cpu().numpy()
            if begin in checks:
                np.testing.assert_allclose(value,checks[begin],atol=config['audit']['cpu_atol'],rtol=config['audit']['cpu_rtol'])
                maximum=max(maximum,float(np.abs(value-checks[begin]).max()))
            x[begin:begin+batch]=value
    del model,state;torch.cuda.empty_cache()
    return dict(x=x,y=d['y'],j1_cpu_max=maximum)


def new_heads(device,seed=0):
    heads=torch.nn.ModuleDict()
    for name,dim in [('G0',2048),('G1',131),('G2',2179)]:
        layer=torch.nn.Linear(dim,4);torch.manual_seed(seed)
        torch.nn.init.normal_(layer.weight,0,.01);torch.nn.init.zeros_(layer.bias)
        heads[name]=layer
    return heads.to(device)


def features(d,name,index):
    if name=='G0':return d['x'][index]
    if name=='G1':return d['g'][index]
    return torch.cat((d['x'][index],d['g'][index]),dim=1)


def predict(head,d,name,batch):
    with torch.no_grad():
        return np.concatenate([head(features(d,name,slice(i,i+batch))).cpu().numpy()
                               for i in range(0,len(d['y']),batch)])


def train_weights(y):
    counts=torch.bincount(y,minlength=4).float().clamp_min(1);w=counts.rsqrt();return w/w.mean()


def to_device(d,device):
    return {k:torch.from_numpy(d[k]).to(device) for k in ('x','g','y')}


def fit(root,train,val,config,contract,device,smoke=False):
    p=config['training'];heads=new_heads(device,p['seed'])
    optimizer=torch.optim.SGD(heads.parameters(),lr=p['lr'],momentum=p['momentum'],weight_decay=p['weight_decay'])
    io.require({id(v) for g in optimizer.param_groups for v in g['params']}=={id(v) for v in heads.parameters()},'optimizer scope')
    cw=train_weights(train['y']);history=[];best={};start=1;path=root/'latest.pt'
    if path.exists():
        io.require(io.sha256_file(path)==io.read(root/'progress.json')['checkpoint_sha256'],'checkpoint commit mismatch')
        state=torch.load(path,map_location='cpu',weights_only=True)
        io.require(state['contract']==contract,'checkpoint contract mismatch')
        heads.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        history,best,start=state['history'],state['best'],state['epoch']+1
        del state
    epochs=1 if smoke else p['epochs']
    for epoch in range(start,epochs+1):
        data.safety(config);heads.train();generator=torch.Generator().manual_seed(p['seed']+epoch)
        order=torch.randperm(len(train['y']),generator=generator);loss_sum=np.zeros(3);steps=0
        for begin in range(0,len(order),p['batch_size']):
            index=order[begin:begin+p['batch_size']].to(device);optimizer.zero_grad(set_to_none=True)
            losses=[torch.nn.functional.cross_entropy(heads[name](features(train,name,index)),train['y'][index],weight=cw)
                    for name in VARIANTS]
            loss=sum(losses);io.require(bool(torch.isfinite(loss)),'nonfinite training loss')
            loss.backward();optimizer.step();loss_sum+=np.array([x.item() for x in losses]);steps+=1
        heads.eval();row=dict(epoch=epoch,loss=(loss_sum/steps).tolist(),variants={})
        for name in VARIANTS:
            dest=root/name;dest.mkdir(exist_ok=True)
            logits=predict(heads[name],val,name,p['eval_batch_size'])
            scored=metrics.classification_metrics(val['y'].cpu().numpy(),logits)
            weight=dest/f'epoch_{epoch:03d}.pt';out=dest/f'val_{epoch:03d}.npy'
            io.save_tensor(weight,io.cpu_state(heads[name]));io.save_array(out,logits)
            info=dict(metrics=scored,weight_sha256=io.sha256_file(weight),logits_sha256=io.sha256_file(out))
            row['variants'][name]=info
            if name not in best or metrics.selection_key(scored)>metrics.selection_key(best[name]['metrics']):
                best[name]=dict(epoch=epoch,**info)
        history.append(row)
        io.save_tensor(path,dict(contract=contract,state_dict=io.cpu_state(heads),optimizer=optimizer.state_dict(),
                                  history=history,best=best,epoch=epoch))
        io.save_json(root/'progress.json',dict(epoch=epoch,checkpoint_sha256=io.sha256_file(path)))
        io.save_json(root/'history.json',history)
        publish(root,'train',epoch=epoch,validation_macro={n:row['variants'][n]['metrics']['macro_f1'] for n in VARIANTS})
    io.require(len(history)==epochs,'epoch count')
    gate=config['training']['adoption_gate'];g0=best['G0']['metrics'];g2=best['G2']['metrics']
    gains=dict(macro=g2['macro_f1']-g0['macro_f1'],
               fall_recall=g2['per_class']['fall']['recall']-g0['per_class']['fall']['recall'],
               lie_down_f1=g2['per_class']['lie_down']['f1']-g0['per_class']['lie_down']['f1'])
    eligible=(not smoke and gains['macro']>=gate['macro_gain_min'] and
              gains['fall_recall']>=-gate['fall_recall_drop_max'] and gains['lie_down_f1']>=-gate['lie_down_f1_drop_max'])
    lock=dict(best=best,epochs=epochs,history_sha256=io.sha256_file(root/'history.json'),
               scaler_sha256=io.sha256_file(root/'train_scaler.npz'),contract=contract,
               validation_gains=gains,validation_adoption_eligible=eligible,holdout_used_for_selection=False)
    lock_path=root/'selection.json'
    if lock_path.exists():io.require(io.read(lock_path)==lock,'selection changed')
    else:io.save_json(lock_path,lock)
    publish(root,'selection',epochs={n:best[n]['epoch'] for n in VARIANTS},validation_adoption_eligible=eligible)
    return lock


def selected_head(root,lock,name,device):
    best=lock['best'][name];path=root/name/f'epoch_{best["epoch"]:03d}.pt'
    io.require(io.sha256_file(path)==best['weight_sha256'],'selected head changed')
    state=torch.load(path,map_location='cpu',weights_only=True)
    head=torch.nn.Linear(state['weight'].shape[1],4).to(device).eval().requires_grad_(False)
    head.load_state_dict(state,strict=True);return head,state


def audit_logits(cpu,name,state,logits,config):
    maximum=0.;batch=config['training']['eval_batch_size']
    for begin in range(0,len(cpu['y']),batch):
        piece=slice(begin,begin+batch)
        x=cpu['x'][piece] if name=='G0' else cpu['g'][piece] if name=='G1' else np.concatenate((cpu['x'][piece],cpu['g'][piece]),1)
        expected=(torch.from_numpy(x)@state['weight'].T+state['bias']).numpy()
        np.testing.assert_allclose(logits[piece],expected,atol=config['audit']['cpu_atol'],rtol=config['audit']['cpu_rtol'])
        maximum=max(maximum,float(np.abs(logits[piece]-expected).max()))
    return maximum


def run(config,root,contract,prepared,device,smoke):
    publish(root,'source');limit=2048 if smoke else None
    # CPU data may coexist with the GPU copies; large adapted arrays are never duplicated on disk.
    cpu={};global_arrays={}
    for split in ('train','val'):
        data.cache(config,split,lambda stage,**kw:publish(root,stage,**kw))
        global_arrays[split]=np.load(ROOT/config['data_output']/split/'global131.npy',mmap_mode='r')
    mean,scale=scaler_fit(global_arrays['train'])
    scaler_path=root/'train_scaler.npz'
    if scaler_path.exists():
        with np.load(scaler_path,allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['mean'],mean);np.testing.assert_array_equal(saved['scale'],scale)
    else:
        with scaler_path.with_suffix('.tmp').open('wb') as f:np.savez(f,mean=mean,scale=scale)
        os.replace(scaler_path.with_suffix('.tmp'),scaler_path)
    for split in ('train','val'):
        publish(root,'representation',split=split)
        cpu[split]=adapt_source(config,prepared,split,device,limit=limit)
        g=global_arrays[split][:limit] if limit else global_arrays[split]
        cpu[split]['g']=((g-mean)/scale).astype(np.float32)
    dev={s:to_device(v,device) for s,v in cpu.items()}
    lock=fit(root,dev['train'],dev['val'],config,contract,device,smoke)
    selection_sha=io.sha256_file(root/'selection.json')
    del dev;torch.cuda.empty_cache()
    # Every saved validation metric is recomputed, independently checking confusion/F1.
    publish(root,'audit');history=io.read(root/'history.json')
    for row in history:
        for name,record in row['variants'].items():
            path=root/name/f'val_{row["epoch"]:03d}.npy';weight=root/name/f'epoch_{row["epoch"]:03d}.pt'
            io.require(io.sha256_file(path)==record['logits_sha256'] and io.sha256_file(weight)==record['weight_sha256'],'epoch output changed')
            io.independent_metrics(cpu['val']['y'],np.load(path,allow_pickle=False),record['metrics'])
    for name in VARIANTS:
        expected=max(history,key=lambda r:metrics.selection_key(r['variants'][name]['metrics']))
        io.require(expected['epoch']==lock['best'][name]['epoch'],'selection audit failed')
    del cpu['train'];gc.collect();results={};cpu_max=0.;j1_max=cpu['val']['j1_cpu_max']
    for split in (('val',) if smoke else ('val','test','ood')):
        data.safety(config)
        io.require(io.sha256_file(root/'selection.json')==selection_sha,'selection lock changed')
        if split!='val':
            data.cache(config,split,lambda stage,**kw:publish(root,stage,**kw),locked=True)
            publish(root,'representation',split=split)
            current=adapt_source(config,prepared,split,device,locked=True)
            global_feature=np.load(ROOT/config['data_output']/split/'global131.npy',mmap_mode='r')
            current['g']=((global_feature-mean)/scale).astype(np.float32)
        else:current=cpu['val']
        j1_max=max(j1_max,current['j1_cpu_max']);gpu=to_device(current,device);results[split]={}
        publish(root,'evaluate',split=split)
        for name in VARIANTS:
            head,state=selected_head(root,lock,name,device)
            logits=predict(head,gpu,name,config['training']['eval_batch_size'])
            scored=metrics.classification_metrics(current['y'],logits)
            io.independent_metrics(current['y'],logits,scored)
            cpu_max=max(cpu_max,audit_logits(current,name,state,logits,config))
            if split=='val':
                old=np.load(root/name/f'val_{lock["best"][name]["epoch"]:03d}.npy',allow_pickle=False)
                np.testing.assert_array_equal(logits,old)
                io.require(scored==lock['best'][name]['metrics'],'validation replay')
            dest=root/'evaluation'/split;dest.mkdir(parents=True,exist_ok=True)
            io.save_array(dest/f'{name}_logits.npy',logits);results[split][name]=scored
            del head,state
        del gpu,current;torch.cuda.empty_cache();gc.collect()
    audit=dict(passed=True,all_epoch_metric_replay=True,independent_confusion_f1=True,
                selected_cpu_logits_max=cpu_max,j1_cpu_check_max=j1_max,validation_replay_exact=True,
                scaler_fit_split='train',scaler_fit_windows=config['split_counts']['train'],
                no_backbone_adl_or_j1_optimizer=True,selection_immutable=True)
    io.save_json(root/'independent_audit.json',audit)
    io.require(source_contract(config,smoke)[0]==contract,'source/code/config changed during run')
    report=dict(passed=True,research_usable=not smoke,historical_exact_reproduction=False,
                 selection=lock,metrics=results,independent_audit=audit,
                 selection_sha256=selection_sha,contract=contract,source_models_frozen=True)
    io.save_json(root/'final_report.json',report);publish(root,'completed',research_usable=not smoke,
                                                         validation_adoption_eligible=lock['validation_adoption_eligible'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('raw','cache','smoke','all'),default='all')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config=io.read(CONFIG);smoke=args.stage=='smoke';contract,prepared=source_contract(config,smoke)
    root=ROOT/(config['output_dir']+('_smoke' if smoke else ''))
    root.mkdir(parents=True,exist_ok=True);path=root/'run_contract.json'
    if path.exists():io.require(args.resume and io.read(path)==contract,'run contract changed or resume required')
    else:io.save_json(path,contract)
    io.require(Path(sys.prefix).name=='fall_detect','fall_detect required');torch.set_num_threads(2)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            data.safety(config)
            existing=sum(p.stat().st_size for p in (ROOT/config['data_output']).rglob('*') if p.is_file()) if (ROOT/config['data_output']).exists() else 0
            io.disk_gate(config,max(0,config['execution']['additional_budget_gib']*1024**3-existing))
            if args.stage in ('raw','cache'):
                io.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preprocessing required')
                data.raw_assets(config,lambda stage,**kw:publish(root,stage,**kw))
                if args.stage=='cache':
                    for split in ('train','val'):data.cache(config,split,lambda stage,**kw:publish(root,stage,**kw))
                return
            io.require((ROOT/config['data_output']/'raw_audit.json').exists(),'raw source audit required')
            io.require(io.read(ROOT/config['data_output']/'raw_audit.json')['passed'],'raw audit failed')
            device=io.device_for(config);run(config,root,contract,prepared,device,smoke)
        except BaseException as error:
            publish(root,'paused' if isinstance(error,io.PauseRequested) else 'failed',
                    error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':main()
