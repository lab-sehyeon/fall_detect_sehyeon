"""Serial J0/J1 development, locked SAFER, nested FU and fixed-epoch final fit."""
from __future__ import annotations
import argparse
import fcntl
from pathlib import Path
from datetime import datetime,timezone
import os
import numpy as np
import torch

from fall_pipeline.safer import v2_controls_reconstruction as io
from fall_pipeline.joint import reconstruction_core as core
from fall_pipeline.joint import reconstruction_inputs as inputs

ROOT=io.ROOT
CONFIG=ROOT/'configs/joint_document_reconstruction_v1.json'
CODE=['fall_pipeline/joint/run_reconstruction.py','fall_pipeline/joint/reconstruction_core.py',
      'fall_pipeline/joint/reconstruction_inputs.py','fall_pipeline/joint/models.py']+io.CODE


def publish(root,stage,**details):
    state={'stage':stage,'lineage':root.name.removesuffix('_smoke'),'time':datetime.now(timezone.utc).isoformat(),**details}
    io.save_json(root/'status.json',state);print(io.json.dumps(state,ensure_ascii=False,allow_nan=False),flush=True)
    if root.name.endswith('_smoke'): return
    status='completed' if stage=='completed' else 'paused' if stage in ('paused','failed') else 'in_progress'
    head=f'- 문서 ID: `DOC-20260922-joint-run-R1`\n- 기준일: 2026-09-22\n- 상태: `{status}`\n'
    internal=f'# 공동학습 자동 실행 기록\n\n{head}\n```json\n{io.json.dumps(state,indent=2,ensure_ascii=False)}\n```\n\nroot: `{root}`\n\n[계약](2026-09-22_joint_reconstruction_internal.md)을 따른다.\n'
    shared=f'# 공동학습 실행 진행\n\n{head}\n입력 계보: {state["lineage"]}, 현재 단계: {stage}.\n'
    if 'epoch' in details: shared+=f'\n{details.get("phase", "")} 완료 epoch: {details["epoch"]}.\n'
    if 'fold' in details: shared+=f'현재 fold: {details["fold"]}.\n'
    shared+='\n[방법과 평가 경계](2026-09-22_joint_reconstruction_shared.md)를 따른다. 원본 동등성이나 전체 프로젝트 완료를 주장하지 않는다.\n'
    for kind,value in (('internal',internal),('shared',shared)):
        path=ROOT/f'docs/{kind}/2026-09-22_joint_run_{kind}.md';temporary=path.with_suffix('.tmp')
        temporary.write_text(value);os.replace(temporary,path)


def contract_for(config,prepared,smoke):
    baseline=io.read(io.CONFIG)
    pins={baseline['encoder']:baseline['encoder_sha256'],
          baseline['adl_root']+'/best_adl_head.pth':baseline['adl_head_sha256']}
    for name,wanted in pins.items(): io.require(io.sha256_file(ROOT/name)==wanted,'baseline weights changed')
    return {'config_sha256':io.sha256_file(CONFIG),'code_sha256':{p:io.sha256_file(ROOT/p) for p in CODE},
            'inputs':prepared,'fu':config['fu'],'baseline_pins':pins,'smoke':smoke,
            'torch':str(torch.__version__),'historical_exact_reproduction':False}


def phase_record(dest,epoch,model,metrics,sl,fl,loss):
    weight=dest/f'epoch_{epoch:03d}.pt';safer=dest/f'safer_{epoch:03d}.npy';fu=dest/f'fu_{epoch:03d}.npy'
    io.save_tensor(weight,io.cpu_state(model));io.save_array(safer,sl);io.save_array(fu,fl)
    return {'epoch':epoch,'metrics':metrics,'loss':loss,'weight_sha256':io.sha256_file(weight),
            'safer_sha256':io.sha256_file(safer),'fu_sha256':io.sha256_file(fu)}


def fit_phase(root,dest,stage,safer,fu,train,val,config,device,contract,smoke,initial=None):
    dest.mkdir(parents=True,exist_ok=True);model=core.model_for(stage,config,device,initial)
    identity=None
    if stage=='j1':
        j0=core.model_for('j0',config,device);j0.load_state_dict(initial,strict=True)
        identity=core.epoch0_exact(j0,model,safer['val'],fu,val,config);del j0
    optimizer=core.optimizer_for(model,stage,config);history=[];start=1
    latest=dest/'latest.pt';progress=dest/'progress.json';epochs=1 if smoke else config['training'][stage+'_epochs']
    if latest.exists():
        io.require(io.sha256_file(latest)==io.read(progress)['checkpoint_sha256'],'uncommitted epoch checkpoint')
        state=torch.load(latest,map_location='cpu',weights_only=True);io.require(state['contract']==contract,'epoch contract')
        model.load_state_dict(state['state_dict'],strict=True);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];start=state['epoch']+1
        torch.set_rng_state(state['rng'])
        if device.type=='cuda': torch.cuda.set_rng_state(state['cuda_rng'],device)
    elif stage=='j1':
        metrics,sl,fl=core.evaluation(model,safer['val'],fu,val,config)
        history.append(phase_record(dest,0,model,metrics,sl,fl,None))
    for epoch in range(start,epochs+1):
        io.pause(root)
        loss=core.train_epoch(model,optimizer,safer['train'],fu,train,epoch,stage,config)
        metrics,sl,fl=core.evaluation(model,safer['val'],fu,val,config)
        history.append(phase_record(dest,epoch,model,metrics,sl,fl,loss))
        io.save_tensor(latest,{'contract':contract,'epoch':epoch,'state_dict':io.cpu_state(model),'optimizer':optimizer.state_dict(),
                             'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None})
        io.save_json(progress,{'epoch':epoch,'checkpoint_sha256':io.sha256_file(latest)})
        io.save_json(dest/'history.json',history)
        publish(root,dest.parent.parent.name,phase=stage,fold=int(dest.parent.name.removeprefix('fold')),epoch=epoch,
                safer_macro_f1=metrics['safer']['macro_f1'],fu_f1=metrics['fu']['f1'])
    io.require([row['epoch'] for row in history]==list(range(0 if stage=='j1' else 1,epochs+1)),'epoch coverage')
    best=max(history,key=core.selection_key)
    result={'phase':stage,'best':best,'identity':identity,'train_indices':train.tolist(),'val_indices':val.tolist(),
            'history_sha256':io.sha256_file(dest/'history.json'),'source_contract':contract,'passed':True}
    if (dest/'result.json').exists(): io.require(io.read(dest/'result.json')==result,'selected result changed')
    else: io.save_json(dest/'result.json',result)
    selected=torch.load(dest/f'epoch_{best["epoch"]:03d}.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(selected,strict=True);model.eval()
    return model,result


def fit_folds(root,context,safer,fu,config,device,contract,smoke):
    results={};nested=context=='nested';fold_ids=[0] if smoke else list(range(5))
    for k in fold_ids:
        io.pause(root);train,val,outer=core.split_indices(fu['folds'],fu['subjects'],k,nested)
        dest=root/context/f'fold{k}';dest.mkdir(parents=True,exist_ok=True);results[str(k)]={}
        initial=None
        for stage in ('j0','j1'):
            model,result=fit_phase(root,dest/stage,stage,safer,fu,train,val,config,device,contract,smoke,initial)
            results[str(k)][stage]=result
            if stage=='j0': initial=io.cpu_state(model)
            del model
        # Lock both selected phases before even forming outer inference input.
        fixed={'j0_result_sha256':io.sha256_file(dest/'j0/result.json'),
               'j1_result_sha256':io.sha256_file(dest/'j1/result.json'),'outer_unread_before_selection':nested}
        if (dest/'selection_lock.json').exists(): io.require(io.read(dest/'selection_lock.json')==fixed,'fold lock changed')
        else: io.save_json(dest/'selection_lock.json',fixed)
        if nested:
            outer_result={}
            for stage in ('j0','j1'):
                model=load_selected(dest/stage,stage,config,device)
                logits=core.predict(model,fu['x'][outer],'fu',config['training']['eval_batch_size'])
                io.save_array(dest/f'{stage}_outer_logits.npy',logits)
                outer_result[stage]={'metrics':core.binary_metrics(fu['y'][outer].cpu().numpy(),fu['actions'][outer],logits),
                                     'logits_sha256':io.sha256_file(dest/f'{stage}_outer_logits.npy')}
                del model
            io.save_array(dest/'outer_indices.npy',outer)
            io.save_json(dest/'outer_result.json',{'indices_sha256':io.sha256_file(dest/'outer_indices.npy'),
                         'selection_lock_sha256':io.sha256_file(dest/'selection_lock.json'),'results':outer_result})
        publish(root,context,completed_folds=k+1,total_folds=len(fold_ids))
    return results


def load_selected(dest,stage,config,device):
    result=io.read(dest/'result.json');row=result['best'];path=dest/f'epoch_{row["epoch"]:03d}.pt'
    io.require(io.sha256_file(path)==row['weight_sha256'],'selected weight changed')
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=core.model_for(stage,config,device,state if stage=='j1' else None);model.load_state_dict(state,strict=True)
    return model.eval().requires_grad_(False)


def locked_safer(root,prepared,config,device,smoke):
    dest=root/'locked';dest.mkdir(exist_ok=True);fold_ids=[0] if smoke else list(range(5))
    fixed={f'{k}/{phase}':io.sha256_file(root/'development'/f'fold{k}'/phase/'result.json') for k in fold_ids for phase in ('j0','j1')}
    lock_path=dest/'selection_lock.json'
    if lock_path.exists(): io.require(io.read(lock_path)==fixed,'locked selection changed')
    else: io.save_json(lock_path,fixed)
    report={}
    for split in ('test','ood'):
        io.pause(root);source=inputs.load_safer(prepared,split,locked=True,limit=256 if smoke else None)
        data=inputs.to_device(source,device);out=dest/split;out.mkdir(exist_ok=True);report[split]={}
        io.save_array(out/'labels.npy',source['y'])
        for phase in ('j0','j1'):
            members=[]
            for k in fold_ids:
                model=load_selected(root/'development'/f'fold{k}'/phase,phase,config,device)
                logits=core.predict(model,data['x'],'safer',config['training']['eval_batch_size']);del model
                io.save_array(out/f'{phase}_fold{k}.npy',logits);members.append(logits)
            ensemble=np.mean(np.stack(members).astype(np.float64),axis=0).astype(np.float32)
            io.save_array(out/f'{phase}_ensemble.npy',ensemble)
            report[split][phase]=core.classification_metrics(source['y'],ensemble)
        io.save_json(out/'payload.json',{p.name:io.sha256_file(p) for p in out.glob('*.npy')})
        del data,source;torch.cuda.empty_cache()
        publish(root,'locked',split=split)
    io.save_json(dest/'report.json',report);return report


def aggregate(root,context,fu,config,smoke):
    fold_ids=[0] if smoke else list(range(5));report={}
    for stage in ('j0','j1'):
        cover=np.zeros(len(fu['actions']),np.int64);logits=np.zeros((len(cover),2),np.float32);epochs=[]
        for k in fold_ids:
            dest=root/context/f'fold{k}';r=io.read(dest/stage/'result.json');epochs.append(r['best']['epoch'])
            if context=='development': idx=np.array(r['val_indices']);value=np.load(dest/stage/f'fu_{r["best"]["epoch"]:03d}.npy',allow_pickle=False)
            else: idx=np.load(dest/'outer_indices.npy',allow_pickle=False);value=np.load(dest/f'{stage}_outer_logits.npy',allow_pickle=False)
            cover[idx]+=1;logits[idx]=value
        io.require(np.all(cover<=1) and (smoke or np.all(cover==1)),'OOF exact once')
        valid=np.flatnonzero(cover==1);io.save_array(root/context/f'{stage}_oof_indices.npy',valid)
        io.save_array(root/context/f'{stage}_oof_logits.npy',logits[valid])
        report[stage]={'best_epochs':epochs,'metrics':core.binary_metrics(fu['y'][valid].cpu().numpy(),fu['actions'][valid],logits[valid]),
                       'oof_count':len(valid),'exact_once':True}
    io.save_json(root/context/'report.json',report);return report


def final_fit(root,safer,fu,nested,config,device,contract,smoke):
    dest=root/'final';dest.mkdir(exist_ok=True)
    plan={stage:int(np.median(nested[stage]['best_epochs'])) for stage in ('j0','j1')}
    plan['nested_report_sha256']=io.sha256_file(root/'nested/report.json')
    path=dest/'epoch_plan.json'
    if path.exists(): io.require(io.read(path)==plan,'final epoch plan changed')
    else: io.save_json(path,plan)
    train=np.arange(len(fu['actions']));initial=None;diagnostics={}
    for stage in ('j0','j1'):
        base=dest/stage;base.mkdir(exist_ok=True);model=core.model_for(stage,config,device,initial)
        if stage=='j1':
            j0=core.model_for('j0',config,device);j0.load_state_dict(initial,strict=True)
            equality=core.epoch0_exact(j0,model,safer['train'],fu,train,config);del j0
            equality['scope']='all final-fit training inputs; not held-out data';io.save_json(base/'epoch0.json',equality)
        opt=core.optimizer_for(model,stage,config);start=1;history=[];latest=base/'latest.pt'
        if latest.exists():
            io.require(io.sha256_file(latest)==io.read(base/'progress.json')['checkpoint_sha256'],'final checkpoint commit')
            state=torch.load(latest,map_location='cpu',weights_only=True);io.require(state['contract']==contract and state['plan']==plan,'final contract/plan')
            model.load_state_dict(state['state_dict'],strict=True);opt.load_state_dict(state['optimizer']);history=state['history'];start=state['epoch']+1
            torch.set_rng_state(state['rng'])
            if device.type=='cuda': torch.cuda.set_rng_state(state['cuda_rng'],device)
        for epoch in range(start,plan[stage]+1):
            io.pause(root);loss=core.train_epoch(model,opt,safer['train'],fu,train,epoch,stage,config)
            history.append({'epoch':epoch,'loss':loss})
            io.save_tensor(latest,{'contract':contract,'plan':plan,'state_dict':io.cpu_state(model),'optimizer':opt.state_dict(),
                                 'epoch':epoch,'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None})
            io.save_json(base/'progress.json',{'epoch':epoch,'checkpoint_sha256':io.sha256_file(latest)})
            publish(root,'final',phase=stage,epoch=epoch,target=plan[stage])
        io.require(len(history)==plan[stage],'fixed final epochs')
        io.save_json(base/'history.json',history);io.save_tensor(dest/f'final_{stage}.pt',io.cpu_state(model))
        logits=core.predict(model,fu['x'],'fu',config['training']['eval_batch_size']);io.save_array(base/'training_fu_logits.npy',logits)
        diagnostics[stage]=core.binary_metrics(fu['y'].cpu().numpy(),fu['actions'],logits)
        if stage=='j0': initial=io.cpu_state(model)
        del model,opt
    result={'passed':True,'epochs':plan,'no_validation_selection':True,'fu_training_only_diagnostics':diagnostics,
            'weights':{stage:io.sha256_file(dest/f'final_{stage}.pt') for stage in ('j0','j1')}}
    io.save_json(dest/'report.json',result);return result


def audit(root,safer_cpu,fu_cpu,prepared,config,smoke):
    fold_ids=[0] if smoke else list(range(5));maximum=0.
    for context in ('development','nested'):
        for k in fold_ids:
            expected_train,expected_val,outer=core.split_indices(fu_cpu['folds'],fu_cpu['subjects'],k,context=='nested')
            dest=root/context/f'fold{k}'
            for stage in ('j0','j1'):
                result=io.read(dest/stage/'result.json');history=io.read(dest/stage/'history.json')
                io.require(io.sha256_file(dest/stage/'history.json')==result['history_sha256'],'phase history hash')
                np.testing.assert_array_equal(result['train_indices'],expected_train);np.testing.assert_array_equal(result['val_indices'],expected_val)
                for row in history:
                    e=row['epoch'];base=dest/stage
                    for name,key in ((f'epoch_{e:03d}.pt','weight_sha256'),(f'safer_{e:03d}.npy','safer_sha256'),(f'fu_{e:03d}.npy','fu_sha256')):
                        io.require(io.sha256_file(base/name)==row[key],'epoch payload hash')
                    sl=np.load(base/f'safer_{e:03d}.npy',allow_pickle=False);fl=np.load(base/f'fu_{e:03d}.npy',allow_pickle=False)
                    io.independent_metrics(safer_cpu['val']['y'],sl,row['metrics']['safer'])
                    io.require(core.binary_metrics(fu_cpu['y'][expected_val],fu_cpu['actions'][expected_val],fl)==row['metrics']['fu'],'FU metrics')
                io.require(max(history,key=core.selection_key)==result['best'],'best selection')
                model=load_selected(dest/stage,stage,config,torch.device('cpu'));e=result['best']['epoch']
                for dataset,x,path in (('safer',safer_cpu['val']['x'],dest/stage/f'safer_{e:03d}.npy'),
                                       ('fu',fu_cpu['x'][expected_val],dest/stage/f'fu_{e:03d}.npy')):
                    expected=core.predict(model,torch.from_numpy(x),dataset,config['training']['eval_batch_size'])
                    actual=np.load(path,allow_pickle=False)
                    np.testing.assert_allclose(actual,expected,atol=config['audit']['cpu_prediction_atol'],rtol=config['audit']['cpu_prediction_rtol'])
                    maximum=max(maximum,float(np.abs(expected-actual).max()))
                if context=='nested':
                    result_outer=io.read(dest/'outer_result.json');indices=np.load(dest/'outer_indices.npy',allow_pickle=False)
                    np.testing.assert_array_equal(indices,outer)
                    io.require(io.sha256_file(dest/'selection_lock.json')==result_outer['selection_lock_sha256'],'outer before lock')
                    actual=np.load(dest/f'{stage}_outer_logits.npy',allow_pickle=False)
                    io.require(io.sha256_file(dest/f'{stage}_outer_logits.npy')==result_outer['results'][stage]['logits_sha256'],'outer payload')
                    expected=core.predict(model,torch.from_numpy(fu_cpu['x'][outer]),'fu',config['training']['eval_batch_size'])
                    np.testing.assert_allclose(actual,expected,atol=config['audit']['cpu_prediction_atol'],rtol=config['audit']['cpu_prediction_rtol'])
                    io.require(core.binary_metrics(fu_cpu['y'][outer],fu_cpu['actions'][outer],actual)==result_outer['results'][stage]['metrics'],'outer metrics')
                del model
    locked=io.read(root/'locked/report.json')
    fixed=io.read(root/'locked/selection_lock.json')
    for k in fold_ids:
        for stage in ('j0','j1'):
            io.require(io.sha256_file(root/'development'/f'fold{k}'/stage/'result.json')==fixed[f'{k}/{stage}'],'locked model selection changed')
    for split in ('test','ood'):
        source=inputs.load_safer(prepared,split,locked=True,limit=256 if smoke else None);dest=root/'locked'/split
        for name,wanted in io.read(dest/'payload.json').items(): io.require(io.sha256_file(dest/name)==wanted,'locked payload hash')
        np.testing.assert_array_equal(np.load(dest/'labels.npy',allow_pickle=False),source['y'])
        for stage in ('j0','j1'):
            members=[np.load(dest/f'{stage}_fold{k}.npy',allow_pickle=False) for k in fold_ids]
            for k,actual in zip(fold_ids,members):
                model=load_selected(root/'development'/f'fold{k}'/stage,stage,config,torch.device('cpu'))
                expected=core.predict(model,torch.from_numpy(source['x']),'safer',config['training']['eval_batch_size'])
                np.testing.assert_allclose(actual,expected,atol=config['audit']['cpu_prediction_atol'],rtol=config['audit']['cpu_prediction_rtol'])
                maximum=max(maximum,float(np.abs(expected-actual).max()));del model
            mean=np.mean(np.stack(members).astype(np.float64),axis=0).astype(np.float32)
            np.testing.assert_array_equal(mean,np.load(dest/f'{stage}_ensemble.npy',allow_pickle=False))
            io.independent_metrics(source['y'],mean,locked[split][stage])
        del source
    for context in ('development','nested'):
        report=io.read(root/context/'report.json')
        for stage in ('j0','j1'):
            pairs=[];epochs=[]
            for k in fold_ids:
                dest=root/context/f'fold{k}';result=io.read(dest/stage/'result.json');e=result['best']['epoch'];epochs.append(e)
                idx=np.array(result['val_indices']) if context=='development' else np.load(dest/'outer_indices.npy',allow_pickle=False)
                logits=np.load(dest/stage/f'fu_{e:03d}.npy',allow_pickle=False) if context=='development' else np.load(dest/f'{stage}_outer_logits.npy',allow_pickle=False)
                pairs.extend(zip(idx.tolist(),logits))
            pairs.sort(key=lambda item:item[0]);ids=np.array([p[0] for p in pairs]);logits=np.stack([p[1] for p in pairs])
            io.require(len(set(ids))==len(ids) and (smoke or np.array_equal(ids,np.arange(993))),'independent OOF coverage')
            np.testing.assert_array_equal(ids,np.load(root/context/f'{stage}_oof_indices.npy',allow_pickle=False))
            np.testing.assert_array_equal(logits,np.load(root/context/f'{stage}_oof_logits.npy',allow_pickle=False))
            io.require(report[stage]['metrics']==core.binary_metrics(fu_cpu['y'][ids],fu_cpu['actions'][ids],logits),'OOF metrics')
            io.require(report[stage]['best_epochs']==epochs,'OOF selected epochs')
    nested=io.read(root/'nested/report.json');final=io.read(root/'final/report.json')
    for stage in ('j0','j1'):
        io.require(final['epochs'][stage]==int(np.median(nested[stage]['best_epochs'])),'nested final median')
        io.require(len(io.read(root/'final'/stage/'history.json'))==final['epochs'][stage],'final count')
        io.require(io.sha256_file(root/'final'/f'final_{stage}.pt')==final['weights'][stage],'final weight hash')
        state=torch.load(root/'final'/f'final_{stage}.pt',map_location='cpu',weights_only=True)
        model=core.model_for(stage,config,torch.device('cpu'),state if stage=='j1' else None);model.load_state_dict(state,strict=True)
        expected=core.predict(model,torch.from_numpy(fu_cpu['x']),'fu',config['training']['eval_batch_size']);del model
        actual=np.load(root/'final'/stage/'training_fu_logits.npy',allow_pickle=False)
        np.testing.assert_allclose(actual,expected,atol=config['audit']['cpu_prediction_atol'],rtol=config['audit']['cpu_prediction_rtol'])
        io.require(core.binary_metrics(fu_cpu['y'],fu_cpu['actions'],actual)==final['fu_training_only_diagnostics'][stage],'final training diagnostic')
    report={'passed':True,'epoch_metric_and_selection_audit':True,'subject_splits_verified':True,
            'cpu_best_forward_max_abs_error':maximum,'nested_median_verified':True,'locked_ensemble_exact':True}
    io.save_json(root/'independent_audit.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--lineage',choices=('v1','v2'),required=True)
    parser.add_argument('--stage',choices=('preflight','smoke','all'),default='all');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();config=io.read(CONFIG);smoke=args.stage=='smoke';prepared=inputs.prepare(config,args.lineage)
    contract=contract_for(config,prepared,smoke);root=io.guard(ROOT/config['output_dir']/(args.lineage+('_smoke' if smoke else '')))
    existing=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    io.disk_gate(config,max(0,(1 if smoke else 6)*1024**3-existing))
    if root.exists() and any(root.iterdir()): io.require(args.resume and io.read(root/'run_contract.json')==contract,'output/contract mismatch')
    else: root.mkdir(parents=True,exist_ok=True);io.save_json(root/'run_contract.json',contract)
    torch.set_num_threads(2)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            publish(root,'source');fu_cpu=inputs.load_fu(config)
            safer_cpu={split:inputs.load_safer(prepared,split,limit=256 if smoke else None) for split in ('train','val')}
            io.save_json(root/'source_audit.json',{'passed':True,'train':len(safer_cpu['train']['y']),'val':len(safer_cpu['val']['y']),
                         'fu':len(fu_cpu['y']),'holdout_arrays_unopened':True})
            if args.stage=='preflight': return
            device=io.device_for(config);safer={s:inputs.to_device(v,device) for s,v in safer_cpu.items()};fu=inputs.to_device(fu_cpu,device)
            fit_folds(root,'development',safer,fu,config,device,contract,smoke)
            development=aggregate(root,'development',fu,config,smoke)
            locked=locked_safer(root,prepared,config,device,smoke)
            fit_folds(root,'nested',safer,fu,config,device,contract,smoke)
            nested=aggregate(root,'nested',fu,config,smoke)
            final=final_fit(root,safer,fu,nested,config,device,contract,smoke)
            del safer,fu;torch.cuda.empty_cache();publish(root,'audit')
            audit(root,safer_cpu,fu_cpu,prepared,config,smoke)
            io.require(contract_for(config,prepared,smoke)==contract,'source/code/config changed during run')
            io.save_json(root/'final_report.json',{'passed':True,'research_usable':not smoke,'historical_exact_reproduction':False,
                          'development':development,'locked_safer':locked,'nested':nested,'final_fit':final,
                          'audit_sha256':io.sha256_file(root/'independent_audit.json'),'contract':contract,
                          'dste_adl_forward_or_optimizer':False})
            publish(root,'completed',research_usable=not smoke)
        except BaseException as error:
            publish(root,'paused' if isinstance(error,io.PauseRequested) else 'failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__': main()
