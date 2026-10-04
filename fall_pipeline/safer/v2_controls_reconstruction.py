"""Pinned reconstructed V2 F0A/F0B; never edits V1 or fits on holdouts."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from fall_pipeline.common.integrity import sha256_file, hash_named_tensors
from fall_pipeline.common.eval_safer_legacy_v1_candidates import load_adl_model, f0a_metrics
from fall_pipeline.common.fall_safer_zeroshot_eval import validate_adl_run_config, forward_adl_logits
from fall_pipeline.safer.extract_f0b_safer_features import pooled_backbone_features
from fall_pipeline.safer import train_f0b_safer_heads as head_ops

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'configs/safer_v2_controls_document_reconstruction_v1.json'
CODE = ['fall_pipeline/safer/v2_controls_reconstruction.py',
        'fall_pipeline/safer/train_f0b_safer_heads.py', 'fall_pipeline/safer/extract_f0b_safer_features.py',
        'fall_pipeline/common/eval_safer_legacy_v1_candidates.py',
        'fall_pipeline/common/fall_safer_zeroshot_eval.py','fall_pipeline/common/integrity.py',
        'fall_pipeline/fu/fall_fu_linear_eval.py', 'tools.py',
        'model/DSTE.py','model/STTR.py', 'data_gen/safer_legacy_v1_gendata.py',
        'data_gen/preflight_safer_legacy_v1.py','data_gen/inspect_safer_activities.py',
        'data_gen/safer_legacy_v1_candidates.py','data_gen/h36m17_to_ntu25.py']
META = ('center_coarse_labels.npy','center_derived_labels.npy','sequence_index.npy','window_start.npy')
FEATURES = ('temporal_features.npy','spatial_features.npy','adl_logits.npy')


def require(value, message):
    if not value: raise RuntimeError(message)


def read(path): return json.loads(Path(path).read_text())


def save_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(temporary,path)


def save_tensor(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);os.replace(temporary,path)


def save_array(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream: np.save(stream,value,allow_pickle=False)
    os.replace(temporary,path)


def digest(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def guard(path):
    path=Path(path).resolve()
    require(path.is_relative_to(ROOT),'path outside project')
    require('cauca' not in str(path).lower(),'CAUCA forbidden')
    return path


class PauseRequested(RuntimeError): pass


def pause(root):
    if (root/'PAUSE_REQUESTED').exists(): raise PauseRequested('pause flag; resume only after explicit continuation')


def disk_gate(config, additional=0):
    require(shutil.disk_usage(ROOT).free >= additional+config['execution']['reserve_gib']*1024**3,'disk reserve gate')


def publish(root,stage,**details):
    state={'stage':stage,'time':datetime.now(timezone.utc).isoformat(),**details}
    save_json(root/'status.json',state);print(json.dumps(state,ensure_ascii=False,allow_nan=False),flush=True)
    if root.name.endswith('_smoke'): return
    words={'preflight':'입력·모델 동일성 검사','extract':'동결 특징 추출','train':'두 linear head 학습',
           'selection':'validation 선택 고정','evaluate':'고정 모델 평가','audit':'독립 검산',
           'completed':'전체 비교·독립 검산 완료','failed':'실행 중단','paused':'일시 중단'}
    status='completed' if stage=='completed' else 'paused' if stage in ('failed','paused') else 'in_progress'
    header=f'- 문서 ID: `DOC-20260922-safer-v2-controls-run-R1`\n- 기준일: 2026-09-22\n- 상태: `{status}`\n'
    internal=f'# V2 controls 자동 실행 기록\n\n{header}\n```json\n{json.dumps(state,indent=2,ensure_ascii=False)}\n```\n\n실행 root: `{root}`\n\n[실행 계약](2026-09-22_safer_v2_controls_internal.md)을 따른다.\n'
    shared=f'# V2 controls 실행 진행\n\n{header}\n현재 단계: {words[stage]}.\n'
    if 'split' in details: shared+=f'\n대상 split: {details["split"]}.\n'
    if 'windows' in details: shared+=f'완료 windows: {details["windows"]}/{details["total"]}.\n'
    if 'epoch' in details: shared+=f'완료 학습 epoch: {details["epoch"]}/50.\n'
    shared+='\n[방법과 한계](2026-09-22_safer_v2_controls_shared.md)를 따른다. 원본 동등성이나 전체 연구 완료를 주장하지 않는다.\n'
    for kind,value in (('internal',internal),('shared',shared)):
        path=ROOT/f'docs/{kind}/2026-09-22_safer_v2_controls_run_{kind}.md'
        temporary=path.with_suffix('.tmp');temporary.write_text(value);os.replace(temporary,path)


def make_contract(config,smoke):
    for path,key in ((ROOT/config['encoder'],'encoder_sha256'),
                     (ROOT/config['adl_root']/'best_adl_head.pth','adl_head_sha256'),
                     (ROOT/config['adl_root']/'run_config.json','adl_run_config_sha256'),
                     (ROOT/config['matched_training_config'],'matched_training_config_sha256')):
        require(sha256_file(guard(path))==config[key],f'pin changed: {key}')
    validate_adl_run_config(ROOT/config['adl_root']/'run_config.json')
    return {'config_sha256':sha256_file(CONFIG),'code_sha256':{name:sha256_file(ROOT/name) for name in CODE},
            'torch':str(torch.__version__),'numpy':str(np.__version__),'smoke':smoke,
            'materialization_sha256':config['materialization_manifest_sha256'],
            'historical_exact_reproduction':False}


def initialize(root,contract,resume):
    guard(root);require(root.parent==ROOT/'checkpoint/fall','output scope')
    if root.exists() and any(root.iterdir()):
        require(resume and read(root/'run_contract.json')==contract,'nonempty output or changed contract')
    else: root.mkdir(parents=True,exist_ok=True);save_json(root/'run_contract.json',contract)


def verify_sources(config):
    source=guard(ROOT/config['data_root']);manifest_path=source/'materialized/materialization_manifest.json'
    require(sha256_file(source/'final_report.json')==config['preprocessing_final_report_sha256'],'V2 final pin')
    final=read(source/'final_report.json');require(final['passed'] and final['model_frozen'],'V2 audit incomplete')
    require(sha256_file(manifest_path)==config['materialization_manifest_sha256'],'V2 manifest pin')
    manifest=read(manifest_path)
    require(manifest['schema']=='safer_v2_document_reconstruction_v1' and manifest['research_usable'] and manifest['integrity']['passed'],'V2 schema/gate')
    total=0
    for split,count in config['split_counts'].items():
        dest=source/'materialized'/split;path=dest/'split_manifest.json'
        require(sha256_file(path)==manifest['splits'][split]['manifest_sha256'],'split lineage')
        spec=read(path);require(spec['written_windows']==count==manifest['splits'][split]['windows'],'count')
        for name,item in spec['payload'].items():
            target=dest/name
            require(target.stat().st_size==item['bytes'] and sha256_file(target)==item['file_sha256'],'source payload hash: '+split+'/'+name)
        for name,wanted in spec['small_file_sha256'].items(): require(sha256_file(dest/name)==wanted,'source metadata hash')
        total+=count
        print(json.dumps({'source_verified':split,'windows':count}),flush=True)
    require(total==1007723,'full V2 count')
    return {'passed':True,'windows':total,'all_source_files_hash_verified':True,'holdout_hash_for_integrity_only':True}


def source_split(config,split,smoke):
    dest=guard(ROOT/config['data_root']/'materialized'/split)
    count=config['split_counts'][split];count=min(count,512) if smoke else count
    arrays={name:np.load(dest/name,mmap_mode='r',allow_pickle=False)[:count] for name in META}
    data=np.load(dest/'data_joint.npy',mmap_mode='r',allow_pickle=False)
    require(data.shape==(config['split_counts'][split],3,64,25,2) and data.dtype==np.float32,'source shape')
    derived=np.array([0]*10+[1,2,3]+[0]*3)[arrays['center_coarse_labels.npy']]
    np.testing.assert_array_equal(derived,arrays['center_derived_labels.npy'])
    return {'root':dest,'count':count,'data':data,'arrays':arrays}


def device_for(config):
    require(Path(sys.prefix).name=='fall_detect','fall_detect required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','GPU0/runtime contract')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and '3090' in torch.cuda.get_device_name(0),'GPU0 RTX3090')
    require(torch.cuda.mem_get_info()[0]>=config['execution']['minimum_gpu_free_gib']*1024**3,'GPU free memory')
    torch.cuda.set_per_process_memory_fraction(config['execution']['gpu_memory_fraction'],0)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch.device('cuda:0')


def payload(path):
    value=np.load(path,mmap_mode='r',allow_pickle=False)
    require(all(np.isfinite(value[i:i+4096]).all() for i in range(0,len(value),4096)),'nonfinite output')
    return {'sha256':sha256_file(path),'bytes':path.stat().st_size,'shape':list(value.shape),'dtype':str(value.dtype)}


def verify_cache(dest,source,model_hash,selection_sha):
    report=read(dest/'manifest.json')
    require(report['count']==source['count'] and report['model_hash']==model_hash and report['selection_sha256']==selection_sha,'cache lineage')
    for name,spec in report['payload'].items():
        dim={'temporal_features.npy':1024,'spatial_features.npy':1024,'adl_logits.npy':60}.get(name)
        require(spec['shape']==([source['count'],dim] if dim else [source['count']]),'expected cache shape')
        require(spec['dtype']==('float32' if dim else 'int64'),'expected cache dtype')
        require(payload(dest/name)==spec,'cache payload changed')
    for name in META: np.testing.assert_array_equal(np.load(dest/name,allow_pickle=False),source['arrays'][name])
    require(set(report['payload'])==set(FEATURES+META),'cache inventory')
    return {'count':source['count'],'arrays':{name:np.load(dest/name,mmap_mode='r',allow_pickle=False) for name in FEATURES+META}}


def extract(root,split,source,model,device,config,model_hash,selection_sha=None):
    if split in ('test','ood'):
        require(selection_sha is not None and sha256_file(root/'selection.json')==selection_sha,'holdout before selection lock')
    dest=root/'cache'/split;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists(): return verify_cache(dest,source,model_hash,selection_sha)
    progress_path=dest/'progress.json';existing=progress_path.exists()
    progress=read(progress_path) if existing else {'next':0,'blocks':[],'model_hash':model_hash,'selection_sha256':selection_sha}
    require(progress['model_hash']==model_hash and progress['selection_sha256']==selection_sha,'resume cache lineage')
    count=source['count'];arrays={}
    for name,dim in zip(FEATURES,(1024,1024,60)):
        path=dest/name
        require(existing or not path.exists(),'orphan cache without progress; preserve and inspect')
        arrays[name]=np.load(path,mmap_mode='r+',allow_pickle=False) if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(count,dim))
        require(arrays[name].shape==(count,dim) and arrays[name].dtype==np.float32,'cache shape')
    cursor=0
    for block in progress['blocks']:
        require(block['begin']==cursor and block['end']<=count,'resume block order')
        for name in FEATURES: require(digest(arrays[name][cursor:block['end']])==block['hashes'][name],'resume prefix changed')
        cursor=block['end']
    require(cursor==progress['next'],'resume cursor')
    for name in META:
        if (dest/name).exists(): np.testing.assert_array_equal(np.load(dest/name,allow_pickle=False),source['arrays'][name])
        else: save_array(dest/name,source['arrays'][name])
    save_json(progress_path,progress)
    batch_size=config['execution']['extract_batch_size'];flush=config['execution']['flush_batches']*batch_size
    with torch.inference_mode():
        for block_begin in range(cursor,count,flush):
            pause(root);block_end=min(block_begin+flush,count)
            for begin in range(block_begin,block_end,batch_size):
                end=min(begin+batch_size,count)
                x=torch.from_numpy(np.array(source['data'][begin:end],copy=True)).to(device)
                t,s=pooled_backbone_features(model,x);adl=model.fc(torch.cat((t,s),1))
                for name,value in zip(FEATURES,(t,s,adl)): arrays[name][begin:end]=value.cpu().numpy()
            for value in arrays.values(): value.flush()
            progress['blocks'].append({'begin':block_begin,'end':block_end,'hashes':{name:digest(value[block_begin:block_end]) for name,value in arrays.items()}})
            progress['next']=block_end;save_json(progress_path,progress)
            publish(root,'extract',split=split,windows=block_end,total=count);disk_gate(config)
    spot_forward(model,source,arrays,device,batch_size)
    save_json(dest/'manifest.json',{'count':count,'model_hash':model_hash,'selection_sha256':selection_sha,
              'sample_order_exact':True,'spot_forward_exact':True,'payload':{name:payload(dest/name) for name in FEATURES+META}})
    return verify_cache(dest,source,model_hash,selection_sha)


def spot_forward(model,source,arrays,device,batch_size):
    count=source['count'];batches=(count+batch_size-1)//batch_size
    with torch.inference_mode():
        for index in sorted({0,batches//2,batches-1}):
            begin=index*batch_size;end=min(begin+batch_size,count)
            x=torch.from_numpy(np.array(source['data'][begin:end],copy=True)).to(device)
            # Independent input layout, pooling and public ADL forward comparison.
            jt=x.permute(0,2,4,3,1).reshape(len(x),64,150);js=x.permute(0,4,3,2,1).reshape(len(x),50,192)
            t,s=model.backbone(jt,js);t=t.max(1).values;s=s.max(1).values
            for name,value in zip(FEATURES[:2],(t,s)): np.testing.assert_array_equal(arrays[name][begin:end],value.cpu().numpy())
            np.testing.assert_array_equal(arrays['adl_logits.npy'][begin:end],forward_adl_logits(model,x).cpu().numpy())


def cpu_state(module): return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def fit_heads(root,cache,training,device,smoke,contract):
    dest=root/'training';dest.mkdir(exist_ok=True)
    torch.manual_seed(training['optimization']['seed'])
    heads=head_ops.initialize_heads(training,device);optimizers=head_ops.create_optimizers(heads,training)
    rng=torch.Generator().manual_seed(training['optimization']['seed']);histories=[];start=1
    latest=dest/'latest.pt';progress_path=dest/'progress.json'
    if latest.exists():
        require(progress_path.exists() and sha256_file(latest)==read(progress_path)['checkpoint_sha256'],'latest checkpoint not committed')
        state=torch.load(latest,map_location='cpu',weights_only=True)
        require(state['contract']==contract,'training contract')
        for name in head_ops.CANDIDATE_ORDER:
            heads[name].load_state_dict(state['heads'][name],strict=True);optimizers[name].load_state_dict(state['optimizers'][name])
        rng.set_state(state['rng']);histories=state['history'];start=state['epoch']+1
    epochs=1 if smoke else training['optimization']['epochs']
    for epoch in range(start,epochs+1):
        pause(root);head_ops.set_learning_rate(optimizers,head_ops.learning_rate_for_epoch(training,epoch-1))
        permutation=torch.randperm(cache['train']['count'],generator=rng)
        losses=head_ops.train_epoch(heads,optimizers,cache['train'],permutation,training['optimization']['batch_size'],device,0)
        metrics,logits,labels=head_ops.evaluate_heads(heads,cache['val'],training['evaluation']['batch_size'],device,0)
        row={'epoch':epoch,'loss':losses,'candidates':{}}
        for name in head_ops.CANDIDATE_ORDER:
            target=dest/name;target.mkdir(exist_ok=True)
            weight_path=target/f'epoch_{epoch:03d}.pt';logit_path=target/f'val_{epoch:03d}.npy'
            save_tensor(weight_path,cpu_state(heads[name]));save_array(logit_path,logits[name])
            row['candidates'][name]={'metrics':metrics[name],'weight_sha256':sha256_file(weight_path),'logits_sha256':sha256_file(logit_path)}
        histories.append(row)
        save_tensor(latest,{'contract':contract,'epoch':epoch,'heads':{n:cpu_state(h) for n,h in heads.items()},
                           'optimizers':{n:o.state_dict() for n,o in optimizers.items()},'rng':rng.get_state(),'history':histories})
        save_json(progress_path,{'epoch':epoch,'checkpoint_sha256':sha256_file(latest)})
        save_json(dest/'history.json',histories)
        publish(root,'train',epoch=epoch,validation={n:m['macro_f1'] for n,m in metrics.items()})
    require([h['epoch'] for h in histories]==list(range(1,epochs+1)),'training epoch coverage')
    # Candidate ordering takes precedence only for exactly equal best metrics.
    candidate_bests={}
    for name in head_ops.CANDIDATE_ORDER:
        candidate_bests[name]=max(histories,key=lambda h:head_ops.selection_key(h['candidates'][name]['metrics']))
    selected=max(head_ops.CANDIDATE_ORDER,key=lambda n:head_ops.selection_key(candidate_bests[n]['candidates'][n]['metrics']))
    row=candidate_bests[selected];epoch=row['epoch']
    selection={'candidate':selected,'epoch':epoch,'metrics':row['candidates'][selected]['metrics'],
               'history_sha256':sha256_file(dest/'history.json'),'weight_sha256':row['candidates'][selected]['weight_sha256'],
               'feature_manifest_sha256':{s:sha256_file(root/'cache'/s/'manifest.json') for s in ('train','val')},
               'contract':contract,'frozen_before_holdout':True,'research_usable':not smoke}
    path=root/'selection.json'
    if path.exists(): require(read(path)==selection,'selection changed on resume')
    else: save_json(path,selection)
    publish(root,'selection',candidate=selected,epoch=epoch)
    return selection


def independent_metrics(labels,logits,reported):
    require(head_ops.classification_metrics(labels,logits)==reported,'metric recomputation')
    pred=logits.argmax(1)
    require(confusion_matrix(labels,pred,labels=[0,1,2,3]).tolist()==reported['confusion'],'independent confusion')
    require(abs(f1_score(labels,pred,labels=[0,1,2,3],average='macro',zero_division=0)-reported['macro_f1'])<1e-12,'independent macro F1')
    require(abs(f1_score(labels==1,pred==1,zero_division=0)-reported['fall_f1'])<1e-12,'independent fall F1')


def cpu_logit_audit(split,state,candidate,logits,config):
    maximum=0.
    for begin in range(0,split['count'],512):
        end=min(begin+512,split['count']);arrays=split['arrays']
        x=np.asarray(arrays['temporal_features.npy'][begin:end])
        if candidate=='temporal_spatial': x=np.concatenate((x,arrays['spatial_features.npy'][begin:end]),axis=1)
        expected=(torch.from_numpy(np.array(x,copy=True))@state['weight'].T+state['bias']).numpy()
        np.testing.assert_allclose(logits[begin:end],expected,atol=config['audit']['cpu_head_logit_atol'],rtol=config['audit']['cpu_head_logit_rtol'])
        maximum=max(maximum,float(np.abs(logits[begin:end]-expected).max()))
    return maximum


def evaluate(root,cache,selection,device):
    name=selection['candidate'];path=root/'training'/name/f'epoch_{selection["epoch"]:03d}.pt'
    require(sha256_file(path)==selection['weight_sha256'],'selected weight changed')
    state=torch.load(path,map_location='cpu',weights_only=True)
    head=torch.nn.Linear(state['weight'].shape[1],4).to(device).eval().requires_grad_(False);head.load_state_dict(state,strict=True)
    before=hash_named_tensors(head.state_dict().items());results={'f0a':{},'f0b':{}}
    dest=root/'evaluation';dest.mkdir(exist_ok=True)
    for split in ('val','test','ood'):
        pause(root);values=cache[split];arrays=values['arrays']
        results['f0a'][split]=f0a_metrics(arrays['center_coarse_labels.npy'],arrays['adl_logits.npy'])
        if split=='val':
            logits=np.load(root/'training'/name/f'val_{selection["epoch"]:03d}.npy',allow_pickle=False)
        else:
            parts=[]
            with torch.inference_mode():
                for begin in range(0,values['count'],512):
                    t=torch.from_numpy(np.array(arrays['temporal_features.npy'][begin:begin+512],copy=True)).to(device)
                    s=torch.from_numpy(np.array(arrays['spatial_features.npy'][begin:begin+512],copy=True)).to(device)
                    parts.append(head(head_ops.features_for_candidate(name,t,s)).cpu().numpy())
            logits=np.concatenate(parts)
        save_array(dest/f'{split}_logits.npy',logits)
        results['f0b'][split]=head_ops.classification_metrics(arrays['center_derived_labels.npy'],logits)
        publish(root,'evaluate',split=split,windows=values['count'],total=values['count'])
    require(before==hash_named_tensors(head.state_dict().items()),'selected head mutated')
    save_json(dest/'report.json',results)
    return results


def audit_outputs(root,cache,selection,config):
    history=read(root/'training/history.json');best={};max_error=0.
    for split,wanted in selection['feature_manifest_sha256'].items():
        require(sha256_file(root/'cache'/split/'manifest.json')==wanted,'selected feature lineage changed')
    for name in head_ops.CANDIDATE_ORDER:
        for row in history:
            pause(root);epoch=row['epoch'];spec=row['candidates'][name]
            weight=root/'training'/name/f'epoch_{epoch:03d}.pt';path=weight.with_name(f'val_{epoch:03d}.npy')
            require(sha256_file(weight)==spec['weight_sha256'] and sha256_file(path)==spec['logits_sha256'],'epoch payload')
            logits=np.load(path,allow_pickle=False);state=torch.load(weight,map_location='cpu',weights_only=True)
            max_error=max(max_error,cpu_logit_audit(cache['val'],state,name,logits,config))
            independent_metrics(cache['val']['arrays']['center_derived_labels.npy'],logits,spec['metrics'])
        best[name]=max(history,key=lambda r:head_ops.selection_key(r['candidates'][name]['metrics']))
    name=max(head_ops.CANDIDATE_ORDER,key=lambda n:head_ops.selection_key(best[n]['candidates'][n]['metrics']))
    require(selection['candidate']==name and selection['epoch']==best[name]['epoch'],'independent selection')
    require(sha256_file(root/'training/history.json')==selection['history_sha256'],'history changed')
    state=torch.load(root/'training'/name/f'epoch_{selection["epoch"]:03d}.pt',map_location='cpu',weights_only=True)
    results=read(root/'evaluation/report.json')
    for split in ('val','test','ood'):
        logits=np.load(root/'evaluation'/f'{split}_logits.npy',allow_pickle=False);arrays=cache[split]['arrays']
        max_error=max(max_error,cpu_logit_audit(cache[split],state,name,logits,config))
        independent_metrics(arrays['center_derived_labels.npy'],logits,results['f0b'][split])
        require(f0a_metrics(arrays['center_coarse_labels.npy'],arrays['adl_logits.npy'])==results['f0a'][split],'F0A metric recomputation')
        truth=arrays['center_coarse_labels.npy']==10;pred=arrays['adl_logits.npy'].argmax(1)==42
        require(abs(f1_score(truth,pred,zero_division=0)-results['f0a'][split]['f1'])<1e-12,'independent F0A F1')
    report={'passed':True,'all_epoch_outputs_and_selection_verified':True,'independent_confusion_f1':True,
            'cpu_head_max_abs_error':max_error,'results':results}
    save_json(root/'independent_audit.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('preflight','smoke','all'),default='all')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config=read(CONFIG);smoke=args.stage=='smoke';contract=make_contract(config,smoke)
    root=guard(ROOT/(config['output_dir']+('_smoke' if smoke else '')))
    existing=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    disk_gate(config,max(0,(1 if smoke else 12)*1024**3-existing))
    initialize(root,contract,args.resume);torch.set_num_threads(2)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            publish(root,'preflight');save_json(root/'source_audit.json',verify_sources(config))
            if args.stage=='preflight': return
            device=device_for(config);torch.manual_seed(0)
            model=load_adl_model(ROOT/config['encoder'],ROOT/config['adl_root']/'best_adl_head.pth',device)
            before=hash_named_tensors(model.state_dict().items());cache={}
            for split in ('train','val'):
                source=source_split(config,split,smoke)
                cache[split]=extract(root,split,source,model,device,config,before)
            require(hash_named_tensors(model.state_dict().items())==before,'DSTE/ADL mutated')
            training=read(ROOT/config['matched_training_config'])
            selection=fit_heads(root,cache,training,device,smoke,contract);selection_sha=sha256_file(root/'selection.json')
            for split in ('test','ood'):
                source=source_split(config,split,smoke)
                cache[split]=extract(root,split,source,model,device,config,before,selection_sha)
            after=hash_named_tensors(model.state_dict().items());require(after==before,'frozen model changed')
            save_json(root/'invariance.json',{'before':before,'after':after,'passed':True})
            del model;torch.cuda.empty_cache()
            results=evaluate(root,cache,selection,device)
            publish(root,'audit');audit=audit_outputs(root,cache,selection,config)
            require(make_contract(config,smoke)==contract,'source code/config changed during run')
            save_json(root/'final_report.json',{'passed':True,'research_usable':not smoke,'selection':selection,
                      'results':results,'audit_sha256':sha256_file(root/'independent_audit.json'),
                      'model_frozen':True,'historical_exact_reproduction':False,'contract':contract})
            publish(root,'completed',research_usable=not smoke)
        except BaseException as error:
            publish(root,'paused' if isinstance(error,PauseRequested) else 'failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__': main()
