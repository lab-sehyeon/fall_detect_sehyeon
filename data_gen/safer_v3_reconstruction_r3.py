"""Train-only V3 pilot, gated full lifting and compact sequence cache."""
from __future__ import annotations
import argparse
import fcntl
from pathlib import Path
from datetime import datetime,timezone
import os
import numpy as np
import torch
from data_gen import safer_v2_reconstruction as v2
from data_gen import safer_v3_geometry_r3 as geometry
from data_gen import safer_v3_io as io

ROOT=v2.ROOT
CONFIG=ROOT/'configs/safer_v3_document_reconstruction_v3.json'
CODE=sorted(set(['data_gen/safer_v3_reconstruction_r3.py','data_gen/safer_v3_geometry_r3.py','data_gen/safer_v3_geometry.py']+v2.CODE+io.CODE))


def publish(root,stage,**details):
    state={'stage':stage,'time':datetime.now(timezone.utc).isoformat(),**details}
    io.save_json(root/'status.json',state);print(io.json.dumps(state,ensure_ascii=False),flush=True)
    status='completed' if stage=='completed' else 'not_selected' if stage=='not_selected' else 'paused' if stage in ('failed','paused') else 'in_progress'
    header=f'- 문서 ID: `DOC-20260922-safer-v3-run-R3`\n- 기준일: 2026-09-22\n- 상태: `{status}`\n'
    internal=f'# V3 자동 실행 기록\n\n{header}\n```json\n{io.json.dumps(state,indent=2,ensure_ascii=False)}\n```\n\nroot: `{root}`\n\n[계약](2026-09-22_v3_valid_support_internal.md)을 따른다.\n'
    shared=f'# V3 geometry 실행 진행\n\n{header}\n단계: {stage}.\n'
    if 'completed_sequences' in details: shared+=f'완료 sequence: {details["completed_sequences"]}/{details["total_sequences"]}.\n'
    shared+='\n[연구 방법](2026-09-22_v3_valid_support_shared.md)을 따른다. 의미적 3D 정확도나 낙상 성능 검증은 아니다.\n'
    for kind,value in (('internal',internal),('shared',shared)):
        v2.atomic_text(ROOT/f'docs/{kind}/2026-09-22_safer_v3_r3_run_{kind}.md',value)


def make_contract(config):
    io.require(io.sha256_file(ROOT/config['source_contract'])==config['source_config_sha256'],'V2 config changed')
    source_config=io.read(ROOT/config['source_contract']);old=ROOT/config['v2_root']
    io.require(io.sha256_file(old/'final_report.json')==config['v2_final_sha256'] and io.read(old/'final_report.json')['passed'],'V2 full gate')
    previous=v2.immutable_contract(source_config,False)
    return {'config_sha256':io.sha256_file(CONFIG),'code_sha256':{p:io.sha256_file(ROOT/p) for p in CODE},
            'v2_contract':previous,'v2_final_sha256':config['v2_final_sha256'],'torch':str(torch.__version__),
            'historical_exact_reproduction':False}


def geometry_sources(source_config):
    """The label field and legacy3D are not read into geometry records."""
    datasets={key:v2.load_pinned_pickle(ROOT/source_config['source'][key],key) for key in ('normal','ood')}
    records=[]
    for domain,dataset in datasets.items():
        items=[]
        for i,original in enumerate(dataset['annotations']):
            a={k:original[k] for k in ('keypoint','keypoint_score','width','height','total_frames','frame_dir')}
            a.update(uid=f'{domain}_{i:04d}',domain=domain);items.append(a);records.append(a)
        dataset['annotations']=items
    split_items,train_subjects,test_subjects=v2.split_annotations(datasets['normal'],datasets['ood'])
    io.require(not train_subjects&test_subjects,'official subjects overlap')
    for split,items in split_items.items():
        for a in items: a['split']=split
    io.require(len(records)==497 and sum(int(a['total_frames']) for a in records)==8091357,'source count')
    return records,split_items


def raw_windows(a,stride,model,device):
    values,repairs=v2.model_input(a['keypoint'][0],a['keypoint_score'][0],a['width'],a['height'])
    starts=v2.starts_for(len(values),stride=stride);parts=[]
    with torch.inference_mode():
        for begin in range(0,len(starts),4):
            windows=v2.windows_at(values,starts[begin:begin+4]);x=torch.from_numpy(windows).to(device)
            normal=model(x).cpu().numpy();flipped=model(torch.from_numpy(v2.flip_numpy(windows)).to(device)).cpu().numpy()
            result=(normal+v2.flip_numpy(flipped))/2;result[:,:,0]=0
            io.require(np.isfinite(result).all(),'nonfinite raw predictions');parts.append(result)
    return np.concatenate(parts),starts,{'repairs':repairs,'model_input_sha256':v2.array_digest(values)}


def save_npz(path,**values):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream: np.savez(stream,**values)
    os.replace(temporary,path)


def baseline(a,config):
    base=ROOT/config['v2_root']/'sequences'/a['uid'];meta=io.read(base.with_suffix('.json'))
    io.require(io.sha256_file(base.with_suffix('.npz'))==meta['sha256'],'saved V2 sequence hash')
    with np.load(base.with_suffix('.npz'),allow_pickle=False) as archive:
        return {key:archive[key] for key in ('h36m','window_predictions','starts')}


def pilot(root,records,config,source_config,contract,device):
    chosen=geometry.pilot_records(records,config);dest=root/'pilot';dest.mkdir(exist_ok=True)
    plan=[{'uid':a['uid'],'frame_dir':a['frame_dir'],'frames':int(a['total_frames']),'split':a['split']} for a in chosen]
    if (dest/'plan.json').exists(): io.require(io.read(dest/'plan.json')==plan,'pilot plan changed')
    else: io.save_json(dest/'plan.json',plan)
    model=v2.load_model(source_config,device);before=io.hash_named_tensors(model.state_dict().items())
    measured={f's{s}_{kind}':[] for s,kind in config['pilot']['policies']};old_metrics=[];max_error=0.;diagnostics={}
    for i,a in enumerate(chosen):
        io.pause(root);old=baseline(a,config);n=int(a['total_frames']);old_metrics.append(geometry.measure(old['h36m'],a['keypoint'][0],old['starts'],config))
        raws={}
        for stride in sorted({p[0] for p in config['pilot']['policies']},reverse=True):
            base=dest/f'{a["uid"]}_s{stride}';path=base.with_suffix('.npz');meta_path=base.with_suffix('.json')
            if meta_path.exists():
                metadata=io.read(meta_path);io.require(io.sha256_file(path)==metadata['sha256'],'pilot raw changed')
                io.require(metadata['contract']==contract and metadata['model_hash']==before,'pilot raw lineage')
                with np.load(path,allow_pickle=False) as saved: raw,starts=saved['raw'],saved['starts']
            else:
                raw,starts,details=raw_windows(a,stride,model,device);save_npz(path,raw=raw,starts=starts)
                io.save_json(meta_path,{'sha256':io.sha256_file(path),'model_hash':before,'contract':contract,**details})
            np.testing.assert_array_equal(starts,v2.starts_for(n,stride=stride));raws[stride]=(raw,starts)
            diagnostics[base.name]=geometry.overlap_diagnostics(raw,starts)
        for stride,kind in config['pilot']['policies']:
            raw,starts=raws[stride];h,cover,div=geometry.fuse(raw,starts,n,kind,config['geometry']['edge_floor'])
            independent=geometry.fuse(raw,starts,n,kind,config['geometry']['edge_floor'],True)
            for actual,expected in zip((h,cover,div),independent): np.testing.assert_array_equal(actual,expected)
            measured[f's{stride}_{kind}'].append(geometry.measure(h,a['keypoint'][0],starts,config))
            if stride==243:
                np.testing.assert_array_equal(starts,old['starts'])
                max_error=max(max_error,float(np.max(np.abs(h-old['h36m']))),float(np.max(np.abs(raw-old['window_predictions']))))
        publish(root,'pilot',completed_sequences=i+1,total_sequences=len(chosen));io.disk_gate(config)
    io.require(before==io.hash_named_tensors(model.state_dict().items()),'MotionAGFormer mutated')
    del model;torch.cuda.empty_cache()
    base_metrics=geometry.aggregate(old_metrics);metrics={name:geometry.aggregate(rows) for name,rows in measured.items()}
    checks={name:geometry.gates(m,base_metrics,config) for name,m in metrics.items() if name!='s243_uniform'}
    equality=max_error<=config['pilot']['gate']['baseline_max_abs']
    selected=next((name for name in config['pilot']['ranking'] if equality and checks[name]['passed']),None)
    report={'completed':True,'selected':selected,'saved_v2':base_metrics,'policies':metrics,'gates':checks,
            'raw_baseline_max_abs':max_error,'baseline_equality_passed':equality,'model_frozen':True,
            'labels_or_classifier_used':False,'overlap_diagnostics':diagnostics,'contract':contract,
            'plan_sha256':io.sha256_file(dest/'plan.json')}
    if (dest/'report.json').exists(): io.require(io.read(dest/'report.json')==report,'pilot report changed')
    else: io.save_json(dest/'report.json',report)
    publish(root,'pilot_selected' if selected else 'not_selected',selected=selected)
    return report


def policy_lock(root,config,contract):
    report=io.read(root/'pilot/report.json');io.require(report['contract']==contract,'pilot contract')
    name=report['selected'];io.require(name in config['pilot']['ranking'] and report['gates'][name]['passed'] and report['baseline_equality_passed'],'no passing pilot policy')
    stride,kind=name.split('_');lock={'name':name,'stride':int(stride[1:]),'kind':kind,
             'pilot_report_sha256':io.sha256_file(root/'pilot/report.json'),'contract':contract}
    path=root/'policy_lock.json'
    if path.exists(): io.require(io.read(path)==lock,'full policy changed')
    else: io.save_json(path,lock)
    return lock


def full_lift(root,records,config,source_config,contract,device):
    policy=policy_lock(root,config,contract);model=v2.load_model(source_config,device)
    before=io.hash_named_tensors(model.state_dict().items());dest=root/'sequences';dest.mkdir(exist_ok=True)
    for i,a in enumerate(records):
        io.pause(root);base=dest/a['uid'];meta=base.with_suffix('.json');path=base.with_suffix('.npz')
        if meta.exists():
            saved=io.read(meta);io.require(saved['policy']==policy and saved['model_hash']==before,'sequence lineage')
            io.require(io.sha256_file(path)==saved['sha256'],'completed full sequence changed')
        else:
            raw,starts,details=raw_windows(a,policy['stride'],model,device)
            h,cover,div=geometry.fuse(raw,starts,int(a['total_frames']),policy['kind'],config['geometry']['edge_floor'])
            save_npz(path,raw=raw,starts=starts,h36m=h,coverage=cover,divisor=div)
            io.save_json(meta,{'uid':a['uid'],'source_name':v2.normalized_name(a['frame_dir']),'split':a['split'],
                             'sha256':io.sha256_file(path),'model_hash':before,'policy':policy,**details})
        if i==0 or (i+1)%10==0 or i+1==len(records):
            publish(root,'lifting',completed_sequences=i+1,total_sequences=len(records));io.disk_gate(config)
            io.require(make_contract(config)==contract,'code/config changed')
    io.require(before==io.hash_named_tensors(model.state_dict().items()),'frozen model changed')
    io.save_json(root/'invariance.json',{'before':before,'after':before,'passed':True})
    del model;torch.cuda.empty_cache()


def audit_and_cache(root,records,split_items,config,source_config,contract):
    policy=policy_lock(root,config,contract);dest=root/'sequences';cache=root/'sequence_cache';cache.mkdir(exist_ok=True)
    invariant=io.read(root/'invariance.json');io.require(invariant['passed'] and invariant['before']==invariant['after'],'frozen invariant')
    io.require({p.stem for p in dest.glob('*.npz')}=={a['uid'] for a in records},'sequence inventory')
    # Labels are read only now, after geometry policy lock, solely for exact copying/indexing.
    full_records,full_splits,_=v2.sources(source_config);labels={a['uid']:a['labels'] for a in full_records}
    old_metrics=[];new_metrics=[];files={};frames=windows=0;cover_min=243;cover_max=0
    for i,a in enumerate(records):
        io.pause(root);base=dest/a['uid'];meta=io.read(base.with_suffix('.json'));path=base.with_suffix('.npz')
        io.require(meta['model_hash']==invariant['before'] and meta['policy']==policy and io.sha256_file(path)==meta['sha256'],'sequence model/hash/policy')
        with np.load(path,allow_pickle=False) as saved:
            n=int(a['total_frames']);starts=v2.starts_for(n,stride=policy['stride'])
            np.testing.assert_array_equal(starts,saved['starts'])
            h,cover,div=geometry.fuse(saved['raw'],starts,n,policy['kind'],config['geometry']['edge_floor'],True)
            for key,value in (('h36m',h),('coverage',cover),('divisor',div)): np.testing.assert_array_equal(value,saved[key])
            io.require(np.all(h[:,0]==0) and np.isfinite(h).all(),'root/finite')
        inp,repairs=v2.model_input(a['keypoint'][0],a['keypoint_score'][0],a['width'],a['height'])
        io.require(meta['model_input_sha256']==v2.array_digest(inp) and meta['repairs']==repairs,'source input identity')
        ntu,trajectory,norm=v2.normalize_ntu(h,source_config['ntu_proxy']['reference_torso'])
        mapped=h[:,v2.H36M_TO_NTU].astype(np.float64)*norm['scale'];xz=(mapped[...,0]+1j*mapped[...,2])*np.exp(-1j*norm['yaw'])
        rotated=mapped.copy();rotated[...,0]=xz.real;rotated[...,2]=xz.imag
        np.testing.assert_allclose(ntu,rotated-rotated[:,1:2],atol=2e-6,rtol=2e-6)
        io.require(np.all(ntu[:,1]==0),'NTU root')
        seq=cache/a['uid'];seq.mkdir(exist_ok=True)
        for name,value in (('ntu25.npy',ntu),('root_trajectory.npy',trajectory),('labels.npy',np.asarray(labels[a['uid']],np.int64)),
                           ('lying_origin.npy',v2.lying_origin(labels[a['uid']]))):
            if (seq/name).exists(): np.testing.assert_array_equal(np.load(seq/name,allow_pickle=False),value)
            else: io.save_array(seq/name,value)
        files[a['uid']]={'source_sha256':meta['sha256'],'normalization':norm,'frames':n,
                        'payload':{p.name:io.payload(p) for p in seq.glob('*.npy')}}
        if a['split']=='ood':
            old=baseline(a,config);old_metrics.append(geometry.measure(old['h36m'],a['keypoint'][0],old['starts'],config))
            new_metrics.append(geometry.measure(h,a['keypoint'][0],starts,config))
        frames+=n;windows+=len(starts);cover_min=min(cover_min,int(cover.min()));cover_max=max(cover_max,int(cover.max()))
        if i==0 or (i+1)%25==0 or i+1==len(records): publish(root,'audit_cache',completed_sequences=i+1,total_sequences=len(records));io.disk_gate(config)
    splits={}
    for split,items in full_splits.items():
        index=cache/split;index.mkdir(exist_ok=True);plans=[v2.window_starts(int(a['total_frames'])) for a in items];rows=v2.sequence_rows(items,plans)
        ids=np.concatenate([np.full(len(s),i,np.int64) for i,s in enumerate(plans)]);starts=np.concatenate(plans)
        coarse=np.concatenate([labels[a['uid']][s+32] for a,s in zip(items,plans)]).astype(np.int64)
        values={'sequence_index.npy':ids,'window_start.npy':starts,'center_coarse_labels.npy':coarse,
                'center_derived_labels.npy':v2.derive_four_class(coarse)}
        for name,value in values.items():
            old=np.load(ROOT/config['v2_root']/'materialized'/split/name,mmap_mode='r',allow_pickle=False)
            np.testing.assert_array_equal(old,value);io.save_array(index/name,value)
        io.require(len(starts)==config['full']['expected_windows'][split],'index count')
        io.save_json(index/'sequences.json',[dict(row,uid=a['uid']) for row,a in zip(rows,items)])
        io.save_json(index/'sample_names.json',[f'{v2.normalized_name(a["frame_dir"])}__f{int(s):07d}' for a,plan in zip(items,plans) for s in plan])
        splits[split]={'count':len(starts),'payload':{name:io.payload(index/name) for name in values},
                       'sequences_sha256':io.sha256_file(index/'sequences.json'),'sample_names_sha256':io.sha256_file(index/'sample_names.json')}
    old=geometry.aggregate(old_metrics);new=geometry.aggregate(new_metrics);gate=geometry.gates(new,old,config)
    report={'completed':True,'passed':gate['passed'],'structural_integrity':True,'geometry_gate':gate,'geometry_v2':old,'geometry_v3':new,
            'sequences':len(records),'frames':frames,'lifting_windows':windows,'coverage_min':cover_min,'coverage_max':cover_max,
            'policy':policy,'contract':contract,'model_frozen':True,'research_usable':gate['passed'],
            'semantic_3d_accuracy_verified':False,'historical_exact_reproduction':False,'splits':splits,'sequence_files':files}
    io.save_json(cache/'manifest.json',report);io.save_json(root/'final_report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('preflight','pilot','full','audit','all'),default='all')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();config=io.read(CONFIG);source_config=io.read(ROOT/config['source_contract'])
    contract=make_contract(config);root=io.guard(ROOT/config['output_dir']);existing=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    io.disk_gate(config,max(0,(16 if args.stage in ('all','full') else 1)*1024**3-existing))
    v2.initialize(root,contract,args.resume);torch.set_num_threads(2)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            publish(root,'source');records,splits=geometry_sources(source_config)
            chosen=geometry.pilot_records(records,config);io.save_json(root/'preflight.json',{'passed':True,'count':len(records),'pilot_uids':[a['uid'] for a in chosen],'labels_used':False})
            if args.stage=='preflight': return
            device=None if args.stage=='audit' else v2.device_for(config)
            if args.stage in ('pilot','all'):
                report=pilot(root,records,config,source_config,contract,device)
                if not report['selected']: return
            if args.stage=='pilot': return
            if args.stage!='audit': full_lift(root,records,config,source_config,contract,device)
            report=audit_and_cache(root,records,splits,config,source_config,contract)
            io.require(make_contract(config)==contract,'code/config changed')
            publish(root,'completed' if report['passed'] else 'not_selected',policy=report['policy']['name'])
        except BaseException as error:
            publish(root,'paused' if isinstance(error,io.PauseRequested) else 'failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__': main()
