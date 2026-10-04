"""Guarded V2 lifting -> independent geometry audit -> materialization.

New reconstruction; never changes legacy V1 or evaluates a fall classifier.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickletools
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile
import numpy as np
import torch
import yaml

from data_gen.inspect_safer_activities import load_pinned_pickle, normalized_name, PINNED_FILES
from data_gen.preflight_safer_legacy_v1 import split_annotations, subject_id, window_starts, EXPECTED_SPLITS
from data_gen.safer_legacy_v1_gendata import derive_four_class, ARRAY_SPECS, sequence_rows
from data_gen.safer_v2_geometry import (model_input, starts_for, windows_at, flip_numpy, fuse_windows,
                                      normalize_ntu, lying_origin, H36M_TO_NTU)
from fall_pipeline.common.integrity import sha256_file, hash_named_tensors

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/safer_v2_document_reconstruction_v1.json'
CODE = ['data_gen/safer_v2_geometry.py', 'data_gen/safer_v2_reconstruction.py',
        'data_gen/inspect_safer_activities.py', 'data_gen/preflight_safer_legacy_v1.py',
        'data_gen/safer_legacy_v1_gendata.py', 'data_gen/safer_legacy_v1_candidates.py',
        'data_gen/h36m17_to_ntu25.py', 'fall_pipeline/common/integrity.py']
EXTRA_SPECS = {'root_trajectory.npy': (np.float32, lambda n: (n,64,3)),
               'dense_lying_origin_labels.npy': (np.int64, lambda n: (n,64))}


class PauseRequested(RuntimeError):
    """Stop only at an already-flushed sequence boundary."""


def pause_gate(root):
    if (root/'PAUSE_REQUESTED').exists():
        raise PauseRequested('sequence-boundary pause requested; preserve flag until approved resume')


def require(test, message):
    if not test: raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(value, encoding='utf-8')
    os.replace(temporary, path)


def save_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def guard_path(path):
    resolved = Path(path).resolve()
    require('cauca' not in str(resolved).lower(), 'CAUCA paths are forbidden')
    return resolved


def disk_gate(config, additional=0):
    free = shutil.disk_usage(ROOT).free
    require(free >= additional + config['execution']['reserve_gib'] * 1024**3, 'disk reserve gate')
    return free


def immutable_contract(config, smoke):
    upstream = guard_path(ROOT / config['model']['upstream'])
    commit = subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    require(commit == config['model']['commit'], 'MotionAGFormer revision changed')
    files = list(CODE) + [str(p.relative_to(ROOT)) for p in sorted((upstream/'model').rglob('*.py'))]
    files += [str((upstream/'utils/learning.py').relative_to(ROOT)),str((upstream/config['model']['yaml']).relative_to(ROOT))]
    for key in ('normal','ood'):
        guard_path(ROOT / config['source'][key])
        require(config['source'][key+'_sha256'] == PINNED_FILES[key]['sha256'], 'source pin changed')
    require(sha256_file(upstream/config['model']['yaml']) == config['model']['yaml_sha256'], 'model yaml changed')
    checkpoint = guard_path(ROOT/config['model']['checkpoint'])
    require(checkpoint.stat().st_size == config['model']['checkpoint_bytes'] and sha256_file(checkpoint) == config['model']['checkpoint_sha256'], 'official model asset changed')
    return {'config_sha256':sha256_file(CONFIG),'code_sha256':{p:sha256_file(ROOT/p) for p in files},
            'source':config['source'],'model':config['model'],'smoke_only':smoke,'historical_exact_reproduction':False}


def initialize(root, contract, resume):
    guard_path(root)
    require(root.parent == ROOT/'data/fall_processed/SAFER-Activities', 'output scope')
    if root.exists() and any(root.iterdir()):
        require(resume and (root/'run_contract.json').exists(), 'nonempty output; matching --resume required')
        require(read(root/'run_contract.json') == contract, 'code/config differs from existing run')
    else:
        root.mkdir(parents=True,exist_ok=True);save_json(root/'run_contract.json',contract)


def publish(root, stage, **details):
    from datetime import datetime, timezone
    state = {'stage':stage,'time':datetime.now(timezone.utc).isoformat(),**details}
    save_json(root/'status.json',state)
    print(json.dumps(state,ensure_ascii=False,allow_nan=False),flush=True)
    if root.name.endswith('_smoke'): return
    # Runtime-generated paired progress artifacts, separate from the manual protocol.
    stage_text = {'preflight':'공식 입력 검사','lifting':'3D 추론','lifting_audit':'3D 전체 구조 검산',
                  'materializing':'64-frame 입력 생성','materialization_audit':'입력 전수 검산',
                  'completed':'전처리 생성·전수 검산 완료','failed':'실행 중단·원인 확인 필요',
                  'paused':'완료된 시퀀스를 보존하고 일시 중단'}[stage]
    status = 'completed' if stage == 'completed' else 'paused' if stage in ('failed','paused') else 'in_progress'
    count = details.get('completed_sequences')
    progress = f'\n완료 sequence: {count}/{details.get("total_sequences",497)}.\n' if count is not None else ''
    internal = f'# SAFER V2 자동 실행 기록\n\n- 문서 ID: `DOC-20260920-safer-v2-run-R1`\n- 기준일: 2026-09-20\n- 상태: `{status}`\n\n단계: {stage_text}\n\n실행 root: `{root}`\n\n```json\n{json.dumps(state,ensure_ascii=False,indent=2)}\n```\n\n조건·이력은 [실행 계약](2026-09-20_safer_v2_reconstruction_internal.md)을 따른다.\n'
    shared = f'# SAFER V2 실행 진행\n\n- 문서 ID: `DOC-20260920-safer-v2-run-R1`\n- 기준일: 2026-09-20\n- 상태: `{status}`\n\n단계: {stage_text}.\n{progress}\n별도 문서 기반 재구현이며 원본 동등성이나 낙상 성능을 주장하지 않는다.\n[연구 방법·검증 범위](2026-09-20_safer_v2_reconstruction_shared.md)를 따른다.\n'
    if stage == 'completed':
        shared += '\n497 sequences/8,091,357 frames와 1,007,723개 학습·평가 windows의 구조 검산을 완료했다. F0A/F0B 성능 비교는 후속 과제다.\n'
    atomic_text(ROOT/'docs/internal/2026-09-20_safer_v2_run_internal.md', internal)
    atomic_text(ROOT/'docs/shared/2026-09-20_safer_v2_run_shared.md', shared)


def sources(config):
    datasets = {key:load_pinned_pickle(guard_path(ROOT/config['source'][key]),key) for key in ('normal','ood')}
    records = []
    keys = ('keypoint','keypoint_score','width','height','img_shape','total_frames','frame_dir','labels')
    for domain, dataset in datasets.items():
        selected = []
        for i, original in enumerate(dataset['annotations']):
            a = {k:original[k] for k in keys}
            a['uid'] = f'{domain}_{i:04d}';a['domain']=domain
            n=int(a['total_frames']); xy=np.asarray(a['keypoint']);score=np.asarray(a['keypoint_score']);y=np.asarray(a['labels'])
            require(xy.shape == (1,n,17,2) and score.shape == (1,n,17), '2D source shape')
            require(n > 0 and np.isfinite(xy).all(), 'invalid 2D coordinates')
            require(y.shape == (n,) and np.issubdtype(y.dtype,np.integer) and np.all((y>=0)&(y<16)), 'official labels')
            require(np.isfinite([a['width'],a['height']]).all() and min(a['width'],a['height'])>0, 'source image dimensions')
            # No legacy 3D survives in the inference/evaluation records.
            records.append(a);selected.append(a)
        dataset['annotations']=selected
    split_items, train_subjects, test_subjects = split_annotations(datasets['normal'],datasets['ood'])
    require(not train_subjects & test_subjects, 'official subject leakage')
    require(not {subject_id(a['frame_dir']) for a in split_items['train']} & {subject_id(a['frame_dir']) for a in split_items['val']}, 'validation subject leakage')
    split_report={}
    for split,items in split_items.items():
        for a in items: a['split']=split
        windows=sum(len(window_starts(int(a['total_frames']))) for a in items)
        require(len(items)==EXPECTED_SPLITS[split]['sequences'] and windows==config['materialization']['expected_windows'][split], 'split count/order gate')
        split_report[split]={'sequences':len(items),'windows':windows,'subjects':sorted({subject_id(a['frame_dir']) for a in items})}
    repairs={'sequences':0,'frames':0,'values':0}
    inventory=[]
    for a in records:
        bad=~np.isfinite(a['keypoint_score'][0]);repairs['sequences']+=int(bad.any());repairs['frames']+=int(bad.any(1).sum());repairs['values']+=int(bad.sum())
        inventory.append({'uid':a['uid'],'source_name':normalized_name(a['frame_dir']),'split':a['split'],'frames':int(a['total_frames']),
                          'width':float(a['width']),'height':float(a['height']), 'lifting_windows':len(starts_for(int(a['total_frames']))),
                          'source_array_sha256':{k:array_digest(a[k]) for k in ('keypoint','keypoint_score','labels')}})
    require(len(records)==497 and sum(a['total_frames'] for a in records)==8091357, 'full source count')
    require(repairs==config['lifting']['expected_confidence_repairs'], 'confidence repair count')
    return records,split_items,{'passed':True,'sequences':len(records),'frames':8091357,'repairs':repairs,'splits':split_report,'inventory':inventory}


def device_for(config):
    require(Path(sys.prefix).name=='fall_detect','fall_detect environment required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','GPU0/CUBLAS contract')
    torch.set_num_threads(2);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and '3090' in torch.cuda.get_device_name(0),'physical GPU0 RTX3090 gate')
    require(torch.cuda.mem_get_info()[0]>=config['execution']['minimum_gpu_free_gib']*1024**3,'GPU memory reserve')
    torch.cuda.set_per_process_memory_fraction(config['execution']['gpu_memory_fraction'],0)
    return torch.device('cuda:0')


def load_model(config, device):
    path=ROOT/config['model']['checkpoint']
    allowed={'torch._utils _rebuild_tensor_v2','torch FloatStorage','torch LongStorage','collections OrderedDict',
             'numpy.core.multiarray scalar','numpy dtype','_codecs encode'}
    with zipfile.ZipFile(path) as archive:
        names=[n for n in archive.namelist() if n.endswith('/data.pkl')]
        require(len(names)==1 and archive.getinfo(names[0]).file_size<10*1024**2,'checkpoint pickle envelope')
        for op,arg,_ in pickletools.genops(archive.read(names[0])):
            require(op.name!='STACK_GLOBAL' and (op.name!='GLOBAL' or arg in allowed),'checkpoint global allowlist')
    with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype(np.float64))]):
        saved=torch.load(path,map_location='cpu',weights_only=True)
    upstream=ROOT/config['model']['upstream']
    if 'model' in sys.modules:
        require(str(upstream) in str(getattr(sys.modules['model'],'__file__','')),'ambiguous upstream model import; use a fresh runner process')
    sys.path.insert(0,str(upstream))
    from utils.learning import load_model as official_load
    settings=yaml.safe_load((upstream/config['model']['yaml']).read_text())
    model=official_load(SimpleNamespace(**settings)).to(device).eval().requires_grad_(False)
    state={k.removeprefix('module.'):v for k,v in saved['model'].items()}
    require(len(state)==len(saved['model']) and all(torch.isfinite(v).all() for v in state.values()),'model state integrity')
    model.load_state_dict(state,strict=True)
    return model


def infer_sequence(a, model, device, config):
    inputs,repairs=model_input(a['keypoint'][0],a['keypoint_score'][0],a['width'],a['height'])
    starts=starts_for(len(inputs));predictions=[]
    with torch.inference_mode():
        for begin in range(0,len(starts),4):
            windows=windows_at(inputs,starts[begin:begin+4])
            normal=model(torch.from_numpy(windows).to(device)).cpu().numpy()
            flipped=model(torch.from_numpy(flip_numpy(windows)).to(device)).cpu().numpy()
            predicted=(normal+flip_numpy(flipped))/2
            predicted[:,:,0]=0
            require(np.isfinite(predicted).all(),'nonfinite lifting output')
            predictions.append(predicted)
    windows=np.concatenate(predictions)
    h36m,coverage=fuse_windows(windows,starts,len(inputs))
    ntu,root,normalization=normalize_ntu(h36m,config['ntu_proxy']['reference_torso'])
    return {'window_predictions':windows,'starts':starts,'h36m':h36m,'coverage':coverage,'ntu25':ntu,
            'root_trajectory':root,'labels':np.asarray(a['labels'],np.int64)}, {'repairs':repairs,'normalization':normalization,'model_input_sha256':array_digest(inputs)}


def lift(root,records,config,device,initial):
    model=load_model(config,device);before=hash_named_tensors(model.state_dict().items())
    dest=root/'sequences';dest.mkdir(exist_ok=True)
    for i,a in enumerate(records):
        pause_gate(root)
        path=dest/(a['uid']+'.npz');metadata=dest/(a['uid']+'.json')
        if metadata.exists():
            require(path.exists() and sha256_file(path)==read(metadata)['sha256'],'completed sequence changed')
        else:
            values,details=infer_sequence(a,model,device,config)
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as stream: np.savez(stream,**values)
            os.replace(temporary,path)
            save_json(metadata,{'uid':a['uid'],'source_name':normalized_name(a['frame_dir']),'frames':int(a['total_frames']),
                                'split':a['split'],'sha256':sha256_file(path),'model_state_sha256':before,**details})
        if i==0 or (i+1)%10==0 or i+1==len(records):
            publish(root,'lifting',completed_sequences=i+1,total_sequences=len(records))
            disk_gate(config)
            require(immutable_contract(config,initial['smoke_only'])==initial,'source/config changed during lifting')
    after=hash_named_tensors(model.state_dict().items());require(before==after,'frozen MotionAGFormer changed')
    save_json(root/'invariance.json',{'before':before,'after':after,'passed':True})
    del model;torch.cuda.empty_cache()


def audit_lifting(root,records,config):
    invariant=read(root/'invariance.json');require(invariant['passed'] and invariant['before']==invariant['after'],'frozen invariant')
    require({p.stem for p in (root/'sequences').glob('*.npz')}=={a['uid'] for a in records},'missing/unexpected sequence')
    repaired={'sequences':0,'frames':0,'values':0};frames=0;windows=0;coverage_min=10;coverage_max=0
    for a in records:
        pause_gate(root)
        base=root/'sequences'/a['uid'];meta=read(base.with_suffix('.json'))
        require(sha256_file(base.with_suffix('.npz'))==meta['sha256'] and meta['model_state_sha256']==invariant['before'],'sequence model/artifact lineage')
        with np.load(base.with_suffix('.npz'),allow_pickle=False) as saved:
            starts=starts_for(int(a['total_frames']));np.testing.assert_array_equal(saved['starts'],starts)
            h36m,count=fuse_windows(saved['window_predictions'],starts,int(a['total_frames']),independent=True)
            np.testing.assert_array_equal(h36m,saved['h36m']);np.testing.assert_array_equal(count,saved['coverage'])
            require(np.all(h36m[:,0]==0) and count.min()>=1 and count.max()<=2,'root/coverage gate')
            np.testing.assert_array_equal(saved['labels'],a['labels'])
            inputs,repairs=model_input(a['keypoint'][0],a['keypoint_score'][0],a['width'],a['height'])
            require(meta['model_input_sha256']==array_digest(inputs) and meta['repairs']==repairs,'input/repair identity')
            ntu,trajectory,norm=normalize_ntu(h36m,config['ntu_proxy']['reference_torso'])
            require(norm==meta['normalization'],'normalization metadata')
            np.testing.assert_array_equal(saved['ntu25'],ntu);np.testing.assert_array_equal(saved['root_trajectory'],trajectory)
            # Independently verify mapping and yaw using a complex XZ plane.
            mapped=h36m[:,H36M_TO_NTU].astype(np.float64)*norm['scale']
            rotated=mapped.copy();xz=(mapped[...,0]+1j*mapped[...,2])*np.exp(-1j*norm['yaw'])
            rotated[...,0],rotated[...,2]=xz.real,xz.imag
            require(np.allclose(ntu,rotated-rotated[:,1:2],atol=2e-6,rtol=2e-6),'independent mapping/normalization')
            require(np.all(ntu[:,1]==0) and np.isfinite(ntu).all(),'NTU center/finite')
            repaired['sequences']+=int(repairs['values']>0)
            for key in ('frames','values'): repaired[key]+=repairs[key]
            frames+=len(h36m);windows+=len(starts);coverage_min=min(coverage_min,int(count.min()));coverage_max=max(coverage_max,int(count.max()))
    if len(records)==497: require(repaired==config['lifting']['expected_confidence_repairs'],'repair aggregate')
    report={'passed':True,'sequences':len(records),'frames':frames,'lifting_windows':windows,'repairs':repaired,
            'coverage_min':coverage_min,'coverage_max':coverage_max,'independent_overlap_exact':True,'mapping_and_labels_verified':True,
            'legacy_3d_used':False,'cauca_read_or_write':False,'semantic_3d_accuracy_verified':False}
    save_json(root/'lifting_audit.json',report)
    return report


def chunk_values(a, saved, starts, sequence_index, origin=None):
    frame_indices=starts[:,None]+np.arange(64)
    pose=saved['ntu25'][frame_indices]
    joints=np.zeros((len(starts),3,64,25,2),np.float32)
    joints[...,0]=pose.transpose(0,3,1,2)
    coarse=np.asarray(a['labels'],np.int64)[frame_indices]
    return {'data_joint.npy':joints,'dense_coarse_labels.npy':coarse,'dense_derived_labels.npy':derive_four_class(coarse),
            'center_coarse_labels.npy':coarse[:,32],'center_derived_labels.npy':derive_four_class(coarse[:,32]),
            'sequence_index.npy':np.full(len(starts),sequence_index,np.int64),'window_start.npy':starts,
            'num_frame.npy':np.full(len(starts),64,np.int64),'root_trajectory.npy':saved['root_trajectory'][frame_indices],
            'dense_lying_origin_labels.npy':(lying_origin(a['labels']) if origin is None else origin)[frame_indices]}


def file_array_hash(path, value):
    full=hashlib.sha256();payload=hashlib.sha256();offset=int(value.offset);position=0
    with path.open('rb') as stream:
        while chunk:=stream.read(8*1024**2):
            full.update(chunk)
            if position+len(chunk)>offset: payload.update(chunk[max(0,offset-position):])
            position+=len(chunk)
    return {'bytes':path.stat().st_size,'file_sha256':full.hexdigest(),'array_payload_sha256':payload.hexdigest(),
            'shape':list(value.shape),'dtype':str(value.dtype)}


def materialize(root,split_items,config):
    specs={**ARRAY_SPECS,**EXTRA_SPECS};manifests={}
    for split,items in split_items.items():
        dest=root/'materialized'/split;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'split_manifest.json').exists():
            manifests[split]=read(dest/'split_manifest.json');continue
        plans=[window_starts(int(a['total_frames'])) for a in items]
        rows=sequence_rows(items,plans);count=sum(len(x) for x in plans)
        progress_path=dest/'progress.json';resuming=progress_path.exists()
        next_sequence=read(progress_path)['next_sequence'] if resuming else 0
        require(0<=next_sequence<=len(items),'materialization resume index')
        arrays={}
        for name,(dtype,shape_fn) in specs.items():
            path=dest/name;shape=shape_fn(count)
            require(not path.exists() or resuming,'partial arrays without progress; preserve and inspect')
            arrays[name]=np.load(path,mmap_mode='r+',allow_pickle=False) if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)
            require(arrays[name].shape==shape and arrays[name].dtype==dtype,'materialization array shape/dtype')
        save_json(progress_path,{'next_sequence':next_sequence})
        for i in range(next_sequence,len(items)):
            pause_gate(root)
            a,starts,row=items[i],plans[i],rows[i]
            with np.load(root/'sequences'/(a['uid']+'.npz'),allow_pickle=False) as archive:
                saved={k:archive[k] for k in ('ntu25','root_trajectory')}
            origin=lying_origin(a['labels'])
            for begin in range(0,len(starts),512):
                values=chunk_values(a,saved,starts[begin:begin+512],i,origin)
                lo=row['output_start']+begin;hi=lo+len(values['data_joint.npy'])
                for name,value in values.items(): arrays[name][lo:hi]=value
            for value in arrays.values(): value.flush()
            save_json(progress_path,{'next_sequence':i+1})
            if i==0 or (i+1)%25==0 or i+1==len(items):
                publish(root,'materializing',split=split,completed_sequences=i+1,total_sequences=len(items),windows=row['output_stop'],target=count)
                disk_gate(config)
        save_json(dest/'sequences.json',rows)
        names=[f'{normalized_name(a["frame_dir"])}__f{int(s):07d}' for a,starts in zip(items,plans) for s in starts]
        require(len(set(names))==count,'duplicate sample name')
        save_json(dest/'sample_names.json',names)
        payload={name:file_array_hash(dest/name,value) for name,value in arrays.items()}
        manifest={'schema':'safer_v2_document_reconstruction_v1','written_windows':count,'payload':payload,
                  'small_file_sha256':{n:sha256_file(dest/n) for n in ('sequences.json','sample_names.json')},
                  'integrity':{'writer_completed':True,'independent_audit_required':True}}
        save_json(dest/'split_manifest.json',manifest);manifests[split]=manifest
        del arrays
    return manifests


def audit_materialized(root,split_items,config):
    summaries={}
    for split,items in split_items.items():
        dest=root/'materialized'/split;manifest=read(dest/'split_manifest.json')
        specs={**ARRAY_SPECS,**EXTRA_SPECS}
        require(set(manifest['payload'])==set(specs),'materialized array inventory')
        arrays={name:np.load(dest/name,mmap_mode='r',allow_pickle=False) for name in manifest['payload']}
        for name,expected in manifest['payload'].items():
            dtype,shape_fn=specs[name]
            require(arrays[name].shape==shape_fn(config['materialization']['expected_windows'][split]) and arrays[name].dtype==dtype,'expected array shape/dtype')
            require(file_array_hash(dest/name,arrays[name])==expected,'materialized hash/shape/dtype')
        for name,digest in manifest['small_file_sha256'].items(): require(sha256_file(dest/name)==digest,'small metadata hash')
        rows=read(dest/'sequences.json');names=read(dest/'sample_names.json');cursor=0
        plans=[window_starts(int(a['total_frames'])) for a in items]
        require(rows==sequence_rows(items,plans),'sequence order and subject identity')
        for i,(a,starts) in enumerate(zip(items,plans)):
            pause_gate(root)
            with np.load(root/'sequences'/(a['uid']+'.npz'),allow_pickle=False) as archive:
                ntu,trajectory=archive['ntu25'],archive['root_trajectory']
            # Direct sequence slices and a separate class lookup, not chunk_values.
            lookup=np.array([0]*10+[1,2,3]+[0]*3,np.int64)
            for begin in range(0,len(starts),512):
                block=starts[begin:begin+512];lo=cursor;hi=lo+len(block)
                intervals=[slice(int(s),int(s)+64) for s in block]
                pose=np.stack([ntu[s] for s in intervals]).transpose(0,3,1,2)
                np.testing.assert_array_equal(arrays['data_joint.npy'][lo:hi,:,:,:,0],pose)
                require(np.all(arrays['data_joint.npy'][lo:hi,:,:,:,1]==0),'second person not zero')
                y=np.stack([a['labels'][s] for s in intervals]).astype(np.int64)
                np.testing.assert_array_equal(arrays['dense_coarse_labels.npy'][lo:hi],y)
                np.testing.assert_array_equal(arrays['dense_derived_labels.npy'][lo:hi],lookup[y])
                np.testing.assert_array_equal(arrays['root_trajectory.npy'][lo:hi],np.stack([trajectory[s] for s in intervals]))
                np.testing.assert_array_equal(arrays['center_coarse_labels.npy'][lo:hi],y[:,32])
                np.testing.assert_array_equal(arrays['center_derived_labels.npy'][lo:hi],lookup[y[:,32]])
                require(np.all(arrays['sequence_index.npy'][lo:hi]==i) and np.all(arrays['num_frame.npy'][lo:hi]==64),'window index')
                np.testing.assert_array_equal(arrays['window_start.npy'][lo:hi],block)
                require(names[lo:hi]==[f'{normalized_name(a["frame_dir"])}__f{int(s):07d}' for s in block],'sample names')
                cursor=hi
            origin=lying_origin(a['labels'])
            lo=rows[i]['output_start'];hi=rows[i]['output_stop']
            np.testing.assert_array_equal(arrays['dense_lying_origin_labels.npy'][lo:hi],origin[starts[:,None]+np.arange(64)])
        require(cursor==config['materialization']['expected_windows'][split]==manifest['written_windows']==len(names),'split window count')
        summaries[split]={'windows':cursor,'sequences':len(items),'passed':True,'sequence_slices_exact':True,'manifest_sha256':sha256_file(dest/'split_manifest.json')}
        publish(root,'materialization_audit',split=split,windows_verified=cursor)
    report={'status':'completed','research_usable':True,'schema':'safer_v2_document_reconstruction_v1','splits':summaries,
            'integrity':{'passed':True},'historical_exact_reproduction':False,'run_contract':read(root/'run_contract.json')}
    save_json(root/'materialized/materialization_manifest.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('preflight','smoke','all','audit'),default='all')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config=read(CONFIG);smoke=args.stage=='smoke';initial=immutable_contract(config,smoke)
    root=guard_path(ROOT/(config['output_dir']+('_smoke' if smoke else '')))
    existing=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    disk_gate(config,max(0,(1 if smoke else 50)*1024**3-existing) if args.stage in ('smoke','all') else 0)
    initialize(root,initial,args.resume or args.stage=='audit')
    torch.set_num_threads(2)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            publish(root,'preflight')
            records,split_items,preflight=sources(config);save_json(root/'preflight.json',preflight)
            if args.stage=='preflight': return
            if smoke: records=records[:1]
            if args.stage!='audit':
                device=device_for(config)
                lift(root,records,config,device,initial)
            publish(root,'lifting_audit')
            lifting=audit_lifting(root,records,config)
            if smoke:
                save_json(root/'smoke_report.json',{'passed':True,'research_usable':False,'lifting':lifting})
                return
            if args.stage!='audit': materialize(root,split_items,config)
            publish(root,'materialization_audit')
            materialized=audit_materialized(root,split_items,config)
            require(immutable_contract(config,False)==initial,'source/config changed')
            save_json(root/'final_report.json',{'passed':True,'lifting':lifting,'materialization':materialized['splits'],
                      'materialization_manifest_sha256':sha256_file(root/'materialized/materialization_manifest.json'),
                      'model_frozen':True,'performance_evaluated':False,'historical_exact_reproduction':False})
            publish(root,'completed')
        except BaseException as error:
            publish(root,'paused' if isinstance(error,PauseRequested) else 'failed',error_type=type(error).__name__,error=str(error))
            raise


if __name__=='__main__': main()
