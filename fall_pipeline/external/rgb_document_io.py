"""Isolated read/lineage and geometry contracts for RGB R1 (no model imports)."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/rgb_document_integration_v1.json'
CODE=['fall_pipeline/external/rgb_document_io.py','fall_pipeline/external/rgb_document_frontend.py',
      'fall_pipeline/external/rgb_document_lift.py','fall_pipeline/external/rgb_document_infer.py',
      'fall_pipeline/external/run_rgb_document_integration.py','fall_pipeline/common/global_reconstruction.py',
      'data_gen/safer_v2_geometry.py','data_gen/safer_v3_geometry.py','data_gen/safer_v2_reconstruction.py',
      'fall_pipeline/safer/extract_f0b_safer_features.py','fall_pipeline/joint/models.py',
      'model/DSTE.py','model/STTR.py','tools.py']

def require(value,message):
    if not value:raise RuntimeError(message)

def read(path):return json.loads(Path(path).read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,path)

def npz(path,**arrays):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    os.replace(tmp,path)

def safety(config):
    root=ROOT/config['output']
    require(not (root/'PAUSE_REQUESTED').exists(),'pause requested')
    require(shutil.disk_usage(ROOT).free>=config['execution']['reserve_gib']*1024**3,'disk reserve')

def device(config):
    import sys,torch
    require(Path(sys.prefix).name=='fall_detect','fall_detect required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','physical GPU0 contract')
    torch.set_num_threads(2);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and '3090' in torch.cuda.get_device_name(0),'GPU0 RTX3090')
    require(torch.cuda.mem_get_info()[0]>config['execution']['minimum_gpu_free_gib']*1024**3,'GPU free memory')
    torch.cuda.set_per_process_memory_fraction(config['execution']['gpu_memory_fraction'])
    return torch.device('cuda:0')

def contract(config):
    require(sha(ROOT/config['alignment'])==config['alignment_sha256'],'alignment changed')
    glob=ROOT/config['inference']['global_config'];require(sha(glob)==config['inference']['global_config_sha256'],'global config')
    g=read(glob);report=ROOT/g['output_dir']/'final_report.json'
    require(sha(report)==config['inference']['global_report_sha256'] and read(report)['passed'],'global report')
    assets=read(ROOT/config['asset_root']/'manifest.json');require(assets['passed'],'assets not validated')
    files={str(Path(config['asset_root'])/name):spec['sha256'] for name,spec in assets['files'].items()}
    files[str(Path(g['joint_root'])/'final/final_j1.pt')]=g['j1_weight_sha256']
    selected=read(report)['selection']['best']
    for name,row in selected.items():files[str(Path(g['output_dir'])/name/f'epoch_{row["epoch"]:03d}.pt')]=row['weight_sha256']
    controls=read(ROOT/'configs/safer_v2_controls_document_reconstruction_v1.json')
    files[controls['encoder']]=controls['encoder_sha256']
    files[str(Path(controls['adl_root'])/'best_adl_head.pth')]=controls['adl_head_sha256']
    lift=read(ROOT/config['lifting_source_config']);files[lift['model']['checkpoint']]=lift['model']['checkpoint_sha256']
    for path,wanted in files.items():require(sha(ROOT/path)==wanted,'frozen asset changed: '+path)
    upstream=ROOT/'third_party/ViTPose'
    require(subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()==config['vitpose_commit'],'ViTPose revision')
    require(not subprocess.check_output(['git','-C',str(upstream),'status','--porcelain','--untracked-files=no'],text=True).strip(),'ViTPose modified')
    code=list(CODE)+[str(p.relative_to(ROOT)) for folder in (upstream/'mmpose',ROOT/'third_party/MotionAGFormer/model') for p in sorted(folder.rglob('*.py'))]
    code += [config['pose_config'],config['lifting_source_config'],lift['model']['upstream']+'/'+lift['model']['yaml']]
    return dict(config_sha256=sha(CONFIG),code_sha256={p:sha(ROOT/p) for p in code},frozen_files=files,
                scaler_sha256=sha(ROOT/g['output_dir']/'train_scaler.npz'),alignment_sha256=config['alignment_sha256'],
                historical_exact_reproduction=False)

def locked(config):
    root=ROOT/config['output'];saved=read(root/'contract.json')
    require(saved==contract(config),'code/config/assets changed; do not resume this revision')
    return root,read(root/'plan.json')

def stage_done(dest,stage):
    path=dest/(stage+'.json')
    if not path.exists():return False
    meta=read(path)
    if 'payload_sha256' in meta:require(sha(dest/(stage+'.npz'))==meta['payload_sha256'],'stage payload changed')
    return meta

def iou(box,boxes):
    box=np.asarray(box);boxes=np.asarray(boxes)
    inter=np.maximum(0,np.minimum(box[2:4],boxes[:,2:4])-np.maximum(box[:2],boxes[:,:2])).prod(1)
    area=np.maximum(0,box[2:4]-box[:2]).prod()+np.maximum(0,boxes[:,2:4]-boxes[:,:2]).prod(1)-inter
    return np.divide(inter,area,out=np.zeros_like(inter,dtype=float),where=area>0)

def select_track(candidates,previous):
    c=np.asarray(candidates,np.float32).reshape(-1,5)
    require(np.isfinite(c).all() and np.all(c[:,2:4]>c[:,:2]) and np.all((c[:,4]>=0)&(c[:,4]<=1)),'invalid candidates')
    if not len(c):return None
    scores=c[:,4] if previous is None else .7*iou(previous,c)+.3*c[:,4]
    return c[int(np.argmax(scores))].copy()

def cascade(base,fallback):
    base=np.asarray(base,np.float32).reshape(-1,5)
    if len(base):return base,False
    rescue=np.asarray(fallback(),np.float32).reshape(-1,5)
    return (rescue[[int(np.argmax(rescue[:,4]))]],True) if len(rescue) else (rescue,False)

def quality(boxes,xy,scores,spec):
    b=np.asarray(boxes);x=np.asarray(xy);s=np.asarray(scores)
    require(len(b)>0 and b.shape==(len(s),5) and x.shape==(len(s),17,2) and s.shape[1:]==(17,),'quality shapes')
    bv=np.isfinite(b).all(1)&(b[:,2]>b[:,0])&(b[:,3]>b[:,1])&(b[:,4]>0)
    pv=bv&np.isfinite(x).all((1,2))&np.isfinite(s).all(1)&(s>0).any(1)
    pelvis=np.where(pv,np.minimum(s[:,11],s[:,12]),0)
    out=dict(bbox_coverage=float(bv.mean()),pose_coverage=float(pv.mean()),pelvis_median=float(np.median(pelvis)))
    out['passed']=out['bbox_coverage']>=spec['bbox_coverage_min'] and out['pose_coverage']>=spec['pose_coverage_min'] and out['pelvis_median']>=spec['pelvis_median_min']
    return out

def annotation(saved):
    box=saved['boxes'][:,:4].copy();box[:,2:]-=box[:,:2]
    return dict(keypoint=saved['xy'][None],keypoint_score=saved['scores'][None],bboxes=box,
                width=int(saved['width']),height=int(saved['height']),total_frames=len(box))

def windows(ntu,starts):
    n=np.asarray(ntu,np.float32);s=np.asarray(starts,np.int64)
    require(n.ndim==3 and n.shape[1:]==(25,3) and np.isfinite(n).all(),'NTU shape/finite')
    require(np.all(s>=0) and np.all(s+64<=len(n)),'window bounds')
    out=np.zeros((len(s),3,64,25,2),np.float32)
    out[...,0]=n[s[:,None]+np.arange(64)].transpose(0,3,1,2)
    return out
