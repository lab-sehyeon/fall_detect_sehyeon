"""Three distinct official SAFER architectures versus frozen DSTE/J1/G0.

New external transfer protocol. No target training or threshold selection.
"""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import argparse, copy, csv, fcntl, importlib.util, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for name in ['safer_pyskl_external_20261003','pyskl_runtime_20261003']:
    sys.path.insert(0,str(ROOT/'third_party'/name))
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/three_by_three_mpl')
from fall_pipeline.external import rgb_document_io as io
from fall_pipeline.external import temporal_continuous_20261002 as temporal

OUT=ROOT/'data/fall_processed/RGB/three_by_three_20261003_r1'
CFG=ROOT/'configs/three_by_three_20261003_r1.json'
ASSETS=ROOT/'data/source_archives/ThreeByThree_20261003'
SOURCES=ROOT/'docs/internal/2026-10-03_three_by_three_sources'
META=ROOT/'data/source_archives/OmniFall/candidate_metadata_20261002_r1'
UPSTREAM=ROOT/'third_party/safer_pyskl_external_20261003'
MODELS=['own','stgcnpp','msg3d','cnn1d']
CONFIG_NAMES={'stgcnpp':'stgcn++','msg3d':'msg3d'}

def status(stage,**kw):
    if 'total' in kw:kw['total']=len(io.read(OUT/'plan.json'))
    row=dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),**kw)
    io.save(OUT/'status.json',row);print(json.dumps(row),flush=True)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def comparator(name,device):
    import torch
    import mmcv
    if name in CONFIG_NAMES:
        from pyskl.models import build_model
        from pyskl.datasets.pipelines import Compose
        kind=CONFIG_NAMES[name];cfg=mmcv.Config.fromfile(str(UPSTREAM/f'configs/{kind}/safer_activity_xsub/non-wheelchair.py'))
        path=ASSETS/f'pyskl/2d/{kind}/non-wheelchair/non-wheelchair-epoch_16.pth'
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        actual=mmcv.Config.fromstring(checkpoint['meta']['config'],'.py')
        # Checkpoint records the official BaseDataset default explicitly.
        cfg.data.train.dataset.setdefault('test_mode',False)
        for key in ['model','train_pipeline','test_pipeline','data','optimizer','lr_config','total_epochs']:
            io.require(cfg[key]==actual[key],f'{name} official checkpoint config mismatch: {key}')
        model=build_model(cfg.model);model.load_state_dict(checkpoint['state_dict'],strict=True)
        pipeline=Compose(cfg.test_pipeline)
        def transform(xy,scores,height,width):
            row=dict(keypoint=xy[None].copy(),keypoint_score=scores[None].copy(),img_shape=(height,width),
                original_shape=(height,width),total_frames=48,start_index=0,label=-1,test_mode=True,modality='Pose',
                usable_indices=np.array([0]),usable_label=np.array([1]))
            return pipeline(row)['keypoint']
    else:
        code=module('official_cnn1d',SOURCES/'safer_keypoints_train_models_keypointcnn_1d.py')
        trans=module('official_cnn1d_transforms',SOURCES/'safer_inference_internal_transforms_transforms.py')
        model=code.KeypointCNN1D(num_classes=15,num_frames=48,motion_info=True)
        model.load_state_dict(torch.load(ASSETS/'1dcnn/CNN1D_kp/CNN1D_kp.pt',map_location='cpu',weights_only=True),strict=True)
        pipeline=trans.Compose([trans.ScaleWithNeckMotion(),trans.To1DInputShape()])
        def transform(xy,scores,height,width):
            # Exactly the released cnn1d_infer: XY only and fixed original limits.
            values=xy.copy();values[...,0]=np.clip(values[...,0],0,1919);values[...,1]=np.clip(values[...,1],0,1079)
            return pipeline(torch.from_numpy(values)).float()
    return model.eval().requires_grad_(False).to(device),transform

def synthetic():
    import torch
    torch.set_num_threads(2);torch.manual_seed(0)
    xy=np.zeros((48,17,2),np.float32);xy[...,0]=80+np.arange(17)[None]*4+np.arange(48)[:,None]*.5;xy[...,1]=40+np.arange(17)[None]*7
    scores=np.ones((48,17),np.float32);reports={}
    for name in MODELS[1:]:
        model,transform=comparator(name,'cpu');a=transform(xy,scores,480,640);b=transform(xy,scores,480,640)
        io.require(torch.equal(a,b) and torch.isfinite(a).all(),'transform determinism')
        with torch.inference_mode():
            if name=='cnn1d':p=model(a[None]).softmax(1).numpy()
            else:p=model(keypoint=a[None],return_loss=False)
        io.require(p.shape==(1,15) and np.isfinite(p).all(),'synthetic forward')
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
        reports[name]=dict(passed=True,shape=list(a.shape),parameters=sum(p.numel() for p in model.parameters()),strict_load=True)
        print(name,reports[name],flush=True)
    io.save(OUT/'synthetic_validation.json',dict(passed=True,models=reports,target_inputs_used=False))

def snapshot(cfg):
    io.CONFIG=CFG
    result=io.contract(cfg)
    files=[Path(__file__),ROOT/'scripts/audit_three_by_three_20261003.py',CFG,OUT/'plan.json',OUT/'ground_truth.json',
           ROOT/'data/fall_processed/RGB/three_by_three_20261003_r1/synthetic_validation.json',ROOT/'configs/three_by_three_20261003_r1.json',ROOT/'fall_pipeline/external/temporal_continuous_20261002.py',
           ROOT/'fall_pipeline/external/j1_g0_only.py',ROOT/'fall_pipeline/external/le2i_current_evaluation.py']
    for name in CONFIG_NAMES.values():files.append(UPSTREAM/f'configs/{name}/safer_activity_xsub/non-wheelchair.py')
    files+=list(UPSTREAM.rglob('*.py'))+list((ROOT/'third_party/pyskl_runtime_20261003').rglob('*.py'))
    files+=list(SOURCES.glob('safer_*.py'))+[SOURCES/'model_acquisition.json',SOURCES/'training_exclusion.json']
    files+=[ROOT/r['path'] for r in io.read(SOURCES/'model_acquisition.json')]
    files+=[ROOT/s['path'] for s in cfg['model'].values()]
    for d in cfg['datasets']:
        files+=[ASSETS/(d+'_manifest.json'),META/f'splits/cs/{d}/test.csv',META/f'labels/{d}.csv']
    for row in io.read(OUT/'plan.json'):
        files += [ROOT/row['video'],ROOT/row['mapping'],OUT/row['id']/'source_trace.json']
    result['experiment_files']={str(p.relative_to(ROOT)):io.sha(p) for p in sorted(set(files))}
    return result

def locked(cfg):
    io.require(snapshot(cfg)==io.read(OUT/'contract.json'),'frozen contract changed')
    return OUT,io.read(OUT/'plan.json')

def setup():
    io.CONFIG=CFG;io.locked=locked
    temporal.OUTPUT=OUT;temporal.CONFIG=CFG;temporal.setup=lambda:setup_config();temporal.status=status
    return io.read(CFG)

def setup_config():
    io.CONFIG=CFG;io.locked=locked;return io.read(CFG)

def prepare():
    io.require(not (OUT/'contract.json').exists(),'already frozen')
    cfg=io.read(CFG);io.require(io.read(ROOT/'data/fall_processed/RGB/three_by_three_20261003_r1/synthetic_validation.json')['passed'],'synthetic validation')
    plan=[];truth=[]
    for dataset in cfg['datasets']:
        receipt=io.read(ASSETS/(dataset+'_manifest.json'));io.require(receipt['passed'],'data incomplete')
        records={r['path']:r for r in receipt['records']};wanted=sorted({r['path'] for r in csv.DictReader((META/f'splits/cs/{dataset}/test.csv').open())})
        io.require(set(records)==set(wanted),'test scope differs')
        labels=list(csv.DictReader((META/f'labels/{dataset}.csv').open()))
        def trace_one(path):
            r=records[path];sid=dataset+'__'+path.replace('/','__');dest=OUT/sid;dest.mkdir(exist_ok=True)
            io.require(io.sha(ROOT/r['video'])==r['video_sha256'],'video changed')
            target=dest/'source_trace.json'
            if not target.exists():io.save(target,temporal.trace_video(ROOT/r['video']))
            return path
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(trace_one,path) for path in wanted]
            for i,f in enumerate(as_completed(futures),1):
                f.result()
                if i%20==0:status('tracing',dataset=dataset,videos=i)
        for path in wanted:
            r=records[path];sid=dataset+'__'+path.replace('/','__');dest=OUT/sid;dest.mkdir(exist_ok=True)
            io.require(io.sha(ROOT/r['video'])==r['video_sha256'],'video changed')
            trace=io.read(dest/'source_trace.json')
            duration=Fraction(trace['duration_num'],trace['duration_den']);n=math.ceil(duration*25)
            indices=temporal.exact_indices(trace['pts'],trace['time_base_num'],trace['time_base_den'],trace['start_pts'],n)
            mapping=dest/'mapping.npz';io.npz(mapping,source_indices=indices,canonical_timestamps=np.arange(n)/25,
                source_pts=np.array(trace['pts'],np.int64),time_base_num=np.array(trace['time_base_num']),time_base_den=np.array(trace['time_base_den']),start_pts=np.array(trace['start_pts']))
            item=dict(id=sid,dataset=dataset,path=path,video=r['video'],video_sha256=r['video_sha256'],frames=n,
                source_frames=len(trace['pts']),source_duration_seconds=float(duration),mapping=str(mapping.relative_to(ROOT)),mapping_sha256=io.sha(mapping),trace_sha256=io.sha(dest/'source_trace.json'))
            episodes=[dict(fall_start=float(x['start']),fall_end=float(x['end'])) for x in labels if x['path']==path and int(x['label'])==1]
            for e in episodes:io.require(0<=e['fall_start']<e['fall_end'] and e['fall_start']<float(duration),'invalid event')
            truth.append(dict(id=sid,scope=dataset,path=path,episodes=episodes));plan.append(item)
            status('preparing',dataset=dataset,videos=len(plan))
    io.save(OUT/'plan.json',plan);io.save(OUT/'ground_truth.json',truth)
    cfg.update(sample_count=len(plan),alignment=str((OUT/'plan.json').relative_to(ROOT)),alignment_sha256=io.sha(OUT/'plan.json'))
    io.save(CFG,cfg);io.save(OUT/'contract.json',snapshot(cfg));status('prepared',videos=len(plan),frames=sum(r['frames'] for r in plan))

def comparisons():
    import torch
    from fall_pipeline.common.integrity import hash_named_tensors
    cfg=setup();_,plan=locked(cfg);device=io.device(cfg)
    reports={};started=time.time()
    for name in MODELS[1:]:
        model,transform=comparator(name,device);before=hash_named_tensors(model.state_dict().items());total=0
        captured=[];head=model.fc2 if name=='cnn1d' else model.cls_head
        hook=head.register_forward_hook(lambda m,a,o:captured.append(o.detach().cpu().numpy()))
        for number,row in enumerate(plan,1):
            io.safety(cfg);dest=OUT/row['id'];meta_path=dest/(name+'.json')
            if io.stage_done(dest,name):total+=io.read(meta_path)['windows'];continue
            io.require(io.stage_done(dest,'frontend'),'missing frontend')
            with np.load(dest/'frontend.npz') as z:xy=z['xy'];scores=z['scores'];height=int(z['height']);width=int(z['width'])
            starts=np.arange(0,len(xy)-63,8,dtype=np.int64);passed=bool(len(starts) and np.any(scores>0));prob=[];logits=[]
            with torch.inference_mode():
                for first in range(0,len(starts),32):
                    io.safety(cfg);part=starts[first:first+32]
                    if not passed:
                        prob.append(np.zeros((len(part),15),np.float32));logits.append(np.zeros((len(part),15),np.float32));continue
                    batch=torch.stack([transform(xy[s+16:s+64],scores[s+16:s+64],height,width) for s in part]).to(device)
                    captured.clear()
                    if name=='cnn1d':p=model(batch).softmax(1).cpu().numpy()
                    else:p=model(keypoint=batch,return_loss=False)
                    v=captured[0];io.require(len(captured)==1 and v.shape==(len(part),15),'head shape')
                    ex=np.exp(v.astype(float)-v.max(1,keepdims=True));np.testing.assert_allclose(p,ex/ex.sum(1,keepdims=True),atol=1e-6,rtol=1e-5)
                    prob.append(p);logits.append(v)
            io.npz(dest/(name+'.npz'),probabilities=np.concatenate(prob) if prob else np.empty((0,15),np.float32),logits=np.concatenate(logits) if logits else np.empty((0,15),np.float32),window_starts=starts,window_endpoints=starts+63,input_starts=starts+16)
            io.save(meta_path,dict(passed=passed,windows=len(starts),payload_sha256=io.sha(dest/(name+'.npz')),source_frontend_sha256=io.sha(dest/'frontend.npz'),training=False,annotations_read=False))
            total+=len(starts);status('comparator',model=name,video=number,total=len(plan),windows=total,elapsed_seconds=time.time()-started)
        hook.remove();after=hash_named_tensors(model.state_dict().items());io.require(before==after,'weights mutated')
        reports[name]=dict(passed=True,windows=total,model_before=before,model_after=after);del model;torch.cuda.empty_cache()
    locked(cfg);io.save(OUT/'comparators_completed.json',dict(passed=True,models=reports,elapsed_seconds=time.time()-started))

def score():
    from sklearn.metrics import average_precision_score
    from fall_pipeline.external.le2i_current_evaluation import match_events,metrics
    cfg=setup();_,plan=locked(cfg);items={r['id']:r for r in plan};rows=[]
    for gt in io.read(OUT/'ground_truth.json'):
        dest=OUT/gt['id'];row=items[gt['id']]
        for model in MODELS:
            own=model=='own';meta=io.stage_done(dest,'frontend' if own else model)
            passed=meta['quality']['passed'] if own else meta['passed'];probs=[];alarms=[];positive=False;best=0.;windows=0
            if passed:
                stage='inference' if own else model;io.require(io.stage_done(dest,stage),'missing prediction')
                with np.load(dest/(stage+'.npz')) as z:
                    logits=z['G0' if own else 'logits'].astype(float);endpoints=z['window_endpoints'];windows=len(endpoints)
                ex=np.exp(logits-logits.max(1,keepdims=True));probs=ex/ex.sum(1,keepdims=True);index=1 if own else 9;flags=logits.argmax(1)==index
                edges=np.flatnonzero(flags & ~np.r_[False,flags[:-1]])
                alarms=[dict(frame=int(endpoints[i]),time=float(endpoints[i]/25)) for i in edges]
                positive=bool(flags.any());best=float(probs[:,index].max())
            rows.append(dict(gt,model=model,processed=bool(passed),predictions=alarms,video_prediction=int(positive),max_fall_probability=best,windows=windows,event=match_events(alarms,gt['episodes'])))
    summaries={}
    for model in MODELS:
        summaries[model]={}
        for dataset in cfg['datasets']:
            selected=[r for r in rows if r['model']==model and r['scope']==dataset];y=np.array([bool(r['episodes']) for r in selected]);p=np.array([r['video_prediction'] for r in selected],bool)
            tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
            summaries[model][dataset]=dict(videos=len(selected),fall_events=sum(len(r['episodes']) for r in selected),positive_videos=int(y.sum()),processed=sum(r['processed'] for r in selected),
                event=metrics(*(sum(r['event'][k] for r in selected) for k in ['tp','fp','fn'])),
                video=dict(metrics(tp,fp,fn),tn=tn,accuracy=float((y==p).mean()),specificity=tn/(tn+fp) if tn+fp else None,
                ap=float(average_precision_score(y,[r['max_fall_probability'] for r in selected])) if y.any() else None,
                discriminative_video_metric=bool(y.any() and (~y).any())))
    io.save(OUT/'evaluation.json',dict(passed=True,pending_independent_audit=True,summaries=summaries,rows=rows,contract_sha256=io.sha(OUT/'contract.json')))
    status('scored_pending_audit',summaries=summaries)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['synthetic','prepare','frontend','lift','infer','comparisons','score']);parser.add_argument('--dataset',choices=['edf','occu','OOPS']);args=parser.parse_args()
    if args.dataset:
        OUT=OUT/args.dataset;CFG=ROOT/f'configs/three_by_three_{args.dataset}_20261003_r1.json'
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage in ['frontend','lift','infer']:setup();getattr(temporal,args.stage)()
        else:globals()[args.stage]()
