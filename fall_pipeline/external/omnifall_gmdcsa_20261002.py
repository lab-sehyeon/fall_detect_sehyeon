"""Fixed-model official GMDCSA24-CS segment evaluation; no target training."""
from pathlib import Path
import argparse,copy,csv,hashlib,json,os
from datetime import datetime,timezone
import numpy as np
from . import rgb_document_io as io

ROOT=io.ROOT
ASSET=ROOT/'data/source_archives/OmniFall/gmdcsa24_cs_20261002_r1'
OUTPUT=ROOT/'data/fall_processed/RGB/omnifall_gmdcsa_cs_20261002_r1'
CONFIG=ROOT/'configs/omnifall_gmdcsa_cs_20261002_r1.json'

def status(stage,**kwargs):
    io.save(OUTPUT/'status.json',dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),**kwargs))
    print(stage,kwargs,flush=True)

def contract(config):
    result=io.contract(config)
    result['own_code']={p:io.sha(ROOT/p) for p in ['fall_pipeline/external/omnifall_gmdcsa_20261002.py','fall_pipeline/external/omnifall_sampling_20261002.py']}
    result['ground_truth_sha256']=io.sha(OUTPUT/'ground_truth.json')
    result['official_metadata']={str(p.relative_to(ASSET)):io.sha(p) for p in (ASSET/'metadata').rglob('*') if p.is_file()}
    result['provenance']={p:io.sha(ASSET/p) for p in ['source_manifest.json','video_manifest.json','runtime_manifest.json','package_source/src/omnifall/_decode.py']}
    runtime=io.read(ASSET/'runtime_manifest.json')
    for p,h in runtime['runtime_files'].items():io.require(io.sha(ASSET/'runtime'/p)==h,'isolated runtime changed')
    return result

def setup():
    io.CONFIG=CONFIG;cfg=io.read(CONFIG)
    def locked(config):
        io.require(io.read(OUTPUT/'contract.json')==contract(config),'GMDCSA frozen contract changed')
        return OUTPUT,io.read(OUTPUT/'plan.json')
    io.locked=locked
    return cfg

def prepare():
    from .omnifall_sampling_20261002 import source_trace,sampling_trace,decode
    import pyarrow.parquet as pq
    if CONFIG.exists():cfg=setup();io.locked(cfg);return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    manifest=io.read(ASSET/'video_manifest.json');io.require(manifest['passed'],'source incomplete')
    videos={p['path']:p for p in manifest['files']}
    table=pq.read_table(ASSET/'metadata/parquet/gmdcsa24-cs/test-00000-of-00001.parquet');rows=table.to_pylist()
    paths=[r['path'] for r in csv.DictReader((ASSET/'metadata/splits/cs/gmdcsa24/test.csv').open())]
    csvrows=[r for r in csv.DictReader((ASSET/'metadata/labels/GMDCSA24.csv').open()) if r['path'] in set(paths)]
    key=lambda r:(r['path'],int(r['label']),round(float(r['start']),5),round(float(r['end']),5),int(r['subject']),int(r['cam']))
    io.require(sorted(map(key,rows))==sorted(map(key,csvrows)),'official parquet/CSV mismatch')
    io.require(len(rows)==93 and len(set(paths))==37 and sum(r['label']==1 for r in rows)==17,'unexpected official scope')
    io.require({r['subject'] for r in rows}=={4} and {r['path'] for r in rows}==set(videos),'subject/video mismatch')
    io.require('`1|fall`' in (ASSET/'metadata/LABELS.md').read_text(),'fall label mapping')
    traces={}
    for number,path in enumerate(paths,1):
        v=videos[path];video=ROOT/v['local'];io.require(io.sha(video)==v['sha256'],'video changed')
        dest=OUTPUT/'source_traces'/(path.replace('/','_')+'.json')
        if dest.exists():trace=io.read(dest)
        else:trace=source_trace(video);io.save(dest,trace)
        traces[path]=trace;status('prepare_source_trace',video=number,total=len(paths),path=path,frames=len(trace['frames']))
    plan=[];truth=[]
    for index,row in enumerate(rows):
        video=videos[row['path']];start,end=map(float,(row['start'],row['end']));io.require(end>start,'empty segment')
        sampled=sampling_trace(traces[row['path']],start,end)
        pixels=decode(ROOT/video['local'],start,end,sampled)
        item=dict(id=f'gmdcsa_cs_{index:04d}',row_index=index,path=row['path'],video=video['local'],video_sha256=video['sha256'],
            start=start,end=end,frames=64,sampling=sampled,decoded_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
        plan.append(item);truth.append(dict(id=item['id'],row_index=index,label=int(row['label']),fall=int(row['label']==1),subject=int(row['subject'])))
        status('prepare_segment',segment=index+1,total=93,unique_frames=sampled['unique_frames'])
        del pixels
    io.save(OUTPUT/'plan.json',plan);io.save(OUTPUT/'ground_truth.json',truth)
    cfg=copy.deepcopy(io.read(ROOT/'configs/rgb_document_integration_v1.json'))
    cfg.update(experiment_id='OMNIFALL_GMDCSA_CS_20261002_R1',output=str(OUTPUT.relative_to(ROOT)),
        alignment=str((OUTPUT/'plan.json').relative_to(ROOT)),alignment_sha256=io.sha(OUTPUT/'plan.json'),sample_count=93,
        scope='official ground-truth segment Fall-vs-rest classification; current frozen J1+separate G0; no event/recovery/16class claim',
        historical_exact_reproduction=False)
    cfg['execution'].update(reserve_gib=64,max_added_gib=2)
    cfg['benchmark']=dict(hf_revision=io.read(ASSET/'source_manifest.json')['hf_revision'],config='gmdcsa24-cs',split='test',
        videos=37,segments=93,positive=17,negative=76,subject=[4],gt_positive_class=1,
        sampling='official omnifall0.2.0 uniform RGB64; inclusive endpoints; nearest presentation timestamp; official repeat-last at EOF',
        sampling_is_physical_25fps=False,target_fps_argument=25,target_fps_ignored_by_uniform=True,
        no_video_context_outside_official_sampler=True,reset_track_per_segment=True,
        primary_head='G0 selected before target predictions; separately source-trained after final J1 adapter',
        decision='one G0 4-class argmax per 64-sample segment; class1 positive; others negative',
        quality_fail='negative/no alarm; retain denominator and report failed positive/negative counts',
        labels_only_used_in_scoring=True,target_training=False,target_calibration=False,
        pretrained_overlap_claim='no task-specific GMDCSA training; untouched/pretraining exclusion not proven',
        global_motion_features_used_for_primary=False)
    io.save(CONFIG,cfg);io.CONFIG=CONFIG;io.save(OUTPUT/'contract.json',contract(cfg))
    status('prepared',videos=37,segments=93,positive=17,negative=76)

def frontend():
    from .omnifall_sampling_20261002 import decode
    import cv2,torch
    cfg=setup();_,plan=io.locked(cfg);dev=io.device(cfg);cv2.setNumThreads(1)
    os.environ['YOLO_CONFIG_DIR']=str(OUTPUT/'yolo_runtime');os.environ['MPLCONFIGDIR']=str(OUTPUT/'matplotlib_runtime')
    from ultralytics import YOLO
    from mmcv import Config
    from mmpose.models import build_posenet
    from mmpose.apis import inference_top_down_pose_model
    from mmpose.datasets import DatasetInfo
    from fall_pipeline.common.integrity import hash_named_tensors
    detector=YOLO(str(ROOT/cfg['asset_root']/'yolov8x.pt'))
    detector.predict(np.zeros((64,64,3),np.uint8),device=0,verbose=False,save=False,imgsz=640)
    detector.model.eval().requires_grad_(False)
    conf=Config.fromfile(str(ROOT/cfg['pose_config']));pose=build_posenet(conf.model)
    saved=torch.load(ROOT/cfg['asset_root']/'vitpose-b-multi-coco.pth',map_location='cpu',weights_only=True)
    weights={k.removeprefix('module.'):v for k,v in saved.get('state_dict',saved).items()}
    pose.load_state_dict(weights,strict=True);del saved,weights
    pose.cfg=conf;pose.eval().requires_grad_(False).to(dev);info=DatasetInfo(conf.data.test.dataset_info)
    before={k:hash_named_tensors(m.state_dict().items()) for k,m in [('detector',detector.model),('pose',pose)]}
    settings=cfg['detector'];checked=set()
    def detect(frame,threshold,size):
        with torch.inference_mode():
            r=detector.predict(frame,conf=threshold,imgsz=size,iou=settings['iou'],classes=[0],max_det=settings['max_det'],
                device=0,verbose=False,save=False,augment=False,half=False)[0]
        return np.column_stack((r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy())).astype(np.float32)
    for number,item in enumerate(plan,1):
        io.safety(cfg);dest=OUTPUT/item['id'];dest.mkdir(exist_ok=True)
        if io.stage_done(dest,'frontend'):continue
        video=ROOT/item['video']
        if video not in checked:io.require(io.sha(video)==item['video_sha256'],'video changed');checked.add(video)
        rgb=decode(video,item['start'],item['end'],item['sampling'])
        io.require(hashlib.sha256(rgb.tobytes()).hexdigest()==item['decoded_rgb_sha256'],'decoded pixels changed')
        n,h,w,_=rgb.shape;xy=np.zeros((n,17,2),np.float32);scores=np.zeros((n,17),np.float32);boxes=np.zeros((n,5),np.float32)
        candidates=[];offsets=[0];rescue=np.zeros(n,bool);previous=None
        for i in range(n):
            io.safety(cfg);frame=cv2.cvtColor(rgb[i],cv2.COLOR_RGB2BGR)
            base=detect(frame,settings['base_conf'],settings['base_imgsz'])
            cand,used=io.cascade(base,lambda:detect(frame,settings['fallback_conf'],settings['fallback_imgsz']))
            candidates.append(cand);offsets.append(offsets[-1]+len(cand));rescue[i]=used;chosen=io.select_track(cand,previous)
            if chosen is not None:
                boxes[i]=chosen;previous=chosen
                with torch.inference_mode():
                    output,_=inference_top_down_pose_model(pose,frame,[{'bbox':chosen}],bbox_thr=None,format='xyxy',dataset='TopDownCocoDataset',dataset_info=info)
                io.require(len(output)==1,'pose count');key=output[0]['keypoints'];io.require(key.shape==(17,3) and np.isfinite(key).all(),'pose finite')
                xy[i]=key[:,:2];scores[i]=key[:,2]
        del rgb
        quality=io.quality(boxes,xy,scores,cfg['quality'])
        io.npz(dest/'frontend.npz',xy=xy,scores=scores,boxes=boxes,rescue=rescue,candidates=np.concatenate(candidates),offsets=np.array(offsets),
            width=np.array(w),height=np.array(h),source_indices=np.array(item['sampling']['source_indices']),timestamps=np.array(item['sampling']['actual_seconds']))
        after={k:hash_named_tensors(m.state_dict().items()) for k,m in [('detector',detector.model),('pose',pose)]};io.require(before==after,'frontend model changed')
        io.save(dest/'frontend.json',dict(passed=True,quality=quality,frames=64,width=w,height=h,models_before=before,models_after=after,
            payload_sha256=io.sha(dest/'frontend.npz'),labels_used=False,training=False,decoded_rgb_sha256=item['decoded_rgb_sha256']))
        status('frontend',segment=number,total=93,id=item['id'],quality=quality)
    io.locked(cfg);status('frontend_completed')

def score():
    from sklearn.metrics import average_precision_score
    cfg=setup();_,plan=io.locked(cfg);truth={r['id']:r for r in io.read(OUTPUT/'ground_truth.json')};rows=[]
    for item in plan:
        d=OUTPUT/item['id'];front=io.stage_done(d,'frontend');io.require(front,'missing frontend');pred=0;prob=0.;klass=None
        if front['quality']['passed']:
            io.require(io.stage_done(d,'lift') and io.stage_done(d,'inference'),'missing inference')
            with np.load(d/'inference.npz') as z:
                logits=z['G0'];io.require(logits.shape==(1,4),'one segment one prediction');klass=int(logits[0].argmax());pred=int(klass==1)
                ex=np.exp(logits[0].astype(float)-logits[0].max());prob=float(ex[1]/ex.sum())
        rows.append(dict(**truth[item['id']],prediction=pred,predicted_class=klass,fall_score=prob,quality_passed=front['quality']['passed'],quality=front['quality']))
    y=np.array([r['fall'] for r in rows]);p=np.array([r['prediction'] for r in rows]);s=np.array([r['fall_score'] for r in rows])
    tp=int(sum((y==1)&(p==1)));fp=int(sum((y==0)&(p==1)));fn=int(sum((y==1)&(p==0)));tn=int(sum((y==0)&(p==0)))
    metric=dict(segments=len(rows),tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn),specificity=tn/(tn+fp),
        precision=tp/(tp+fp) if tp+fp else 0.,f1=2*tp/(2*tp+fp+fn),accuracy=float(np.mean(y==p)),ap=float(average_precision_score(y,s)),
        classified=sum(r['quality_passed'] for r in rows),rejected_positive=sum(not r['quality_passed'] and r['fall']==1 for r in rows),
        rejected_negative=sum(not r['quality_passed'] and r['fall']==0 for r in rows))
    report=dict(passed=True,scope='OmniFall gmdcsa24-cs test; 93 GT-provided segments; Fall binary only',metrics=metric,rows=rows,
        config_sha256=io.sha(CONFIG),contract_sha256=io.sha(OUTPUT/'contract.json'),ground_truth_sha256=io.sha(OUTPUT/'ground_truth.json'),
        target_training=False,event_evaluation=False,recovery_evaluation=False)
    io.save(OUTPUT/'evaluation.json',report);status('completed',metrics=metric)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','frontend','lift','infer','score']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='frontend':frontend()
    elif a.stage=='score':score()
    else:
        setup()
        if a.stage=='lift':from .rgb_document_lift import main as run
        else:from .rgb_document_infer import main as run
        run();status(a.stage+'_completed')

if __name__=='__main__':main()
