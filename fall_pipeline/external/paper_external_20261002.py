"""Locked external evaluation of the existing RGB/J1/G0 path; no training."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import copy
import json
import os
import re
import shutil
import zipfile
import numpy as np
from . import rgb_document_io as io

ROOT=io.ROOT
ASSETS=ROOT/'data/source_archives/external_evidence_20261002'
SCRIPT=Path(__file__)


def paths(dataset):
    return ROOT/f'configs/external_{dataset}_20261002_r1.json',ROOT/f'data/fall_processed/RGB/external_{dataset}_20261002_r1'


def status(root,stage,**kwargs):
    io.save(root/'status.json',dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),**kwargs))
    print(stage,kwargs,flush=True)


def digest_contract(config):
    result=io.contract(config)
    result['external_runner_sha256']=io.sha(SCRIPT)
    result['input_plan_sha256']=io.sha(ROOT/config['alignment'])
    result['ground_truth_sha256']=io.sha(ROOT/config['output']/'ground_truth.json')
    result['input_exception_scope_sha256']=io.sha(ROOT/'docs/internal/2026-10-02_urfd_input_scope.json')
    return result


def setup(dataset):
    config_path,root=paths(dataset);io.CONFIG=config_path
    config=io.read(config_path)
    def locked(cfg):
        io.require(io.read(root/'contract.json')==digest_contract(cfg),'external contract changed')
        return root,io.read(root/'plan.json')
    io.locked=locked
    return config,root


def prepare(dataset):
    import cv2
    config_path,root=paths(dataset)
    if config_path.exists():
        config,_=setup(dataset);io.locked(config);return
    root.mkdir(parents=True,exist_ok=True);plan=[];truth=[]
    acquisition=io.read(ASSETS/'manifest.json');io.require(acquisition['passed'],'download incomplete')
    files={Path(r['path']).name:r for r in acquisition['files']}
    if dataset=='urfd':
        for category,count in [('adl',40),('fall',30)]:
            for n in range(1,count+1):
                sid=f'{category}-{n:02d}';zname=f'{sid}-cam0-rgb.zip';csvname=f'{sid}-data.csv'
                row=files[zname];zp=ROOT/row['path'];tp=ROOT/files[csvname]['path']
                csv=np.loadtxt(tp,delimiter=',',ndmin=2);names=sorted(row['zip_members'],key=lambda s:int(re.search(r'-(\d+)\.png$',s).group(1)))
                frame_ids=np.array([int(re.search(r'-(\d+)\.png$',s).group(1)) for s in names])
                if not np.array_equal(frame_ids,csv[:,0].astype(int)):
                    scope=io.read(ROOT/'docs/internal/2026-10-02_urfd_input_scope.json')
                    io.require(dataset=='urfd' and sid=='adl-37' and scope['adl-37']=='retain_as_processing_failure',
                               'unapproved PNG/CSV mismatch '+sid)
                    io.require(np.array_equal(frame_ids,np.arange(1,351)) and np.array_equal(csv[:,0],np.arange(1,331)),
                               'different timestamp defect')
                    plan.append(dict(id=sid,dataset=dataset,archive=row['path'],archive_sha256=row['sha256'],frames=0,
                        source_rgb_frames=350,timestamp_rows=330,preprocessing_rejection='official timestamp missing for frames331-350',
                        timestamp_source=files[csvname]['path'],timestamp_sha256=files[csvname]['sha256']))
                    truth.append(dict(id=sid,label=0,unit='sequence',source='official ADL archive category'))
                    continue
                seconds=csv[:,1]/1000;io.require(seconds[0]==0 and np.all(np.diff(seconds)>0),'URFD timestamp order')
                times=np.arange(int(np.floor(seconds[-1]*25))+1,dtype=np.float64)/25
                indices=np.searchsorted(seconds,times,side='right')-1
                io.require(np.all(indices>=0) and np.all(seconds[indices]<=times),'future PNG')
                mapping=root/f'{sid}_mapping.npz'
                io.npz(mapping,source_indices=indices,canonical_timestamps=times,source_timestamps=seconds)
                plan.append(dict(id=sid,dataset=dataset,archive=row['path'],archive_sha256=row['sha256'],
                    png_members=names,mapping=str(mapping.relative_to(ROOT)),mapping_sha256=io.sha(mapping),frames=len(times),
                    timestamp_source=files[csvname]['path'],timestamp_sha256=files[csvname]['sha256']))
                truth.append(dict(id=sid,label=int(category=='fall'),unit='sequence',source='official fall/adl archive categories'))
    elif dataset=='mcfd':
        # Only after the user's explicit choice on conflicting event annotations.
        approval=ROOT/'docs/internal/2026-10-02_mcfd_evaluation_scope.json'
        io.require(approval.exists() and io.read(approval).get('sequence_level_approved'),'MCFD scope pending')
        row=files['dataset.zip'];zp=ROOT/row['path']
        with zipfile.ZipFile(zp) as z:
            members=sorted([s for s in z.namelist() if s.endswith('.avi')])
            io.require(len(members)==192,'MCFD expected 24x8 views')
            for member in members:
                m=re.search(r'chute(\d+)/cam(\d)\.avi$',member);scenario,camera=map(int,m.groups())
                sid=f'chute{scenario:02d}_cam{camera}'
                temp=root/'metadata_video.avi'
                io.require(not temp.exists(),'metadata temporary file already exists')
                temp.write_bytes(z.read(member))
                try:
                    cap=cv2.VideoCapture(str(temp));io.require(cap.isOpened(),'MCFD video open')
                    fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
                    io.require(fps>0 and n>0,'MCFD metadata')
                finally:temp.unlink()
                seconds=np.arange(n,dtype=np.float64)/fps
                times=np.arange(int(np.floor(seconds[-1]*25))+1,dtype=np.float64)/25
                indices=np.searchsorted(seconds,times,side='right')-1
                mapping=root/f'{sid}_mapping.npz';io.npz(mapping,source_indices=indices,canonical_timestamps=times,source_timestamps=seconds)
                plan.append(dict(id=sid,dataset=dataset,archive=row['path'],archive_sha256=row['sha256'],video_member=member,
                    source_fps=fps,source_frames=n,mapping=str(mapping.relative_to(ROOT)),mapping_sha256=io.sha(mapping),frames=len(times),
                    scenario=scenario,camera=camera))
                truth.append(dict(id=sid,label=int(scenario<=22),unit='camera video',scenario=scenario,camera=camera,
                                  source='official homepage; event annotations unresolved, not used'))
    else:raise ValueError(dataset)
    io.save(root/'plan.json',plan);io.save(root/'ground_truth.json',truth)
    config=copy.deepcopy(io.read(ROOT/'configs/rgb_document_integration_v1.json'))
    config.update(experiment_id=f'EXTERNAL_{dataset.upper()}_20261002_R1',output=str(root.relative_to(ROOT)),
        alignment=str((root/'plan.json').relative_to(ROOT)),alignment_sha256=io.sha(root/'plan.json'),sample_count=len(plan),
        plan='complete official manifest, no prediction-based sample selection',
        scope='offline external sequence classification, G0 primary, no target training or calibration')
    config['execution'].update(reserve_gib=48,max_added_gib=1)
    config['evaluation']=dict(primary='G0 (source validation comparator; G2 not adopted)',
        prediction='any complete 64-frame window with 4-class argmax fall(index1)',
        short_sequences='no padding; <64 canonical frames gives no alarm and explicitly unprocessed',
        input_exception='adl-37 retained as processing failure; no inference; no alarm',
        quality_reject='no alarm; retain every video in confusion matrix and report coverage',
        target_training=False,target_threshold_fitting=False,event_metrics=False,
        urfd='camera0 all70',mcfd='per-camera video all192; scenario grouped; no multi-view fusion')
    io.save(config_path,config);io.CONFIG=config_path
    io.save(root/'contract.json',digest_contract(config))
    status(root,'prepared',videos=len(plan),canonical_frames=sum(p['frames'] for p in plan),
           shorter_than64=sum(p['frames']<64 for p in plan))


def frontend(dataset):
    import cv2,torch
    config,root=setup(dataset);_,plan=io.locked(config);dev=io.device(config)
    cv2.setNumThreads(1);os.environ['YOLO_CONFIG_DIR']=str(root/'yolo_runtime');os.environ['MPLCONFIGDIR']=str(root/'matplotlib_runtime')
    from ultralytics import YOLO
    from mmcv import Config
    from mmpose.models import build_posenet
    from mmpose.apis import inference_top_down_pose_model
    from mmpose.datasets import DatasetInfo
    from fall_pipeline.common.integrity import hash_named_tensors
    detector=YOLO(str(ROOT/config['asset_root']/'yolov8x.pt'))
    detector.predict(np.zeros((64,64,3),np.uint8),device=0,verbose=False,save=False,imgsz=640)
    detector.model.eval().requires_grad_(False)
    cfg=Config.fromfile(str(ROOT/config['pose_config']));pose=build_posenet(cfg.model)
    state=torch.load(ROOT/config['asset_root']/'vitpose-b-multi-coco.pth',map_location='cpu',weights_only=True)
    state={k.removeprefix('module.'):v for k,v in state.get('state_dict',state).items()}
    pose.load_state_dict(state,strict=True);del state
    pose.cfg=cfg;pose.eval().requires_grad_(False).to(dev);info=DatasetInfo(cfg.data.test.dataset_info)
    before={k:hash_named_tensors(m.state_dict().items()) for k,m in [('detector',detector.model),('pose',pose)]}
    settings=config['detector'];checked=set()
    def detect(frame,conf,size):
        with torch.inference_mode():
            r=detector.predict(frame,conf=conf,imgsz=size,iou=settings['iou'],classes=[0],max_det=settings['max_det'],
                               device=0,verbose=False,save=False,augment=False,half=False)[0]
        return np.column_stack((r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy())).astype(np.float32)
    for number,item in enumerate(plan,1):
        io.safety(config);dest=root/item['id'];dest.mkdir(exist_ok=True)
        if io.stage_done(dest,'frontend'):continue
        if item.get('preprocessing_rejection'):
            io.save(dest/'frontend.json',dict(passed=True,quality=dict(passed=False,
                input_rejection=item['preprocessing_rejection']),frames=0,labels_used=False,training=False,
                classifier_run=False,source_rgb_frames=item['source_rgb_frames']))
            status(root,'frontend',video=number,total=len(plan),id=item['id'],input_rejected=True)
            continue
        archive=ROOT/item['archive'];mapping=ROOT/item['mapping']
        if archive not in checked:io.require(io.sha(archive)==item['archive_sha256'],'archive changed');checked.add(archive)
        io.require(io.sha(mapping)==item['mapping_sha256'],'time mapping changed')
        with np.load(mapping,allow_pickle=False) as z:indices=z['source_indices'];times=z['canonical_timestamps']
        n=len(indices);xy=np.zeros((n,17,2),np.float32);scores=np.zeros((n,17),np.float32);boxes=np.zeros((n,5),np.float32)
        rescue=np.zeros(n,bool);previous=None;cap=None;temp=None;candidates=[];offsets=[0]
        with zipfile.ZipFile(archive) as z:
            if dataset=='mcfd':
                temp=root/'active_video.avi';io.require(not temp.exists(),'active video temp exists');temp.write_bytes(z.read(item['video_member']))
                cap=cv2.VideoCapture(str(temp));io.require(cap.isOpened(),'video decode');source_index=-1;frame=None
            try:
                for i,wanted in enumerate(indices):
                    io.safety(config)
                    if dataset=='urfd':
                        frame=cv2.imdecode(np.frombuffer(z.read(item['png_members'][int(wanted)]),np.uint8),cv2.IMREAD_COLOR)
                        io.require(frame is not None,'PNG decode')
                    else:
                        while source_index<int(wanted):
                            ok,frame=cap.read();source_index+=1;io.require(ok,'early video end')
                    height,width=frame.shape[:2]
                    base=detect(frame,settings['base_conf'],settings['base_imgsz'])
                    cand,used=io.cascade(base,lambda:detect(frame,settings['fallback_conf'],settings['fallback_imgsz']))
                    candidates.append(cand);offsets.append(offsets[-1]+len(cand));rescue[i]=used
                    chosen=io.select_track(cand,previous)
                    if chosen is not None:
                        boxes[i]=chosen;previous=chosen
                        with torch.inference_mode():
                            output,_=inference_top_down_pose_model(pose,frame,[{'bbox':chosen}],bbox_thr=None,format='xyxy',dataset='TopDownCocoDataset',dataset_info=info)
                        io.require(len(output)==1,'pose count');key=output[0]['keypoints']
                        io.require(key.shape==(17,3) and np.isfinite(key).all(),'pose shape/finite');xy[i]=key[:,:2];scores[i]=key[:,2]
                    if (i+1)%200==0:status(root,'frontend',video=number,total=len(plan),id=item['id'],frame=i+1,frames=n)
                if cap is not None:
                    while cap.read()[0]:source_index+=1
                    io.require(source_index+1==item['source_frames'],'MCFD frame count changed')
            finally:
                if cap is not None:cap.release()
                if temp is not None:temp.unlink()
        q=io.quality(boxes,xy,scores,config['quality'])
        q['insufficient_frames']=n<64;q['passed']=q['passed'] and n>=64
        io.npz(dest/'frontend.npz',xy=xy,scores=scores,boxes=boxes,rescue=rescue,candidates=np.concatenate(candidates),
            offsets=np.array(offsets,np.int64),width=np.array(width),height=np.array(height),source_indices=indices,timestamps=times)
        after={k:hash_named_tensors(m.state_dict().items()) for k,m in [('detector',detector.model),('pose',pose)]};io.require(before==after,'frontend weight mutation')
        io.save(dest/'frontend.json',dict(passed=True,quality=q,frames=n,width=width,height=height,
            payload_sha256=io.sha(dest/'frontend.npz'),models_before=before,models_after=after,labels_used=False,training=False))
        status(root,'frontend',video=number,total=len(plan),id=item['id'],quality=q)
    io.locked(config);status(root,'frontend_completed',videos=len(plan))


def score(dataset):
    from sklearn.metrics import average_precision_score
    config,root=setup(dataset);_,plan=io.locked(config)
    truth={r['id']:r for r in io.read(root/'ground_truth.json')};rows=[]
    for item in plan:
        d=root/item['id'];front=io.stage_done(d,'frontend');io.require(front,'frontend missing')
        passed=front['quality']['passed'];probability=0.;pred=0;alarms=0
        if passed:
            io.require(io.stage_done(d,'lift') and io.stage_done(d,'inference'),'incomplete inference')
            with np.load(d/'inference.npz',allow_pickle=False) as z:
                logits=z['G0'];classes=logits.argmax(1);fall=classes==1
                exp=np.exp(logits.astype(np.float64)-logits.max(1,keepdims=True));p=exp/exp.sum(1,keepdims=True)
                pred=int(fall.any());probability=float(p[:,1].max());alarms=int(np.sum(np.diff(np.r_[False,fall].astype(int))==1))
        rows.append({**truth[item['id']], 'prediction':pred,'max_fall_probability':probability,'alarm_runs':alarms,
                     'quality_passed':passed,'quality':front['quality'],'frames':item['frames']})
    def metrics(subset):
        y=np.array([r['label'] for r in subset]);p=np.array([r['prediction'] for r in subset]);s=np.array([r['max_fall_probability'] for r in subset])
        tp=int(np.sum((y==1)&(p==1)));fp=int(np.sum((y==0)&(p==1)));fn=int(np.sum((y==1)&(p==0)));tn=int(np.sum((y==0)&(p==0)))
        return dict(videos=len(y),tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,specificity=tn/(tn+fp) if tn+fp else None,
            accuracy=float(np.mean(y==p)),ap=float(average_precision_score(y,s)),quality_pass=sum(r['quality_passed'] for r in subset),
            rejected_positive=sum(r['label']==1 and not r['quality_passed'] for r in subset),
            rejected_negative=sum(r['label']==0 and not r['quality_passed'] for r in subset))
    result=dict(passed=True,dataset=dataset,unit='sequence' if dataset=='urfd' else 'camera video',
                primary='G0, source-trained fixed model',metrics=metrics(rows),rows=rows,event_metrics=False,
                config_sha256=io.sha(io.CONFIG),contract_sha256=io.sha(root/'contract.json'),ground_truth_sha256=io.sha(root/'ground_truth.json'))
    if dataset=='mcfd':result['per_camera']={str(c):metrics([r for r in rows if r['camera']==c]) for c in range(1,9)}
    io.save(root/'evaluation.json',result);status(root,'completed',metrics=result['metrics'])


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','frontend','lift','infer','score']);p.add_argument('--dataset',choices=['urfd','mcfd'],required=True);a=p.parse_args()
    if a.stage=='prepare':prepare(a.dataset)
    elif a.stage=='frontend':frontend(a.dataset)
    elif a.stage=='score':score(a.dataset)
    else:
        config,root=setup(a.dataset)
        if a.stage=='lift':
            from .rgb_document_lift import main as run
        else:
            from .rgb_document_infer import main as run
        run();status(root,a.stage+'_completed')

if __name__=='__main__':main()
