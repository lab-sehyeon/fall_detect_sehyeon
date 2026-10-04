"""Frozen external RGB adapter for the source-trained official FLASH network.

MediaPipe normalized 33-point XYZ, native ordered frames, offline 100-frame
blocks. This adapter is ours, not a released author Le2i/CAUCA evaluator.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

from flash_reproduction_20261004 import ROOT, REPO, EVIDENCE, OUT as TRAIN, sha, read, save, require

OUT = ROOT/'data/fall_processed/RGB/flash_external_20261004_r1'
PARENT = ROOT/'data/fall_processed/RGB/original_baselines_20261004_r2'
RUNTIME = ROOT/'data/source_archives/FLASH_20261004_r1/runtime'
CONFIG = ROOT/'configs/flash_external_20261004_r1.json'


def status(stage,**kw):
    row=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**kw)
    save(OUT/'status.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)


def prepare():
    require(not (OUT/'contract.json').exists(),'external contract already frozen')
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('plan.json','ground_truth.json','own_predictions.json'):
        (OUT/name).write_bytes((PARENT/name).read_bytes())
    cfg=dict(experiment_id='FLASH_EXTERNAL_20261004_R1',
        model='source-trained official FLASH network with explicitly documented IO corrections',
        input={'mediapipe':'0.10.21','model_complexity':1,'static_image_mode':False,'smooth_landmarks':True,
               'enable_segmentation':False,'min_detection_confidence':.5,'min_tracking_confidence':.5,
               'coordinates':'pose_landmarks normalized x/y/z, standard MediaPipe landmark order',
               'image':'full original frame BGR->RGB; no manually annotated crop or GrabCut',
               'timing':'all native decoded frames in order; preserve original PTS; no temporal resampling',
               'missing':'causal carry-forward of latest detected pose in context; zero before first detection; mask undetected output frames false',
               'whole_video_failure':'decode/hash/shape error or no detected pose; keep denominator, empty predictions'},
        temporal={'length':100,'stride':100,'tail':'repeat final frame to 100; ignore padded outputs',
                  'aggregation':'each real frame occurs once; sigmoid logit>0; no smoothing/refractory',
                  'causal':False,'meaning':'offline frame localization; padded temporal convolutions see future frames within block; no online latency claim'},
        evaluation={'early_seconds':.5,'late_seconds':3.,'alarm':'rising edges of detected positive frames',
                    'video_prediction':'one or more alarms','failure':'retain all videos and GT; empty predictions'},
        target_training=False,target_threshold_tuning=False,source_threshold='official logit>0 (probability>0.5)',
        provenance={'frontend_defaults':'https://chuoling.github.io/mediapipe/solutions/pose.html',
                    'skeleton_source':'https://github.com/Tresor-Koffi/3D_skeletons-UP-Fall-Dataset',
                    'external_adapter':'new evaluation integration fixed before target inference; not official author RGB evaluation'},
        limitations=['source historical MediaPipe version and extraction code unavailable',
                     'source dataset README describes crop/GrabCut; no target annotation crop or manual mask used',
                     'source normalized-coordinate interpretation based on supplied coordinate distribution and MediaPipe API; no world-coordinate conversion',
                     'official network topology retained even where joint names in code differ from standard MediaPipe names'],
        execution={'pose_workers':4,'pose_device':'CPU','classifier_device':'physical GPU0','classifier_batch':4})
    save(CONFIG,cfg)
    frozen=[Path(__file__),CONFIG,*[OUT/n for n in ('plan.json','ground_truth.json','own_predictions.json')],
            ROOT/'fall_pipeline/external/le2i_current_evaluation.py',TRAIN/'source_contract.json',
            RUNTIME/'mediapipe/python/solutions/pose.py',
            *RUNTIME.glob('mediapipe/modules/pose*/*.tflite'),
            *RUNTIME.glob('mediapipe/modules/pose*/*.binarypb')]
    save(OUT/'contract.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in frozen},
                                 created=datetime.now(timezone.utc).isoformat()))
    status('prepared',videos=230,native_frames=sum(p['source_frames'] for p in read(OUT/'plan.json')))


def check_contract():
    for p,d in read(OUT/'contract.json')['files'].items():
        require(sha(ROOT/p)==d,'external contract changed: '+p)


def init_worker():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['MPLCONFIGDIR']='/tmp/flash_mpl'
    sys.path.insert(0,str(RUNTIME))
    global mp,cv2,np
    import mediapipe as mp
    import cv2
    import numpy as np
    cv2.setNumThreads(1)


def pose_video(item):
    start=time.monotonic();dest=OUT/item['id'];dest.mkdir(exist_ok=True)
    if (dest/'pose.json').exists():
        r=read(dest/'pose.json');require(r['contract_sha256']==sha(OUT/'contract.json'),'stale pose')
        if r['processed']:require(sha(dest/'pose.npz')==r['payload_sha256'],'pose changed')
        return r
    cap=None;count=0
    try:
        require(sha(ROOT/item['video'])==item['video_sha256'],'video hash')
        require(sha(ROOT/item['trace'])==item['trace_sha256'],'trace hash')
        trace=read(ROOT/item['trace']);n=item['source_frames'];pts=np.asarray(trace['pts'],np.int64)
        require(len(pts)==n==len(trace['bgr_sha256']),'trace lengths')
        timestamps=(pts-trace['start_pts'])*trace['time_base_num']/trace['time_base_den']
        require(np.all(np.diff(timestamps)>0),'time order')
        poses=np.zeros((n,33,3),np.float32);vis=np.zeros((n,33),np.float32);detected=np.zeros(n,bool)
        cap=cv2.VideoCapture(str(ROOT/item['video']));require(cap.isOpened(),'video open')
        with mp.solutions.pose.Pose(static_image_mode=False,model_complexity=1,smooth_landmarks=True,
             enable_segmentation=False,min_detection_confidence=.5,min_tracking_confidence=.5) as extractor:
            for i in range(n):
                ok,frame=cap.read();require(ok,'early EOF')
                require(hashlib.sha256(frame.tobytes()).hexdigest()==trace['bgr_sha256'][i],'decoded frame hash')
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);rgb.flags.writeable=False
                result=extractor.process(rgb)
                if result.pose_landmarks is not None:
                    values=np.asarray([[p.x,p.y,p.z,p.visibility] for p in result.pose_landmarks.landmark],np.float32)
                    require(values.shape==(33,4) and np.isfinite(values).all(),'pose values')
                    poses[i]=values[:,:3];vis[i]=values[:,3];detected[i]=True
                elif i:
                    poses[i]=poses[i-1]
                count+=1
        require(not cap.read()[0],'extra frames');require(detected.any(),'no pose in entire video')
        np.savez_compressed(dest/'pose.npz',xyz=poses,visibility=vis,detected=detected,timestamps=timestamps,source_pts=pts)
        row=dict(id=item['id'],scope=item['scope'],processed=True,frames=n,detected_frames=int(detected.sum()),
                 missing_frames=int((~detected).sum()),pixels_verified=count,payload_sha256=sha(dest/'pose.npz'),
                 seconds=time.monotonic()-start,contract_sha256=sha(OUT/'contract.json'))
    except Exception as exc:
        row=dict(id=item['id'],scope=item['scope'],processed=False,frames=count,error=f'{type(exc).__name__}: {exc}',
                 seconds=time.monotonic()-start,contract_sha256=sha(OUT/'contract.json'))
    finally:
        if cap is not None:cap.release()
    save(dest/'pose.json',row);return row


def poses():
    check_contract();plan=read(OUT/'plan.json');start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker) as pool:
        futures=[pool.submit(pose_video,p) for p in plan]
        for f in as_completed(futures):
            rows.append(f.result())
            status('pose_extraction',completed=len(rows),total=len(plan),failures=sum(not r['processed'] for r in rows),
                   frames=sum(r['frames'] for r in rows),seconds=time.monotonic()-start)
    save(OUT/'pose_summary.json',dict(passed=True,videos=len(rows),processed=sum(r['processed'] for r in rows),
         frames=sum(r['frames'] for r in rows),detected_frames=sum(r.get('detected_frames',0) for r in rows),
         seconds=time.monotonic()-start,rows=sorted(rows,key=lambda r:r['id'])))
    check_contract()


def blocks(x):
    import numpy as np
    parts=[]
    for start in range(0,len(x),100):
        a=x[start:start+100]
        if len(a)<100:a=np.concatenate([a,np.repeat(a[-1:],100-len(a),axis=0)])
        parts.append(a)
    return np.stack(parts)


def infer():
    import numpy as np
    import torch
    check_contract();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','GPU0 only')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'GPU0 unavailable')
    source=read(TRAIN/'source_evaluation.json');require(source['passed'],'source training incomplete')
    require(source['checkpoint_sha256']==sha(TRAIN/'best.pt'),'selected checkpoint changed')
    require(source['scaler_sha256']==sha(TRAIN/'sc1.pkl'),'source scaler changed')
    require(read(OUT/'pose_summary.json')['videos']==230,'pose work incomplete')
    torch.set_num_threads(4)
    sys.path.insert(0,str(REPO));from HGCN.hmamba import HyperMamba
    model=HyperMamba(device='cuda').cuda()
    cp=torch.load(TRAIN/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(cp['model_state_dict'],strict=True);model.eval()
    with np.load(TRAIN/'source.npz') as z:mean=z['mean'];scale=z['scale'];graph=z['graph']
    save(OUT/'model_lock.json',dict(checkpoint_sha256=sha(TRAIN/'best.pt'),scaler_sha256=sha(TRAIN/'sc1.pkl'),
        source_evaluation_sha256=sha(TRAIN/'source_evaluation.json'),external_contract_sha256=sha(OUT/'contract.json'),
        threshold_logit=0.,created=datetime.now(timezone.utc).isoformat()))
    start=time.monotonic();total_blocks=0
    for number,item in enumerate(read(OUT/'plan.json'),1):
        dest=OUT/item['id'];r=read(dest/'pose.json')
        if not r['processed']:
            save(dest/'prediction.json',dict(id=item['id'],processed=False,error=r['error'],model_lock_sha256=sha(OUT/'model_lock.json')))
            continue
        require(sha(dest/'pose.npz')==r['payload_sha256'],'pose changed')
        with np.load(dest/'pose.npz') as z:xyz=z['xyz'];valid=z['detected'];ts=z['timestamps']
        normalized=((xyz.reshape(-1,99)-mean)/scale).astype(np.float32).reshape(xyz.shape)
        features=np.einsum('ij,tjc->tic',graph,normalized).astype(np.float32)
        batches=blocks(features);pred=[]
        with torch.no_grad():
            for k in range(0,len(batches),4):
                out=model(torch.tensor(batches[k:k+4],device='cuda'))
                require(torch.isfinite(out).all().item(),'nonfinite target logits')
                pred.append(out.cpu().numpy())
        logits=np.concatenate(pred).reshape(-1)[:len(xyz)]
        alarms=(logits>0)&valid
        np.savez_compressed(dest/'prediction.npz',logits=logits,alarms=alarms,timestamps=ts,detected=valid)
        save(dest/'prediction.json',dict(id=item['id'],processed=True,frames=len(xyz),blocks=len(batches),
             payload_sha256=sha(dest/'prediction.npz'),pose_sha256=r['payload_sha256'],model_lock_sha256=sha(OUT/'model_lock.json')))
        total_blocks+=len(batches)
        if number%10==0:status('inference',completed=number,total=230,blocks=total_blocks,seconds=time.monotonic()-start)
    require(sha(TRAIN/'best.pt')==source['checkpoint_sha256'],'checkpoint mutated')
    save(OUT/'inference_summary.json',dict(videos=230,blocks=total_blocks,seconds=time.monotonic()-start,passed=True))
    status('inference_complete',videos=230,blocks=total_blocks,seconds=time.monotonic()-start)


def score():
    import numpy as np
    sys.path.insert(0,str(ROOT))
    from fall_pipeline.external.le2i_current_evaluation import match_events,metrics
    check_contract();require(read(OUT/'inference_summary.json')['passed'],'inference incomplete')
    gt={r['id']:r for r in read(OUT/'ground_truth.json')};rows=read(OUT/'own_predictions.json')
    for item in read(OUT/'plan.json'):
        dest=OUT/item['id'];r=read(dest/'prediction.json');pred=[]
        if r['processed']:
            require(sha(dest/'prediction.npz')==r['payload_sha256'],'prediction changed')
            with np.load(dest/'prediction.npz') as z:
                a=z['alarms'];idx=np.flatnonzero(a&~np.r_[False,a[:-1]])
                pred=[dict(frame=int(i),time=float(z['timestamps'][i])) for i in idx]
        g=gt[item['id']]
        rows.append(dict(id=item['id'],scope=item['scope'],model='flash',processed=r['processed'],episodes=g['episodes'],
                         predictions=pred,video_prediction=int(bool(pred)),event=match_events(pred,g['episodes'])))
    summaries={}
    for model in ('own','flash'):
        summaries[model]={}
        for scope in ('le2i130','cauca100'):
            rr=[r for r in rows if r['model']==model and r['scope']==scope]
            event=metrics(*(sum(r['event'][k] for r in rr) for k in ('tp','fp','fn')))
            tp=sum(bool(r['episodes']) and bool(r['video_prediction']) for r in rr)
            fp=sum(not r['episodes'] and bool(r['video_prediction']) for r in rr)
            fn=sum(bool(r['episodes']) and not r['video_prediction'] for r in rr)
            tn=sum(not r['episodes'] and not r['video_prediction'] for r in rr)
            summaries[model][scope]=dict(videos=len(rr),processed=sum(r['processed'] for r in rr),event=event,
                video={**metrics(tp,fp,fn),'tn':tn,'accuracy':(tp+tn)/len(rr)},fall_events=sum(len(r['episodes']) for r in rr))
    save(OUT/'evaluation.json',dict(passed=False,pending_independent_audit=True,summaries=summaries,rows=rows,
                                   target_training=False,target_tuning=False,contract_sha256=sha(OUT/'contract.json')))
    status('scored_pending_audit',summaries=summaries)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','poses','infer','score'])
    a=p.parse_args();globals()[a.stage]()
