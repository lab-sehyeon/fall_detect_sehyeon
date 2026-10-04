"""Frozen G0 evaluation with exact-PTS, continuous full-video input (new revision)."""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import argparse, copy, fcntl, hashlib, math, os, shutil, sys
import numpy as np
from . import rgb_document_io as io
from .le2i_current_evaluation import rising_edges, match_events, metrics

ROOT = io.ROOT
OUTPUT = ROOT/'data/fall_processed/RGB/temporal_continuous_20261002_r1'
CONFIG = ROOT/'configs/temporal_continuous_20261002_r1.json'
RUNTIME = ROOT/'data/source_archives/OmniFall/gmdcsa24_cs_20261002_r1/runtime'
SOURCES = {
    'le2i127': 'le2i_current_evaluation_20261002_r1',
    'le2i38': 'omnifall_le2i_continuous_20261002_r1',
    'gmdcsa37': 'omnifall_gmdcsa_cs_20261002_r1',
    'cauca19': 'omnifall_cauca_cs_20261002_r1',
}
CODE = ['fall_pipeline/external/temporal_continuous_20261002.py',
        'scripts/run_temporal_continuous_20261002.py', 'scripts/audit_temporal_continuous_20261002.py',
        'fall_pipeline/external/j1_g0_only.py', 'fall_pipeline/external/le2i_current_evaluation.py']


def av_module():
    if str(RUNTIME) not in sys.path: sys.path.insert(0, str(RUNTIME))
    import av
    return av


def status(stage, **kwargs):
    row = dict(stage=stage, time=datetime.now(timezone.utc).isoformat(), **kwargs)
    io.save(OUTPUT/'status.json', row)
    print(row, flush=True)


def exact_indices(pts, numerator, denominator, origin, count):
    # Exact integer comparison of (pts-origin)*time_base with j/25.
    ticks = (np.asarray(pts, np.int64)-origin)*int(numerator)*25
    targets = np.arange(count, dtype=np.int64)*int(denominator)
    indices = np.searchsorted(ticks, targets, side='right')-1
    io.require(len(ticks) and np.all(np.diff(ticks)>0) and np.all(indices>=0), 'invalid exact timeline')
    io.require(np.all(ticks[indices]<=targets), 'future frame')
    return indices.astype(np.int64)


def trace_video(path):
    av = av_module()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]; stream.codec_context.thread_count = 2
        base = stream.time_base; origin = stream.start_time or 0
        pts, hashes = [], []
        for frame in container.decode(stream):
            io.require(frame.pts is not None, 'missing PTS')
            pts.append(int(frame.pts))
            bgr = frame.to_ndarray(format='rgb24')[:,:,::-1].copy()
            hashes.append(hashlib.sha256(bgr.tobytes()).hexdigest())
        io.require(pts and pts[0]==origin and all(b>a for a,b in zip(pts,pts[1:])), 'PTS order/origin')
        duration = None if stream.duration is None else Fraction(stream.duration)*base
        return dict(pts=pts, time_base_num=base.numerator, time_base_den=base.denominator,
                    start_pts=origin, bgr_sha256=hashes, width=stream.codec_context.width,
                    height=stream.codec_context.height,
                    duration_num=None if duration is None else duration.numerator,
                    duration_den=None if duration is None else duration.denominator)


def contract(cfg):
    result = io.contract(cfg)
    result['own_code'] = {p:io.sha(ROOT/p) for p in CODE}
    result['own_inputs'] = {name:io.sha(OUTPUT/name) for name in
                            ['plan.json', 'ground_truth.json', 'source_snapshot.json']}
    result['active_model'] = cfg['model']
    for spec in cfg['model'].values(): io.require(io.sha(ROOT/spec['path'])==spec['sha256'], 'active weight changed')
    return result


def setup():
    io.CONFIG = CONFIG
    def locked(cfg):
        io.require(contract(cfg)==io.read(OUTPUT/'contract.json'), 'temporal revision contract changed')
        return OUTPUT, io.read(OUTPUT/'plan.json')
    io.locked = locked
    return io.read(CONFIG)


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if CONFIG.exists(): cfg=setup(); io.locked(cfg); return
    candidates, truth, snapshot = {}, [], {}
    def remember(p):
        snapshot[str(p.relative_to(ROOT))] = io.sha(p)
    for scope, source in SOURCES.items():
        parent = ROOT/'data/fall_processed/RGB'/source
        rows = io.read(parent/'plan.json'); gt = io.read(parent/'ground_truth.json')
        for name in ['plan.json', 'ground_truth.json', 'evaluation.json', 'contract.json']: remember(parent/name)
        if scope.startswith('le2i'):
            for row, label in zip(rows, gt):
                io.require(row['id']==label['id'], 'source label order')
                sid = 'le2i__'+row['id']
                item = dict(id=sid, dataset='le2i', video=row['video'], video_sha256=row['video_sha256'],
                            frames=row['frames'], old_source=str((parent/row['id']).relative_to(ROOT)),
                            old_mapping=row['mapping'], source_duration_seconds=row['source_duration_seconds'])
                if sid in candidates:
                    io.require(candidates[sid]['video_sha256']==item['video_sha256'], 'overlap video mismatch')
                else: candidates[sid]=item
                truth.append(dict(scope=scope,id=sid,path=row.get('path',row['id']),episodes=label['episodes']))
        else:
            labels = {r['id']:r for r in gt}
            groups = {}
            for row in rows: groups.setdefault(row['path'], []).append(row)
            for path, segments in groups.items():
                first = segments[0]; dataset = 'gmdcsa' if scope=='gmdcsa37' else 'cauca'
                sid = dataset+'__'+path.replace('/','__')
                candidates[sid] = dict(id=sid,dataset=dataset,video=first['video'],video_sha256=first['video_sha256'])
                episodes = [dict(fall_start=r['start'],fall_end=r['end'],row_index=r['row_index'])
                            for r in segments if labels[r['id']]['fall']]
                io.require(len(episodes)<=1, 'independent event audit expects <=1 fall per video')
                truth.append(dict(scope=scope,id=sid,path=path,episodes=episodes))
    io.require(len(candidates)==199 and len(truth)==221, 'fixed video scope')
    for scope,expected in {'le2i127':(127,96),'le2i38':(38,22),'gmdcsa37':(37,17),'cauca19':(19,9)}.items():
        selected=[r for r in truth if r['scope']==scope]
        io.require((len(selected),sum(len(r['episodes']) for r in selected))==expected,'scope denominator changed')
    plan = []
    for number, item in enumerate(candidates.values(),1):
        video = ROOT/item['video']; io.require(io.sha(video)==item['video_sha256'], 'source video changed')
        remember(video); dest=OUTPUT/item['id']; dest.mkdir(exist_ok=True)
        tracepath=dest/'source_trace.json'
        if tracepath.exists(): trace=io.read(tracepath)
        else: trace=trace_video(video);io.save(tracepath,trace)
        if item['dataset']=='le2i': count=item['frames']
        else:
            io.require(trace['duration_num'] is not None, 'missing official stream duration')
            duration=Fraction(trace['duration_num'],trace['duration_den'])
            count=math.ceil(duration*25);item['source_duration_seconds']=float(duration)
        indices=exact_indices(trace['pts'],trace['time_base_num'],trace['time_base_den'],trace['start_pts'],count)
        mapping=dest/'mapping.npz';times=np.arange(count)/25
        source_times=(np.array(trace['pts'])-trace['start_pts'])*trace['time_base_num']/trace['time_base_den']
        io.npz(mapping, source_indices=indices,canonical_timestamps=times,source_timestamps=source_times,
               source_pts=np.array(trace['pts'],np.int64),time_base_num=np.array(trace['time_base_num']),
               time_base_den=np.array(trace['time_base_den']),start_pts=np.array(trace['start_pts']))
        item.update(frames=count,source_frames=len(trace['pts']),mapping=str(mapping.relative_to(ROOT)),
                    mapping_sha256=io.sha(mapping),trace_sha256=io.sha(tracepath),reuse_full=False)
        if item.get('old_source'):
            old=ROOT/item['old_source']
            for name in ['frontend.json','frontend.npz']:remember(old/name)
            remember(ROOT/item['old_mapping'])
            with np.load(ROOT/item['old_mapping']) as values:
                old_indices=values['source_indices']; old_times=values['canonical_timestamps']
            io.require(len(old_indices)==count and np.array_equal(old_times,times), 'Le2i grid changed')
            item['changed_positions']=int(np.count_nonzero(indices!=old_indices))
            if item['changed_positions']==0:
                item['reuse_full']=True
                for name in ['frontend.json','frontend.npz','lift.json','lift.npz','quality_rejection.json']:
                    if (old/name).exists():
                        remember(old/name)
                        shutil.copy2(old/name,dest/name)
                        io.require(io.sha(old/name)==io.sha(dest/name), 'cache copy differs')
        plan.append(item)
        if number%20==0:status('prepare',completed=number,total=199)
    for row in truth:
        item=candidates[row['id']]
        for e in row['episodes']:
            io.require(0<=e['fall_start']<e['fall_end'] and e['fall_start']<item['source_duration_seconds'], 'invalid event')
    io.save(OUTPUT/'plan.json',plan);io.save(OUTPUT/'ground_truth.json',truth);io.save(OUTPUT/'source_snapshot.json',snapshot)
    cfg=copy.deepcopy(io.read(ROOT/'configs/omnifall_le2i_continuous_20261002_r1.json'))
    cfg.update(experiment_id='TEMPORAL_CONTINUOUS_20261002_R1',output=str(OUTPUT.relative_to(ROOT)),
        alignment=str((OUTPUT/'plan.json').relative_to(ROOT)),alignment_sha256=io.sha(OUTPUT/'plan.json'),
        sample_count=199,plan='fixed union of Le2i127/38, GMDCSA37, CAUCA19 full videos',
        video_root='data/source_archives',
        sampling='exact rational PTS latest past frame on physical25Hz grid',
        scope='user-authorized continuous input correction; not official segment benchmark reproduction',
        model=io.read(ROOT/'configs/j1_g0_only_evaluation_20261002_r1.json')['model'])
    cfg['execution']['max_added_gib']=8
    cfg['evaluation'].update(heads=['G0'],secondary_video_rule='any window argmax==1, inherited URFD',
        gt_time='inherited original per-scope GT intervals, unchanged',
        ground_truth='original recorded intervals; no clipping or target tuning',
        comparisons='Le2i paired events; GMDCSA/CAUCA continuous events and video, not segment F1')
    cfg['newly_specified']=['exact integer PTS mapping replaces strict float boundary comparison',
        'full video frontend/lifting before64/8 classification for all selected videos',
        'same-pixel detector and same-box pose cache reuse, track always replayed for changed mapping',
        'Le2i D1 event and URFD video rules fixed before target predictions; all failures retained']
    io.save(CONFIG,cfg);io.CONFIG=CONFIG;io.save(OUTPUT/'contract.json',contract(cfg))
    status('prepared',videos=199,memberships=221,frames=sum(r['frames'] for r in plan),
           reused_full=sum(r['reuse_full'] for r in plan))


def frontend():
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
    pc=Config.fromfile(str(ROOT/cfg['pose_config']));pose=build_posenet(pc.model)
    checkpoint=torch.load(ROOT/cfg['asset_root']/'vitpose-b-multi-coco.pth',map_location='cpu',weights_only=True)
    pose.load_state_dict({k.removeprefix('module.'):v for k,v in checkpoint.get('state_dict',checkpoint).items()},strict=True)
    del checkpoint
    pose.cfg=pc;pose.eval().requires_grad_(False).to(dev);info=DatasetInfo(pc.data.test.dataset_info)
    def hashes():return {k:hash_named_tensors(m.state_dict().items()) for k,m in [('detector',detector.model),('pose',pose)]}
    before=hashes();settings=cfg['detector']
    def detect(frame,conf,size):
        with torch.inference_mode():
            r=detector.predict(frame,conf=conf,imgsz=size,iou=settings['iou'],classes=[0],max_det=settings['max_det'],
                device=0,verbose=False,save=False,augment=False,half=False)[0]
        return np.column_stack((r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy())).astype(np.float32)
    def candidates_for(frame):
        return io.cascade(detect(frame,settings['base_conf'],settings['base_imgsz']),
                          lambda:detect(frame,settings['fallback_conf'],settings['fallback_imgsz']))
    def keypoints_for(frame,box):
        with torch.inference_mode():
            result,_=inference_top_down_pose_model(pose,frame,[{'bbox':box}],bbox_thr=None,
                format='xyxy',dataset='TopDownCocoDataset',dataset_info=info)
        io.require(len(result)==1,'pose count');key=result[0]['keypoints']
        io.require(key.shape==(17,3) and np.isfinite(key).all(),'pose shape/finite')
        return key
    av=av_module()
    for number,item in enumerate(plan,1):
        io.safety(cfg);dest=OUTPUT/item['id']
        if io.stage_done(dest,'frontend'):continue
        trace=io.read(dest/'source_trace.json');io.require(io.sha(dest/'source_trace.json')==item['trace_sha256'],'trace changed')
        with np.load(ROOT/item['mapping']) as z:indices=z['source_indices'];times=z['canonical_timestamps']
        n=len(indices);xy=np.zeros((n,17,2),np.float32);scores=np.zeros((n,17),np.float32)
        boxes=np.zeros((n,5),np.float32);rescue=np.zeros(n,bool);all_candidates=[];offsets=[0]
        cached_candidates={};cached_poses={};cache_checks=dict(detector=False,pose=False)
        if item.get('old_source'):
            old=ROOT/item['old_source'];meta=io.stage_done(old,'frontend')
            io.require(meta['models_before']==meta['models_after']==before,'cache models differ')
            with np.load(old/'frontend.npz') as z:
                for i,index in enumerate(z['source_indices']):
                    index=int(index);cand=z['candidates'][z['offsets'][i]:z['offsets'][i+1]].copy();used=bool(z['rescue'][i])
                    if index in cached_candidates:
                        np.testing.assert_array_equal(cached_candidates[index][0],cand)
                        io.require(cached_candidates[index][1]==used,'duplicate detector mismatch')
                    cached_candidates[index]=(cand,used)
                    if z['boxes'][i,4]>0:
                        key=(index,z['boxes'][i].tobytes())
                        pose_value=np.column_stack((z['xy'][i],z['scores'][i]))
                        if key in cached_poses:np.testing.assert_array_equal(cached_poses[key],pose_value)
                        cached_poses[key]=pose_value
        reused_detector=reused_pose=0;previous=None;source_index=-1;frame=None
        with av.open(str(ROOT/item['video'])) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=2;decoder=iter(container.decode(stream))
            for i,wanted in enumerate(indices):
                io.safety(cfg)
                while source_index<int(wanted):
                    decoded=next(decoder);source_index+=1
                    io.require(decoded.pts==trace['pts'][source_index],'decoded PTS changed')
                    frame=decoded.to_ndarray(format='rgb24')[:,:,::-1].copy()
                    io.require(hashlib.sha256(frame.tobytes()).hexdigest()==trace['bgr_sha256'][source_index],'pixels changed')
                if int(wanted) in cached_candidates:
                    cand,used=cached_candidates[int(wanted)];reused_detector+=1
                    if not cache_checks['detector']:
                        fresh,flag=candidates_for(frame);np.testing.assert_array_equal(fresh,cand)
                        io.require(flag==used,'cache fallback mismatch');cache_checks['detector']=True
                else:cand,used=candidates_for(frame)
                all_candidates.append(cand);offsets.append(offsets[-1]+len(cand));rescue[i]=used
                chosen=io.select_track(cand,previous)
                if chosen is not None:
                    boxes[i]=chosen;previous=chosen;key=(int(wanted),chosen.tobytes())
                    if key in cached_poses:
                        points=cached_poses[key];reused_pose+=1
                        if not cache_checks['pose']:
                            np.testing.assert_array_equal(keypoints_for(frame,chosen),points);cache_checks['pose']=True
                    else:points=keypoints_for(frame,chosen)
                    xy[i]=points[:,:2];scores[i]=points[:,2]
                if (i+1)%100==0:status('frontend',video=number,total=199,id=item['id'],frame=i+1,frames=n)
        q=io.quality(boxes,xy,scores,cfg['quality'])
        q['insufficient_frames']=n<64;q['passed']=bool(q['passed'] and n>=64)
        io.npz(dest/'frontend.npz',xy=xy,scores=scores,boxes=boxes,rescue=rescue,candidates=np.concatenate(all_candidates),
            offsets=np.asarray(offsets,np.int64),width=np.array(trace['width']),height=np.array(trace['height']),
            source_indices=indices,timestamps=times)
        after=hashes();io.require(before==after,'frontend weight mutation')
        io.save(dest/'frontend.json',dict(passed=True,quality=q,frames=n,width=trace['width'],height=trace['height'],
            payload_sha256=io.sha(dest/'frontend.npz'),models_before=before,models_after=after,labels_used=False,training=False,
            reused_detector_positions=reused_detector,reused_pose_positions=reused_pose,cache_checks=cache_checks))
        status('frontend',video=number,total=199,id=item['id'],quality=q)
    io.locked(cfg);status('frontend_completed')


def lift():
    cfg=setup();io.locked(cfg)
    from .rgb_document_lift import main
    main();status('lifting_completed')


def infer():
    import torch
    from .j1_g0_only import load_model
    from fall_pipeline.common.integrity import hash_named_tensors
    cfg=setup();_,plan=io.locked(cfg);dev=io.device(cfg);model=load_model(ROOT,cfg['model'],dev)
    before=hash_named_tensors(model.state_dict().items())
    for number,item in enumerate(plan,1):
        io.safety(cfg);dest=OUTPUT/item['id'];front=io.stage_done(dest,'frontend')
        if not front['quality']['passed']:continue
        if io.stage_done(dest,'inference'):continue
        io.require(io.stage_done(dest,'lift'),'missing lift')
        with np.load(dest/'lift.npz') as z:ntu=z['ntu25'];starts=z['window_starts']
        values={k:[] for k in ['pooled','adapted','G0']}
        with torch.inference_mode():
            for first in range(0,len(starts),32):
                io.safety(cfg);batch=torch.from_numpy(io.windows(ntu,starts[first:first+32])).to(dev)
                result=model(batch)
                for k in values:values[k].append(result[k].cpu().numpy())
        arrays={k:np.concatenate(v) for k,v in values.items()}
        io.require(all(np.isfinite(v).all() for v in arrays.values()),'nonfinite outputs')
        io.npz(dest/'inference.npz',**arrays,window_starts=starts,window_endpoints=starts+63)
        after=hash_named_tensors(model.state_dict().items());io.require(before==after,'inference weight mutation')
        io.save(dest/'inference.json',dict(passed=True,windows=len(starts),payload_sha256=io.sha(dest/'inference.npz'),
            model_before=before,model_after=after,training=False,only_head='G0'))
        status('inference',video=number,total=199,id=item['id'])
    io.locked(cfg);status('inference_completed')


def score():
    from sklearn.metrics import average_precision_score
    cfg=setup();_,plan=io.locked(cfg);items={r['id']:r for r in plan};rows=[]
    for gt in io.read(OUTPUT/'ground_truth.json'):
        item=items[gt['id']];dest=OUTPUT/item['id'];front=io.stage_done(dest,'frontend');predictions=[];prob=0.;positive=False
        passed=front['quality']['passed']
        if passed:
            io.require(io.stage_done(dest,'inference'),'missing inference')
            with np.load(dest/'inference.npz') as z:
                predictions=rising_edges(z['window_endpoints'],z['G0']);positive=bool((z['G0'].argmax(1)==1).any())
                logits=z['G0'].astype(float);ex=np.exp(logits-logits.max(1,keepdims=True));prob=float((ex/ex.sum(1,keepdims=True))[:,1].max())
        else:io.require(not (dest/'inference.npz').exists(),'quality bypass')
        rows.append(dict(gt,quality_passed=passed,quality=front['quality'],predictions=predictions,
                         video_prediction=int(positive),max_fall_probability=prob,event=match_events(predictions,gt['episodes'])))
    datasets={}
    for scope in SOURCES:
        selected=[r for r in rows if r['scope']==scope];y=np.array([bool(r['episodes']) for r in selected]);p=np.array([r['video_prediction'] for r in selected],bool)
        tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
        datasets[scope]=dict(videos=len(selected),positive=int(y.sum()),quality_pass=sum(r['quality_passed'] for r in selected),
            rejected_positive=sum(bool(r['episodes']) and not r['quality_passed'] for r in selected),
            event=metrics(*(sum(r['event'][k] for r in selected) for k in ['tp','fp','fn'])),
            video=dict(metrics(tp,fp,fn),tn=tn,accuracy=float((y==p).mean()),specificity=tn/(tn+fp),
                       ap=float(average_precision_score(y,[r['max_fall_probability'] for r in selected]))))
    io.save(OUTPUT/'evaluation.json',dict(passed=True,datasets=datasets,rows=rows,no_training=True,no_target_tuning=True,
        contract_sha256=io.sha(OUTPUT/'contract.json'),official_segment_metric=False))
    status('scored_pending_audit',datasets=datasets)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','frontend','lift','infer','score']);args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        io.require(sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())<8*1024**3,'8GiB budget')
        globals()[args.stage]()


if __name__=='__main__':main()
