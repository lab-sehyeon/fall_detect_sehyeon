"""Frozen common-unseen-set comparison with the SAFER authors' PoseC3D.

New continuous-window transfer protocol; not reproduction of NTU clip accuracy.
Inference reads only input manifests, never annotations or target metrics.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ['safer_pyskl_external_20261003', 'pyskl_runtime_20261003']:
    sys.path.insert(0, str(ROOT / 'third_party' / name))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/posec3d_mpl_20261003')
import numpy as np
from fall_pipeline.external import rgb_document_io as io

PARENT = ROOT / 'data/fall_processed/RGB/temporal_continuous_20261002_r1'
OUTPUT = ROOT / 'data/fall_processed/RGB/safer_posec3d_external_20261003_r1'
CONFIG = ROOT / 'configs/safer_posec3d_external_20261003_r1.json'
UPSTREAM = ROOT / 'third_party/safer_pyskl_external_20261003'
MODEL_CONFIG = UPSTREAM / 'configs/posec3d/safer_activity_xsub/non-wheelchair.py'
WEIGHT = ROOT / 'data/source_archives/SAFER_PoseC3D_20261003/non-wheelchair-epoch_44.pth'
SCOPES = ['le2i38', 'gmdcsa37', 'cauca19']


def save(path, value):
    io.save(path, value)


def status(stage, **kwargs):
    row = dict(stage=stage, time=datetime.now(timezone.utc).isoformat(), **kwargs)
    save(OUTPUT / 'status.json', row)
    print(json.dumps(row), flush=True)


def snapshot():
    files = [Path(__file__), CONFIG, MODEL_CONFIG, WEIGHT,
             PARENT / 'plan.json', PARENT / 'ground_truth.json',
             PARENT / 'evaluation.json', PARENT / 'independent_audit.json',
             PARENT / 'contract.json', ROOT / 'configs/temporal_continuous_20261002_r1.json',
             ROOT / 'fall_pipeline/external/rgb_document_io.py',
             ROOT / 'fall_pipeline/external/le2i_current_evaluation.py']
    files += [ROOT/'docs/internal/2026-10-03_external_sources/safer_mappings.json',
              ROOT/'docs/internal/2026-10-03_external_sources/safer_checkpoint_metadata.json']
    files += sorted((UPSTREAM / 'pyskl').rglob('*.py'))
    files += sorted((ROOT / 'third_party/pyskl_runtime_20261003').rglob('*.py'))
    files += [OUTPUT / 'plan.json', OUTPUT / 'ground_truth.json']
    for item in io.read(OUTPUT / 'plan.json'):
        d = PARENT / item['id']
        files += [d / 'frontend.npz', d / 'frontend.json', d / 'mapping.npz', d / 'source_trace.json']
        if (d / 'inference.npz').exists():
            files += [d / 'inference.npz', d / 'inference.json']
    for s in io.read(ROOT / 'configs/temporal_continuous_20261002_r1.json')['model'].values():
        files.append(ROOT / s['path'])
    return {str(p.relative_to(ROOT)): io.sha(p) for p in sorted(set(files))}


def locked():
    io.require(snapshot() == io.read(OUTPUT / 'contract.json')['files'], 'frozen input/code/asset changed')
    return io.read(CONFIG), io.read(OUTPUT / 'plan.json')


def prepare():
    io.require(not (OUTPUT / 'contract.json').exists(), 'contract already exists')
    audit = io.read(PARENT / 'independent_audit.json')
    io.require(audit['passed'] and audit['evaluation_sha256'] == io.sha(PARENT / 'evaluation.json'), 'parent audit')
    truth = [r for r in io.read(PARENT / 'ground_truth.json') if r['scope'] in SCOPES]
    ids = {r['id'] for r in truth}
    plan = [r for r in io.read(PARENT / 'plan.json') if r['id'] in ids]
    io.require(len(plan) == len(ids) == len(truth) == 94, 'fixed manifest cardinality')
    for row in plan:
        meta = io.stage_done(PARENT / row['id'], 'frontend')
        io.require(meta and meta['passed'], 'frontend integrity')
    save(OUTPUT / 'plan.json', plan)
    save(OUTPUT / 'ground_truth.json', truth)
    cfg = dict(experiment_id='SAFER_POSEC3D_EXTERNAL_20261003_R1', scopes=SCOPES,
        output=str(OUTPUT.relative_to(ROOT)),
        videos=94, windows=2765, fps=25, window=64, stride=8, append_tail=False,
        input='unchanged exact-PTS canonical frontend, shared YOLOv8x and ViTPose COCO17',
        own_model='DSTE/J1/G0; hash-verified existing predictions and existing quality rejections',
        comparator='SAFER author PoseC3D SlowOnly-R50 non-wheelchair epoch44',
        upstream_revision='85525521b85a44c5df79873102192225c9565edc',
        weight_sha256='386d5580d612f9aa40801a2724945eeb4c2d1182c0f81e5e07c8ae5757470124',
        source_training=dict(own=['NTU60', 'SAFER-Activities', 'FU-Kinect-Fall'], posec3d=['SAFER-Activities non-wheelchair sub_train']),
        posec3d_test='official Sequential48,compact,64x64,sigma0.6,confidence,flip,mean2softmax; trailing48 at common endpoints',
        posec3d_fall_index=9, posec3d_window=48, trailing_offset=16, own_fall_index=1, decision='original multiclass argmax mapped to binary',
        inference=dict(max_testing_views=4, dtype='float32', amp=False),
        quality='own original gate; PoseC3D no 3D gate, zero-score missing poses retained; all failures counted',
        evaluation=dict(event='rising edge at window endpoint; early0.5/late3; one-to-one; refractory0',
                        video='any fall argmax window; max fall score for AP',
                        primary='event F1', paired_bootstrap='10000 video draws per dataset, seed20261003; exploratory 95% percentile CI'),
        execution=dict(gpu='0', minimum_gpu_free_gib=8, gpu_memory_fraction=.6, reserve_gib=64),
        target_training=False, target_tuning=False, untouched_target=False, official_ntu_reproduction=False,
        limitations=['different source training and label spaces', 'shared ViTPose-B replaces comparator source ViTPose-H input',
                     'common64-frame decision grid; comparator uses native trailing48 continuous frames', 'offline own lifting and normalization',
                     'existing target diagnostic exposure; 2/1/2 held-out subjects, clustered-video bootstrap is descriptive'])
    save(CONFIG, cfg)
    io.require(io.sha(WEIGHT) == cfg['weight_sha256'], 'weight hash')
    save(OUTPUT / 'contract.json', dict(created_at=datetime.now(timezone.utc).isoformat(), files=snapshot()))
    status('prepared', videos=94, windows=2765)


def model_and_pipeline(device):
    import torch
    import mmcv
    from pyskl.models import build_model
    from pyskl.datasets.pipelines import Compose
    cfg = mmcv.Config.fromfile(str(MODEL_CONFIG))
    labels=list(io.read(ROOT/'docs/internal/2026-10-03_external_sources/safer_mappings.json')['normal']['labels'].values())[1:]
    io.require(len(labels)==15 and labels[9]=='fall','official label order')
    cfg.model.test_cfg.max_testing_views = 4  # Official view chunking; both views retained.
    model = build_model(cfg.model)
    weights = torch.load(WEIGHT, map_location='cpu', weights_only=True)
    model.load_state_dict(weights['state_dict'], strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, Compose(cfg.test_pipeline)


def annotation(xy, scores, height, width):
    return dict(keypoint=xy[None].copy(), keypoint_score=scores[None].copy(),
                img_shape=(height, width), original_shape=(height, width),
                total_frames=len(xy), start_index=0, label=-1, test_mode=True, modality='Pose',
                usable_indices=np.array([0]),usable_label=np.array([1]))


def smoke():
    """Synthetic input only: deterministic official sampler, shape, missing heatmaps."""
    import torch
    torch.set_num_threads(2)
    _, pipeline = model_and_pipeline('cpu')
    from pyskl.datasets.pipelines import SampleSequentialFrames
    sampled=SampleSequentialFrames(48)(dict(total_frames=48,start_index=0,test_mode=True,usable_indices=np.array([0]),usable_label=np.array([1])))['frame_inds'].reshape(1,48)
    io.require(np.all(np.diff(sampled,axis=1)>=0) and sampled.min()>=0 and sampled.max()<64,'temporal order')
    xy = np.zeros((48, 17, 2), np.float32)
    xy[..., 0] = np.arange(17)[None] * 4 + np.arange(48)[:, None] * .5 + 80
    xy[..., 1] = np.arange(17)[None] * 7 + 40
    scores = np.ones((48, 17), np.float32)
    a = pipeline(annotation(xy, scores, 240, 320))['imgs']
    b = pipeline(annotation(xy, scores, 240, 320))['imgs']
    io.require(a.shape == (2,17,48,64,64) and torch.equal(a,b), 'official shape/determinism')
    z = pipeline(annotation(np.zeros_like(xy), np.zeros_like(scores), 240, 320))['imgs']
    io.require(torch.count_nonzero(z).item() == 0, 'missing pose heatmap')
    save(OUTPUT / 'synthetic_validation.json', dict(passed=True, strict_weight_load=True,
        tensor_shape=list(a.shape), deterministic=True, missing_pose_zero=True, target_inputs_used=False,
        official_sample_indices=sampled.tolist(),chronological_sequential48=True))
    status('synthetic_validation_passed')


def infer():
    import torch
    from fall_pipeline.common.integrity import hash_named_tensors
    cfg, plan = locked()
    device = io.device(cfg)
    model, pipeline = model_and_pipeline(device)
    before = hash_named_tensors(model.state_dict().items())
    captured = []
    hook = model.cls_head.register_forward_hook(lambda module, args, output: captured.append(output.detach().cpu().numpy()))
    started = time.time()
    total = 0
    for number, row in enumerate(plan, 1):
        dest = OUTPUT / row['id']; dest.mkdir(exist_ok=True)
        if (dest / 'prediction.json').exists():
            meta = io.read(dest / 'prediction.json')
            io.require(io.sha(dest / 'prediction.npz') == meta['payload_sha256'], 'resume payload')
            total += meta['windows']; continue
        with np.load(PARENT / row['id'] / 'frontend.npz') as z:
            xy=z['xy']; scores=z['scores']; height=int(z['height']); width=int(z['width'])
            np.testing.assert_array_equal(z['timestamps'], np.arange(row['frames']) / 25)
        io.require(np.isfinite(xy).all() and np.isfinite(scores).all(), 'nonfinite frontend')
        starts = np.arange(0, len(xy) - 63, 8, dtype=np.int64)
        passed = bool(len(starts) and np.any(scores > 0))
        probabilities=[]; views=[]; input_hashes=[]
        with torch.inference_mode():
            for j, start in enumerate(starts):
                io.safety(cfg)
                if not passed:
                    probabilities.append(np.zeros(15, np.float32));views.append(np.zeros((2,15),np.float32));continue
                batch = pipeline(annotation(xy[start+16:start+64], scores[start+16:start+64], height, width))['imgs']
                io.require(tuple(batch.shape) == (2,17,48,64,64), 'heatmap shape')
                if j == 0:
                    input_hashes.append(hashlib.sha256(batch.numpy().tobytes()).hexdigest())
                captured.clear()
                p = model(imgs=batch.unsqueeze(0).to(device), return_loss=False)[0]
                v = captured[0]
                io.require(len(captured)==1 and v.shape==(2,15), 'head view logits')
                ex=np.exp(v.astype(float)-v.max(1,keepdims=True))
                np.testing.assert_allclose(p,(ex/ex.sum(1,keepdims=True)).mean(0),atol=1e-6,rtol=1e-5)
                io.require(np.isfinite(p).all(), 'nonfinite prediction')
                probabilities.append(p); views.append(v)
                if (j+1)%20==0:
                    status('inferring', video=number,total_videos=94,id=row['id'],window=j+1,windows=len(starts),completed_windows=total+j+1,elapsed_seconds=time.time()-started)
        prob=np.asarray(probabilities,np.float32).reshape(-1,15)
        vlog=np.asarray(views,np.float32).reshape(-1,2,15)
        io.npz(dest/'prediction.npz', probabilities=prob,view_logits=vlog,window_starts=starts,window_endpoints=starts+63,input_starts=starts+16)
        meta=dict(passed=passed,windows=len(starts),payload_sha256=io.sha(dest/'prediction.npz'),
                  source_frontend_sha256=io.sha(PARENT/row['id']/'frontend.npz'),first_heatmap_sha256=input_hashes,
                  training=False,annotations_read=False)
        save(dest/'prediction.json',meta)
        total+=len(starts)
        status('video_completed', video=number,total_videos=94,id=row['id'],completed_windows=total,elapsed_seconds=time.time()-started)
    hook.remove()
    after=hash_named_tensors(model.state_dict().items())
    io.require(before==after and total==2765,'weights/windows changed')
    locked()
    save(OUTPUT/'inference_completed.json',dict(passed=True,windows=total,videos=len(plan),elapsed_seconds=time.time()-started,
        model_before=before,model_after=after,torch_version=torch.__version__,device=torch.cuda.get_device_name(0),
        official_model_code_unmodified=True,no_training=True,no_target_tuning=True))
    status('inference_completed',windows=total)


def metric(tp, fp, fn):
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=float(tp/(tp+fp)) if tp+fp else 0.,
        recall=float(tp/(tp+fn)) if tp+fn else 0., f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)


def aggregate(rows):
    from sklearn.metrics import average_precision_score
    answer={}
    for model in ['own','posec3d']:
        answer[model]={}
        for scope in SCOPES:
            r=[x for x in rows if x['scope']==scope and x['model']==model]
            y=np.array([bool(x['episodes']) for x in r]);p=np.array([x['video_prediction'] for x in r],bool)
            tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
            answer[model][scope]=dict(videos=len(r),positive=int(y.sum()),processed=sum(x['processed'] for x in r),
                event=metric(*(sum(x['event'][k] for x in r) for k in ['tp','fp','fn'])),
                video=dict(metric(tp,fp,fn),tn=tn,accuracy=float((y==p).mean()),specificity=tn/(tn+fp),
                           ap=float(average_precision_score(y,[x['max_fall_probability'] for x in r]))))
    return answer


def score():
    from fall_pipeline.external.le2i_current_evaluation import match_events
    locked();io.require(io.read(OUTPUT/'inference_completed.json')['passed'],'incomplete inference')
    previous={(r['scope'],r['id']):r for r in io.read(PARENT/'evaluation.json')['rows']}
    rows=[]
    for gt in io.read(OUTPUT/'ground_truth.json'):
        old=previous[(gt['scope'],gt['id'])]
        rows.append(dict(gt,model='own',processed=old['quality_passed'],event=old['event'],
            predictions=old['predictions'],video_prediction=old['video_prediction'],max_fall_probability=old['max_fall_probability']))
        dest=OUTPUT/gt['id'];meta=io.read(dest/'prediction.json')
        io.require(meta['payload_sha256']==io.sha(dest/'prediction.npz'),'prediction changed')
        with np.load(dest/'prediction.npz') as z:
            prob=z['probabilities'];positive=(prob.argmax(1)==9) if len(prob) else np.zeros(0,bool)
            edges=positive & ~np.r_[False,positive[:-1]] if len(positive) else positive
            predictions=[dict(time=float(t/25),score=float(s)) for t,s in zip(z['window_endpoints'][edges],prob[edges,9])]
        rows.append(dict(gt,model='posec3d',processed=meta['passed'],event=match_events(predictions,gt['episodes']),
            predictions=predictions,video_prediction=int(positive.any()),max_fall_probability=float(prob[:,9].max()) if len(prob) else 0.))
    summaries=aggregate(rows)
    rng=np.random.default_rng(20261003);paired={}
    for scope in SCOPES:
        a=[r for r in rows if r['scope']==scope and r['model']=='own'];b=[r for r in rows if r['scope']==scope and r['model']=='posec3d']
        io.require([r['id'] for r in a]==[r['id'] for r in b],'pair order')
        index=rng.integers(0,len(a),size=(10000,len(a)))
        f=[]
        for r in [a,b]:
            counts=np.array([[x['event'][k] for k in ['tp','fp','fn']] for x in r])[index].sum(1)
            den=2*counts[:,0]+counts[:,1]+counts[:,2]
            f.append(np.divide(2*counts[:,0],den,out=np.zeros(len(den)),where=den>0))
        paired[scope]=dict(event_f1_difference=summaries['own'][scope]['event']['f1']-summaries['posec3d'][scope]['event']['f1'],
            descriptive_video_bootstrap95=np.quantile(f[0]-f[1],[.025,.975]).tolist(),resamples=10000,
            not_subject_independent=True)
    save(OUTPUT/'evaluation.json',dict(passed=True,summaries=summaries,paired=paired,rows=rows,
        no_training=True,no_target_tuning=True,contract_sha256=io.sha(OUTPUT/'contract.json')))
    status('scored_pending_independent_audit',summaries=summaries)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','infer','score']);args=p.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        globals()[args.stage]()


if __name__=='__main__':
    main()
