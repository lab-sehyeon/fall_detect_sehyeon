"""Author RTHFD rules + original Thunder v3; frozen 230-video external test.

No training, target calibration, color correction, or rule correction.
The author's exact decision block executes with print/drawing instrumentation.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSET = ROOT/'data/source_archives/OriginalBaselines_20261004_r1'
OUT = ROOT/'data/fall_processed/RGB/original_baselines_20261004_r1'
EVIDENCE = ROOT/'docs/internal/2026-10-04_original_baseline_evaluation_evidence'
CONFIG = ROOT/'configs/original_baselines_20261004_r1.json'
MODEL = ASSET/'lite-model_movenet_singlepose_thunder_3.tflite'
REPO = ROOT/'third_party/original_rthfd_20261004'
COMMIT = '63181df4e1d8172e91ee7b3c958c15b5860ceb29'
PARENTS = {n: ROOT/f'data/fall_processed/RGB/{n}_20261003_r1'
           for n in ('le2i130', 'cauca100')}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def require(condition, message):
    if not condition: raise RuntimeError(message)


def status(stage, **kw):
    row = dict(stage=stage, time=datetime.now(timezone.utc).isoformat(), **kw)
    save(OUT/'status.json', row); print(json.dumps(row), flush=True)


def extract_block(source):
    start = source.index('        if (kws[0][0][0][0] > 0.5):')
    end = source.index('# Rendering', start)
    return '\n'.join(s[8:] if s.startswith(' '*8) else s
                     for s in source[start:end].splitlines())+'\n'


class SilentDrawing:
    FONT_HERSHEY_COMPLEX = 3
    def putText(self, *args): return args[0]


class Rules:
    def __init__(self, block, drawing=None):
        self.code = compile(block, '<original RTHFD decision block>', 'exec')
        self.alarm = False
        self.env = dict(cv2=drawing or SilentDrawing(), frame=None,
                        print=self.capture, flag=0, fc=0)

    def capture(self, *args, **kwargs):
        if args and args[0] == 'Fall': self.alarm = True

    def step(self, pose):
        self.alarm = False; self.env['kws'] = pose
        exec(self.code, self.env)
        return self.alarm, int(self.env['fc']), int(self.env['flag'])


def prepare():
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True); EVIDENCE.mkdir(parents=True, exist_ok=True)
    if (OUT/'contract.json').exists(): check_contract(); return
    blocks = {}; originals = []
    for name in ('Thunder_GMDCSA.ipynb', 'Thunder_URFD.ipynb'):
        payload = subprocess.check_output(['git', '-C', str(REPO), 'show', f'{COMMIT}:{name}'])
        dest = ASSET/name
        if dest.exists(): require(dest.read_bytes() == payload, 'notebook overwrite refused')
        else: dest.write_bytes(payload)
        notebook = json.loads(payload)
        source = '\n\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        block = extract_block(source); blocks[name] = block
        (ASSET/(name+'.rules.py')).write_text(block)
        originals.extend([dest, ASSET/(name+'.rules.py')])
    class StripDisplay(ast.NodeTransformer):
        def visit_Expr(self, node):
            return None if isinstance(node.value, ast.Call) else node
    trees = [ast.dump(StripDisplay().visit(ast.parse(s)), include_attributes=False) for s in blocks.values()]
    require(trees[0] == trees[1], 'notebook decision semantics differ')
    # Compare the instrumented rule against literal code + real OpenCV drawing.
    import cv2
    rng = np.random.RandomState(20261004)
    poses = rng.uniform(0, 1, size=(1000, 1, 1, 17, 3)).astype(np.float32)
    poses[:8] = np.array([0, .01, .05, .5, .51, .6, .61, 1], np.float32)[:, None, None, None, None]
    alarm_checks = 0
    for name, block in blocks.items():
        rule = Rules(block); reference_alarm = [False]
        def captured(*args, **kwargs):
            if args and args[0] == 'Fall': reference_alarm[0] = True
        env = dict(cv2=cv2, frame=np.zeros((256, 256, 3), np.uint8), print=captured, flag=0, fc=0)
        code = compile(block, name, 'exec')
        for p in poses:
            reference_alarm[0] = False; env['kws'] = p; exec(code, env)
            observed = rule.step(p)
            require(observed == (reference_alarm[0], env['fc'], env['flag']), 'rule instrumentation changed output')
            alarm_checks += int(observed[0])
    require(alarm_checks > 0, 'synthetic audit lacks positive alarms')
    save(EVIDENCE/'rule_preflight.json', dict(passed=True, synthetic_frames_per_notebook=1000,
         alarm_frames=alarm_checks, real_drawing_equivalent=True, notebook_decisions_identical=True,
         decision_ast_sha256=hashlib.sha256(trees[0].encode()).hexdigest()))
    plan=[]; truth=[]; own=[]; parents=[]
    for scope, parent in PARENTS.items():
        audit=read(parent/'independent_audit.json'); ev=read(parent/'evaluation.json')
        require(audit['passed'] and audit['evaluation_sha256'] == sha(parent/'evaluation.json'), 'parent audit mismatch')
        p=read(parent/'plan.json'); g=[r for r in read(parent/'ground_truth.json') if r['scope']==scope]
        e=[r for r in ev['rows'] if r.get('model','own')=='own' and r['scope']==scope]
        require({r['id'] for r in p}=={r['id'] for r in g}=={r['id'] for r in e}, 'parent membership mismatch')
        for r in p:
            r=dict(r,scope=scope,trace=str((parent/r['id']/'source_trace.json').relative_to(ROOT)))
            require(sha(ROOT/r['trace'])==r['trace_sha256'], 'trace changed')
            plan.append(r)
        truth.extend(g)
        own.extend(dict(r,model='own',processed=r.get('processed',r.get('quality_passed'))) for r in e)
        parents.extend(parent/f for f in ('plan.json','ground_truth.json','evaluation.json','independent_audit.json','contract.json'))
    require(len(plan)==230 and len({r['id'] for r in plan})==230, 'denominator')
    save(OUT/'plan.json',plan);save(OUT/'ground_truth.json',truth);save(OUT/'own_predictions.json',own)
    config=dict(experiment_id='ORIGINAL_BASELINES_20261004_R1',videos=230,
        datasets={'le2i130':130,'cauca100':100}, model='RTHFD',author_commit=COMMIT,
        source='https://github.com/ekramalam/RTHFD', weights_url='https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/3?lite-format=tflite',
        weights_sha256='8014d8fe22285265f52aa1cea84056b7704f75adf12341a7712d4cb28bd1d9b6',
        preprocessing='original cv2 BGR, tf.image.resize_with_pad(256,256), float32; no RGB conversion',
        rules='exact GMDCSA notebook block; URFD decision AST identical; preserve abs(bool) and final else counter reset',
        alarm='any original print(Fall) call per frame, rising edges; no refractory or smoothing',
        timing='every native frame in order; exact existing rational source PTS, verified decoded frame hashes',
        evaluation={'early_seconds':.5,'late_seconds':3.,'failure':'no predictions, retain all videos and GT',
                    'video_prediction':'one or more alarm frames anywhere in video'},
        execution={'workers':4,'tflite_threads_per_worker':2,'tf_intra_threads':2,'tf_inter_threads':1,'device':'CPU'},
        runtime={'tensorflow_cpu':'2.15.1','numpy':'1.23.5'},
        no_training=True,no_target_tuning=True,own='reuse independently audited parent predictions',
        external_score_is_original_paper_metric=False,
        unavailable={'HFD':'original fitted fall SVM not obtained','SDFA':'original fall checkpoint not obtained'})
    require(sha(MODEL)==config['weights_sha256'], 'wrong MoveNet version')
    save(CONFIG,config)
    files=[Path(__file__),CONFIG,MODEL,*originals,*parents,OUT/'plan.json',OUT/'ground_truth.json',OUT/'own_predictions.json',
           ROOT/'fall_pipeline/external/le2i_current_evaluation.py',EVIDENCE/'rule_preflight.json',EVIDENCE/'runtime_install.json']
    save(OUT/'contract.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in files},'created':datetime.now(timezone.utc).isoformat()})
    status('prepared',videos=230,source_frames=sum(r['source_frames'] for r in plan),no_training=True)


def check_contract():
    for path,digest in read(OUT/'contract.json')['files'].items():
        require(sha(ROOT/path)==digest, 'frozen file changed: '+path)


def init_worker():
    global tf, np, cv2, interpreter, inputs, outputs, block
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    sys.path.insert(0,str(ASSET/'runtime'))
    import tensorflow as tf
    import numpy as np
    import cv2
    cv2.setNumThreads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    interpreter=tf.lite.Interpreter(model_path=str(MODEL),num_threads=2)
    interpreter.allocate_tensors();inputs=interpreter.get_input_details()[0];outputs=interpreter.get_output_details()[0]
    require(tuple(inputs['shape'])==(1,256,256,3) and inputs['dtype']==np.float32,'model input')
    require(tuple(outputs['shape'])==(1,1,17,3),'model output')
    block=(ASSET/'Thunder_GMDCSA.ipynb.rules.py').read_text()


def infer_video(item):
    started=time.monotonic();dest=OUT/item['id'];dest.mkdir(exist_ok=True)
    receipt=dest/'rthfd.json';payload=dest/'rthfd.npz'
    if receipt.exists():
        old=read(receipt)
        require(old['contract_sha256']==sha(OUT/'contract.json'),'stale receipt')
        if old['processed']: require(sha(payload)==old['payload_sha256'],'stale output')
        return old
    count=0;cap=None
    try:
        require(sha(ROOT/item['video'])==item['video_sha256'],'video changed')
        require(sha(ROOT/item['trace'])==item['trace_sha256'],'trace changed')
        trace=read(ROOT/item['trace']);n=item['source_frames']
        require(len(trace['pts'])==n==len(trace['bgr_sha256']),'source frame count')
        timestamps=(np.array(trace['pts'],np.int64)-trace['start_pts'])*trace['time_base_num']/trace['time_base_den']
        require(np.all(np.diff(timestamps)>0),'non-monotonic input')
        poses=np.zeros((n,17,3),np.float32);alarms=np.zeros(n,bool);counters=np.zeros(n,np.int64);flags=np.zeros(n,np.int64)
        rules=Rules(block);cap=cv2.VideoCapture(str(ROOT/item['video']));require(cap.isOpened(),'video open failed')
        for index in range(n):
            ok,frame=cap.read();require(ok,'early decoder EOF')
            require(hashlib.sha256(frame.tobytes()).hexdigest()==trace['bgr_sha256'][index],f'decoded BGR differs at {index}')
            img=tf.image.resize_with_pad(np.expand_dims(frame.copy(),axis=0),256,256)
            input_image=tf.cast(img,dtype=tf.float32)
            interpreter.set_tensor(inputs['index'],np.array(input_image));interpreter.invoke()
            kws=interpreter.get_tensor(outputs['index']);require(np.isfinite(kws).all(),'nonfinite pose')
            poses[index]=kws[0,0];alarms[index],counters[index],flags[index]=rules.step(kws);count+=1
        require(not cap.read()[0],'extra decoded frames')
        tmp=dest/'rthfd.tmp.npz'
        np.savez_compressed(tmp,keypoints=poses,alarms=alarms,counters=counters,flags=flags,timestamps=timestamps,
                            source_pts=np.array(trace['pts'],np.int64))
        tmp.replace(payload)
        row=dict(id=item['id'],scope=item['scope'],processed=True,frames=count,pixels_verified=count,
                 payload_sha256=sha(payload),seconds=time.monotonic()-started,contract_sha256=sha(OUT/'contract.json'))
    except Exception as exc:
        row=dict(id=item['id'],scope=item['scope'],processed=False,frames=count,
                 error=f'{type(exc).__name__}: {exc}',seconds=time.monotonic()-started,
                 contract_sha256=sha(OUT/'contract.json'))
    finally:
        if cap is not None: cap.release()
    save(receipt,row);return row


def run():
    check_contract();plan=read(OUT/'plan.json');completed=0;failed=0;started=time.monotonic()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker) as pool:
        futures=[pool.submit(infer_video,r) for r in plan]
        for f in as_completed(futures):
            row=f.result();completed+=1;failed+=not row['processed']
            status('inference',completed=completed,total=230,failures=failed,id=row['id'],elapsed_seconds=time.monotonic()-started)
    check_contract();status('inference_complete',videos=230,failures=failed,elapsed_seconds=time.monotonic()-started)


def score():
    import numpy as np
    from fall_pipeline.external.le2i_current_evaluation import match_events,metrics
    check_contract();plan=read(OUT/'plan.json');gt={r['id']:r for r in read(OUT/'ground_truth.json')}
    rows=[]
    for row in read(OUT/'own_predictions.json'):
        row=dict(row);event=match_events(row['predictions'],gt[row['id']]['episodes'])
        require(event==row['event'],'own parent score changed');rows.append(row)
    for item in plan:
        dest=OUT/item['id'];receipt=read(dest/'rthfd.json');pred=[]
        if receipt['processed']:
            require(sha(dest/'rthfd.npz')==receipt['payload_sha256'],'prediction changed')
            with np.load(dest/'rthfd.npz',allow_pickle=False) as z:
                a=z['alarms'];edges=np.flatnonzero(a & ~np.r_[False,a[:-1]])
                pred=[dict(frame=int(i),time=float(z['timestamps'][i])) for i in edges]
        g=gt[item['id']]
        rows.append(dict(id=item['id'],scope=item['scope'],path=item['path'],model='rthfd',
                         episodes=g['episodes'],processed=receipt['processed'],predictions=pred,
                         video_prediction=int(bool(pred)),event=match_events(pred,g['episodes'])))
    summaries={}
    for model in ('own','rthfd'):
        summaries[model]={}
        for scope in PARENTS:
            subset=[r for r in rows if r['model']==model and r['scope']==scope]
            event=metrics(*(sum(r['event'][k] for r in subset) for k in ('tp','fp','fn')))
            tp=sum(bool(r['episodes']) and bool(r['video_prediction']) for r in subset)
            fp=sum(not r['episodes'] and bool(r['video_prediction']) for r in subset)
            fn=sum(bool(r['episodes']) and not r['video_prediction'] for r in subset)
            tn=sum(not r['episodes'] and not r['video_prediction'] for r in subset)
            summaries[model][scope]=dict(videos=len(subset),processed=sum(r['processed'] for r in subset),event=event,
                video={**metrics(tp,fp,fn),'tn':tn,'accuracy':(tp+tn)/len(subset)},
                fall_events=sum(len(r['episodes']) for r in subset))
    save(OUT/'evaluation.json',dict(pending_independent_audit=True,summaries=summaries,rows=rows,
                                  no_training=True,no_target_tuning=True,contract_sha256=sha(OUT/'contract.json')))
    status('scored_pending_audit',summaries=summaries)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','run','score','all'],default='all');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage in ('prepare','all'): prepare()
        if args.stage in ('run','all'): run()
        if args.stage in ('score','all'): score()


if __name__=='__main__': main()
