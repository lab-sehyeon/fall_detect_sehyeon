"""Independent literal-rule, timestamp, model replay and common metric audit."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'data/source_archives/OriginalBaselines_20261004_r1'
OUT=ROOT/'data/fall_processed/RGB/original_baselines_20261004_r2'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.insert(0,str(ASSET/'runtime'))
import numpy as np
import cv2
import tensorflow as tf


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''): h.update(b)
    return h.hexdigest()


def require(x,m):
    if not x: raise RuntimeError(m)


def count_events(predictions,events):
    used=set();tp=0;fp=0
    for pred in sorted(predictions,key=lambda x:x['time']):
        candidates=[i for i,e in enumerate(events) if i not in used and
                    e['fall_start']-.5<=pred['time']<=max(e['fall_start'],e['fall_end'])+3.]
        if candidates:
            selected=min(candidates,key=lambda i:(events[i]['fall_start'],i));used.add(selected);tp+=1
        else: fp+=1
    return tp,fp,len(events)-len(used)


def metric(tp,fp,fn):
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def main():
    started=time.monotonic();contract=read(OUT/'contract.json')
    for p,h in contract['files'].items():require(sha(ROOT/p)==h,'contract changed '+p)
    evaluation=read(OUT/'evaluation.json');plan=read(OUT/'plan.json')
    gt={r['id']:r for r in read(OUT/'ground_truth.json')}
    rows={(r['id'],r['model']):r for r in evaluation['rows']}
    require(len(rows)==460 and len(plan)==230,'duplicate/missing rows')
    tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
    cv2.setNumThreads(1)
    net=tf.lite.Interpreter(model_path=str(ASSET/'lite-model_movenet_singlepose_thunder_3.tflite'),num_threads=2)
    net.allocate_tensors();inp=net.get_input_details()[0]['index'];out=net.get_output_details()[0]['index']
    # Use the other official notebook, independently extracted from its JSON.
    nb=read(ASSET/'Thunder_URFD.ipynb')
    source='\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    a=source.index('        if (kws[0][0][0][0] > 0.5):');b=source.index('# Rendering',a)
    block='\n'.join(s[8:] if s.startswith(' '*8) else s for s in source[a:b].splitlines())
    code=compile(block,'<independent original URFD notebook>','exec')
    class Drawing:
        FONT_HERSHEY_COMPLEX=3
        def __init__(self): self.alarm=False
        def putText(self,*args):
            if args[1].startswith('T Fall'): self.alarm=True
            return args[0]
    total_frames=0;replays=0;max_error=0.;failures=[];case_rows=[]
    for number,item in enumerate(plan,1):
        sid=item['id'];receipt=read(OUT/sid/'rthfd.json');row=rows[(sid,'rthfd')]
        require(sha(ROOT/item['video'])==item['video_sha256'],'input bytes changed')
        require(sha(ROOT/item['trace'])==item['trace_sha256'],'timestamp trace changed')
        t=read(ROOT/item['trace']);pred=[]
        if receipt['processed']:
            require(sha(OUT/sid/'rthfd.npz')==receipt['payload_sha256'],'output bytes changed')
            with np.load(OUT/sid/'rthfd.npz',allow_pickle=False) as z: saved={k:z[k] for k in z.files}
            n=item['source_frames'];require(n==receipt['frames']==len(saved['keypoints']),'source count')
            np.testing.assert_array_equal(saved['source_pts'],t['pts'])
            times=(np.array(t['pts'])-t['start_pts'])*t['time_base_num']/t['time_base_den']
            np.testing.assert_array_equal(saved['timestamps'],times)
            captured=[False]
            def emit(*args,**kw):
                # All URFD true-fall print variants: T Fall, T Fall 2,
                # T Fall , FC:. All 68 branches also draw T Fall labels.
                if args and isinstance(args[0],str) and args[0].startswith('T Fall'):captured[0]=True
            drawing=Drawing()
            env=dict(cv2=drawing,frame=None,print=emit,fc=0,flag=0)
            previous=False
            for i,pose in enumerate(saved['keypoints']):
                captured[0]=False;drawing.alarm=False;env['kws']=pose[None,None];exec(code,env)
                require(captured[0]==drawing.alarm,'printed and displayed fall mismatch')
                require(captured[0]==bool(saved['alarms'][i]),'rule alarm mismatch')
                require(env['fc']==saved['counters'][i] and env['flag']==saved['flags'][i],'counter/flag mismatch')
                if captured[0] and not previous:pred.append(dict(frame=i,time=float(times[i])))
                previous=captured[0]
            cap=cv2.VideoCapture(str(ROOT/item['video']))
            selected={0,n//2,n-1}
            for i in range(n):
                ok,frame=cap.read();require(ok,'audit decode EOF')
                require(hashlib.sha256(frame.tobytes()).hexdigest()==t['bgr_sha256'][i],'audit pixel mismatch')
                if i in selected:
                    # Same literal official preprocessing, independently replayed.
                    tensor=tf.cast(tf.image.resize_with_pad(np.expand_dims(frame.copy(),0),256,256),tf.float32)
                    net.set_tensor(inp,np.array(tensor));net.invoke();actual=net.get_tensor(out)[0,0]
                    err=float(np.max(np.abs(actual-saved['keypoints'][i])));max_error=max(max_error,err)
                    np.testing.assert_allclose(actual,saved['keypoints'][i],rtol=0,atol=1e-6)
                    replays+=1
            require(not cap.read()[0],'audit extra frames');cap.release();total_frames+=n
        else:
            failures.append(dict(id=sid,error=receipt['error']))
        require(row['predictions']==pred,'event extraction mismatch')
        for model in ('own','rthfd'):
            r=rows[(sid,model)];g=gt[sid]['episodes'];require(r['episodes']==g,'GT mismatch')
            tp,fp,fn=count_events(r['predictions'],g)
            require([tp,fp,fn]==[r['event'][k] for k in ('tp','fp','fn')],'case score mismatch')
            require(int(bool(r['predictions']))==r['video_prediction'],'video decision mismatch')
            case_rows.append(dict(dataset=item['scope'],id=sid,model=model,processed=r['processed'],
                gt_fall=int(bool(g)),video_prediction=r['video_prediction'],event_tp=tp,event_fp=fp,event_fn=fn,
                event_count=len(r['predictions']),alarm_times_seconds=';'.join(format(p['time'],'.9g') for p in r['predictions'])))
        if number%10==0: print(json.dumps(dict(stage='audit',completed=number,total=230,frames=total_frames,
                                               model_replays=replays,max_pose_error=max_error)),flush=True)
    summary_rows=[];error_breakdown={}
    for model,scopes in evaluation['summaries'].items():
        error_breakdown[model]={}
        for scope,expected in scopes.items():
            selected=[r for r in evaluation['rows'] if r['scope']==scope and r['model']==model]
            triples=[count_events(r['predictions'],r['episodes']) for r in selected]
            event=metric(*map(sum,zip(*triples)));require(event==expected['event'],'aggregate event mismatch')
            tp=sum(bool(r['episodes']) and bool(r['predictions']) for r in selected)
            fp=sum(not r['episodes'] and bool(r['predictions']) for r in selected)
            fn=sum(bool(r['episodes']) and not r['predictions'] for r in selected)
            tn=sum(not r['episodes'] and not r['predictions'] for r in selected)
            video={**metric(tp,fp,fn),'tn':tn,'accuracy':(tp+tn)/len(selected)}
            require(video==expected['video'],'aggregate video mismatch')
            error_breakdown[model][scope]=dict(
                failed_positive_videos=sum(bool(r['episodes']) and not r['processed'] for r in selected),
                processed_positive_no_alarm=sum(bool(r['episodes']) and r['processed'] and not r['predictions'] for r in selected),
                positive_alarm_without_match=sum(bool(r['episodes']) and bool(r['predictions']) and r['event']['tp']==0 for r in selected),
                negative_alarm_videos=fp,
                negative_unmatched_events=sum(r['event']['fp'] for r in selected if not r['episodes']),
                positive_unmatched_events=sum(r['event']['fp'] for r in selected if r['episodes']))
            summary_rows.append(dict(dataset=scope,model=model,videos=len(selected),processed=expected['processed'],
                event_tp=event['tp'],event_fp=event['fp'],event_fn=event['fn'],event_precision_pct=100*event['precision'],
                event_recall_pct=100*event['recall'],event_f1_pct=100*event['f1'],video_tp=tp,video_fp=fp,video_fn=fn,video_tn=tn,
                video_accuracy_pct=100*video['accuracy'],video_f1_pct=100*video['f1']))
    for path,records in [(OUT/'cases.csv',case_rows),(OUT/'comparison.csv',summary_rows)]:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    for p,h in contract['files'].items():require(sha(ROOT/p)==h,'post-audit contract changed')
    evaluation['pending_independent_audit']=False;evaluation['passed']=True
    (OUT/'evaluation.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2)+'\n')
    audit=dict(passed=True,videos=230,model_case_rows=460,decoded_frames_verified=total_frames,
        original_rules_replayed_frames=total_frames,pose_model_replayed_frames=replays,max_pose_abs_error=max_error,
        predictions_recomputed=True,metrics_independently_recomputed=True,own_parent_predictions_unchanged=True,
        failures=failures,error_breakdown=error_breakdown,seconds=time.monotonic()-started,
        evaluation_sha256=sha(OUT/'evaluation.json'),contract_sha256=sha(OUT/'contract.json'),audit_code_sha256=sha(__file__),
        cases_sha256=sha(OUT/'cases.csv'),comparison_sha256=sha(OUT/'comparison.csv'))
    (OUT/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit),flush=True)


if __name__=='__main__': main()
