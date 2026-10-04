"""Read-only post-hoc diagnosis of saved predictions; no inference or tuning."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/fall_processed/RGB/three_by_three_20261003_r1'
OUTPUT = ROOT / 'data/fall_processed/RGB/three_by_three_diagnosis_20261003_r1'
MODELS = ['own', 'stgcnpp', 'msg3d', 'cnn1d']
CLASSES = ['other', 'fall', 'lie_down_state', 'lying_down_motion']
HASHES = {}


def record(path):
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    record(path)
    return json.loads(path.read_text())


def aggregate(rows):
    result = {k: sum(r['event'][k] for r in rows) for k in ['tp', 'fp', 'fn']}
    tp, fp, fn = (result[k] for k in ['tp', 'fp', 'fn'])
    result.update(videos=len(rows), events=tp+fn,
                  precision=tp/(tp+fp) if tp+fp else 0,
                  recall=tp/(tp+fn) if tp+fn else 0,
                  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return result


def main():
    results = {}
    for dataset in ['edf', 'occu', 'OOPS']:
        out = SOURCE/dataset
        ev = read(out/'evaluation.json')
        audit = read(out/'independent_audit.json')
        assert audit['passed'] and audit['evaluation_sha256'] == HASHES[str((out/'evaluation.json').relative_to(ROOT))]
        own = [r for r in ev['rows'] if r['model']=='own']
        accepted = {r['id'] for r in own if r['processed']}
        rejected = {r['id'] for r in own if not r['processed']}
        rows_by_model = {m:{r['id']:r for r in ev['rows'] if r['model']==m} for m in MODELS}
        subsets = {}
        for name, ids in [('all',accepted|rejected), ('own_quality_accepted',accepted), ('own_quality_rejected',rejected)]:
            subsets[name] = {m: aggregate([r for r in ev['rows'] if r['model']==m and r['id'] in ids]) for m in MODELS}
        rejection_patterns = Counter()
        fn_reasons = Counter()
        fn_dominant = Counter()
        class_counts = Counter()
        event_details = []
        video_details = []
        gaps = {m:Counter() for m in MODELS[1:]}
        delays = []
        for row in own:
            folder = out/row['id']
            q = read(folder/'frontend.json')['quality']
            flags = [k for k in ['bbox_coverage','pose_coverage'] if q[k]<.8]
            if q['pelvis_median']<.3: flags.append('pelvis_median')
            if q['insufficient_frames']: flags.append('insufficient_frames')
            assert bool(flags) != row['processed']
            if flags: rejection_patterns['+'.join(flags)] += 1
            detail = dict(id=row['id'],quality=q,events=len(row['episodes']),event_by_model={m:rows_by_model[m][row['id']]['event'] for m in MODELS})
            for model in MODELS[1:]:
                own_hits={v['gt_index'] for v in row['event']['matches']}
                their_hits={v['gt_index'] for v in rows_by_model[model][row['id']]['event']['matches']}
                prefix='accepted' if row['processed'] else 'rejected'
                gaps[model][prefix+'_both_detected']+=len(own_hits&their_hits)
                gaps[model][prefix+'_own_only']+=len(own_hits-their_hits)
                gaps[model][prefix+'_comparator_only']+=len(their_hits-own_hits)
            if not row['processed']:
                fn_reasons['quality_rejected'] += len(row['episodes'])
                video_details.append(detail)
                continue
            record(folder/'inference.npz')
            with np.load(folder/'inference.npz') as z:
                logits=z['G0'].astype(float)
                times=z['window_endpoints']/25
                centers=(z['window_starts']+32)/25
            pred=logits.argmax(1)
            prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
            counts=np.bincount(pred,minlength=4)
            detail['all_window_classes'] = dict(zip(CLASSES,map(int,counts)))
            detail['max_fall_probability'] = float(prob[:,1].max())
            class_counts.update(detail['all_window_classes'])
            missed=set(row['event']['unmatched_gt_indices'])
            alarm_times=np.array([v['time'] for v in row['predictions']])
            for i, episode in enumerate(row['episodes']):
                a,b=episode['fall_start'],episode['fall_end']
                eligible=(times>=a-.5)&(times<=b+3)
                central=(centers>=a)&(centers<=b)
                cnt=np.bincount(pred[eligible],minlength=4)
                central_cnt=np.bincount(pred[central],minlength=4)
                reason='detected'
                if i in missed:
                    if not eligible.any():reason='no_decision_time_in_tolerance'
                    elif not (pred[eligible]==1).any():reason='no_fall_prediction_in_tolerance'
                    elif not ((alarm_times>=a-.5)&(alarm_times<=b+3)).any():reason='fall_active_but_no_rising_edge_in_tolerance'
                    else:reason='eligible_alarm_assigned_to_another_event'
                    fn_reasons[reason]+=1
                    if eligible.any():fn_dominant[CLASSES[int(cnt.argmax())]]+=1
                event_details.append(dict(id=row['id'],event_index=i,start=a,end=b,reason=reason,
                    endpoint_tolerance_window_classes=dict(zip(CLASSES,map(int,cnt))),
                    gt_center_window_classes=dict(zip(CLASSES,map(int,central_cnt))),
                    max_fall_probability_in_tolerance=float(prob[eligible,1].max()) if eligible.any() else None))
            delays.extend(m['onset_delay'] for m in row['event']['matches'])
            video_details.append(detail)
        assert sum(fn_reasons.values())==subsets['all']['own']['fn']
        for model in MODELS[1:]:
            for prefix, subset in [('accepted','own_quality_accepted'),('rejected','own_quality_rejected')]:
                assert gaps[model][prefix+'_comparator_only']-gaps[model][prefix+'_own_only']==subsets[subset][model]['tp']-subsets[subset]['own']['tp']
        no_fall = [e for e in event_details if e['reason']=='no_fall_prediction_in_tolerance']
        no_fall_probs = [e['max_fall_probability_in_tolerance'] for e in no_fall]
        probability_diagnostic = dict(events=len(no_fall),
            below_0_1=sum(p<.1 for p in no_fall_probs),below_0_3=sum(p<.3 for p in no_fall_probs),
            quantiles=dict(zip(['min','p25','median','p75','max'],map(float,np.quantile(no_fall_probs,[0,.25,.5,.75,1])))) if no_fall else {},
            gt_center_window_classes={k:sum(e['gt_center_window_classes'][k] for e in no_fall) for k in CLASSES})
        results[dataset]=dict(subsets=subsets,quality_rejection_patterns=dict(rejection_patterns),
            own_fn_reasons=dict(fn_reasons),missed_event_dominant_class_in_tolerance=dict(fn_dominant),
            no_fall_prediction_probability_diagnostic=probability_diagnostic,
            all_accepted_window_classes=dict(class_counts),paired_event_gaps={m:dict(c) for m,c in gaps.items()},
            own_matched_onset_delay_quantiles=dict(zip(['min','median','p90','max'],map(float,np.quantile(delays,[0,.5,.9,1])))) if delays else {},
            events=event_details,videos=video_details)
        print(dataset, json.dumps({k:v for k,v in results[dataset].items() if k not in ['events','videos']},ensure_ascii=False),flush=True)
    record(Path(__file__))
    for path, expected in HASHES.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected, path
    OUTPUT.mkdir(parents=True,exist_ok=True)
    payload=dict(status='completed',created_utc=datetime.now(timezone.utc).isoformat(),post_hoc=True,
                 inference=False,threshold_changes=False,primary_results_unchanged=True,
                 definitions=dict(conditional_subset='all models evaluated on own quality-passed video IDs',
                     fn_dominant_class='most frequent argmax among endpoints in [GT start-.5, GT end+3]; ties lowest index; descriptive, not ground-truth frame confusion',
                     no_rising_edge='fall remains active from an alarm before tolerance; no eligible new alarm',
                     matched_delay='window endpoint minus GT onset; offline pipeline, not real-time latency'),
                 datasets=results,input_sha256=HASHES)
    (OUTPUT/'diagnosis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
