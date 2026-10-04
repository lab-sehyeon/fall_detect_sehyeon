"""CPU independent audit of reused predictions and three fresh external comparators."""
from pathlib import Path
import csv, os, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
from fall_pipeline.external import rgb_document_io as io
import run_external_comparison_expansion_20261003 as run
from audit_three_by_three_20261003 import counts, ap


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU audit only'
    cfg = run.setup()
    out, plan = run.locked(cfg)
    evaluation = io.read(out / 'evaluation.json')
    truth = io.read(out / 'ground_truth.json')
    assert truth == [r for r in io.read(run.PARENT / 'ground_truth.json') if r['scope'] in run.SCOPES]
    assert plan == io.read(run.PARENT / 'plan.json')
    parent_audit = io.read(run.PARENT / 'independent_audit.json')
    assert parent_audit['passed'] and parent_audit['evaluation_sha256'] == io.sha(run.PARENT / 'evaluation.json')
    parent_eval = io.read(run.PARENT / 'evaluation.json')
    parent_rows = {(r['scope'], r['id']): r for r in parent_eval['rows']}
    completed = io.read(out / 'comparators_completed.json')
    assert completed['passed']
    for value in completed['models'].values():
        assert value['passed'] and value['model_before'] == value['model_after'] and value['windows'] == 6112
    records = {}
    windows = {}
    for item in plan:
        dest = out / item['id']
        src = run.PARENT / item['id']
        trace = io.read(dest / 'source_trace.json')
        assert io.sha(dest / 'source_trace.json') == item['trace_sha256']
        with np.load(dest / 'mapping.npz') as mapping, np.load(dest / 'frontend.npz') as front:
            indices = mapping['source_indices']; times = mapping['canonical_timestamps']
            assert io.sha(dest / 'mapping.npz') == item['mapping_sha256']
            np.testing.assert_array_equal(times, np.arange(item['frames']) / 25)
            ticks = (np.array(trace['pts']) - trace['start_pts']) * trace['time_base_num'] * 25
            targets = np.arange(item['frames']) * trace['time_base_den']
            assert (ticks[indices] <= targets).all() and (np.diff(indices) >= 0).all()
            has_next = indices + 1 < len(ticks)
            assert (ticks[indices[has_next] + 1] > targets[has_next]).all()
            np.testing.assert_array_equal(front['source_indices'], indices)
            np.testing.assert_array_equal(front['timestamps'], times)
            boxes, xy, scores = front['boxes'], front['xy'], front['scores']
            bv = np.isfinite(boxes).all(1) & (boxes[:,2] > boxes[:,0]) & (boxes[:,3] > boxes[:,1]) & (boxes[:,4] > 0)
            pv = bv & np.isfinite(xy).all((1,2)) & np.isfinite(scores).all(1) & (scores > 0).any(1)
            pelvis = np.where(pv, np.minimum(scores[:,11], scores[:,12]), 0)
            own_pass = bool(bv.mean() >= .8 and pv.mean() >= .8 and np.median(pelvis) >= .3 and len(xy) >= 64)
            compare_pass = bool(len(xy) >= 64 and np.any(scores > 0))
        front_meta = io.stage_done(dest, 'frontend')
        assert front_meta['quality']['passed'] == own_pass
        assert front_meta['models_before'] == front_meta['models_after']
        for model in ['own', 'stgcnpp', 'msg3d', 'cnn1d']:
            own = model == 'own'
            passed = own_pass if own else compare_pass
            stage = 'inference' if own else model
            alarms, best, positive = [], 0., False
            if passed:
                meta = io.stage_done(dest, stage)
                if own:
                    assert meta['model_before'] == meta['model_after']
                    assert io.sha(dest / 'inference.npz') == io.sha(src / 'inference.npz')
                else:
                    assert meta['passed'] and not meta['training'] and not meta['annotations_read']
                    assert meta['source_frontend_sha256'] == io.sha(dest / 'frontend.npz')
                with np.load(dest / (stage + '.npz')) as z:
                    starts, ends = z['window_starts'], z['window_endpoints']
                    np.testing.assert_array_equal(starts, np.arange(0, item['frames'] - 63, 8))
                    np.testing.assert_array_equal(ends, starts + 63)
                    logits = z['G0' if own else 'logits'].astype(float)
                    assert logits.shape == (len(starts), 4 if own else 15) and np.isfinite(logits).all()
                    ex = np.exp(logits - logits.max(1, keepdims=True)); prob = ex / ex.sum(1, keepdims=True)
                    if not own:
                        np.testing.assert_array_equal(z['input_starts'], starts + 16)
                        np.testing.assert_allclose(prob, z['probabilities'], atol=1e-6, rtol=1e-5)
                previous = False; index = 1 if own else 9
                for end, logit in zip(ends, logits):
                    current = int(np.argmax(logit)) == index
                    if current and not previous:
                        alarms.append(float(end / 25))
                    previous = current
                best = float(prob[:,index].max()); positive = bool((logits.argmax(1) == index).any())
                windows[model] = windows.get(model, 0) + len(starts)
            elif own:
                assert not (dest / 'inference.npz').exists()
            else:
                assert not io.stage_done(dest, model)['passed']
            records[(model, item['id'])] = dict(passed=passed, times=alarms, best=best, positive=positive)
    reported = {(r['model'], r['scope'], r['id']): r for r in evaluation['rows']}
    assert len(reported) == len(truth) * 4 == len(evaluation['rows'])
    independent = []; cases = []
    for gt in truth:
        case = dict(dataset=gt['scope'], video=gt['path'], fall_events=len(gt['episodes']))
        for model in ['own', 'stgcnpp', 'msg3d', 'cnn1d']:
            r = records[(model, gt['id'])]; got = reported[(model, gt['scope'], gt['id'])]
            c = counts(r['times'], gt['episodes'])
            assert tuple(got['event'][k] for k in ['tp','fp','fn']) == c
            assert [p['time'] for p in got['predictions']] == r['times']
            assert got['processed'] == r['passed'] and bool(got['video_prediction']) == r['positive']
            assert abs(got['max_fall_probability'] - r['best']) < 1e-12
            if model == 'own':
                old = parent_rows[(gt['scope'], gt['id'])]
                for k in ['predictions', 'video_prediction', 'event', 'max_fall_probability']:
                    assert got[k] == old[k], k
            independent.append(dict(model=model, scope=gt['scope'], counts=c, truth=bool(gt['episodes']), **r))
            case.update({model+'_processed':int(r['passed']), model+'_tp':c[0], model+'_fp':c[1], model+'_fn':c[2],
                         model+'_alarms':';'.join(map(str,r['times']))})
        cases.append(case)
    for model, scopes in evaluation['summaries'].items():
        for scope, s in scopes.items():
            rows = [r for r in independent if r['model'] == model and r['scope'] == scope]
            y = [r['truth'] for r in rows]; p = [r['positive'] for r in rows]
            tp = sum(a and b for a,b in zip(y,p)); fp = sum(not a and b for a,b in zip(y,p))
            fn = sum(a and not b for a,b in zip(y,p)); tn = sum(not a and not b for a,b in zip(y,p))
            assert s['videos'] == len(rows) and s['processed'] == sum(r['passed'] for r in rows)
            assert s['positive_videos'] == sum(y)
            for unit, c in [('event', np.array([r['counts'] for r in rows]).sum(0)), ('video', (tp,fp,fn))]:
                assert list(c) == [s[unit][k] for k in ['tp','fp','fn']]
                a,b,c = c
                expected = [a/(a+b) if a+b else 0., a/(a+c) if a+c else 0., 2*a/(2*a+b+c) if 2*a+b+c else 0.]
                for key, value in zip(['precision','recall','f1'],expected):
                    assert abs(s[unit][key] - value) < 1e-12
            assert s['video']['tn'] == tn and s['video']['accuracy'] == (tp+tn)/len(rows)
            assert s['video']['specificity'] == tn/(tn+fp)
            assert abs(s['video']['ap'] - ap(y,[r['best'] for r in rows])) < 1e-12
            if model == 'own':
                assert s['event'] == parent_eval['datasets'][scope]['event']
    run.locked(cfg)
    with (out / 'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    report = dict(passed=True, unique_videos=len(plan), memberships=len(truth), models=4, windows=windows,
                  evaluation_sha256=io.sha(out/'evaluation.json'), contract_sha256=io.sha(out/'contract.json'),
                  temporal_order=True, quality_recomputed=True, source_predictions_identical=True,
                  raw_probabilities_verified=True, event_video_ap_independently_recomputed=True)
    io.save(out / 'independent_audit.json', report)
    run.base.status('completed', audit=report)
    print('AUDIT PASS', report, flush=True)


if __name__ == '__main__':
    main()
