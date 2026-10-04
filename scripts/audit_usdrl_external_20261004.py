"""Independent float64 linear replay, alarm reconstruction, and metric audit."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fall_pipeline.external import rgb_document_io as io

OUT = ROOT / 'data/fall_processed/RGB/usdrl_external_20261004_r1'


def counts(times, episodes):
    remaining = sorted(enumerate(episodes), key=lambda pair: (pair[1]['fall_start'], pair[0]))
    tp, fp = 0, 0
    for t in sorted(times):
        matched = None
        for i, (_, event) in enumerate(remaining):
            if event['fall_start'] - .5 <= t <= max(event['fall_start'], event['fall_end']) + 3:
                matched = i
                break
        if matched is None:
            fp += 1
        else:
            tp += 1
            remaining.pop(matched)
    return tp, fp, len(remaining)


def measures(tp, fp, fn):
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0., f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def timestamps(labels, endpoints, fall_index):
    times = []
    previous = False
    for label, frame in zip(labels, endpoints):
        current = int(label) == fall_index
        if current and not previous:
            times.append(float(frame)/25.)
        previous = current
    return times


def main():
    import torch
    cfg = io.read(ROOT / 'configs/usdrl_external_20261004_r1.json')
    for path, digest in io.read(OUT / 'contract.json')['files'].items():
        io.require(io.sha(ROOT / path) == digest, 'changed contract ' + path)
    evaluation = io.read(OUT / 'evaluation.json')
    truth = {r['id']: r for r in io.read(OUT / 'ground_truth.json')}
    plan = io.read(OUT / 'plan.json')
    rows = {r['id']: r for r in evaluation['rows']}
    own = {r['id']: r for r in io.read(OUT / 'own_predictions.json')}
    io.require(len(rows) == len(truth) == len(plan) == len(own) == 230, 'membership count')
    state = torch.load(ROOT / cfg['head'], weights_only=True, map_location='cpu')
    w = state['weight'].numpy().astype(np.float64)
    b = state['bias'].numpy().astype(np.float64)
    maximum_error = 0.
    maximum_pooled_error = 0.
    checked_windows = 0
    accum = {(name, scope): [] for name in ('usdrl_ntu60', 'own') for scope in cfg['datasets']}
    cases = []
    for item in plan:
        sid = item['id']
        gt = truth[sid]
        row = rows[sid]
        io.require(row['episodes'] == gt['episodes'] and row['scope'] == gt['scope'], 'GT mismatch')
        dest = OUT / sid
        receipt = io.read(dest / 'inference.json')
        io.require(receipt['contract_sha256'] == io.sha(OUT/'contract.json'), 'receipt contract mismatch')
        io.require(receipt['processed'] == row['processed'] == item['processed'], 'quality mismatch')
        times, own_times = [], []
        labels = np.array([], dtype=np.int64)
        hist = np.zeros(60, np.int64)
        own_positive = False
        max_probability = 0.
        if item['processed']:
            io.require(io.sha(dest/'inference.npz') == receipt['payload_sha256'], 'new output hash')
            with np.load(dest/'inference.npz', allow_pickle=False) as z:
                logits, pooled, endpoints, starts = z['logits'], z['pooled'], z['window_endpoints'], z['window_starts']
            io.require(logits.shape == (item['windows'], 60) and pooled.shape == (item['windows'], 2048), 'shape')
            np.testing.assert_array_equal(endpoints, starts+63)
            with np.load(ROOT/item['parent']/sid/'lift.npz', allow_pickle=False) as lift:
                np.testing.assert_array_equal(starts, np.arange(0, len(lift['ntu25'])-63, 8))
            cpu = pooled.astype(np.float64) @ w.T + b
            error = float(np.max(np.abs(cpu-logits)))
            maximum_error = max(maximum_error, error)
            np.testing.assert_allclose(cpu, logits, atol=cfg['audit']['atol'], rtol=cfg['audit']['rtol'])
            np.testing.assert_array_equal(cpu.argmax(1), logits.argmax(1))
            labels = cpu.argmax(1)
            times = timestamps(labels, endpoints, 42)
            hist = np.bincount(labels, minlength=60)
            exp = np.exp(logits.astype(float)-logits.max(1, keepdims=True))
            max_probability = float((exp[:,42]/exp.sum(1)).max())
            parent = ROOT/item['parent']/sid
            with np.load(parent/'inference.npz', allow_pickle=False) as prior:
                np.testing.assert_array_equal(endpoints, prior['window_endpoints'])
                np.testing.assert_allclose(pooled, prior['pooled'], atol=cfg['audit']['atol'], rtol=cfg['audit']['rtol'])
                maximum_pooled_error = max(maximum_pooled_error, float(np.max(np.abs(pooled-prior['pooled']))))
                own_times = timestamps(prior['G0'].argmax(1), prior['window_endpoints'], 1)
                own_positive = bool(np.any(prior['G0'].argmax(1) == 1))
            checked_windows += len(labels)
        else:
            io.require(not (dest/'inference.npz').exists(), 'rejection has payload')
        io.require(times == [p['time'] for p in row['predictions']], 'alarm edges mismatch')
        io.require(own_times == [p['time'] for p in own[sid]['predictions']], 'own alarm mismatch')
        io.require(hist.tolist() == row['class_histogram'] and int(hist[42]) == row['fall_windows'], 'class histogram mismatch')
        io.require(abs(row['max_fall_probability']-max_probability) < 1e-12, 'score mismatch')
        positive = bool(np.any(labels == 42))
        io.require(positive == bool(row['video_prediction']), 'video prediction mismatch')
        for name, tt, pp, rr in [('usdrl_ntu60', times, positive, row), ('own', own_times, own_positive, own[sid])]:
            ec = counts(tt, gt['episodes'])
            io.require(ec == tuple(rr['event'][k] for k in ('tp','fp','fn')), 'event mismatch')
            io.require(pp == bool(rr['video_prediction']), 'own video prediction mismatch')
            accum[name, item['scope']].append(dict(truth=bool(gt['episodes']), pred=pp, event=ec,
                                                 processed=item['processed'], windows=item['windows'],
                                                 fall_windows=int(hist[42]), hist=hist))
        cases.append(dict(dataset=item['scope'], sample_id=sid, gt_fall=int(bool(gt['episodes'])),
                          processed=int(item['processed']), windows=item['windows'], fall_windows=int(hist[42]),
                          predicted_fall=int(positive), event_tp=row['event']['tp'], event_fp=row['event']['fp'],
                          event_fn=row['event']['fn'], max_fall_probability=max_probability,
                          most_frequent_class=f'A{int(hist.argmax())+1:03d}' if len(labels) else '',
                          alarm_seconds=json.dumps(times), own_predicted_fall=int(own_positive)))
    for (name, scope), rr in accum.items():
        s = evaluation['summaries'][name][scope]
        ec = tuple(sum(r['event'][i] for r in rr) for i in range(3))
        tp = sum(r['truth'] and r['pred'] for r in rr)
        fp = sum(not r['truth'] and r['pred'] for r in rr)
        fn = sum(r['truth'] and not r['pred'] for r in rr)
        tn = sum(not r['truth'] and not r['pred'] for r in rr)
        io.require(measures(*ec) == s['event'], 'aggregate event mismatch')
        v = dict(measures(tp,fp,fn), tn=tn, accuracy=(tp+tn)/len(rr), specificity=tn/(tn+fp))
        io.require(v == s['video'], 'aggregate video mismatch')
        io.require(s['processed'] == sum(r['processed'] for r in rr), 'processed count')
        if name == 'usdrl_ntu60':
            io.require(s['windows'] == sum(r['windows'] for r in rr), 'window count')
            io.require(s['fall_windows'] == sum(r['fall_windows'] for r in rr), 'fall window count')
            io.require(s['class_histogram'] == np.sum([r['hist'] for r in rr], axis=0).tolist(), 'aggregate classes')
    model_audit = io.read(OUT/'model_audit.json')
    io.require(model_audit['passed'] and model_audit['before'] == model_audit['after'], 'weight mutation')
    io.require(checked_windows == 6286, 'missing evaluated windows')
    with (OUT/'cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    report = dict(passed=True, videos=len(plan), windows=checked_windows, cpu_head_max_abs_error=maximum_error,
                  parent_pooled_max_abs_error=maximum_pooled_error, all_argmax_equal=True,
                  independent_event_video_metrics=True, no_target_tuning=True,
                  evaluation_sha256=io.sha(OUT/'evaluation.json'), cases_sha256=io.sha(OUT/'cases.csv'))
    io.save(OUT/'independent_audit.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
