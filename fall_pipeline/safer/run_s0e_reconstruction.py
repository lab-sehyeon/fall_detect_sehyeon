"""72 fixed, CPU-only hard-rule candidates with immutable val-only selection."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from fall_pipeline.safer import run_s0c_reconstruction_r3 as c
from fall_pipeline.safer import s0e_core as core

s, io = c.s, c.io
CONFIG = io.ROOT / 'configs/s0e_document_reconstruction_v1.json'
CODE = c.CODE + ['fall_pipeline/safer/s0e_core.py', 'fall_pipeline/safer/run_s0e_reconstruction.py']
GLOBAL_PAUSE = io.ROOT / 'checkpoint/fall/STATE_REMAINING_20260930_R1/PAUSE_REQUESTED'


def pause(root, cfg):
    if s.STOP or (root/'PAUSE_REQUESTED').exists() or GLOBAL_PAUSE.exists():
        raise io.PauseRequested('explicit state-remaining pause')
    try:
        io.disk_gate(cfg)
    except RuntimeError as error:
        raise io.PauseRequested(str(error)) from error


def publish(root, stage, **details):
    state = {'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **details}
    io.save_json(root/'status.json', state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    if root.name.endswith('_smoke'):
        return
    status = 'completed' if stage == 'completed' else 'paused' if stage in ('failed', 'paused') else 'in_progress'
    header = f'- 문서 ID: `DOC-20260930-s0e-reconstruction-run-R1`\n- 기준일: 2026-09-30\n- 상태: `{status}`\n'
    internal = '# S0-E 자동 실행 기록\n\n' + header + '\n```json\n' + json.dumps(state, ensure_ascii=False, indent=2) + '\n```\n'
    internal += f'\nOutput: `{root}`\n'
    shared = '# S0-E 실험 진행\n\n' + header + f'\n현재 단계: {stage}.\n'
    for key in ('completed_candidates', 'total_candidates', 'split', 'adoption'):
        if key in details:
            shared += f'\n- {key}: {details[key]}\n'
    for kind, value in (('internal', internal), ('shared', shared)):
        value += f'\n[사전 계약과 한계](2026-09-30_s0e_reconstruction_{kind}.md). 전체 연구 완료와 구분한다.\n'
        path = io.ROOT/f'docs/{kind}/2026-09-30_s0e_reconstruction_run_{kind}.md'
        tmp = path.with_suffix('.tmp'); tmp.write_text(value); os.replace(tmp, path)


def parent(cfg):
    root = io.ROOT/cfg['s0c_root']; report = io.read(root/'final_report.json')
    io.require(io.sha256_file(root/'final_report.json') == cfg['s0c_report_sha256'], 'C report pin')
    io.require(report['passed'] and report['research_usable'], 'C not complete')
    contract = io.read(root/'run_contract.json')
    io.require(io.sha256_file(io.ROOT/cfg['s0c_config']) == contract['config_sha256'], 'C config changed')
    io.require(s.fingerprint(c.CODE) == contract['source'], 'C source changed')
    ac, sources = c.sources(io.read(io.ROOT/cfg['s0c_config']))
    for name, item in sources.items():
        for filename, sha in contract['parents'][name].items():
            io.require(io.sha256_file(item['root']/filename) == sha, 'C parent changed')
    io.require(io.sha256_file(root/'selection.json') == report['selection_sha256'], 'C selection changed')
    return ac, root, report


def records(ac, source, split, cfg, smoke):
    data = s.core.verify_split(ac, split); manifest = io.read(source/'evaluation'/split/'manifest.json')
    out = []
    for row, begin, end in data.sequences():
        path = source/'evaluation'/split/(row['uid']+'.npy')
        io.require(io.sha256_file(path) == manifest['payload'][path.name], 'C prediction changed')
        pred = np.load(path, allow_pickle=False)
        _, labels = data.arrays(row['sequence_index'])
        truth = np.asarray(labels[s.core.coverage(len(labels), data.starts[begin:end]) > 0])
        io.require(pred.shape == truth.shape, 'C covered support differs')
        if smoke:
            pred, truth = pred[:cfg['execution']['smoke_frames']], truth[:cfg['execution']['smoke_frames']]
        out.append({'uid': row['uid'], 'pred': pred, 'truth': truth})
        if smoke and len(out) >= cfg['execution']['smoke_sequences']:
            break
    return out


def summarize_latency(events):
    out = {}
    for pair in ('10->12', '6->4', '12->7'):
        rows = [e for e in events if e['pair'] == pair]; delay = [e['delay_frames'] for e in rows if e['detected']]
        out[pair] = {'events': len(rows), 'detected': len(delay), 'missed': len(rows)-len(delay),
                     'recall': len(delay)/len(rows) if rows else None,
                     'premature': sum(e['premature'] for e in rows), 'boundary_correct': sum(e['boundary_correct'] for e in rows),
                     'delay_median_frames': float(np.median(delay)) if delay else None,
                     'delay_p90_frames': float(np.percentile(delay, 90)) if delay else None,
                     'miss_dominant_counts': {str(k): sum(e['miss_dominant_class'] == k for e in rows) for k in range(16)}}
    return out


def evaluate(rows, spec, ac, cfg, root, dest=None):
    metric = s.a.metric_instance(ac); digests = {}; events = []
    for row in rows:
        pause(root, cfg)
        pred = row['pred'] if spec is None else core.decode(row['pred'], spec, cfg['rules'])
        np.testing.assert_array_equal(pred == 10, row['pred'] == 10)
        if spec:
            for cut in (1, min(128, len(pred)), len(pred)//2):
                np.testing.assert_array_equal(pred[:cut], core.decode(row['pred'][:cut], spec, cfg['rules']))
        metric.add(pred, row['truth']); digests[row['uid']] = io.digest(pred)
        events += [{'uid': row['uid'], **e} for e in c.core.latency_events(pred, row['truth'], [(10,12),(6,4),(12,7)], cfg['latency'])]
        if dest is not None:
            path = dest/(row['uid']+'.npy')
            if path.exists():
                np.testing.assert_array_equal(np.load(path, allow_pickle=False), pred)
            else:
                io.save_array(path, pred)
    result = {'metrics': metric.result(), 'latency': summarize_latency(events), 'prediction_digests': digests,
              'fall_mask_exact': True, 'prefix_invariance': True}
    if dest is not None:
        io.save_json(dest/'manifest.json', result); io.save_json(dest/'latency_events.json', events)
    return result


def run(args, cfg, root):
    io.require(Path(sys.prefix).name == 'fall_detect' and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only fall_detect required')
    torch.set_num_threads(cfg['execution']['cpu_threads'])
    ac, source, source_report = parent(cfg)
    contract = {'source': s.fingerprint(CODE), 'config_sha256': io.sha256_file(CONFIG),
                'parent_sha256': cfg['s0c_report_sha256'], 'smoke': args.smoke, 'numpy': str(np.__version__),
                'historical_exact_reproduction': False}
    s.lock_contract(root, contract, args.resume)
    if (root/'final_report.json').exists():
        report = io.read(root/'final_report.json')
        io.require(report['passed'], 'existing report failed')
        io.require(io.sha256_file(root/'final_report.json') == io.read(root/'status.json')['final_report_sha256'], 'existing report changed')
        return
    pause(root, cfg)
    if not args.smoke:
        smoke = Path(str(root)+'_smoke')
        io.require(io.read(smoke/'final_report.json')['passed'], 'smoke required')
        io.require({**io.read(smoke/'run_contract.json'), 'smoke': False} == contract, 'smoke contract mismatch')
    publish(root, 'preflight', holdouts_opened=False)
    val = records(ac, source, 'val', cfg, args.smoke)
    baseline = evaluate(val, None, ac, cfg, root)
    if not args.smoke:
        io.require(baseline['metrics'] == source_report['metrics']['val'], 'C baseline metrics differ')
        io.require(baseline['latency'] == source_report['latency']['val'], 'C latency definition differs')
    specs = core.grid(cfg); io.require(len(specs) == 72 == cfg['grid']['count'], '72 candidates required')
    if args.smoke:
        specs = specs[:1]
    path = root/'validation/candidates.json'; rows = io.read(path) if path.exists() else []
    io.require(len(rows) <= len(specs), 'candidate count changed')
    for i, spec in enumerate(specs):
        result = evaluate(val, spec, ac, cfg, root)
        row = {'spec': spec, **result, 'gate': core.gates(result['metrics'], baseline['metrics'], result['latency'], baseline['latency'], result['fall_mask_exact'], cfg['evaluation'])}
        if i < len(rows):
            io.require(rows[i] == row, 'candidate deterministic replay differs')
        else:
            rows.append(row); io.save_json(path, rows)
        publish(root, 'decode', completed_candidates=i+1, total_candidates=len(specs))
    best = core.choose(rows)
    if args.smoke:
        report = {'passed': True, 'research_usable': False, 'candidates': 1, 'holdouts_opened': False}
    else:
        selection = {**best, 'selected_using': 'val only', 'holdouts_opened_at_selection': False,
                     'run_contract_sha256': io.sha256_file(root/'run_contract.json'), 'candidates_sha256': io.sha256_file(path)}
        selected = root/'selection.json'
        if selected.exists():
            io.require(io.read(selected) == selection, 'immutable selection differs')
        else:
            io.save_json(selected, selection)
        selected_sha = io.sha256_file(selected)
        publish(root, 'selection', adoption=best['adoption'], candidate=best['spec']['name'])
        final = {}
        for split in ('val', 'test', 'ood'):
            pause(root, cfg); io.require(io.sha256_file(selected) == selected_sha, 'selection changed before holdout')
            current = val if split == 'val' else records(ac, source, split, cfg, False)
            final[split] = evaluate(current, best['spec'], ac, cfg, root, root/'evaluation'/split)
            if split == 'val':
                io.require(final[split]['metrics'] == best['metrics'], 'selected val replay differs')
            publish(root, 'evaluate', split=split)
        publish(root, 'audit')
        # Reproduce every candidate digest and replay all saved final arrays.
        for row in rows:
            for record in val:
                pause(root, cfg)
                io.require(io.digest(core.decode(record['pred'], row['spec'], cfg['rules'])) == row['prediction_digests'][record['uid']], 'candidate digest replay')
        for split in ('val', 'test', 'ood'):
            for uid, digest in final[split]['prediction_digests'].items():
                io.require(io.digest(np.load(root/'evaluation'/split/(uid+'.npy'), allow_pickle=False)) == digest, 'final saved digest')
        parent(cfg)
        io.require(s.fingerprint(CODE) == contract['source'] and io.sha256_file(CONFIG) == contract['config_sha256'], 'E source/config changed')
        report = {'passed': True, 'research_usable': True, 'historical_exact_reproduction': False,
                  'candidates': 72, 'adoption': best['adoption'], 'eligible_count': best['eligible_count'],
                  'selected_spec': best['spec'], 'gate': best['gate'], 'selection_sha256': selected_sha,
                  'metrics': {k:v['metrics'] for k,v in final.items()}, 'latency': {k:v['latency'] for k,v in final.items()},
                  'training': False, 'model_inference': False, 'fall_mask_exact': True, 'frozen_parent_unchanged': True,
                  'all_candidate_digest_replay': True, 'end_to_end_causal': False, 'automatic_deployment': False}
    io.save_json(root/'final_report.json', report)
    publish(root, 'completed', adoption=report.get('adoption', False), final_report_sha256=io.sha256_file(root/'final_report.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); cfg = io.read(CONFIG)
    root = io.guard(io.ROOT/(cfg['output']+('_smoke' if args.smoke else '')))
    io.require(root.parent == io.ROOT/'checkpoint/fall', 'output scope')
    root.mkdir(parents=True, exist_ok=True); s.install_signals()
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(args, cfg, root)
        except io.PauseRequested as error:
            publish(root, 'paused', reason=str(error)); raise SystemExit(75)
        except Exception as error:
            publish(root, 'failed', reason=str(error)); raise


if __name__ == '__main__':
    main()
