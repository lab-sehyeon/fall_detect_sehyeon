"""Frozen current RGB pipeline, Le2i D1 fall-event evaluation, historical comparison.

No target training, calibration, recovery scoring or legacy code mutation.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

from . import rgb_document_io as io

ROOT = io.ROOT
CONFIG = ROOT/'configs/le2i_current_evaluation_20261002_r1.json'
OUTPUT = ROOT/'data/fall_processed/RGB/le2i_current_evaluation_20261002_r1'
PREP = ROOT/'data/source_archives/Le2i-FDD/ffv1_20261002_r2'
SOURCE = ROOT/'data/source_archives/Le2i-FDD/kaggle_v2_20261002_r1'
HISTORY = {'G2': dict(tp=79, fp=14, fn=17), 'G0': dict(tp=80, fp=14, fn=16)}


def status(stage, **values):
    io.save(OUTPUT/'status.json', dict(stage=stage, time=datetime.now(timezone.utc).isoformat(), **values))
    print(stage, values, flush=True)


def metrics(tp, fp, fn):
    return dict(tp=int(tp), fp=int(fp), fn=int(fn),
                precision=float(tp/(tp+fp)) if tp+fp else 0.,
                recall=float(tp/(tp+fn)) if tp+fn else 0.,
                f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)


def rising_edges(endpoints, logits, fps=25.):
    endpoints = np.asarray(endpoints, np.int64)
    logits = np.asarray(logits)
    io.require(logits.shape == (len(endpoints), 4) and np.isfinite(logits).all(), 'invalid logits')
    io.require(len(endpoints) == 0 or np.all(np.diff(endpoints) > 0), 'invalid endpoints')
    fall = logits.argmax(1) == 1
    edges = np.flatnonzero(fall & ~np.r_[False, fall[:-1]]) if len(fall) else np.array([], dtype=np.int64)
    return [dict(frame=int(endpoints[i]), time=float(endpoints[i]/fps)) for i in edges]


def match_events(predictions, episodes, early=.5, late=3.):
    used = set(); matches = []; unmatched = []
    for prediction in sorted(predictions, key=lambda p: p['time']):
        candidates = [i for i, event in enumerate(episodes) if i not in used and
                      event['fall_start']-early <= prediction['time'] <= max(event['fall_start'], event['fall_end'])+late]
        if candidates:
            index = min(candidates, key=lambda i: (episodes[i]['fall_start'], i)); used.add(index)
            matches.append(dict(prediction=prediction, gt_index=index,
                                onset_delay=prediction['time']-episodes[index]['fall_start']))
        else: unmatched.append(prediction)
    return {**metrics(len(matches), len(unmatched), len(episodes)-len(used)), 'matches': matches,
            'unmatched_predictions': unmatched, 'unmatched_gt_indices': [i for i in range(len(episodes)) if i not in used]}


def digest_contract(config):
    result = io.contract(config)
    result.update(runner_sha256=io.sha(Path(__file__)),
                  preparation_manifest_sha256=io.sha(PREP/'manifest.json'),
                  preparation_contract_sha256=io.sha(PREP/'contract.json'),
                  plan_sha256=io.sha(OUTPUT/'plan.json'),
                  ground_truth_sha256=io.sha(OUTPUT/'ground_truth.json'),
                  scope_sha256=io.sha(CONFIG))
    return result


def setup():
    io.CONFIG = CONFIG
    config = io.read(CONFIG)
    def locked(cfg):
        io.require(io.read(OUTPUT/'contract.json') == digest_contract(cfg), 'Le2i contract changed; do not resume')
        return OUTPUT, io.read(OUTPUT/'plan.json')
    io.locked = locked
    return config


def initialize():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if CONFIG.exists():
        config = setup(); io.locked(config); return config
    prepared = io.read(PREP/'manifest.json')
    io.require(prepared['passed'] and prepared['pixel_identical'] == 127, 'pixel validation incomplete')
    plan, truth = [], []
    for row in prepared['records']:
        io.require(row['pixel_identical'], 'pixel validation failed')
        item = {k: row[k] for k in ('id', 'scene', 'number', 'video', 'video_sha256', 'mapping',
                                    'mapping_sha256', 'frames', 'source_frames', 'source_fps',
                                    'source_width', 'source_height', 'source_duration_seconds')}
        starts = np.arange(0, item['frames']-63, 8)
        io.require(len(starts) > 0, 'short sequence outside approved subset')
        start, end = row['raw_header']
        io.require(start == end == 0 or 0 < start <= end <= row['source_frames'], 'GT outside decoded source frames')
        episodes = [] if start == end == 0 else [dict(fall_start=start/row['source_fps'], fall_end=end/row['source_fps'])]
        for episode in episodes:
            accepted = (starts+63)/25
            io.require(np.any((accepted >= episode['fall_start']-.5) & (accepted <= episode['fall_end']+3)), 'unscorable event')
        plan.append(item)
        truth.append(dict(id=item['id'], scene=item['scene'], episodes=episodes,
                          annotation=row['annotation'], annotation_sha256=row['annotation_sha256'],
                          raw_header=row['raw_header'], recovery_annotation_available=False))
    io.require(len(plan) == 127 and sum(len(r['episodes']) for r in truth) == 96, 'evaluation denominator')
    io.save(OUTPUT/'plan.json', plan); io.save(OUTPUT/'ground_truth.json', truth)
    config = copy.deepcopy(io.read(ROOT/'configs/rgb_document_integration_v1.json'))
    config.update(experiment_id='LE2I_CURRENT_EVALUATION_20261002_R1', output=str(OUTPUT.relative_to(ROOT)),
                  alignment=str((OUTPUT/'plan.json').relative_to(ROOT)), alignment_sha256=io.sha(OUTPUT/'plan.json'),
                  video_root=str(PREP.relative_to(ROOT)), plan='fixed127 historical subset; no prediction selection',
                  sample_count=127, scope='current frozen pipeline vs documented historical Le2i; not bit-exact reproduction')
    config['execution']['max_added_gib'] = 16
    config['newly_specified'] = [
        'Current reconstructed DSTE/J1/global131/scaler/heads; not original lost checkpoint bytes',
        'Existing current-pipeline past-only25fps sampling and no appended tail, retained without target tuning',
        'Historical127 subset and D1 event matching; source frame header divided by actual rational FPS',
        'Known3 missing annotation headers excluded before model inference; all quality failures retained',
        'G2 and G0 reported separately; current G2 source-validation non-adoption unchanged',
        'Single physicalGPU0; no training/calibration; recovery unscored because no GT']
    config['evaluation'] = dict(heads=['G2', 'G0'], event='fall_trigger', decoder='D1_DECOUPLED rising edge',
        early_seconds=.5, late_seconds=3., refractory_seconds=0., gt_time='header_frame / source_fps; no subtract1',
        quality_rejection='no alarm; retain in primary denominator', recovery_scored=False,
        target_training=False, target_calibration=False,
        historical_counts=HISTORY, historical_quality_pass=109, historical_g2_quality_counts=dict(tp=79, fp=14, fn=2),
        caveat='current past-only sampling, no appended tail, reconstructed weights/features; pipeline-level comparison only')
    io.save(CONFIG, config); io.CONFIG = CONFIG
    io.save(OUTPUT/'contract.json', digest_contract(config))
    config = setup(); io.locked(config)
    status('prepared', videos=127, fall_events=96, negatives=31, frames=sum(r['frames'] for r in plan))
    return config


def score():
    config = setup(); _, plan = io.locked(config)
    truth = {r['id']: r for r in io.read(OUTPUT/'ground_truth.json')}
    rows = []
    for item in plan:
        sid = item['id']; dest = OUTPUT/sid; front = io.stage_done(dest, 'frontend')
        io.require(front, 'missing frontend')
        gt = truth[sid]
        io.require(io.sha(ROOT/gt['annotation']) == gt['annotation_sha256'], 'annotation changed')
        row = dict(id=sid, scene=item['scene'], source_fps=item['source_fps'],
                   source_height=item['source_height'], frames=item['frames'], duration=item['source_duration_seconds'],
                   episodes=gt['episodes'], quality=front['quality'], quality_passed=front['quality']['passed'], heads={})
        if front['quality']['passed']:
            io.require(io.stage_done(dest, 'lift') and io.stage_done(dest, 'inference'), 'missing classification')
            with np.load(dest/'inference.npz', allow_pickle=False) as saved:
                expected = np.arange(0, item['frames']-63, 8)+63
                np.testing.assert_array_equal(saved['window_endpoints'], expected)
                for head in config['evaluation']['heads']:
                    predictions = rising_edges(expected, saved[head])
                    row['heads'][head] = dict(predictions=predictions, **match_events(predictions, gt['episodes']))
        else:
            io.require(not (dest/'inference.npz').exists(), 'quality gate bypass')
            for head in config['evaluation']['heads']:
                row['heads'][head] = dict(predictions=[], **match_events([], gt['episodes']))
        rows.append(row)
    def aggregate(subset, head):
        total = metrics(*(sum(r['heads'][head][k] for r in subset) for k in ('tp', 'fp', 'fn')))
        delays = [m['onset_delay'] for r in subset for m in r['heads'][head]['matches']]
        return {**total, 'videos': len(subset), 'fall_events': sum(len(r['episodes']) for r in subset),
                'quality_pass': sum(r['quality_passed'] for r in subset),
                'fp_per_hour': total['fp']/(sum(r['duration'] for r in subset)/3600) if subset else None,
                'onset_delay_median_seconds': float(np.median(delays)) if delays else None}
    evaluations = {}
    for head in config['evaluation']['heads']:
        primary = aggregate(rows, head); old = metrics(**HISTORY[head])
        evaluations[head] = dict(primary=primary, quality_pass=aggregate([r for r in rows if r['quality_passed']], head),
            per_scene={scene: aggregate([r for r in rows if r['scene'] == scene], head) for scene in sorted({r['scene'] for r in rows})},
            historical_primary=old, difference_percentage_points={k: (primary[k]-old[k])*100 for k in ('precision', 'recall', 'f1')})
    result = dict(passed=True, videos=len(rows), total_fall_events=96, negative_videos=31,
                  quality_pass=sum(r['quality_passed'] for r in rows),
                  quality_rejected_fall_events=sum(len(r['episodes']) for r in rows if not r['quality_passed']),
                  heads=evaluations, rows=rows, recovery_scored=False, training=False, calibration=False,
                  historical_protocol_identical=False, historical_weight_identical=False,
                  contract_sha256=io.sha(OUTPUT/'contract.json'), config_sha256=io.sha(CONFIG))
    io.locked(config); io.save(OUTPUT/'evaluation.json', result)
    status('scored_pending_independent_audit', quality_pass=result['quality_pass'],
           heads={head: result['heads'][head]['primary'] for head in evaluations})


def worker(stage):
    config = setup(); io.locked(config); io.safety(config)
    if stage == 'score': score(); return
    # Observe existing stage receipts without changing the frozen implementation.
    original_save = io.save
    stage_name = 'inference' if stage == 'infer' else stage
    def save_and_report(path, value):
        original_save(path, value)
        if Path(path).name == stage_name+'.json':
            original_save(OUTPUT/'status.json', dict(stage=stage, time=datetime.now(timezone.utc).isoformat(),
                completed=len(list(OUTPUT.glob('*/'+stage_name+'.json'))), total=127, id=Path(path).parent.name))
    io.save = save_and_report
    if stage == 'frontend':
        from .rgb_document_frontend import main as run
    elif stage == 'lift':
        from .rgb_document_lift import main as run
    else:
        from .rgb_document_infer import main as run
    run()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['prepare', 'all', 'frontend', 'lift', 'infer', 'score'], default='all')
    p.add_argument('--worker', action='store_true'); args = p.parse_args()
    if args.worker:
        io.require(args.stage not in ('prepare', 'all'), 'worker stage'); worker(args.stage); return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            config = initialize(); io.safety(config)
            if args.stage == 'prepare': return
            stages = ['frontend', 'lift', 'infer', 'score'] if args.stage == 'all' else [args.stage]
            for stage in stages:
                io.safety(config); status(stage, total=127)
                subprocess.run([sys.executable, '-m', 'fall_pipeline.external.le2i_current_evaluation',
                                '--stage', stage, '--worker'], cwd=ROOT, check=True)
        except Exception as exc:
            status('paused', error=f'{type(exc).__name__}: {exc}'); raise


if __name__ == '__main__': main()
