"""Official Le2i-CS videos, unchanged continuous RGB inference, D1 events."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, fcntl, shutil
import numpy as np
from . import rgb_document_io as io
from . import le2i_current_evaluation as old
from .omnifall_le2i_20261002 import ASSET, official_rows

ROOT = io.ROOT
OUTPUT = ROOT/'data/fall_processed/RGB/omnifall_le2i_continuous_20261002_r1'
PREP = ROOT/'data/source_archives/OmniFall/le2i_continuous_20261002_r1'
CONFIG = ROOT/'configs/omnifall_le2i_continuous_20261002_r1.json'
INPUT_KEYS = ['id','scene','number','video','video_sha256','mapping','mapping_sha256','frames',
              'source_frames','source_fps','source_width','source_height','source_duration_seconds']
CODE = ['fall_pipeline/external/omnifall_le2i_continuous_20261002.py',
        'fall_pipeline/external/omnifall_le2i_20261002.py',
        'fall_pipeline/external/le2i_current_evaluation.py',
        'scripts/prepare_omnifall_le2i_continuous_20261002.py',
        'scripts/run_omnifall_le2i_continuous_20261002.py',
        'scripts/audit_omnifall_le2i_continuous_20261002.py',
        'fall_pipeline/external/rgb_recovery_common.py']


def status(stage, **values):
    io.save(OUTPUT/'status.json', dict(stage=stage, time=datetime.now(timezone.utc).isoformat(), **values))
    print(stage, values, flush=True)


def contract(cfg):
    result = io.contract(cfg)
    result['own_code'] = {p:io.sha(ROOT/p) for p in CODE}
    result['inputs'] = {str(p.relative_to(ROOT)):io.sha(p) for p in [
        PREP/'manifest.json', PREP/'contract.json', OUTPUT/'plan.json', OUTPUT/'ground_truth.json',
        OUTPUT/'cache_provenance.json', ASSET/'original_manifest.json', ASSET/'source_manifest.json',
        old.OUTPUT/'contract.json', old.OUTPUT/'independent_audit.json']}
    result['metadata'] = {str(p.relative_to(ASSET)):io.sha(p) for p in sorted((ASSET/'metadata').rglob('*')) if p.is_file()}
    return result


def setup():
    io.CONFIG = CONFIG
    def locked(cfg):
        io.require(io.read(OUTPUT/'contract.json') == contract(cfg), 'continuous evaluation contract changed')
        return OUTPUT, io.read(OUTPUT/'plan.json')
    io.locked = locked
    return io.read(CONFIG)


def make_truth(rows, plan):
    truth = []
    for item in plan:
        segments = [(n,r) for n,r in enumerate(rows) if r['path'] == item['path']]
        episodes = [dict(fall_start=float(r['start']), fall_end=float(r['end']), row_index=n)
                    for n,r in segments if int(r['label']) == 1]
        episodes.sort(key=lambda r:r['fall_start'])
        io.require(len(episodes) <= 1, 'independent single-event audit scope changed')
        for e in episodes:
            times = (np.arange(0,item['frames']-63,8)+63)/25
            io.require(0 <= e['fall_start'] < e['fall_end'] <= item['source_duration_seconds'], 'fall GT outside video')
            io.require(np.any((times >= e['fall_start']-.5)&(times <= e['fall_end']+3)), 'unscorable fall')
        truth.append(dict(id=item['id'], path=item['path'], episodes=episodes,
                          official_segment_indices=[n for n,_ in segments]))
    io.require(len(truth) == 38 and sum(len(r['episodes']) for r in truth) == 22, 'GT denominator')
    return truth


def initialize():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if CONFIG.exists():
        cfg = setup(); io.locked(cfg); return
    prepared = io.read(PREP/'manifest.json')
    io.require(prepared['passed'] and prepared['pixel_identical'] == 38, 'input validation incomplete')
    rows, paths = official_rows()
    plan = [{**{k:r[k] for k in INPUT_KEYS}, 'path':r['path']} for r in prepared['records']]
    io.require([r['path'] for r in plan] == paths, 'sample order')
    io.save(OUTPUT/'plan.json', plan); io.save(OUTPUT/'ground_truth.json', make_truth(rows, plan))
    cfg = copy.deepcopy(io.read(old.CONFIG))
    cfg.update(experiment_id='OMNIFALL_LE2I_CONTINUOUS_20261002_R1', output=str(OUTPUT.relative_to(ROOT)),
        alignment=str((OUTPUT/'plan.json').relative_to(ROOT)), alignment_sha256=io.sha(OUTPUT/'plan.json'),
        video_root=str(PREP.relative_to(ROOT)), sample_count=38,
        plan='all official Le2i-CS38 test videos, continuous processing,22 official fall events',
        scope='restore current127-video input contract on official38-video test; not historical bit-exact')
    cfg['execution']['max_added_gib'] = 4
    cfg['newly_specified'] = ['fixed official38 test videos; same source models and continuous input contract',
        'official dense label1 intervals instead of legacy TXT; D1 event scoring unchanged',
        'reuse22 hash-verified identical caches;16 new full-video pipelines',
        'no invented segment reducer; event score not directly comparable to prior segment F1']
    cfg['evaluation'] = dict(heads=['G0','G2'], event='fall_trigger', decoder='D1_DECOUPLED rising edge',
        early_seconds=.5, late_seconds=3., refractory_seconds=0., gt_time='official parquet seconds verified against CSV',
        quality_rejection='no alarm; retain in primary denominator', recovery_scored=False,
        target_training=False, target_calibration=False, segment_classification=False)
    # Check all inference-affecting settings and implementation/asset contracts before inheriting traces.
    original_cfg = io.read(old.CONFIG)
    for key in ['asset_root','sampling','detector','pose_config','vitpose_commit','pose_inference',
                'quality','lifting_source_config','lifting','inference']:
        io.require(cfg[key] == original_cfg[key], 'pipeline setting changed: '+key)
    io.CONFIG = old.CONFIG
    io.require(old.digest_contract(original_cfg) == io.read(old.OUTPUT/'contract.json'), 'old frozen run changed')
    audit = io.read(old.OUTPUT/'independent_audit.json')
    io.require(audit['passed'] and audit['evaluation_sha256'] == io.sha(old.OUTPUT/'evaluation.json'), 'old audit invalid')
    previous = {r['id']:r for r in io.read(old.OUTPUT/'plan.json')}
    inherited = []
    for item in plan:
        if item['id'] not in previous: continue
        io.require({k:item[k] for k in INPUT_KEYS} == previous[item['id']], 'cached input differs')
        sid = item['id']; src = old.OUTPUT/sid; dest = OUTPUT/sid; dest.mkdir(exist_ok=True)
        front = io.stage_done(src,'frontend'); io.require(front and front['passed'], 'cached frontend missing')
        filenames = ['frontend.json','frontend.npz']
        if front['quality']['passed']:
            for stage in ['lift','inference']:
                io.require(io.stage_done(src,stage)['passed'], 'cached stage invalid')
                filenames += [stage+'.json',stage+'.npz']
        else: filenames += ['quality_rejection.json']
        hashes = {}
        for name in filenames:
            digest = io.sha(src/name)
            if (dest/name).exists(): io.require(io.sha(dest/name) == digest, 'destination collision')
            else: shutil.copy2(src/name, dest/name)
            io.require(io.sha(dest/name) == digest, 'copy validation')
            hashes[name] = digest
        inherited.append(dict(id=sid, source=str(src.relative_to(ROOT)), hashes=hashes))
    io.require(len(inherited) == 22, 'cache count')
    io.save(OUTPUT/'cache_provenance.json', dict(reused=22, new_videos=16, records=inherited,
        old_contract_sha256=io.sha(old.OUTPUT/'contract.json'), old_audit_sha256=io.sha(old.OUTPUT/'independent_audit.json')))
    io.save(CONFIG,cfg); io.CONFIG = CONFIG
    io.save(OUTPUT/'contract.json',contract(cfg))
    status('prepared', videos=38, fall_events=22, cache_reused=22, fresh_videos=16)


def aggregate(rows, head):
    m = old.metrics(*(sum(r['heads'][head][k] for r in rows) for k in ('tp','fp','fn')))
    return dict(m, videos=len(rows), fall_events=sum(len(r['episodes']) for r in rows),
                quality_pass=sum(r['quality_passed'] for r in rows))


def score():
    cfg = setup(); _, plan = io.locked(cfg)
    truth = {r['id']:r for r in io.read(OUTPUT/'ground_truth.json')}
    rows = []
    for item in plan:
        dest = OUTPUT/item['id']; front = io.stage_done(dest,'frontend')
        io.require(front and front['passed'], 'frontend missing')
        gt = truth[item['id']]
        row = dict(id=item['id'], path=item['path'], scene=item['scene'], episodes=gt['episodes'],
            quality=front['quality'], quality_passed=front['quality']['passed'], heads={})
        predictions = {h:[] for h in cfg['evaluation']['heads']}
        inside = {h:[] for h in predictions}
        if front['quality']['passed']:
            io.require(io.stage_done(dest,'lift') and io.stage_done(dest,'inference'), 'inference incomplete')
            with np.load(dest/'inference.npz') as z:
                endpoints = np.arange(0,item['frames']-63,8)+63
                np.testing.assert_array_equal(endpoints,z['window_endpoints'])
                for h in predictions:
                    predictions[h] = old.rising_edges(endpoints,z[h])
                    fall_times = endpoints[z[h].argmax(1)==1]/25
                    inside[h] = [float(t) for t in fall_times if any(e['fall_start']<=t<=e['fall_end'] for e in gt['episodes'])]
        else:
            io.require(not (dest/'inference.npz').exists(), 'quality gate bypass')
        for h in predictions:
            row['heads'][h] = dict(predictions=predictions[h], fall_window_times_inside_gt=inside[h],
                **old.match_events(predictions[h],gt['episodes']))
        rows.append(row)
    heads = {h:dict(primary=aggregate(rows,h),quality_pass=aggregate([r for r in rows if r['quality_passed']],h),
        per_scene={s:aggregate([r for r in rows if r['scene']==s],h) for s in sorted({r['scene'] for r in rows})})
        for h in cfg['evaluation']['heads']}
    result = dict(passed=True, videos=38, total_fall_events=22, negative_videos=16,
        quality_pass=sum(r['quality_passed'] for r in rows),
        quality_rejected_fall_events=sum(len(r['episodes']) for r in rows if not r['quality_passed']),
        rows=rows, heads=heads, target_training=False, target_calibration=False,
        segment_classification=False, event_evaluation=True, recovery_scored=False,
        contract_sha256=io.sha(OUTPUT/'contract.json'),config_sha256=io.sha(CONFIG))
    io.locked(cfg); io.save(OUTPUT/'evaluation.json',result)
    status('scored_pending_independent_audit',heads={h:v['primary'] for h,v in heads.items()})


def worker(stage):
    cfg = setup(); io.locked(cfg); io.safety(cfg)
    saved = io.save
    stage_name = 'inference' if stage=='infer' else stage
    def report(path,value):
        saved(path,value)
        if Path(path).name == stage_name+'.json':
            saved(OUTPUT/'status.json',dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),
                completed=len(list(OUTPUT.glob('*/'+stage_name+'.json'))),total=38,id=Path(path).parent.name))
    io.save = report
    if stage=='frontend': from .rgb_document_frontend import main as run
    elif stage=='lift': from .rgb_document_lift import main as run
    else: from .rgb_document_infer import main as run
    run(); status(stage+'_completed')


def main():
    p = argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','frontend','lift','infer','score']);a=p.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        io.require(sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file()) < 4*1024**3, 'output budget')
        if a.stage=='prepare': initialize()
        elif a.stage=='score': score()
        else: worker(a.stage)


if __name__=='__main__': main()
