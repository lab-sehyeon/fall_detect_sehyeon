"""CAUCA100: frozen common protocol, audited parent inputs, three fresh comparators."""
from pathlib import Path
import argparse, fcntl, os, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
from fall_pipeline.external import rgb_document_io as io
import run_three_by_three_20261003 as base

OUT = ROOT / 'data/fall_processed/RGB/cauca100_comparison_20261003_r1'
CFG = ROOT / 'configs/cauca100_comparison_20261003_r1.json'
PARENT = ROOT / 'data/fall_processed/RGB/cauca100_20261003_r1'
PRIOR = ROOT / 'data/fall_processed/RGB/external_comparison_expansion_20261003_r1'
PRIOR_CFG = ROOT / 'configs/external_comparison_expansion_20261003_r1.json'
SCOPES = ['cauca100', 'cauca19']
CODE = [Path(__file__), ROOT / 'scripts/audit_cauca100_comparison_20261003.py',
        ROOT / 'scripts/run_three_by_three_20261003.py', ROOT / 'scripts/audit_three_by_three_20261003.py',
        ROOT / 'fall_pipeline/external/rgb_document_io.py',
        ROOT / 'fall_pipeline/external/le2i_current_evaluation.py']


def verify_parent():
    import run_cauca100_20261003 as own
    import run_external_comparison_expansion_20261003 as prior
    own.locked(io.read(own.CFG))
    prior.locked(io.read(prior.CFG))
    for parent in [PARENT, PRIOR]:
        audit = io.read(parent / 'independent_audit.json')
        io.require(audit['passed'] and audit['evaluation_sha256'] == io.sha(parent / 'evaluation.json'), 'parent audit')
        io.require(audit['contract_sha256'] == io.sha(parent / 'contract.json'), 'parent audit contract')
    for spec in io.read(base.SOURCES / 'model_acquisition.json'):
        io.require(io.sha(ROOT / spec['path']) == spec['sha256'], 'comparator checkpoint')
    io.require(io.read(ROOT / 'data/fall_processed/RGB/three_by_three_20261003_r1/synthetic_validation.json')['passed'], 'prior synthetic validation')
    original = io.read(PRIOR_CFG)
    parent_cfg = io.read(own.CFG)
    for key in ['detector', 'pose_config', 'quality', 'lifting', 'inference', 'model', 'sampling']:
        io.require(original[key] == parent_cfg[key], 'inherited protocol differs: ' + key)


def snapshot():
    files = CODE + [CFG, OUT / 'plan.json', OUT / 'ground_truth.json', OUT / 'training_exclusion.json',
                   PARENT / 'contract.json', PARENT / 'independent_audit.json', PARENT / 'evaluation.json',
                   PARENT / 'plan.json', PARENT / 'ground_truth.json',
                   base.SOURCES / 'model_acquisition.json', base.SOURCES / 'training_exclusion.json',
                   ROOT / 'configs/active_fall_model.json', PRIOR_CFG,
                   *[PRIOR / n for n in ['contract.json', 'independent_audit.json', 'evaluation.json', 'comparators_completed.json']],
                   ROOT / 'scripts/run_cauca100_20261003.py',
                   ROOT / 'scripts/run_external_comparison_expansion_20261003.py']
    for folder in [base.UPSTREAM, ROOT / 'third_party/pyskl_runtime_20261003']:
        files += list(folder.rglob('*.py'))
    files += list(base.SOURCES.glob('safer_*.py'))
    files += [ROOT / r['path'] for r in io.read(base.SOURCES / 'model_acquisition.json')]
    files += [ROOT / r['path'] for r in io.read(CFG)['model'].values()]
    files += [ROOT / 'fall_pipeline/common/integrity.py']
    for item in io.read(OUT / 'plan.json'):
        src = OUT / item['id']
        files += [src / name for name in ['frontend.npz', 'frontend.json', 'mapping.npz', 'source_trace.json']]
        if (src / 'inference.npz').exists():
            files += [src / 'inference.npz', src / 'inference.json']
        else:
            files += [src / 'quality_rejection.json']
    for row in io.read(OUT / 'ground_truth.json'):
        if row['scope'] == 'cauca19':
            files += [PRIOR / row['id'] / (m + ext) for m in base.MODELS[1:] for ext in ['.npz', '.json']]
    return dict(files={str(p.relative_to(ROOT)): io.sha(p) for p in sorted(set(files))},
                target_training=False, target_tuning=False, own_prediction_reused=True)


def locked(cfg):
    io.require(snapshot() == io.read(OUT / 'contract.json'), 'CAUCA100 comparison contract changed')
    return OUT, io.read(OUT / 'plan.json')


def setup():
    base.OUT = OUT
    base.CFG = CFG
    base.locked = locked
    base.setup = lambda: io.read(CFG)
    return io.read(CFG)


def prepare():
    io.require(not (OUT / 'contract.json').exists(), 'already frozen')
    verify_parent()
    truth = [r for r in io.read(PARENT / 'ground_truth.json') if r['scope'] in SCOPES]
    ids = {r['id'] for r in truth}
    plan = [r for r in io.read(PARENT / 'plan.json') if r['id'] in ids]
    io.require(len(plan) == 100 and len(truth) == 119 and len(ids) == 100, 'scope size')
    for scope, size in [('cauca100', 100), ('cauca19', 19)]:
        io.require(sum(r['scope'] == scope for r in truth) == size, 'scope count')
    for item in plan:
        src = PARENT / item['id']
        dst = OUT / item['id']
        dst.mkdir(parents=True, exist_ok=True)
        fm = io.stage_done(src, 'frontend')
        io.require(fm and fm['models_before'] == fm['models_after'], 'frontend provenance')
        names = ['frontend.npz', 'frontend.json', 'source_trace.json', 'mapping.npz']
        if fm['quality']['passed']:
            im = io.stage_done(src, 'inference')
            io.require(im and im['model_before'] == im['model_after'], 'own inference provenance')
            names += ['inference.npz', 'inference.json']
        else:
            io.require(not (src / 'inference.npz').exists(), 'quality bypass')
            names += ['quality_rejection.json']
        for name in names:
            dest = dst / name
            if dest.is_symlink():
                io.require(dest.resolve() == (src / name).resolve(), 'wrong source link')
            else:
                io.require(not dest.exists(), 'existing non-link')
                dest.symlink_to(os.path.relpath(src / name, dst))
    io.save(OUT / 'plan.json', plan)
    io.save(OUT / 'ground_truth.json', truth)
    cfg = io.read(PRIOR_CFG)
    cfg.update(experiment_id='CAUCA100_COMPARISON_20261003_R1',
               output=str(OUT.relative_to(ROOT)), datasets=SCOPES, sample_count=len(plan),
               alignment=str((OUT / 'plan.json').relative_to(ROOT)), alignment_sha256=io.sha(OUT / 'plan.json'),
               plan='100 unique CAUCA videos; primary100, nested19 consistency check',
               reuse_parent=str(PARENT.relative_to(ROOT)), historical_exact_reproduction=False)
    cfg['evaluation']['secondary_video_rule'] = 'any window argmax equals model fall index (own1/comparator9)'
    cfg['evaluation']['ground_truth'] = 'unchanged audited CAUCA100 GT; fixed OmniFall CSV100 with historical19 memberships'
    cfg['newly_specified'] = ['Scope expansion only. All inherited inference and scoring settings unchanged.',
                              'CAUCA19 is a subset of CAUCA100; never combine the denominators.',
                              'No new training, score threshold, frontend, or confidence adjustment.']
    cfg['limitations'] = [v for v in cfg['limitations'] if not any(d in v for d in ['EDF', 'OCCU', 'OOPS'])]
    cfg['limitations'] += ['CAUCA previously observed and diagnosed; not a blind target', 'Full100 includes OmniFall train/val/test and3 split-unlisted videos; no target training in this evaluation']
    io.save(CFG, cfg)
    exclusion = io.read(base.SOURCES / 'training_exclusion.json')
    exclusion.update(targets=['CAUCAFall100'], previously_observed_targets=True,
                     not_blind_test=True, parent_experiment='THREE_BY_THREE_20261003_R1')
    io.save(OUT / 'training_exclusion.json', exclusion)
    io.save(OUT / 'contract.json', snapshot())
    setup()
    base.status('prepared', unique_videos=len(plan), memberships=len(truth),
                windows_per_comparator=sum(len(range(0, r['frames'] - 63, 8)) for r in plan))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'comparisons', 'score'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'prepare':
            prepare()
        else:
            setup()
            getattr(base, args.stage)()
