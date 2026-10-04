"""User-requested post-hoc EDF quality gate ablation; frozen model and inputs."""
from pathlib import Path
import argparse
import copy
import csv
import fcntl
import os
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_three_by_three_20261003 as base
from audit_three_by_three_20261003 import counts
from fall_pipeline.external import rgb_document_io as io

SOURCE=ROOT/'data/fall_processed/RGB/three_by_three_20261003_r1/edf'
OUT=ROOT/'data/fall_processed/RGB/edf_confidence_ablation_20261003_r1'
SOURCE_CONFIG=ROOT/'configs/three_by_three_edf_20261003_r1.json'
CFG=ROOT/'configs/edf_confidence_ablation_20261003_r1.json'
ORIGINAL_SNAPSHOT=base.snapshot


def snapshot(cfg):
    result=ORIGINAL_SNAPSHOT(cfg)
    result['diagnostic_files']={str(p.relative_to(ROOT)):io.sha(p) for p in
        [Path(__file__),SOURCE_CONFIG,*sorted(SOURCE.rglob('*'))] if p.is_file()}
    # Lock copied/recomputed frontend metadata and all cached outputs before inference.
    result['prepared_inputs']={p:io.sha(ROOT/p) for p in io.read(OUT/'prepared_inputs.json')}
    result['prepared_manifest_sha256']=io.sha(OUT/'prepared_inputs.json')
    return result


def configure():
    base.OUT=OUT;base.CFG=CFG;base.snapshot=snapshot
    return base.setup()


def prepare():
    assert not (OUT/'contract.json').exists(),'revision already frozen'
    base.OUT=SOURCE;base.CFG=SOURCE_CONFIG;base.setup()
    original=io.read(SOURCE_CONFIG)
    base.locked(original)
    assert io.read(SOURCE/'independent_audit.json')['passed']
    cfg=copy.deepcopy(original)
    cfg.update(experiment_id='EDF_CONFIDENCE_ABLATION_20261003_R1',output=str(OUT.relative_to(ROOT)),
        alignment=str((OUT/'plan.json').relative_to(ROOT)),
        scope='user-requested post-hoc pelvis quality gate0.30->0.05; original external benchmark retained')
    cfg['quality']['pelvis_median_min']=.05
    cfg['evaluation']['target_calibration']=True
    cfg['diagnostic']=dict(post_hoc=True,source_config=str(SOURCE_CONFIG.relative_to(ROOT)),
        only_functional_change='whole-video pelvis median confidence gate0.30->0.05',
        selected_before_new_inference=True,selection_reason='previously observed rejected median0.0735025853;0.10 would still reject',
        threshold_sweep=False,model_or_detection_threshold_changed=False,
        original_results_retained=True,view1_prediction_reused=True)
    for name in ['plan.json','ground_truth.json']:
        shutil.copyfile(SOURCE/name,OUT/name)
    cfg['alignment_sha256']=io.sha(OUT/'plan.json')
    prepared=[];quality_rows=[]
    for row in io.read(OUT/'plan.json'):
        source=SOURCE/row['id'];dest=OUT/row['id'];dest.mkdir(exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and path.name!='quality_rejection.json':
                target=dest/path.name
                assert not target.exists(),'refuse overwrite '+str(target)
                shutil.copyfile(path,target)
        original_front=io.stage_done(source,'frontend')
        with np.load(dest/'frontend.npz') as z:
            old=io.quality(z['boxes'],z['xy'],z['scores'],original['quality'])
            new=io.quality(z['boxes'],z['xy'],z['scores'],cfg['quality'])
            for q in [old,new]:
                q['insufficient_frames']=len(z['xy'])<64
                q['passed']=bool(q['passed'] and not q['insufficient_frames'])
        assert old==original_front['quality']
        assert new['passed'],'lowered gate still rejects'
        changed=copy.deepcopy(original_front);changed['quality']=new
        changed['ablation']=dict(source_metadata_sha256=io.sha(source/'frontend.json'),
            source_payload_sha256=io.sha(source/'frontend.npz'),quality_only_recomputed=True)
        io.save(dest/'frontend.json',changed)
        if old['passed']:assert io.sha(dest/'inference.npz')==io.sha(source/'inference.npz')
        else:assert not (dest/'inference.npz').exists() and not (dest/'lift.npz').exists()
        prepared.extend(str(p.relative_to(ROOT)) for p in dest.iterdir() if p.is_file())
        quality_rows.append(dict(id=row['id'],before=old,after=new))
    io.save(OUT/'prepared_inputs.json',sorted(prepared))
    io.save(CFG,cfg);configure()
    io.save(OUT/'contract.json',snapshot(cfg))
    io.save(OUT/'preflight_validation.json',dict(passed=True,quality=quality_rows,
        single_functional_change=True,new_forward_not_started=True))
    base.status('prepared',quality=quality_rows)


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','audit is CPU only'
    cfg=configure();_,plan=base.locked(cfg)
    ev=io.read(OUT/'evaluation.json');original=io.read(SOURCE/'evaluation.json')
    truth={r['id']:r for r in io.read(OUT/'ground_truth.json')}
    assert io.sha(OUT/'ground_truth.json')==io.sha(SOURCE/'ground_truth.json')
    rows={(r['model'],r['id']):r for r in ev['rows']}
    oldrows={(r['model'],r['id']):r for r in original['rows']}
    event_rows=[];per_video=[]
    for item in plan:
        dest=OUT/item['id'];source=SOURCE/item['id'];gt=truth[item['id']]['episodes']
        with np.load(dest/'frontend.npz') as z:
            q=io.quality(z['boxes'],z['xy'],z['scores'],cfg['quality'])
        assert q['passed'] and io.sha(dest/'frontend.npz')==io.sha(source/'frontend.npz')
        with np.load(ROOT/item['mapping']) as z:
            assert (np.diff(z['source_indices'])>=0).all()
            np.testing.assert_array_equal(z['canonical_timestamps'],np.arange(item['frames'])/25)
        meta=io.stage_done(dest,'inference');assert meta['model_before']==meta['model_after']
        lm=io.stage_done(dest,'lift');assert lm['model_before']==lm['model_after']
        with np.load(dest/'inference.npz') as z:
            logits=z['G0'].astype(float);ends=z['window_endpoints'];adapted=z['adapted']
            np.testing.assert_array_equal(z['window_starts'],np.arange(0,item['frames']-63,8))
            np.testing.assert_array_equal(ends,z['window_starts']+63)
        # Independent CPU reconstruction of the final linear head, all windows.
        import torch
        state=torch.load(ROOT/cfg['model']['g0']['path'],map_location='cpu',weights_only=True)
        rebuilt=adapted.astype(float)@state['weight'].numpy().astype(float).T+state['bias'].numpy().astype(float)
        np.testing.assert_allclose(rebuilt,logits,atol=1e-4,rtol=1e-5)
        flags=logits.argmax(1)==1;times=[];previous=False
        for end,flag in zip(ends,flags):
            if flag and not previous:times.append(float(end/25))
            previous=flag
        row=rows[('own',item['id'])]
        assert times==[p['time'] for p in row['predictions']]
        assert counts(times,gt)==tuple(row['event'][k] for k in ['tp','fp','fn'])
        matched={m['gt_index']:m for m in row['event']['matches']}
        for i,e in enumerate(gt):
            event_rows.append(dict(view=item['path'].split('/')[1],event=i+1,
                gt_start=e['fall_start'],gt_end=e['fall_end'],detected=int(i in matched),
                alarm_time=matched[i]['prediction']['time'] if i in matched else '',
                onset_offset=matched[i]['onset_delay'] if i in matched else ''))
        old=oldrows[('own',item['id'])]
        if old['processed']:
            assert io.sha(dest/'inference.npz')==io.sha(source/'inference.npz')
            assert row==old
        for m in ['stgcnpp','msg3d','cnn1d']:
            assert rows[(m,item['id'])]==oldrows[(m,item['id'])]
        per_video.append(dict(id=item['id'],previously_processed=old['processed'],
            before=old['event'],after=row['event'],windows=len(logits),
            class_counts=np.bincount(logits.argmax(1),minlength=4).tolist()))
    for model in ['own','stgcnpp','msg3d','cnn1d']:
        selected=[r for r in ev['rows'] if r['model']==model]
        summary=ev['summaries'][model]['edf']['event']
        tp,fp,fn=[sum(r['event'][k] for r in selected) for k in ['tp','fp','fn']]
        assert [tp,fp,fn]==[summary[k] for k in ['tp','fp','fn']]
        assert abs(summary['f1']-2*tp/(2*tp+fp+fn))<1e-12
        assert tp+fn==16
    with (OUT/'events.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(event_rows[0]));w.writeheader();w.writerows(event_rows)
    base.locked(cfg)
    report=dict(passed=True,post_hoc=True,only_functional_change='pelvis gate0.30->0.05',
        original_files_unchanged=True,raw_event_counts_recomputed=True,cpu_linear_all_windows=True,
        original_view1_and_comparators_exact=True,evaluation_sha256=io.sha(OUT/'evaluation.json'),
        original_evaluation_sha256=io.sha(SOURCE/'evaluation.json'),events_csv_sha256=io.sha(OUT/'events.csv'),
        videos=per_video)
    io.save(OUT/'independent_audit.json',report);base.status('completed',videos=per_video,summaries=ev['summaries'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','lift','infer','score','audit']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='prepare':prepare()
        elif args.stage=='audit':audit()
        else:
            configure()
            if args.stage in ['lift','infer']:getattr(base.temporal,args.stage)()
            else:base.score()
