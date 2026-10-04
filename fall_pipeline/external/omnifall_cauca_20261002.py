"""Frozen G0 on official CAUCA-CS GT segments; separate from full-video events."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, csv, fcntl, hashlib, os
import numpy as np
from . import rgb_document_io as io

ROOT = io.ROOT
ASSET = ROOT/'data/source_archives/OmniFall/cauca_cs_20261002_r1'
RUNTIME = ROOT/'data/source_archives/OmniFall/gmdcsa24_cs_20261002_r1'
OUTPUT = ROOT/'data/fall_processed/RGB/omnifall_cauca_cs_20261002_r1'
CONFIG = ROOT/'configs/omnifall_cauca_cs_20261002_r1.json'


def status(stage, **kwargs):
    if stage == 'frontend' and 'segment' in kwargs: kwargs['total'] = 47
    io.save(OUTPUT/'status.json',dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),**kwargs))
    print(stage,kwargs,flush=True)


def official_rows():
    from . import omnifall_sampling_20261002  # isolated pinned PyArrow runtime
    import pyarrow.parquet as pq
    rows=pq.read_table(ASSET/'metadata/parquet/caucafall-cs/test-00000-of-00001.parquet').to_pylist()
    splits={s:{r['path'] for r in csv.DictReader((ASSET/f'metadata/splits/cs/caucafall/{s}.csv').read_text().splitlines())}
            for s in ['train','val','test']}
    io.require(all(not splits[a]&splits[b] for a,b in [('train','val'),('train','test'),('val','test')]),'split overlap')
    csvrows=[r for r in csv.DictReader((ASSET/'metadata/labels/caucafall.csv').read_text().splitlines()) if r['path'] in splits['test']]
    key=lambda r:(r['path'],int(r['label']),round(float(r['start']),5),round(float(r['end']),5),int(r['subject']),int(r['cam']))
    io.require(sorted(map(key,rows))==sorted(map(key,csvrows)),'official CSV/parquet mismatch')
    io.require(len(rows)==47 and len(splits['test'])==19 and sum(r['label']==1 for r in rows)==9,'scope changed')
    io.require({r['subject'] for r in rows}=={8,9} and {r['path'] for r in rows}==splits['test'],'test subject/path mismatch')
    io.require('`1|fall`' in (ASSET/'metadata/LABELS.md').read_text(),'fall label mapping')
    return rows,sorted(splits['test'])


def contract(cfg):
    result=io.contract(cfg)
    global_cfg=io.read(ROOT/cfg['inference']['global_config'])
    global_root=ROOT/global_cfg['output_dir']
    selection=io.read(global_root/'selection.json')
    io.require(selection['best']==io.read(global_root/'final_report.json')['selection']['best'],'source selection mismatch')
    result['source_selection_sha256']=io.sha(global_root/'selection.json')
    result['own_code']={p:io.sha(ROOT/p) for p in [
        'fall_pipeline/external/omnifall_cauca_20261002.py',
        'fall_pipeline/external/omnifall_gmdcsa_20261002.py',
        'fall_pipeline/external/omnifall_sampling_20261002.py',
        'scripts/prepare_omnifall_cauca_20261002.py',
        'scripts/prepare_omnifall_le2i_20261002.py',
        'scripts/run_omnifall_cauca_20261002.py',
        'scripts/audit_omnifall_cauca_20261002.py']}
    result['ground_truth_sha256']=io.sha(OUTPUT/'ground_truth.json')
    result['metadata']={str(p.relative_to(ASSET)):io.sha(p) for p in sorted((ASSET/'metadata').rglob('*')) if p.is_file()}
    result['provenance']={p:io.sha(ASSET/p) for p in [
        'source_manifest.json','original_manifest.json','archive_selected_members.json','video_manifest.json','ffmpeg_gpl/manifest.json']}
    result['source_traces']={p.name:io.sha(p) for p in sorted((OUTPUT/'source_traces').glob('*.json'))}
    result['runtime_manifest_sha256']=io.sha(RUNTIME/'runtime_manifest.json')
    binary_manifest=io.read(ASSET/'ffmpeg_gpl/manifest.json')
    for name,h in binary_manifest['files'].items():
        io.require(io.sha(ROOT/binary_manifest['reused_root']/name)==h,'FFmpeg binary changed')
    result['official_decoder_sha256']=io.sha(RUNTIME/'package_source/src/omnifall/_decode.py')
    for p,h in io.read(RUNTIME/'runtime_manifest.json')['runtime_files'].items():
        io.require(io.sha(RUNTIME/'runtime'/p)==h,'isolated runtime changed')
    return result


def setup():
    io.CONFIG=CONFIG
    def locked(cfg):
        io.require(io.read(OUTPUT/'contract.json')==contract(cfg),'CAUCA frozen contract changed')
        return OUTPUT,io.read(OUTPUT/'plan.json')
    io.locked=locked
    return io.read(CONFIG)


def prepare():
    from .omnifall_sampling_20261002 import source_trace,sampling_trace,decode
    if CONFIG.exists(): cfg=setup(); io.locked(cfg); return
    rows,paths=official_rows(); manifest=io.read(ASSET/'video_manifest.json')
    io.require(manifest['passed'],'source incomplete');videos={r['path']:r for r in manifest['files']}
    io.require(set(paths)==set(videos),'video manifest scope')
    traces={}
    for n,path in enumerate(paths,1):
        v=videos[path];video=ROOT/v['local'];io.require(io.sha(video)==v['sha256'],'video changed')
        io.require(io.sha(ROOT/v['source']['local'])==v['source']['sha256'],'original changed')
        dest=OUTPUT/'source_traces'/(path.replace('/','_')+'.json')
        if dest.exists(): trace=io.read(dest)
        else: trace=source_trace(video);io.save(dest,trace)
        io.require(len(trace['frames'])==v['source_frames'],'frame count mismatch')
        traces[path]=trace;status('prepare_source_trace',video=n,total=19,path=path)
    plan=[];truth=[]
    for i,row in enumerate(rows):
        v=videos[row['path']];start,end=float(row['start']),float(row['end']);trace=traces[row['path']]
        io.require(0<=start<end,'bad segment')
        io.require(trace['duration'] is not None and start<trace['duration'],'segment starts beyond source')
        # The official decoder defines an end timestamp past EOF by
        # repeat-last, not a GT edit; any excursion is recorded.
        sampled=sampling_trace(trace,start,end);pixels=decode(ROOT/v['local'],start,end,sampled)
        item=dict(id=f'cauca_cs_{i:04d}',row_index=i,path=row['path'],video=v['local'],video_sha256=v['sha256'],
                  start=start,end=end,frames=64,sampling=sampled,
                  annotation_end_past_stream_seconds=max(0.,end-trace['duration']),
                  decoded_rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
        plan.append(item);truth.append(dict(id=item['id'],row_index=i,label=int(row['label']),fall=int(row['label']==1),subject=int(row['subject'])))
        status('prepare_segment',segment=i+1,total=47,unique_frames=sampled['unique_frames'])
    io.save(OUTPUT/'plan.json',plan);io.save(OUTPUT/'ground_truth.json',truth)
    cfg=copy.deepcopy(io.read(ROOT/'configs/omnifall_gmdcsa_cs_20261002_r1.json'))
    cfg.update(experiment_id='OMNIFALL_CAUCA_CS_20261002_R1',output=str(OUTPUT.relative_to(ROOT)),
        alignment=str((OUTPUT/'plan.json').relative_to(ROOT)),alignment_sha256=io.sha(OUTPUT/'plan.json'),
        video_root=str((ASSET/'video').relative_to(ROOT)),sample_count=47,
        plan='all official caucafall-cs test segments in official parquet order; no prediction selection',
        sampling='official companion0.2.0 uniform64 nearestPTS with inclusive endpoints; NOT physical25fps',
        newly_specified=['CAUCA official CS47 GT segments rather than historical100 full-video diagnosis',
                         'author CAUCA conversion recipe GOP20 CRF24, previously verified FFmpeg build, two CPU threads',
                         'G0 only, identical predeclared GMDCSA segment adapter; no target tuning',
                         'explicit physical25fps mismatch excludes G1/G2 motion features from scoring'])
    cfg['execution'].update(max_added_gib=4,reserve_gib=64)
    cfg['benchmark'].update(config='caucafall-cs',videos=19,segments=47,positive=9,negative=38,subject=[8,9],
        hf_revision=io.read(ASSET/'source_manifest.json')['hf_revision'],
        pretrained_overlap_claim='no task-specific CAUCA training; previously examined engineering target, not untouched; pretraining exclusion unproven',
        comparison='same GT test segments and binary metrics; different training data/input representation/sampling, not exact paper reproduction',
        excluded_heads=['G1','G2'],excluded_heads_reason='global131 assumes physical25fps but uniform segment sampler warps time')
    io.save(CONFIG,cfg);io.CONFIG=CONFIG;io.save(OUTPUT/'contract.json',contract(cfg))
    status('prepared',videos=19,segments=47,positive=9,negative=38)


def frontend():
    from . import omnifall_gmdcsa_20261002 as common_front
    # Reuse the already-validated frontend without changing its on-disk code.
    common_front.OUTPUT=OUTPUT;common_front.CONFIG=CONFIG;common_front.setup=setup;common_front.status=status
    common_front.frontend()


def binary_metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    io.require(y.shape==p.shape and len(y)>0 and set(y)<= {0,1} and set(p)<= {0,1},'binary vectors')
    tp=int(sum((y==1)&(p==1)));fp=int(sum((y==0)&(p==1)));fn=int(sum((y==1)&(p==0)));tn=int(sum((y==0)&(p==0)))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn) if tp+fn else 0.,
        specificity=tn/(tn+fp) if tn+fp else 0.,precision=tp/(tp+fp) if tp+fp else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,accuracy=float(np.mean(y==p)))


def score():
    from sklearn.metrics import average_precision_score
    cfg=setup();_,plan=io.locked(cfg);truth=io.read(OUTPUT/'ground_truth.json');rows=[]
    for item,gt in zip(plan,truth):
        io.require(item['id']==gt['id'],'GT order');dest=OUTPUT/item['id'];front=io.stage_done(dest,'frontend')
        io.require(front and front['passed'],'frontend missing');pred=0;prob=0.;klass=None
        if front['quality']['passed']:
            io.require(io.stage_done(dest,'lift') and io.stage_done(dest,'inference'),'inference missing')
            with np.load(dest/'inference.npz',allow_pickle=False) as z:
                logits=z['G0'];io.require(logits.shape==(1,4) and np.isfinite(logits).all(),'one prediction per segment')
                klass=int(logits[0].argmax());pred=int(klass==1)
                ex=np.exp(logits[0].astype(float)-logits[0].max());prob=float(ex[1]/ex.sum())
        else:
            io.require(not (dest/'inference.json').exists(),'failed quality was classified')
            io.require(io.read(dest/'quality_rejection.json')['classifier_run'] is False,'missing rejection receipt')
        rows.append(dict(**gt,prediction=pred,predicted_class=klass,fall_score=prob,quality_passed=front['quality']['passed'],quality=front['quality']))
    io.require(len(rows)==47,'denominator changed')
    y=[r['fall'] for r in rows];p=[r['prediction'] for r in rows];s=[r['fall_score'] for r in rows]
    m=dict(segments=len(rows),**binary_metrics(y,p),ap=float(average_precision_score(y,s)),
           classified=sum(r['quality_passed'] for r in rows),
           rejected_positive=sum(not r['quality_passed'] and r['fall']==1 for r in rows),
           rejected_negative=sum(not r['quality_passed'] and r['fall']==0 for r in rows))
    io.save(OUTPUT/'evaluation.json',dict(passed=True,scope='caucafall-cs test47 GT-provided segments; Fall binary only',
        metrics=m,rows=rows,config_sha256=io.sha(CONFIG),contract_sha256=io.sha(OUTPUT/'contract.json'),
        ground_truth_sha256=io.sha(OUTPUT/'ground_truth.json'),target_training=False,event_evaluation=False,recovery_evaluation=False))
    status('scored_audit_pending',metrics=m)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','frontend','lift','infer','score']);a=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        io.require(sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file()) < 4*1024**3,'output budget')
        if a.stage=='prepare':prepare()
        elif a.stage=='frontend':frontend()
        elif a.stage=='score':score()
        else:
            setup()
            if a.stage=='lift':from .rgb_document_lift import main as run
            else:from .rgb_document_infer import main as run
            run();status(a.stage+'_completed')


if __name__=='__main__':main()
