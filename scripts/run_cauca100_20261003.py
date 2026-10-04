"""CAUCA100 own-model evaluation; frozen prior19 protocol and inference reuse."""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import argparse, copy, csv, fcntl, math, os, sys, subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from fall_pipeline.external import rgb_document_io as io
from fall_pipeline.external import temporal_continuous_20261002 as temporal
import prepare_cauca100_20261003 as assets

OUT=ROOT/'data/fall_processed/RGB/cauca100_20261003_r1'
CFG=ROOT/'configs/cauca100_20261003_r1.json'
PARENT=ROOT/'data/fall_processed/RGB/temporal_continuous_20261002_r1'
PARENT_CFG=ROOT/'configs/temporal_continuous_20261002_r1.json'
SCOPES={'cauca100':'cauca100_20261003_r1','cauca19':'temporal_continuous_20261002_r1'}
LABELS=assets.DEST/'metadata/labels/caucafall.csv'


def status(stage,**kw):
    if 'total' in kw:kw['total']=100
    row=dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),**kw)
    io.save(OUT/'status.json',row);print(row,flush=True)


def setup():
    io.CONFIG=CFG;io.locked=locked
    temporal.OUTPUT=OUT;temporal.CONFIG=CFG;temporal.SOURCES=SCOPES
    temporal.setup=lambda:io.read(CFG);temporal.status=status
    return io.read(CFG)


def snapshot(cfg):
    io.CONFIG=CFG;result=io.contract(cfg)
    files=[Path(__file__),ROOT/'scripts/audit_cauca100_20261003.py',Path(assets.__file__),CFG,
           ROOT/'scripts/prepare_omnifall_cauca_20261002.py',ROOT/'scripts/prepare_omnifall_le2i_20261002.py',
           ROOT/'fall_pipeline/external/j1_g0_only.py',ROOT/'fall_pipeline/external/temporal_continuous_20261002.py',
           ROOT/'fall_pipeline/external/le2i_current_evaluation.py',ROOT/'fall_pipeline/common/integrity.py',
           ROOT/'configs/active_fall_model.json',PARENT_CFG,
           *[PARENT/n for n in ['contract.json','independent_audit.json','evaluation.json','plan.json','ground_truth.json']],
           *[OUT/n for n in ['plan.json','ground_truth.json','cache_reuse.json','source_snapshot.json']],
           *[assets.DEST/n for n in ['source_manifest.json','original_manifest.json','video_manifest.json','archive_selected_members.json','ffmpeg_gpl/manifest.json']]]
    files += [ROOT/spec['path'] for spec in cfg['model'].values()]
    result['experiment_files']={str(p.relative_to(ROOT)):io.sha(p) for p in sorted(set(files))}
    for path,wanted in io.read(OUT/'source_snapshot.json').items():io.require(io.sha(ROOT/path)==wanted,'source changed: '+path)
    result['source_snapshot_sha256']=io.sha(OUT/'source_snapshot.json')
    result['target_training']=result['target_tuning']=False
    return result


def locked(cfg):
    io.require(snapshot(cfg)==io.read(OUT/'contract.json'),'CAUCA100 contract changed')
    return OUT,io.read(OUT/'plan.json')


def verify_parent():
    audit=io.read(PARENT/'independent_audit.json')
    io.require(audit['passed'] and audit['evaluation_sha256']==io.sha(PARENT/'evaluation.json'),'parent audit')
    c=io.read(PARENT/'contract.json')
    for category in ['code_sha256','frozen_files','own_code','own_inputs']:
        for path,wanted in c[category].items():
            io.require(io.sha((PARENT if category=='own_inputs' else ROOT)/path)==wanted,'parent changed: '+path)
    io.require(io.sha(PARENT_CFG)==c['config_sha256'],'parent config')
    for spec in c['active_model'].values():io.require(io.sha(ROOT/spec['path'])==spec['sha256'],'parent model')


def prepare():
    if (OUT/'contract.json').exists():setup();locked(io.read(CFG));return
    verify_parent()
    originals=io.read(assets.DEST/'original_manifest.json');videos=io.read(assets.DEST/'video_manifest.json')
    io.require(originals['passed'] and videos['passed'] and len(originals['files'])==len(videos['files'])==100,'all100 media')
    originals={r['path']:r for r in originals['files']};videos={r['path']:r for r in videos['files']}
    label_rows=list(csv.DictReader(LABELS.open()));by_path={}
    for i,row in enumerate(label_rows):by_path.setdefault(row['path'],[]).append(dict(row,csv_row=i))
    io.require(len(by_path)==100 and set(by_path)==set(originals)==set(videos),'label/media mapping')
    prior={r['id']:r for r in io.read(PARENT/'plan.json') if r['dataset']=='cauca'}
    old_gt={r['id']:r for r in io.read(PARENT/'ground_truth.json') if r['scope']=='cauca19'}
    splits={}
    for split in ['train','val','test']:
        for row in csv.DictReader((assets.DEST/f'metadata/splits/cs/caucafall/{split}.csv').open()):
            io.require(row['path'] not in splits,'split overlap');splits[row['path']]=split
    plan=[];truth=[];reuse=[];sources={}
    def remember(p):sources[str(p.relative_to(ROOT))]=io.sha(p)
    for p in (assets.DEST/'metadata').rglob('*'):
        if p.is_file():remember(p)
    for number,path in enumerate(sorted(by_path),1):
        video=videos[path];original=originals[path];sid='cauca__'+path.replace('/','__');dest=OUT/sid;dest.mkdir(exist_ok=True)
        raw=ROOT/original['local'];movie=ROOT/video['local']
        io.require(io.sha(raw)==original['sha256'] and assets.source.crc(raw)==original['official_member_crc32'],'original identity')
        io.require(io.sha(movie)==video['sha256'],'converted identity')
        remember(raw);remember(movie);remember(movie.with_suffix('.json'))
        if sid in prior:
            item=copy.deepcopy(prior[sid]);io.require(item['video_sha256']==video['sha256'],'prior video differs')
            src=PARENT/sid
            for name in ['source_trace.json','mapping.npz','frontend.json','frontend.npz','lift.json','lift.npz','inference.json','inference.npz','quality_rejection.json']:
                if (src/name).exists():
                    assets.copy_checked(src/name,dest/name);remember(src/name);remember(dest/name)
                    reuse.append(dict(source=str((src/name).relative_to(ROOT)),copy=str((dest/name).relative_to(ROOT)),sha256=io.sha(src/name)))
            item.pop('old_source',None);item.pop('old_mapping',None)
            trace=io.read(dest/'source_trace.json')
        else:
            trace=temporal.trace_video(movie);io.require(trace['duration_num'] is not None,'missing video duration')
            io.save(dest/'source_trace.json',trace)
            duration=Fraction(trace['duration_num'],trace['duration_den']);count=math.ceil(duration*25)
            indices=temporal.exact_indices(trace['pts'],trace['time_base_num'],trace['time_base_den'],trace['start_pts'],count)
            io.npz(dest/'mapping.npz',source_indices=indices,canonical_timestamps=np.arange(count)/25,
                   source_pts=np.array(trace['pts'],np.int64),time_base_num=np.array(trace['time_base_num']),
                   time_base_den=np.array(trace['time_base_den']),start_pts=np.array(trace['start_pts']))
            item=dict(id=sid,frames=count,source_frames=len(trace['pts']),source_duration_seconds=float(duration))
        io.require(len(trace['pts'])==video['source_frames'],'source/output frame count')
        item.update(dataset='cauca',path=path,video=str(movie.relative_to(ROOT)),video_sha256=video['sha256'],
                    source_video=original['local'],source_sha256=original['sha256'],subject=int(by_path[path][0]['subject']),
                    benchmark_split=splits.get(path,'outside_split'),cached_prediction=sid in prior,
                    mapping=str((dest/'mapping.npz').relative_to(ROOT)),mapping_sha256=io.sha(dest/'mapping.npz'),
                    trace_sha256=io.sha(dest/'source_trace.json'))
        remember(dest/'mapping.npz');remember(dest/'source_trace.json')
        # Prior19 originated in official Parquet float32 seconds; retain the same precision for all100.
        episodes=[dict(fall_start=float(np.float32(r['start'])),fall_end=float(np.float32(r['end'])),row_index=r['csv_row'])
                  for r in by_path[path] if r['label']=='1']
        io.require(len(episodes)<=1,'one-fall scope')
        for e in episodes:io.require(0<=e['fall_start']<e['fall_end']<=item['source_duration_seconds']+.05,'GT beyond media')
        for row in by_path[path]:io.require(0<=float(row['start'])<float(row['end'])<=item['source_duration_seconds']+.05,'label outside video')
        if sid in old_gt:
            io.require([(e['fall_start'],e['fall_end']) for e in episodes]==[(e['fall_start'],e['fall_end']) for e in old_gt[sid]['episodes']],'prior GT differs')
        plan.append(item);truth.append(dict(scope='cauca100',id=sid,path=path,episodes=episodes))
        if number%10==0:status('prepare',completed=number,total=100)
    io.require(len(plan)==100 and sum(bool(r['episodes']) for r in truth)==50,'100/50 denominator')
    io.require(sum(r['cached_prediction'] for r in plan)==19,'cache19')
    truth += [copy.deepcopy(old_gt[r['id']]) for r in plan if r['cached_prediction']]
    io.save(OUT/'plan.json',plan);io.save(OUT/'ground_truth.json',truth);io.save(OUT/'cache_reuse.json',reuse)
    io.save(OUT/'source_snapshot.json',sources)
    cfg=copy.deepcopy(io.read(PARENT_CFG))
    cfg.update(experiment_id='CAUCA100_OWN_20261003_R1',output=str(OUT.relative_to(ROOT)),
               alignment=str((OUT/'plan.json').relative_to(ROOT)),alignment_sha256=io.sha(OUT/'plan.json'),sample_count=100,
               datasets=['cauca100','cauca19'],plan='all100 original CAUCA videos,50fall/50nonfall; own DSTE/J1/G0 only',
               reuse_parent=str(PARENT.relative_to(ROOT)),scope='user-authorized full100 external evaluation, not official19 benchmark')
    cfg['execution']['max_added_gib']=4
    cfg['evaluation'].update(gt_time='fixed OmniFall CSV seconds represented as official-Parquet float32, prior19 identical',
                             ground_truth=str(LABELS.relative_to(ROOT)),comparisons='own only; prior19 consistency check')
    cfg['newly_specified']=['Scope100 only; all input/model/decision/evaluation settings unchanged.',
                           'All81 additional originals included, including three paths outside official CS split.',
                           '19 verified cached predictions,81 new inference; no target tuning/training.']
    io.save(CFG,cfg);setup();io.save(OUT/'contract.json',snapshot(cfg))
    status('prepared',videos=100,fall_events=50,reused=19,new_pipeline_videos=81,frames=sum(r['frames'] for r in plan))


def orchestrate():
    io.require(Path(sys.prefix).name=='fall_detect','fall_detect required')
    (OUT/'logs').mkdir(exist_ok=True);env=os.environ.copy()
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    journal=[]
    for stage in ['prepare','frontend','lift','infer','score','audit']:
        env['CUDA_VISIBLE_DEVICES']='0' if stage in ['frontend','lift','infer'] else ''
        env['PYTHONPATH']=str(ROOT/'third_party/ViTPose')+':'+str(ROOT) if stage=='frontend' else str(ROOT)
        command=[sys.executable,str(Path(__file__)),stage]
        if stage=='audit':command=[sys.executable,str(ROOT/'scripts/audit_cauca100_20261003.py')]
        began=datetime.now(timezone.utc).isoformat();print('START',stage,began,flush=True)
        with (OUT/'logs'/(stage+'.log')).open('a') as stream:
            stream.write('\nSTART '+began+'\n');stream.flush()
            rc=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        record=dict(stage=stage,started=began,ended=datetime.now(timezone.utc).isoformat(),returncode=rc)
        journal.append(record);io.save(OUT/'runner_journal.json',journal);print(record,flush=True)
        if rc:io.save(OUT/'runner_failure.json',record);raise SystemExit(rc)
    io.require(io.read(OUT/'independent_audit.json')['passed'],'audit incomplete')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['all','prepare','frontend','lift','infer','score']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/('runner.lock' if args.stage=='all' else 'stage.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='all':orchestrate()
        elif args.stage=='prepare':prepare()
        else:setup();getattr(temporal,args.stage)()
