"""Le2i130: recovered original annotations, unchanged common comparison protocol."""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import argparse, copy, fcntl, hashlib, math, os, re, shutil, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
from fall_pipeline.external import rgb_document_io as io
from fall_pipeline.external import temporal_continuous_20261002 as temporal
import run_three_by_three_20261003 as base
import run_external_comparison_expansion_20261003 as parent
import prepare_le2i_current_evaluation as video_prep

OUT = ROOT/'data/fall_processed/RGB/le2i130_20261003_r1'
CFG = ROOT/'configs/le2i130_20261003_r1.json'
EVIDENCE = ROOT/'docs/internal/2026-10-03_le2i_missing_annotation_search_evidence'
SOURCE = ROOT/'data/source_archives/Le2i-FDD/kaggle_v2_20261002_r1'
MEDIA = ROOT/'data/source_archives/Le2i-FDD/le2i130_ffv1_20261003_r1'
NEW = {'le2i__Coffee_room_01_026', 'le2i__Coffee_room_02_050', 'le2i__Coffee_room_02_052'}
CACHE_NAMES = ['source_trace.json','mapping.npz','frontend.json','frontend.npz',
               'inference.json','inference.npz','stgcnpp.json','stgcnpp.npz',
               'msg3d.json','msg3d.npz','cnn1d.json','cnn1d.npz']


def copy_checked(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists(): shutil.copy2(src, dst)
    io.require(io.sha(src) == io.sha(dst), 'cache copy differs: '+str(src))


def snapshot(cfg):
    io.CONFIG = CFG
    result = io.contract(cfg)
    files = [Path(__file__), ROOT/'scripts/audit_le2i130_20261003.py', CFG,
             OUT/'plan.json', OUT/'ground_truth.json', OUT/'cache_reuse.json',
             EVIDENCE/'recovered_annotation_manifest_130.json', EVIDENCE/'official_archive_check.json',
             parent.OUT/'contract.json', parent.OUT/'evaluation.json', parent.OUT/'independent_audit.json',
             parent.CFG, ROOT/'configs/active_fall_model.json',
             ROOT/'scripts/run_three_by_three_20261003.py',
             ROOT/'scripts/run_external_comparison_expansion_20261003.py',
             ROOT/'scripts/prepare_le2i_current_evaluation.py',
             ROOT/'fall_pipeline/external/j1_g0_only.py',
             ROOT/'fall_pipeline/external/temporal_continuous_20261002.py',
             ROOT/'fall_pipeline/external/le2i_current_evaluation.py',
             ROOT/'fall_pipeline/common/integrity.py',
             base.SOURCES/'model_acquisition.json', base.SOURCES/'training_exclusion.json']
    files += list(base.UPSTREAM.rglob('*.py'))
    files += list((ROOT/'third_party/pyskl_runtime_20261003').rglob('*.py'))
    files += list(base.SOURCES.glob('safer_*.py'))
    files += [ROOT/r['path'] for r in io.read(base.SOURCES/'model_acquisition.json')]
    files += [ROOT/r['path'] for r in cfg['model'].values()]
    for row in io.read(OUT/'plan.json'):
        files += [ROOT/row['video'], ROOT/row['mapping'], ROOT/row['source_annotation'],
                  OUT/row['id']/'source_trace.json', OUT/row['id']/'source_media.json']
    for row in io.read(OUT/'cache_reuse.json'):
        files += [ROOT/row['source'], ROOT/row['copy']]
    result['experiment_files'] = {str(p.relative_to(ROOT)):io.sha(p) for p in sorted(set(files))}
    result['target_training'] = result['target_tuning'] = False
    return result


def locked(cfg):
    io.require(snapshot(cfg) == io.read(OUT/'contract.json'), 'Le2i130 contract changed')
    return OUT, io.read(OUT/'plan.json')


def setup():
    base.OUT = OUT; base.CFG = CFG; base.locked = locked; base.setup = lambda: io.read(CFG)
    io.CONFIG = CFG; io.locked = locked
    temporal.OUTPUT = OUT; temporal.CONFIG = CFG
    temporal.setup = lambda: io.read(CFG); temporal.status = base.status
    return io.read(CFG)


def converted(record):
    """Lossless decode in FFmpeg, with full BGR equality before inference."""
    sid = f'{record["scene"]}_{record["video_number"]:03d}'
    raw = SOURCE/'source'/record['scene']/'Videos'/f'video ({record["video_number"]}).avi'
    target = MEDIA/(sid+'.mkv'); receipt = MEDIA/(sid+'.json')
    if receipt.exists():
        saved = io.read(receipt)
        io.require(io.sha(raw)==saved['raw_sha256'] and io.sha(target)==saved['video_sha256'], 'media changed')
        return target, saved
    MEDIA.mkdir(parents=True, exist_ok=True)
    ffmpeg = video_prep.TOOLS/'ffmpeg'
    log = MEDIA/(sid+'.ffmpeg.log')
    if not target.exists():
        with log.open('ab') as stream:
            subprocess.run([str(ffmpeg),'-nostdin','-n','-v','error','-threads','2','-i',str(raw),
                            '-map','0:v:0','-an','-sn','-dn','-c:v','ffv1','-level','3','-pix_fmt','bgr0',
                            '-threads','2',str(target)],stdout=stream,stderr=stream,check=True,timeout=300)
    original, size = video_prep.raw_pixel_digest(raw,log)
    derived, size2 = video_prep.raw_pixel_digest(target,log)
    meta = video_prep.probe(raw)['streams'][0]
    count = int(meta['nb_frames']); fps = Fraction(meta['avg_frame_rate'])
    io.require(original==derived and size==size2==count*meta['width']*meta['height']*3, 'pixel mismatch')
    saved = dict(raw=str(raw.relative_to(ROOT)),raw_sha256=io.sha(raw),video_sha256=io.sha(target),
                 frames=count,fps_num=fps.numerator,fps_den=fps.denominator,
                 bgr_sha256=original,lossless_pixel_equality=True)
    io.save(receipt,saved)
    return target,saved


def prepare():
    if (OUT/'contract.json').exists(): setup(); locked(io.read(CFG)); return
    parent.verify_parent(); parent.locked(io.read(parent.CFG))
    audit = io.read(parent.OUT/'independent_audit.json')
    io.require(audit['passed'] and audit['evaluation_sha256']==io.sha(parent.OUT/'evaluation.json'), 'parent audit')
    recovered = io.read(EVIDENCE/'recovered_annotation_manifest_130.json')
    official = io.read(EVIDENCE/'official_archive_check.json')
    io.require(official['completed'] and all(x['same_as_local'] for x in official['annotations']), 'official annotations')
    io.require(all(x['size_crc_match'] for x in official['videos']), 'official video identities')
    prior = {r['id']:r for r in io.read(parent.OUT/'plan.json')}
    old_gt = {r['id']:r for r in io.read(parent.OUT/'ground_truth.json') if r['scope']=='le2i127'}
    plan,truth,reuse = [],[],[]
    for record in recovered['records']:
        sid=f'le2i__{record["scene"]}_{record["video_number"]:03d}';dest=OUT/sid;dest.mkdir(exist_ok=True)
        ann=ROOT/record['source_annotation'];io.require(io.sha(ann)==record['source_sha256'],'annotation changed')
        values=[int(s.strip()) for s in ann.read_text().splitlines() if re.fullmatch('[0-9]+',s.strip())]
        io.require(values==[record['start_frame'],record['end_frame']], 'annotation parse')
        raw=SOURCE/'source'/record['scene']/'Videos'/f'video ({record["video_number"]}).avi'
        media_path=dest/'source_media.json'
        if media_path.exists():media=io.read(media_path)
        else:
            media=dict(raw_video=str(raw.relative_to(ROOT)),metadata=video_prep.probe(raw)['streams'][0])
            io.save(media_path,media)
        fps=float(Fraction(media['metadata']['avg_frame_rate']))
        nominal=25 if record['scene'].startswith('Coffee') else 24
        io.require(abs(fps-nominal)<.001,'unexpected source fps')
        episodes=[] if values==[0,0] else [dict(fall_start=values[0]/fps,fall_end=values[1]/fps)]
        if sid in old_gt: io.require(episodes==old_gt[sid]['episodes'],'legacy GT changed')
        if sid in prior:
            item=copy.deepcopy(prior[sid]);src=parent.OUT/sid
            for name in CACHE_NAMES:
                if (src/name).exists():
                    copy_checked(src/name,dest/name)
                    reuse.append(dict(source=str((src/name).relative_to(ROOT)),copy=str((dest/name).relative_to(ROOT)),sha256=io.sha(src/name)))
            # Geometry was audited in the temporal parent and is checked again independently.
            for name in ['lift.npz','lift.json','quality_rejection.json']:
                src2=parent.PARENT/sid/name
                if src2.exists():
                    copy_checked(src2,dest/name)
                    reuse.append(dict(source=str(src2.relative_to(ROOT)),copy=str((dest/name).relative_to(ROOT)),sha256=io.sha(src2)))
            item.pop('old_source',None);item.pop('old_mapping',None)
            item['cached_prediction']=True
        else:
            video,receipt=converted(record);trace=temporal.trace_video(video)
            count=receipt['frames'];io.require(len(trace['pts'])==count,'frame count')
            duration=Fraction(count*receipt['fps_den'],receipt['fps_num'])
            trace.update(duration_num=duration.numerator,duration_den=duration.denominator)
            io.save(dest/'source_trace.json',trace)
            n=math.ceil(duration*25);indices=temporal.exact_indices(trace['pts'],trace['time_base_num'],trace['time_base_den'],trace['start_pts'],n)
            io.npz(dest/'mapping.npz',source_indices=indices,canonical_timestamps=np.arange(n)/25,
                   source_pts=np.asarray(trace['pts'],np.int64),time_base_num=np.array(trace['time_base_num']),
                   time_base_den=np.array(trace['time_base_den']),start_pts=np.array(trace['start_pts']))
            item=dict(id=sid,video=str(video.relative_to(ROOT)),video_sha256=io.sha(video),frames=n,
                      source_frames=count,source_duration_seconds=float(duration),cached_prediction=False)
        item.update(dataset='le2i130',path=record['id'],source_annotation=record['source_annotation'],
                    annotation_sha256=record['source_sha256'],source_fps=fps,
                    source_fps_rational=media['metadata']['avg_frame_rate'],raw_header=values,
                    added_video=sid in NEW,mapping=str((dest/'mapping.npz').relative_to(ROOT)),
                    mapping_sha256=io.sha(dest/'mapping.npz'),trace_sha256=io.sha(dest/'source_trace.json'))
        plan.append(item);truth.append(dict(id=sid,scope='le2i130',path=record['id'],episodes=episodes))
    io.require(len(plan)==130 and len({r['id'] for r in plan})==130,'130 distinct videos')
    io.require(sum(bool(r['episodes']) for r in truth)==99 and sum(r['added_video'] for r in plan)==3,'99 falls')
    io.require(sum(r['cached_prediction'] for r in plan)==128 and set(old_gt)=={r['id'] for r in plan if not r['added_video']},'reuse scope')
    io.save(OUT/'plan.json',plan);io.save(OUT/'ground_truth.json',truth);io.save(OUT/'cache_reuse.json',reuse)
    cfg=copy.deepcopy(io.read(parent.CFG))
    cfg.update(experiment_id='LE2I130_20261003_R1',output=str(OUT.relative_to(ROOT)),
               alignment=str((OUT/'plan.json').relative_to(ROOT)),alignment_sha256=io.sha(OUT/'plan.json'),
               sample_count=130,datasets=['le2i130'],plan='130 original videos,99falls/31nonfall; recovered original scalar lines',
               reuse_parent=str(parent.OUT.relative_to(ROOT)),
               newly_specified=['Only inclusion of three original-annotation videos; no model or metric change.',
                                '128 cached predictions verified; two newly inferred videos; all130 rescored.'])
    cfg['evaluation']['gt_time']='original frame/source fps, no minus-one adjustment, unchanged from Le2i127'
    cfg['evaluation']['ground_truth']='130 original annotation files; scalar pair may be inside bbox rows'
    io.save(CFG,cfg);io.CONFIG=CFG;io.save(OUT/'contract.json',snapshot(cfg))
    setup();base.status('prepared',videos=130,fall_events=99,reused_videos=128,new_inference_videos=2,
                        frames=sum(r['frames'] for r in plan))


def orchestrate():
    io.require(Path(sys.prefix).name=='fall_detect','fall_detect required')
    logs=OUT/'logs';logs.mkdir(exist_ok=True);env=os.environ.copy()
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
               MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    journal=[]
    for stage in ['prepare','frontend','lift','infer','comparisons','score','audit']:
        env['CUDA_VISIBLE_DEVICES']='0' if stage in ['frontend','lift','infer','comparisons'] else ''
        env['PYTHONPATH']=str(ROOT/'third_party/ViTPose')+':'+str(ROOT) if stage=='frontend' else str(ROOT)
        command=[sys.executable,str(Path(__file__)),stage]
        if stage=='audit':command=[sys.executable,str(ROOT/'scripts/audit_le2i130_20261003.py')]
        began=datetime.now(timezone.utc).isoformat();print('START',stage,began,flush=True)
        with (logs/(stage+'.log')).open('a') as log:
            log.write('\nSTART '+began+'\n');log.flush()
            rc=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        record=dict(stage=stage,started=began,ended=datetime.now(timezone.utc).isoformat(),returncode=rc)
        journal.append(record);io.save(OUT/'runner_journal.json',journal);print(record,flush=True)
        if rc:io.save(OUT/'runner_failure.json',record);raise SystemExit(rc)
    io.require(io.read(OUT/'independent_audit.json')['passed'],'audit incomplete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','prepare','frontend','lift','infer','comparisons','score']);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/('runner.lock' if args.stage=='all' else 'stage.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='all':orchestrate()
        elif args.stage=='prepare':prepare()
        else:
            setup()
            if args.stage in ['frontend','lift','infer']:getattr(temporal,args.stage)()
            else:getattr(base,args.stage)()
