#!/usr/bin/env python3
"""Serial, fail-closed V3 follow-up after audited V1/V2 joint completion.

No download, deletion, system change or additional GPU. This is not an all-project
completion marker: RGB/external/LaDy and other historical branches are separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
QUEUE=ROOT/'checkpoint/fall/RECOVERY_V3_QUEUE_20260922_R3'
GEOMETRY=ROOT/'data/fall_processed/SAFER-Activities/v3_document_reconstruction_20260922_r3'
CONTROL=ROOT/'checkpoint/fall/F0_CONTROLS_SAFER_V3_DOCUMENT_RECONSTRUCTION_20260922_R3'
JOINT=ROOT/'checkpoint/fall/JOINT_V3_DOCUMENT_RECONSTRUCTION_20260922_R3'
DEPENDENCY=ROOT/'checkpoint/fall/JOINT_DOCUMENT_RECONSTRUCTION_20260922_R1'
STAGES=[
 ('v3_pilot','data_gen.safer_v3_reconstruction_r3',['--stage','pilot','--resume'],GEOMETRY,GEOMETRY/'pilot/report.json'),
 ('v3_full','data_gen.safer_v3_reconstruction_r3',['--stage','full','--resume'],GEOMETRY,GEOMETRY/'final_report.json'),
 ('v3_controls_smoke','fall_pipeline.safer.v3_controls_reconstruction_r3',['--stage','smoke'],Path(str(CONTROL)+'_smoke'),Path(str(CONTROL)+'_smoke')/'final_report.json'),
 ('v3_controls_full','fall_pipeline.safer.v3_controls_reconstruction_r3',['--stage','all'],CONTROL,CONTROL/'final_report.json'),
 ('v3_joint_smoke','fall_pipeline.joint.run_v3_reconstruction_r3',['--stage','smoke'],JOINT/'v3_smoke',JOINT/'v3_smoke/final_report.json'),
 ('v3_joint_full','fall_pipeline.joint.run_v3_reconstruction_r3',['--stage','all'],JOINT/'v3',JOINT/'v3/final_report.json')]


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8*1024**2): h.update(block)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(temporary,path)


def publish(stage,**details):
    state={'stage':stage,'time':datetime.now(timezone.utc).isoformat(),**details}
    save(QUEUE/'status.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)


def pause():
    if (QUEUE/'PAUSE_REQUESTED').exists(): raise RuntimeError('queue pause requested; no next stage')


def verify_report(path,pilot=False,smoke=False):
    r=read(path)
    if pilot:
        name=r.get('selected')
        if not (r.get('completed') and name and r.get('baseline_equality_passed') and r['gates'][name]['passed']):
            raise RuntimeError('V3 pilot not selected: preserve report, do not start full')
    else:
        if not r.get('passed') or (not smoke and not r.get('research_usable')): raise RuntimeError('stage did not pass full gate')
        if 'audit_sha256' in r:
            audit=path.parent/'independent_audit.json'
            if sha(audit)!=r['audit_sha256'] or not read(audit)['passed']: raise RuntimeError('stage audit mismatch')
        if 'geometry_gate' in r and not r['geometry_gate']['passed']: raise RuntimeError('V3 full geometry failed')
    return sha(path)


def dependency_ready():
    for lineage in ('v1','v2'):
        root=DEPENDENCY/lineage
        if (root/'status.json').exists() and read(root/'status.json')['stage'] in ('failed','paused'):
            raise RuntimeError('joint dependency failed/paused; no V3 start')
        if not (root/'final_report.json').exists(): return False
        verify_report(root/'final_report.json')
    result=subprocess.run(['tmux','-S',str(ROOT/'checkpoint/fall/joint_20260922.tmux.sock'),
                           'list-panes','-t','joint-full-v1-v2','-F','#{pane_dead} #{pane_dead_status}'],capture_output=True,text=True)
    if result.returncode or result.stdout.strip()!='1 0':
        if result.returncode or result.stdout.strip().startswith('1 '): raise RuntimeError('dependency tmux not a successful exit')
        return False
    return True


def fingerprint():
    # Pin all new/shared executable dependencies without pinning mutable documents.
    from data_gen import safer_v3_reconstruction_r3 as geometry
    from fall_pipeline.safer import v3_controls_reconstruction_r3 as controls
    from fall_pipeline.joint import run_v3_reconstruction_r3 as joint
    paths=set(geometry.CODE+controls.CODE+joint.CODE)
    paths.update(['scripts/run_recovery_v3_queue_r3.py','configs/safer_v3_document_reconstruction_v3.json',
                  'configs/safer_v3_controls_document_reconstruction_v3.json','configs/joint_v3_document_reconstruction_v3.json',
                  'configs/joint_document_reconstruction_v1.json','configs/safer_v2_controls_document_reconstruction_v1.json'])
    return {p:sha(ROOT/p) for p in sorted(paths)}


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if Path(sys.prefix).name!='fall_detect' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0': raise RuntimeError('fall_detect/GPU0 required')
    QUEUE.mkdir(parents=True,exist_ok=True);contract=fingerprint()
    if (QUEUE/'run_contract.json').exists():
        if not args.resume or read(QUEUE/'run_contract.json')!=contract: raise RuntimeError('queue contract/resume mismatch')
    else: save(QUEUE/'run_contract.json',contract)
    with (QUEUE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            publish('waiting_for_joint_v1_v2');deadline=time.monotonic()+24*3600
            while not dependency_ready():
                pause()
                if time.monotonic()>deadline: raise RuntimeError('dependency wait expired')
                time.sleep(15)
            for name,module,arguments,root,report in STAGES:
                pause()
                if fingerprint()!=contract: raise RuntimeError('queue pinned source changed')
                marker=QUEUE/(name+'.json')
                if marker.exists():
                    if sha(report)!=read(marker)['report_sha256']: raise RuntimeError('completed stage report changed')
                    verify_report(report,pilot=name=='v3_pilot',smoke=name.endswith('_smoke'));continue
                command=[sys.executable,'-m',module]+arguments
                if args.resume and (root/'run_contract.json').exists() and '--resume' not in command: command.append('--resume')
                log=ROOT/'logs'/f'recovery_20260922_queue_r3_{name}.log'
                publish('running',experiment=name,output=str(root),log=str(log))
                with log.open('a') as stream:
                    process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        if (QUEUE/'PAUSE_REQUESTED').exists():
                            # Child stops itself at a flushed sequence/batch/epoch boundary.
                            if root.exists(): (root/'PAUSE_REQUESTED').touch(exist_ok=True)
                        time.sleep(5)
                if process.returncode: raise RuntimeError(f'{name} exited {process.returncode}; next stages blocked')
                digest=verify_report(report,pilot=name=='v3_pilot',smoke=name.endswith('_smoke'))
                save(marker,{'passed':True,'report_sha256':digest});publish('stage_completed',experiment=name)
            publish('completed',scope='V3 geometry,matched controls,joint only; NOT entire historical recovery')
        except BaseException as error:
            publish('paused' if (QUEUE/'PAUSE_REQUESTED').exists() else 'failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__': main()
