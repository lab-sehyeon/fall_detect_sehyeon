"""Attach to the identified S0-B worker, then CPU-only S0-C smoke/full, fail closed."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from fall_pipeline.safer import state_followup_io as s
from fall_pipeline.safer import run_s0b_reconstruction as b
from fall_pipeline.safer import run_s0c_reconstruction as c

io=s.io
ROOT=io.ROOT/'checkpoint/fall/STATE_RECOVERY_QUEUE_20260929_R1'
BROOT=io.ROOT/'checkpoint/fall/S0B_DOCUMENT_RECONSTRUCTION_20260929_R1'
CROOT=io.ROOT/'checkpoint/fall/S0C_DOCUMENT_RECONSTRUCTION_20260929_R1'
GTROOT=io.ROOT/'data/fall_processed/SAFER-Activities/s0gt0_document_reconstruction_20260929_r1'
CODE=c.CODE+['scripts/run_state_recovery_queue.py','configs/s0c_document_reconstruction_v1.json']


def identity(pid):
    process=Path('/proc')/str(pid)
    try:
        cmd=(process/'cmdline').read_bytes().split(b'\0')
        raw=(process/'stat').read_text();start=raw[raw.rfind(')')+2:].split()[19]
    except FileNotFoundError:return None
    return {'pid':pid,'start_ticks':start,'command':[x.decode() for x in cmd if x]}


def verify_b():
    cfg=io.read(b.CONFIG)
    io.require(b.make_contract(cfg,False)==io.read(BROOT/'run_contract.json'),'B source/contract changed')
    report=io.read(BROOT/'final_report.json')
    io.require(report['passed'] and report['research_usable'],'B integrity failed')
    io.require(io.sha256_file(BROOT/'final_report.json')==io.read(BROOT/'status.json')['final_report_sha256'],'B report changed')
    io.require(io.sha256_file(BROOT/'selection.json')==report['selection_sha256'],'B selection changed')
    # Adoption is a research result, not an integrity prerequisite for matched S0-C comparison.
    return io.sha256_file(BROOT/'final_report.json')


def state(stage,**details):
    io.save_json(ROOT/'status.json',{'stage':stage,**details})
    print({'stage':stage,**details},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--b-pid',type=int,required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    io.require(Path(sys.prefix).name=='fall_detect' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','queue CPU-only')
    ROOT.mkdir(parents=True,exist_ok=True);s.install_signals()
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=ROOT/'run_contract.json'
        process=identity(args.b_pid)
        if path.exists():
            contract=io.read(path)
            io.require(args.resume and contract['source']==s.fingerprint(CODE),'queue resume changed')
            io.require(contract['worker']['pid']==args.b_pid,'worker PID changed')
        else:
            io.require(process is not None and 'fall_pipeline.safer.run_s0b_reconstruction' in process['command'],'not the S0-B worker')
            contract={'source':s.fingerprint(CODE),'worker':process,'b_contract_sha256':io.sha256_file(BROOT/'run_contract.json'),
                      'gt0_report_sha256':io.sha256_file(GTROOT/'final_report.json'),'maximum_wait_hours':24}
            s.lock_contract(ROOT,contract,False)
        config=io.read(c.CONFIG)
        try:
            io.require(io.read(GTROOT/'final_report.json')['passed'],'GT0 prerequisite failed')
            state('waiting_for_s0b',pid=args.b_pid)
            deadline=time.monotonic()+24*3600
            while not (BROOT/'final_report.json').exists():
                s.pause(ROOT,config)
                io.require(io.sha256_file(BROOT/'run_contract.json')==contract['b_contract_sha256'],'B contract changed')
                io.require(identity(args.b_pid)==contract['worker'],'B worker exited/replaced before completion')
                io.require(io.read(BROOT/'status.json')['stage'] not in ('failed','paused'),'B stopped; no downstream run')
                io.require(time.monotonic()<deadline,'B wait limit reached')
                time.sleep(10)
            io.save_json(ROOT/'s0b_complete.json',{'report_sha256':verify_b()})
            for smoke in (True,False):
                s.pause(ROOT,config)
                io.require(s.fingerprint(CODE)==contract['source'],'queue source changed')
                io.require(io.sha256_file(GTROOT/'final_report.json')==contract['gt0_report_sha256'],'GT0 report changed')
                root=Path(str(CROOT)+'_smoke') if smoke else CROOT
                marker=ROOT/('s0c_smoke.json' if smoke else 's0c_full.json')
                if marker.exists():
                    io.require(io.sha256_file(root/'final_report.json')==io.read(marker)['report_sha256'],'C completion changed');continue
                command=[sys.executable,'-m','fall_pipeline.safer.run_s0c_reconstruction']
                if smoke:command.append('--smoke')
                if args.resume and (root/'run_contract.json').exists():command.append('--resume')
                log=io.ROOT/'logs'/('recovery_20260929_s0c_smoke.log' if smoke else 'recovery_20260929_s0c_full.log')
                state('running_s0c',smoke=smoke)
                with log.open('a') as stream:
                    child=subprocess.Popen(command,cwd=io.ROOT,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
                    while child.poll() is None:
                        if s.STOP or (ROOT/'PAUSE_REQUESTED').exists():
                            if root.exists():(root/'PAUSE_REQUESTED').touch(exist_ok=True)
                        time.sleep(5)
                io.require(child.returncode==0,f'S0-C returned {child.returncode}; preserve and inspect')
                report=io.read(root/'final_report.json')
                io.require(report['passed'] and report['research_usable'] is (not smoke),'C final gate')
                io.save_json(marker,{'report_sha256':io.sha256_file(root/'final_report.json')})
            state('completed',scope='S0-B/S0-C and preceding GT0 only, not all historical recovery')
        except io.PauseRequested as error:
            if not (BROOT/'final_report.json').exists():(BROOT/'PAUSE_REQUESTED').touch(exist_ok=True)
            state('paused',reason=str(error));raise SystemExit(75)
        except Exception as error:
            state('failed',reason=str(error));raise


if __name__=='__main__':main()
