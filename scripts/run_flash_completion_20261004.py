"""Complete an already-running FLASH experiment without selecting on target scores."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from flash_reproduction_20261004 import ROOT,OUT as TRAIN,EVIDENCE,read,save,sha,require
from evaluate_flash_20261004 import OUT


def alive(pid):
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False


def status(stage,**extra):
    d=dict(stage=stage,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**extra)
    save(OUT/'completion_status.json',d)
    print(d,flush=True)


def run(script,stage=None):
    command=[sys.executable,str(ROOT/'scripts'/script)]
    if stage:command.append(stage)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MPLCONFIGDIR='/tmp/flash_mpl',PYTHONUNBUFFERED='1')
    name=Path(script).stem+('_'+stage if stage else '')
    status(name)
    with (EVIDENCE/(name+'.log')).open('w') as log:
        r=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    require(r.returncode==0,f'{name} failed with exit{r.returncode}; see {name}.log')


def main():
    p=argparse.ArgumentParser();p.add_argument('--training-pid',type=int,required=True);p.add_argument('--pose-pid',type=int,required=True)
    a=p.parse_args();started=time.monotonic();deadline=started+12*3600
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'completion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status('waiting_for_source_training_and_pose',training_pid=a.training_pid,pose_pid=a.pose_pid)
            while True:
                training_done=(TRAIN/'source_evaluation.json').exists()
                pose_done=(OUT/'pose_summary.json').exists()
                if training_done and pose_done:break
                require(training_done or alive(a.training_pid),'training process ended without final source evaluation')
                require(pose_done or alive(a.pose_pid),'pose process ended without complete manifest')
                require(time.monotonic()<deadline,'12-hour completion monitor timeout')
                save(OUT/'completion_heartbeat.json',dict(utc=datetime.now(timezone.utc).isoformat(),
                     source=read(TRAIN/'status.json'),external=read(OUT/'status.json')))
                time.sleep(15)
            require(read(TRAIN/'source_evaluation.json')['passed'],'source training not passed')
            require(read(OUT/'pose_summary.json')['passed'],'pose preparation not passed')
            run('evaluate_flash_20261004.py','infer')
            run('evaluate_flash_20261004.py','score')
            run('audit_flash_20261004.py')
            run('report_flash_20261004.py')
            require(read(EVIDENCE/'report_validation.json')['passed'],'report verification failed')
            status('completed',seconds=time.monotonic()-started,evaluation_sha256=sha(OUT/'evaluation.json'),
                   summaries=read(OUT/'evaluation.json')['summaries'])
        except Exception as exc:
            status('failed',error=str(exc),seconds=time.monotonic()-started)
            traceback.print_exc()
            internal=ROOT/'docs/internal/2026-10-04_flash_reproduction_internal.md'
            internal.write_text(internal.read_text().replace('상태: `in_progress`','상태: `paused`',1)+
                                '\n## 자동 실행 중단\n\n'+str(exc)+'\n최종 성공 상태와 비교 결과로 제시하지 않는다. 로그 확인 후 재개한다.\n')
            shared=ROOT/'docs/shared/2026-10-04_flash_reproduction_shared.md'
            shared.write_text(shared.read_text().replace('상태: `in_progress`','상태: `paused`',1)+
                              '\n학습·평가·독립 검산의 전체 완료가 확인되지 않아 새 외부 성능을 확정하지 않았다.\n')
            raise


if __name__=='__main__':main()
