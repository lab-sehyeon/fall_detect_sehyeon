"""The active J1+G0 evaluation entrypoint, with isolated GPU0 inference."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,os,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fall_pipeline.external import j1_g0_evaluation_20261002 as run
from fall_pipeline.external import rgb_document_io as io


def main():
    io.require(Path(sys.prefix).name=='fall_detect','fall_detect required')
    active=io.read(ROOT/'configs/active_fall_model.json')
    io.require(active['architecture']=='DSTE_J1_G0_ONLY' and active['heads']==['G0'],'active model mismatch')
    io.require(ROOT/active['evaluation_config']==run.CONFIG,'active configuration mismatch')
    run.OUTPUT.mkdir(parents=True,exist_ok=True)
    with (run.OUTPUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.nice(10)
        logs=run.OUTPUT/'logs';logs.mkdir(exist_ok=True)
        env=os.environ.copy();env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT))
        journal=io.read(run.OUTPUT/'runner_journal.json') if (run.OUTPUT/'runner_journal.json').exists() else []
        for stage in ['prepare','infer','score','audit']:
            io.require(not (run.OUTPUT/'PAUSE_REQUESTED').exists(),'pause requested')
            env['CUDA_VISIBLE_DEVICES']='0' if stage=='infer' else ''
            cmd=[sys.executable,'-m','fall_pipeline.external.j1_g0_evaluation_20261002',stage]
            if stage=='audit':cmd=[sys.executable,str(ROOT/'scripts/audit_j1_g0_evaluation_20261002.py')]
            started=datetime.now(timezone.utc).isoformat();print('start',stage,started,flush=True)
            with (logs/(stage+'.log')).open('a') as log:
                log.write('\nSTART '+started+'\n');log.flush()
                rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            record=dict(stage=stage,started=started,ended=datetime.now(timezone.utc).isoformat(),returncode=rc)
            journal.append(record);io.save(run.OUTPUT/'runner_journal.json',journal);print(record,flush=True)
            if rc:io.save(run.OUTPUT/'runner_failure.json',record);raise SystemExit(rc)
        io.require(io.read(run.OUTPUT/'status.json')['stage']=='completed','audit not completed')


if __name__=='__main__':main()
