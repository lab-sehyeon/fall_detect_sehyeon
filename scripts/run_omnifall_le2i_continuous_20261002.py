"""Separate frozen stages, exclusive lock, GPU0 only, stop on failure."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fall_pipeline.external.omnifall_le2i_continuous_20261002 import OUTPUT, PREP
from fall_pipeline.external import rgb_document_io as io


def main():
    io.require(Path(sys.prefix).name=='fall_detect','fall_detect required')
    io.require(io.read(PREP/'manifest.json')['passed'],'prepared input missing')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.nice(10); logs=OUTPUT/'logs';logs.mkdir(exist_ok=True)
        env=os.environ.copy()
        env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                   MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        journal=io.read(OUTPUT/'runner_journal.json') if (OUTPUT/'runner_journal.json').exists() else []
        for stage in ['prepare','frontend','lift','infer','score','audit']:
            io.require(not (OUTPUT/'PAUSE_REQUESTED').exists(),'pause requested')
            env['CUDA_VISIBLE_DEVICES']='0' if stage in ['frontend','lift','infer'] else ''
            env['PYTHONPATH']=str(ROOT/'third_party/ViTPose')+':'+str(ROOT) if stage=='frontend' else str(ROOT)
            cmd=[sys.executable,'-m','fall_pipeline.external.omnifall_le2i_continuous_20261002',stage]
            if stage=='audit':cmd=[sys.executable,str(ROOT/'scripts/audit_omnifall_le2i_continuous_20261002.py')]
            started=datetime.now(timezone.utc).isoformat(); print('start',stage,started,flush=True)
            with (logs/(stage+'.log')).open('a') as log:
                log.write('\nSTART '+started+'\n');log.flush()
                rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            record=dict(stage=stage,started=started,ended=datetime.now(timezone.utc).isoformat(),returncode=rc)
            journal.append(record);io.save(OUTPUT/'runner_journal.json',journal);print(record,flush=True)
            if rc:
                io.save(OUTPUT/'runner_failure.json',record);raise SystemExit(rc)
        io.require(io.read(OUTPUT/'status.json')['stage']=='completed','audit incomplete')


if __name__=='__main__':main()
