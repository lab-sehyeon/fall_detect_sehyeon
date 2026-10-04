"""Separate-process frozen benchmark pipeline; stop immediately on any failure."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,os,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fall_pipeline.external.omnifall_cauca_20261002 import OUTPUT,ASSET
from fall_pipeline.external import rgb_document_io as io


def main():
    assert Path(sys.prefix).name=='fall_detect'
    assert io.read(ASSET/'video_manifest.json')['passed']
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.nice(10);logs=OUTPUT/'logs';logs.mkdir(exist_ok=True)
        env=os.environ.copy()
        env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                   MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        journal=io.read(OUTPUT/'runner_journal.json') if (OUTPUT/'runner_journal.json').exists() else []
        for stage in ['prepare','frontend','lift','infer','score','audit']:
            assert not (OUTPUT/'PAUSE_REQUESTED').exists(),'pause requested'
            gpu=stage in ['frontend','lift','infer']
            env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
            env['PYTHONPATH']=str(ROOT/'third_party/ViTPose')+':'+str(ROOT) if stage=='frontend' else str(ROOT)
            command=[sys.executable,'-m','fall_pipeline.external.omnifall_cauca_20261002',stage]
            if stage=='audit':command=[sys.executable,str(ROOT/'scripts/audit_omnifall_cauca_20261002.py')]
            started=datetime.now(timezone.utc).isoformat();print('start',stage,started,flush=True)
            with (logs/(stage+'.log')).open('a') as log:
                log.write('\nSTART '+started+'\n');log.flush()
                rc=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            record=dict(stage=stage,started=started,ended=datetime.now(timezone.utc).isoformat(),returncode=rc)
            journal.append(record);io.save(OUTPUT/'runner_journal.json',journal);print(record,flush=True)
            if rc:
                io.save(OUTPUT/'runner_failure.json',record)
                raise SystemExit(rc)
        assert io.read(OUTPUT/'status.json')['stage']=='completed'
        print('evaluation and independent audit completed',flush=True)


if __name__=='__main__':main()
