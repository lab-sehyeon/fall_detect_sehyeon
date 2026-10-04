"""Sequential GPU0 evaluation after independently prepared data; stops on errors."""
from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/fall_processed/RGB/three_by_three_20261003_r1'
SOURCES=ROOT/'docs/internal/2026-10-03_three_by_three_sources'
PYTHON=sys.executable

def mark(stage,**kwargs):
    value=dict(stage=stage,**kwargs);p=OUT/'orchestrator_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,p);print(json.dumps(value),flush=True)

def wait_for(path,dataset):
    while not path.exists():
        if (OUT/'PAUSE_REQUESTED').exists():raise RuntimeError('pause requested')
        mark('waiting_for_preparation',dataset=dataset,file=str(path.relative_to(ROOT)));time.sleep(20)

def run(dataset,stage,audit=False):
    if audit:cmd=[PYTHON,'scripts/audit_three_by_three_20261003.py',dataset]
    else:cmd=[PYTHON,'-u','scripts/run_three_by_three_20261003.py',stage,'--dataset',dataset]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='' if audit else '0';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['MPLCONFIGDIR']=str(OUT/'matplotlib_runtime')
    log=SOURCES/f'{dataset}_{stage}.log';mark('running',dataset=dataset,step=stage)
    with log.open('a') as stream:subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    mark('step_completed',dataset=dataset,step=stage)

try:
    for dataset in ['OOPS','edf','occu']:
        if (OUT/dataset/'independent_audit.json').exists():continue
        if dataset=='OOPS':wait_for(OUT/dataset/'contract.json',dataset)
        else:
            wait_for(ROOT/f'data/source_archives/ThreeByThree_20261003/{dataset}_manifest.json',dataset)
            if not (OUT/dataset/'contract.json').exists():run(dataset,'prepare')
        for stage in ['frontend','lift','infer','comparisons','score']:run(dataset,stage)
        run(dataset,'audit',True)
    subprocess.run([PYTHON,'scripts/report_three_by_three_20261003.py'],cwd=ROOT,check=True)
    mark('completed')
except Exception as e:
    mark('failed',error=str(e));raise
